package latentimage

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestResidentImagePipelineRetainsTextAndLatentOnDevice(t *testing.T) {
	cudatest.Require(t)
	const (
		fixtureSeq   = 3
		fixtureGrid  = 1
		fixturePatch = 2
		fixtureSigma = 0.9
		fixtureDelta = -0.125
	)
	encoderSpec := syntheticEncoderSpec()
	transformerSpec := syntheticImageTransformerSpec(encoderSpec)
	encoderWeights := encoderStore(encoderSpec)
	transformerWeights := syntheticStore(transformerSpec)
	modelDir := t.TempDir()
	writePipelineWeights(t, filepath.Join(modelDir, "text_encoder"), encoderWeights, EncoderTensorShapes(encoderSpec))
	writePipelineWeights(t, filepath.Join(modelDir, "transformer"), transformerWeights, DenoiserTensorShapes(transformerSpec))

	encoder, err := CompileEncoderProgram(
		encoderSpec, float32(encoderSpec.RMSNormEps), fixtureSeq, dtype.F32,
	)
	if err != nil {
		t.Fatal(err)
	}
	fusion, err := CompileFusionProgram(transformerSpec, 1e-5, attendedTextMask(fixtureSeq), dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	denoiser, err := CompileDenoiserProgram(
		transformerSpec, 1e-5, attendedTextMask(fixtureSeq), fixtureGrid, fixtureGrid, dtype.F32,
	)
	if err != nil {
		t.Fatal(err)
	}
	vaeDecoder := syntheticVAEDecoder(t)
	vaeExtent := fixtureGrid * fixturePatch
	vae, err := CompileVAEProgram(vaeDecoder, vaeExtent, vaeExtent, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	pipeline, err := NewResidentImagePipeline(ctx, encoder, fusion, denoiser, vae, modelDir, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := pipeline.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	}()
	embed := f32slice(syntheticEncoderEmbed(fixtureSeq, encoderSpec.Hidden))
	conditioning, err := pipeline.Condition(ctx, embed)
	if err != nil {
		t.Fatal(err)
	}
	defer conditioning.Release(ctx)
	initial := make([]float32, transformerSpec.InChannels)
	for index := range initial {
		initial[index] = float32(index+1) / float32(len(initial))
	}
	if err := pipeline.Begin(ctx, initial); err != nil {
		t.Fatal(err)
	}
	if err := pipeline.Advance(ctx, fixtureSigma, fixtureDelta); err != nil {
		t.Fatal(err)
	}
	got, err := pipeline.Latent(ctx)
	if err != nil {
		t.Fatal(err)
	}

	hidden, err := encoder.RunHostFeed(
		GraphRunner(reference.Execute), func(name string) ([]float32, error) { return encoderWeights[name], nil }, embed,
	)
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewDenoiser(transformerSpec, 1e-5, numTrainTimesteps, transformerWeights)
	if err != nil {
		t.Fatal(err)
	}
	velocity, err := denoiser.Forward(
		GraphRunner(reference.Execute), host, f64of(initial), hidden.Data, fixtureSigma,
	)
	if err != nil {
		t.Fatal(err)
	}
	const deviceTolerance = 5e-3
	for index, value := range got {
		want := initial[index] + fixtureDelta*velocity.Velocity[index]
		if delta := math.Abs(float64(value - want)); delta > deviceTolerance {
			t.Fatalf("latent[%d] = %g, want %g (delta %g)", index, value, want, delta)
		}
	}
	pixels, height, width, err := pipeline.Decode(ctx)
	if err != nil {
		t.Fatal(err)
	}
	unpacked, err := unpackLatent(got, vaeDecoder.ZDim, fixtureGrid, fixtureGrid, fixturePatch)
	if err != nil {
		t.Fatal(err)
	}
	wantPixels, wantHeight, wantWidth, err := vaeDecoder.DecodeImage(unpacked, vaeExtent, vaeExtent)
	if err != nil {
		t.Fatal(err)
	}
	if height != wantHeight || width != wantWidth {
		t.Fatalf("decoded geometry = %dx%d, want %dx%d", width, height, wantWidth, wantHeight)
	}
	const vaeTolerance = 5e-3
	for index, value := range pixels {
		if delta := math.Abs(float64(value - wantPixels[index])); delta > vaeTolerance {
			t.Fatalf("pixel[%d] = %g, want %g (delta %g)", index, value, wantPixels[index], delta)
		}
	}
}

func syntheticImageTransformerSpec(encoder TextEncoderSpec) TransformerSpec {
	const transformerHeadDim = 8
	return TransformerSpec{
		Layers: 1, Heads: 4, KVHeads: 2, HeadDim: transformerHeadDim,
		Hidden: 4 * transformerHeadDim, KVDim: 2 * transformerHeadDim, InChannels: 16, Intermediate: 8,
		RopeAxes: [3]int{2, 2, 4}, RopeTheta: 1000, NormEps: 1e-5, TimestepEmbed: 8, ModFields: 6,
		TextLayers: len(encoder.SelectLayers), TextHidden: encoder.Hidden, TextIntermediate: 8,
		TextHeads: encoder.Hidden / transformerHeadDim, TextKVHeads: 1, LayerwiseTextBlocks: 1, RefinerTextBlocks: 1,
	}
}

func writePipelineWeights(t *testing.T, directory string, weights map[string][]float32, shapes map[string][]int) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
}

func TestSelectedHiddenBridgePreservesTokenLayerOrder(t *testing.T) {
	const (
		fixtureHidden = 2
		fixtureSeq    = 2
		fixtureLayers = 3
	)
	builder := tensor.NewBuilder()
	selected := make([]*tensor.Tensor, fixtureLayers)
	for index := range selected {
		selected[index] = builder.Input(
			"selected", dtype.F32, tensor.MustShape(fixtureHidden, fixtureSeq),
		)
	}
	program := &EncoderProgram{
		E: TextEncoderSpec{Hidden: fixtureHidden}, Seq: fixtureSeq, Selected: selected,
	}
	inputs, output, err := selectedHiddenBridge(program, fixtureSeq)
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, fixtureLayers)
	for layer, input := range inputs {
		data := make([]float32, fixtureHidden*fixtureSeq)
		for token := range fixtureSeq {
			for feature := range fixtureHidden {
				data[token*fixtureHidden+feature] = float32(
					layer*fixtureSeq*fixtureHidden + token*fixtureHidden + feature,
				)
			}
		}
		feeds[input] = reference.Value{Shape: input.Shape, Data: data}
	}
	results, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	got := results[output].Data
	want := []float32{0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11}
	if !equalFloat32(got, want) {
		t.Fatalf("bridge output = %v, want %v", got, want)
	}
}

func TestSelectedHiddenBridgeDropsEncoderPrefix(t *testing.T) {
	const (
		fixtureHidden = 2
		fixtureSeq    = 3
		outputSeq     = 2
	)
	builder := tensor.NewBuilder()
	selected := builder.Input("selected", dtype.F32, tensor.MustShape(fixtureHidden, fixtureSeq))
	program := &EncoderProgram{
		E: TextEncoderSpec{Hidden: fixtureHidden}, Seq: fixtureSeq,
		Selected: []*tensor.Tensor{selected},
	}
	inputs, output, err := selectedHiddenBridge(program, outputSeq)
	if err != nil {
		t.Fatal(err)
	}
	results, err := reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		inputs[0]: {Shape: inputs[0].Shape, Data: []float32{0, 1, 2, 3, 4, 5}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 3, 4, 5}
	if !equalFloat32(results[output].Data, want) {
		t.Fatalf("cropped output = %v, want %v", results[output].Data, want)
	}
}

func equalFloat32(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
