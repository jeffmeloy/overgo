package latentimage

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
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

	encoder := testcheck.Must(CompileEncoderProgram(
		encoderSpec, float32(encoderSpec.RMSNormEps), fixtureSeq, dtype.F32,
	))(t)
	fusion := testcheck.Must(CompileFusionProgram(transformerSpec, 1e-5, attendedTextMask(fixtureSeq), dtype.F32))(t)
	denoiser := testcheck.Must(CompileDenoiserProgram(
		transformerSpec, 1e-5, attendedTextMask(fixtureSeq), fixtureGrid, fixtureGrid, dtype.F32,
	))(t)
	vaeDecoder := syntheticVAEDecoder(t)
	vaeExtent := fixtureGrid * fixturePatch
	vae := testcheck.Must(CompileVAEProgram(vaeDecoder, vaeExtent, vaeExtent, dtype.F32))(t)
	ctx := t.Context()
	pipeline := testcheck.Must(NewResidentImagePipeline(ctx, encoder, fusion, denoiser, vae, fixtureTimestepProgram(transformerSpec), modelDir, 0))(t)
	defer closeReporting(t, ctx, pipeline)
	embed := f32slice(syntheticEncoderEmbed(fixtureSeq, encoderSpec.Hidden))
	conditioning := testcheck.Must(pipeline.Condition(ctx, "a fixture", embed))(t)
	defer conditioning.Release(ctx)
	initial := make([]float32, transformerSpec.InChannels)
	for index := range initial {
		initial[index] = float32(index+1) / float32(len(initial))
	}
	testcheck.NoError(t, pipeline.Begin(ctx, initial))
	testcheck.NoError(t, pipeline.Advance(ctx, fixtureSigma, fixtureDelta))
	got := testcheck.Must(pipeline.Latent(ctx))(t)

	hidden := testcheck.Must(encoder.RunHostFeed(
		reference.Execute, func(name string) ([]float32, error) { return encoderWeights[name], nil }, embed,
	))(t)
	host := testcheck.Must(NewDenoiser(transformerSpec, 1e-5, fixtureTimestepProgram(transformerSpec), transformerWeights))(t)
	velocity := testcheck.Must(denoiser.Forward(
		reference.Execute, host, f64of(initial), hidden.Data, fixtureSigma,
	))(t)
	const deviceTolerance = 5e-3
	for index, value := range got {
		want := initial[index] + fixtureDelta*velocity.Velocity[index]
		if delta := math.Abs(float64(value - want)); !(delta <= deviceTolerance) {
			t.Fatalf("latent[%d] = %g, want %g (delta %g)", index, value, want, delta)
		}
	}
	pixels, height, width, err := pipeline.DecodeHWC(ctx)
	testcheck.NoError(t, err)
	unpacked := testcheck.Must(media.UnpackPlanar(got, vaeDecoder.ZDim, fixtureGrid, fixtureGrid, fixturePatch, media.PatchChannelsFirst))(t)
	wantPixels, wantHeight, wantWidth, err := vaeDecoder.DecodeImage(unpacked, vaeExtent, vaeExtent)
	testcheck.NoError(t, err)
	if height != wantHeight || width != wantWidth {
		t.Fatalf("decoded geometry = %dx%d, want %dx%d", width, height, wantWidth, wantHeight)
	}
	const vaeTolerance = 5e-3
	plane := height * width
	for position := range plane {
		for channel := range vaeDecoder.OutChannels {
			gotIndex, wantIndex := position*vaeDecoder.OutChannels+channel, channel*plane+position
			if delta := math.Abs(float64(pixels[gotIndex] - wantPixels[wantIndex])); !(delta <= vaeTolerance) {
				t.Fatalf("pixel[%d,%d] = %g, want %g (delta %g)", position, channel, pixels[gotIndex], wantPixels[wantIndex], delta)
			}
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
	testcheck.NoError(t, os.MkdirAll(directory, 0o700))
	testcheck.NoError(t, safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil))
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
	testcheck.NoError(t, err)
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
	results := testcheck.Must(reference.Execute([]*tensor.Tensor{output}, feeds))(t)
	got := results[output].Data
	want := []float32{0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11}
	if !slices.Equal(got, want) {
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
	testcheck.NoError(t, err)
	results := testcheck.Must(reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		inputs[0]: {Shape: inputs[0].Shape, Data: []float32{0, 1, 2, 3, 4, 5}},
	}))(t)
	want := []float32{2, 3, 4, 5}
	if !slices.Equal(results[output].Data, want) {
		t.Fatalf("cropped output = %v, want %v", results[output].Data, want)
	}
}
