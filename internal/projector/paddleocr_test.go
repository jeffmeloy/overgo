package projector

import (
	"context"
	"image"
	"image/color"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

type paddleOCRPromptTokenizer struct{}

func (paddleOCRPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	const placeholder = "\x00"
	text = strings.ReplaceAll(text, PaddleOCRImagePad, placeholder)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for _, value := range text {
		if value == 0 {
			ids = append(ids, 9001)
		} else {
			ids = append(ids, tokenizer.TokenID(value))
		}
	}
	return ids, nil
}

func TestPaddleOCRRunnerTinyFixture(t *testing.T) {
	path := writeTinyPaddleOCR(t, tinyPaddleOCRTensors())
	runner, err := openImageProjectorAs[*PaddleOCRRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	output, err := runner.EncodeImage(context.Background(), input, RasterPatchOptions{
		MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(6, 1)) {
		t.Fatalf("output shape = %v", output.Embeddings.Shape)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	if !slices.Equal(output.Embeddings.Data, want) {
		t.Fatalf("output = %v, want %v", output.Embeddings.Data, want)
	}
	if output.GridH != 2 || output.GridW != 2 || output.MergeSize != fixtureSpatialMerge {
		t.Fatalf("output grid = %d,%d merge=%d", output.GridH, output.GridW, output.MergeSize)
	}
}

func TestOpenImageProjectorDispatchesPaddleOCR(t *testing.T) {
	projector, err := OpenAs[Projector](context.Background(), writeTinyPaddleOCR(t, tinyPaddleOCRTensors()), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*PaddleOCRRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestPaddleOCRMultipleImagePromptAndPositions(t *testing.T) {
	runner, err := openImageProjectorAs[*PaddleOCRRunner](writeTinyPaddleOCR(t, tinyPaddleOCRTensors()), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	first := image.NewRGBA(image.Rect(0, 0, 4, 4))
	second := image.NewRGBA(image.Rect(0, 0, 8, 4))
	prompt, err := testSession(t, runner).BuildImagesPrompt(
		context.Background(), paddleOCRPromptTokenizer{}, []image.Image{first, second},
		[]string{"OCR:", " and ", "Table Recognition:"}, PromptOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if prompt.EmbeddingWidth != 6 || len(prompt.Embeddings) != 18 {
		t.Fatalf("embedding contract = width %d values %d", prompt.EmbeddingWidth, len(prompt.Embeddings))
	}
	if len(prompt.EmbeddingTokenIndices) != 3 {
		t.Fatalf("embedding indices = %v", prompt.EmbeddingTokenIndices)
	}
	for axis := range prompt.MultiAxisPositions {
		if len(prompt.MultiAxisPositions[axis]) != len(prompt.TokenIDs) {
			t.Fatalf("axis %d positions = %d, tokens = %d", axis, len(prompt.MultiAxisPositions[axis]), len(prompt.TokenIDs))
		}
	}
	secondStart := int(prompt.EmbeddingTokenIndices[1])
	base := prompt.MultiAxisPositions[0][secondStart]
	if got := []uint32{
		prompt.MultiAxisPositions[0][secondStart] - base,
		prompt.MultiAxisPositions[1][secondStart] - base,
		prompt.MultiAxisPositions[2][secondStart] - base,
		prompt.MultiAxisPositions[0][secondStart+1] - base,
		prompt.MultiAxisPositions[1][secondStart+1] - base,
		prompt.MultiAxisPositions[2][secondStart+1] - base,
	}; !slices.Equal(got, []uint32{0, 0, 0, 0, 0, 1}) {
		t.Fatalf("second image coordinates = %v", got)
	}
}

func TestPaddleOCRCatalogRejectsIncompletePreNorm(t *testing.T) {
	tensors := append(tinyPaddleOCRTensors(), f32Tensor("v.pre_ln.weight", []uint64{4}, []float32{1, 1, 1, 1}))
	path := writeTinyPaddleOCR(t, tensors)
	if _, err := openImageProjectorAs[*PaddleOCRRunner](path, OpenOptions{}); err == nil || !strings.Contains(err.Error(), "must be paired") {
		t.Fatalf("error = %v", err)
	}
}

func TestPaddleOCRCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyPaddleOCR(t, nonzeroTinyPaddleOCRTensors())
	cpu, err := openImageProjectorAs[*PaddleOCRRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := openImageProjectorAs[*PaddleOCRRunner](path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 8, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 25), G: uint8(y * 50), B: 80, A: fixtureOpaqueAlpha})
		}
	}
	options := RasterPatchOptions{MinPixels: fixtureMediumPixelBudget, MaxPixels: fixtureMediumPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio}
	want, err := cpu.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "PaddleOCR", got.Embeddings.Data, want.Embeddings.Data, 2e-3)
}

func writeTinyPaddleOCR(t *testing.T, tensors []gguf.TensorData) string {
	return writeProjectorFixture(t, "paddleocr-mmproj.gguf", tinyPaddleOCRMetadata(), tensors)
}

func tinyPaddleOCRMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: paddleOCRProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(8)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(6)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: visionMinPixelsKey, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(fixtureSmallPixelBudget)}},
		{Key: visionMaxPixelsKey, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(fixtureLargePixelBudget)}},
		{Key: visionSpatialMergeKey, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(fixtureSpatialMerge)}},
		{Key: visionRopeFrequencyKey, Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: fixtureRopeFrequency}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
	}
}

func tinyPaddleOCRSpec() PaddleOCRSpec {
	return PaddleOCRSpec{
		visionBackboneSpec: fixtureVisionBackbone(4, 2, 4, 8, 1, 1), ProjectorIntermediate: 16,
		OutputHidden: 6, MergeSize: fixtureSpatialMerge, MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureLargePixelBudget,
		FusedQKV: []bool{true},
	}
}

func tinyPaddleOCRTensors() []gguf.TensorData {
	tensors := []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{2, 2, 3, 4}, nil),
		f32Tensor("v.patch_embd.bias", []uint64{4}, nil),
		f32Tensor("v.position_embd.weight", []uint64{4, 4}, nil),
		f32Tensor("mm.input_norm.weight", []uint64{4}, []float32{1, 1, 1, 1}),
		f32Tensor("mm.input_norm.bias", []uint64{4}, nil),
		f32Tensor("mm.1.weight", []uint64{16, 16}, nil),
		f32Tensor("mm.1.bias", []uint64{16}, nil),
		f32Tensor("mm.2.weight", []uint64{16, 6}, nil),
		f32Tensor("mm.2.bias", []uint64{6}, []float32{1, 2, 3, 4, 5, 6}),
	}
	for name, shape := range map[string][]uint64{
		"attn_qkv.weight": {4, 12}, "attn_qkv.bias": {12},
		"attn_out.weight": {4, 4}, "attn_out.bias": {4},
		"ffn_up.weight": {4, 8}, "ffn_up.bias": {8},
		"ffn_down.weight": {8, 4}, "ffn_down.bias": {4},
		"ln1.weight": {4}, "ln1.bias": {4}, "ln2.weight": {4}, "ln2.bias": {4},
	} {
		values := []float32(nil)
		if name == "ln1.weight" || name == "ln2.weight" {
			values = []float32{1, 1, 1, 1}
		}
		tensors = append(tensors, f32Tensor("v.blk.0."+name, shape, values))
	}
	return tensors
}

func nonzeroTinyPaddleOCRTensors() []gguf.TensorData {
	base := tinyPaddleOCRTensors()
	result := make([]gguf.TensorData, len(base))
	for tensorIndex, item := range base {
		elements := uint64(1)
		for _, dimension := range item.Shape {
			elements *= dimension
		}
		values := make([]float32, int(elements))
		isNorm := strings.HasSuffix(item.Name, "ln.weight") || strings.Contains(item.Name, "ln1.weight") ||
			strings.Contains(item.Name, "ln2.weight") || strings.Contains(item.Name, "input_norm.weight")
		for index := range values {
			value := float32((index+tensorIndex*3)%13-6) * 0.0075
			if isNorm {
				value += 1
			}
			values[index] = value
		}
		result[tensorIndex] = f32Tensor(item.Name, item.Shape, values)
	}
	return result
}
