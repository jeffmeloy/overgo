package projector

import (
	"context"
	"image"
	"image/color"
	"slices"
	"strings"
	"testing"

	cudatest "llamacpp2go/internal/cuda/testutil"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

type hunyuanVLPromptTokenizer struct{}

func (hunyuanVLPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	const placeholder = "\x00"
	text = strings.ReplaceAll(text, HunyuanVLImagePad, placeholder)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for _, value := range text {
		if value == 0 {
			ids = append(ids, 9002)
		} else {
			ids = append(ids, tokenizer.TokenID(value))
		}
	}
	return ids, nil
}

func TestHunyuanVLRunnerTinyFixture(t *testing.T) {
	path := writeTinyHunyuanVL(t, tinyHunyuanVLTensors())
	runner, err := OpenHunyuanVL(path)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 4, 4)), HunyuanVLPreprocessOptions{
		MinPixels: 16, MaxPixels: 16, MaxAspectRatio: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(6, 4)) {
		t.Fatalf("output shape = %v", output.Embeddings.Shape)
	}
	if !slices.Equal(output.Embeddings.Data, make([]float32, 24)) {
		t.Fatalf("output = %v", output.Embeddings.Data)
	}
	if output.GridH != 2 || output.GridW != 2 || output.MergeSize != 2 {
		t.Fatalf("output grid = %d,%d merge=%d", output.GridH, output.GridW, output.MergeSize)
	}
}

func TestOpenImageProjectorDispatchesHunyuanVL(t *testing.T) {
	projector, err := OpenImageProjector(writeTinyHunyuanVL(t, tinyHunyuanVLTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*HunyuanVLRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestHunyuanVLMultipleImagePromptAndPositions(t *testing.T) {
	runner, err := OpenHunyuanVL(writeTinyHunyuanVL(t, tinyHunyuanVLTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	prompt, err := runner.BuildImagesPrompt(
		context.Background(), hunyuanVLPromptTokenizer{},
		[]image.Image{image.NewRGBA(image.Rect(0, 0, 4, 4)), image.NewRGBA(image.Rect(0, 0, 8, 4))},
		[]string{"first ", " then ", " question"}, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if prompt.EmbeddingWidth != 6 || len(prompt.Embeddings) != 9*6 || len(prompt.EmbeddingTokenIndices) != 9 {
		t.Fatalf("embedding contract = width %d values %d indices %d", prompt.EmbeddingWidth, len(prompt.Embeddings), len(prompt.EmbeddingTokenIndices))
	}
	for axis := range prompt.MultiAxisPositions {
		if len(prompt.MultiAxisPositions[axis]) != len(prompt.TokenIDs) {
			t.Fatalf("axis %d positions = %d, tokens = %d", axis, len(prompt.MultiAxisPositions[axis]), len(prompt.TokenIDs))
		}
	}
	start := int(prompt.EmbeddingTokenIndices[4])
	base := prompt.MultiAxisPositions[0][start]
	want := [][4]uint32{
		{0, base, base, base}, {1, 0, 0, 1}, {2, 1, 0, 1}, {3, 2, 0, 1}, {4, base + 4, base + 4, base + 4},
	}
	for offset, expected := range want {
		got := [4]uint32{
			prompt.MultiAxisPositions[0][start+offset] - base,
			prompt.MultiAxisPositions[1][start+offset],
			prompt.MultiAxisPositions[2][start+offset],
			prompt.MultiAxisPositions[3][start+offset],
		}
		if got != expected {
			t.Fatalf("second image position %d = %v, want %v", offset, got, expected)
		}
	}
}

func TestPreprocessHunyuanVLImageRasterPatchOrder(t *testing.T) {
	spec := tinyHunyuanVLSpec()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(y*16 + x), A: 255})
		}
	}
	processed, err := PreprocessHunyuanVLImage(input, spec, HunyuanVLPreprocessOptions{MinPixels: 16, MaxPixels: 16, MaxAspectRatio: 10})
	if err != nil {
		t.Fatal(err)
	}
	patchArea := spec.PatchSize * spec.PatchSize
	redStarts := []float32{
		processed.PixelValues[0], processed.PixelValues[3*patchArea],
		processed.PixelValues[6*patchArea], processed.PixelValues[9*patchArea],
	}
	want := []float32{0, 2.0 / 255, 32.0 / 255, 34.0 / 255}
	for index := range want {
		if difference := redStarts[index] - want[index]; difference < -1e-6 || difference > 1e-6 {
			t.Fatalf("patch %d red start = %g, want %g", index, redStarts[index], want[index])
		}
	}
}

func TestHunyuanVLCatalogRejectsIncompletePreNorm(t *testing.T) {
	tensors := append(tinyHunyuanVLTensors(), f32Tensor("v.pre_ln.weight", []uint64{4}, []float32{1, 1, 1, 1}))
	if _, err := OpenHunyuanVL(writeTinyHunyuanVL(t, tensors)); err == nil || !strings.Contains(err.Error(), "must be paired") {
		t.Fatalf("error = %v", err)
	}
}

func TestHunyuanVLCatalogKeepsHostReorderedWeightOffDevice(t *testing.T) {
	file, err := gguf.Open(writeTinyHunyuanVL(t, tinyHunyuanVLTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	spec, err := ReadHunyuanVLSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	names, err := validateHunyuanVLCatalog(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(names, "mm.0.weight") {
		t.Fatal("host-reordered convolution entered device catalog")
	}
	if !slices.IsSorted(names) {
		t.Fatalf("device catalog is not ordered: %v", names)
	}
	for _, name := range []string{"v.blk.0.attn_qkv.weight", "mm.0.bias", "mm.2.weight"} {
		if !slices.Contains(names, name) {
			t.Errorf("device catalog missing %q", name)
		}
	}
}

func TestHunyuanVLCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyHunyuanVL(t, nonzeroTinyHunyuanVLTensors())
	cpu, err := OpenHunyuanVL(path)
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := OpenHunyuanVLWithOptions(path, HunyuanVLOpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 8, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 25), G: uint8(y * 50), B: 80, A: 255})
		}
	}
	options := HunyuanVLPreprocessOptions{MinPixels: 32, MaxPixels: 32, MaxAspectRatio: 10}
	want, err := cpu.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "Hunyuan-VL", got.Embeddings.Data, want.Embeddings.Data, 2e-3)
}

func writeTinyHunyuanVL(t *testing.T, tensors []gguf.TensorData) string {
	return writeProjectorFixture(t, "hunyuanvl-mmproj.gguf", tinyHunyuanVLMetadata(), tensors)
}

func tinyHunyuanVLMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: hunyuanVLProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(8)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(6)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.spatial_merge_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.image_min_pixels", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(16)}},
		{Key: "clip.vision.image_max_pixels", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(64)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
	}
}

func tinyHunyuanVLSpec() HunyuanVLSpec {
	return HunyuanVLSpec{
		ImageSize: 4, PatchSize: 2, Hidden: 4, Intermediate: 8, OutputHidden: 6,
		Layers: 1, Heads: 1, MergeSize: 2, MinPixels: 16, MaxPixels: 64,
		ConvIntermediate: 8, ProjectorInput: 5, LayerNormEpsilon: 1e-6,
		ImageStd: [3]float32{1, 1, 1}, FusedQKV: []bool{true},
	}
}

func tinyHunyuanVLTensors() []gguf.TensorData {
	tensors := []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{2, 2, 3, 4}, nil),
		f32Tensor("v.patch_embd.bias", []uint64{4}, nil),
		f32Tensor("v.position_embd.weight", []uint64{4, 4}, nil),
		f32Tensor("mm.pre_norm.weight", []uint64{4}, []float32{1, 1, 1, 1}),
		f32Tensor("mm.0.weight", []uint64{2, 2, 4, 8}, nil),
		f32Tensor("mm.0.bias", []uint64{8}, nil),
		f32Tensor("mm.2.weight", []uint64{1, 1, 8, 5}, nil),
		f32Tensor("mm.2.bias", []uint64{5}, nil),
		f32Tensor("v.image_newline", []uint64{5}, nil),
		f32Tensor("mm.model.fc.weight", []uint64{5, 6}, nil),
		f32Tensor("mm.model.fc.bias", []uint64{6}, nil),
		f32Tensor("mm.image_begin", []uint64{6}, nil),
		f32Tensor("mm.image_end", []uint64{6}, nil),
		f32Tensor("mm.post_norm.weight", []uint64{6}, []float32{1, 1, 1, 1, 1, 1}),
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

func nonzeroTinyHunyuanVLTensors() []gguf.TensorData {
	base := tinyHunyuanVLTensors()
	result := make([]gguf.TensorData, len(base))
	for tensorIndex, item := range base {
		elements := uint64(1)
		for _, dimension := range item.Shape {
			elements *= dimension
		}
		values := make([]float32, int(elements))
		isNorm := strings.Contains(item.Name, "ln1.weight") || strings.Contains(item.Name, "ln2.weight") ||
			strings.Contains(item.Name, "pre_norm.weight") || strings.Contains(item.Name, "post_norm.weight")
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
