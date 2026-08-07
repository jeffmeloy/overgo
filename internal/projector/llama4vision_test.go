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

type llama4PromptTokenizer struct{}

func (llama4PromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	const placeholder = "\x00"
	text = strings.ReplaceAll(text, Llama4ImagePad, placeholder)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for _, value := range text {
		if value == 0 {
			ids = append(ids, 9090)
		} else {
			ids = append(ids, tokenizer.TokenID(value))
		}
	}
	return ids, nil
}

func TestLlama4VisionRunnerTinyFixture(t *testing.T) {
	runner, err := OpenLlama4Vision(writeTinyLlama4Vision(t, tinyLlama4VisionTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 4, 4)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(6, 1)) {
		t.Fatalf("output shape = %v", output.Embeddings.Shape)
	}
	if !slices.Equal(output.Embeddings.Data, make([]float32, 6)) {
		t.Fatalf("output = %v", output.Embeddings.Data)
	}
	if output.TileCount != 1 || output.GridH != 0 || output.GridW != 0 {
		t.Fatalf("tiles=%d grid=%d,%d", output.TileCount, output.GridH, output.GridW)
	}
}

func TestOpenImageProjectorDispatchesLlama4(t *testing.T) {
	projector, err := OpenImageProjector(writeTinyLlama4Vision(t, tinyLlama4VisionTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*Llama4VisionRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestPreprocessLlama4VisionUHDOrder(t *testing.T) {
	input := image.NewRGBA(image.Rect(0, 0, 8, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 30), G: uint8(y * 50), A: fixtureOpaqueAlpha})
		}
	}
	processed, err := PreprocessLlama4VisionImage(input, tinyLlama4VisionSpec())
	if err != nil {
		t.Fatal(err)
	}
	if processed.GridW != 2 || processed.GridH != 1 || len(processed.Tiles) != 3 {
		t.Fatalf("grid=%d,%d tiles=%d", processed.GridW, processed.GridH, len(processed.Tiles))
	}
	if processed.Tiles[0].PixelValues[0] >= processed.Tiles[1].PixelValues[0] {
		t.Fatalf("refined tile order is not left-to-right")
	}
	if processed.Tiles[2].PixelValues[0] != 0 {
		t.Fatalf("overview padding start = %g", processed.Tiles[2].PixelValues[0])
	}
}

func TestLlama4MultipleImagePrompt(t *testing.T) {
	runner, err := OpenLlama4Vision(writeTinyLlama4Vision(t, tinyLlama4VisionTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	prompt, err := runner.BuildImagesPrompt(
		context.Background(), llama4PromptTokenizer{},
		[]image.Image{image.NewRGBA(image.Rect(0, 0, 4, 4)), image.NewRGBA(image.Rect(0, 0, 8, 4))},
		[]string{"first ", " then ", " question"}, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if prompt.EmbeddingWidth != 6 || len(prompt.Embeddings) != 4*6 || len(prompt.EmbeddingTokenIndices) != 4 {
		t.Fatalf("embedding contract = width %d values %d indices %d", prompt.EmbeddingWidth, len(prompt.Embeddings), len(prompt.EmbeddingTokenIndices))
	}
	for index := 1; index < len(prompt.EmbeddingTokenIndices); index++ {
		if prompt.EmbeddingTokenIndices[index] <= prompt.EmbeddingTokenIndices[index-1] {
			t.Fatalf("indices = %v", prompt.EmbeddingTokenIndices)
		}
	}
}

func TestLlama4VisionRoPEUsesWidthAndHeight(t *testing.T) {
	qkv := make([]float32, 5*12)
	qkv[12] = 1
	qkv[14] = 1
	qkv[2*12] = 1
	qkv[2*12+2] = 1
	llama4VisionRoPE(qkv, 2, 2, 4, 1, 10000)
	if qkv[12] == qkv[2*12] || qkv[14] == qkv[2*12+2] {
		t.Fatalf("2D RoPE axes not separated: row0=%v row1=%v", qkv[12:16], qkv[2*12:2*12+4])
	}
}

func TestLlama4CatalogRejectsIncompletePreNorm(t *testing.T) {
	tensors := append(tinyLlama4VisionTensors(), f32Tensor("v.pre_ln.weight", []uint64{4}, []float32{1, 1, 1, 1}))
	if _, err := OpenLlama4Vision(writeTinyLlama4Vision(t, tensors)); err == nil || !strings.Contains(err.Error(), "must be paired") {
		t.Fatalf("error = %v", err)
	}
}

func TestLlama4VisionCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyLlama4Vision(t, nonzeroTinyLlama4VisionTensors())
	cpu, err := OpenLlama4Vision(path)
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := OpenLlama4VisionWithOptions(path, Llama4VisionOpenOptions{CUDA: true})
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
	want, err := cpu.EncodeImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "Llama-4", got.Embeddings.Data, want.Embeddings.Data, 2e-3)
}

func writeTinyLlama4Vision(t *testing.T, tensors []gguf.TensorData) string {
	return writeProjectorFixture(t, "llama4-mmproj.gguf", tinyLlama4VisionMetadata(), tensors)
}

func tinyLlama4VisionMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: llama4ProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(8)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(6)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.projector.scale_factor", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
	}
}

func tinyLlama4VisionSpec() Llama4VisionSpec {
	return Llama4VisionSpec{
		ImageSize: 4, PatchSize: 2, Hidden: 4, Intermediate: 8, OutputHidden: 6,
		AdapterIntermediate: 8, AdapterHidden: 5, Layers: 1, Heads: 1, MergeSize: 2,
		LayerNormEpsilon: 1e-6, RopeTheta: 10000, ImageStd: [3]float32{1, 1, 1}, FusedQKV: []bool{true},
	}
}

func tinyLlama4VisionTensors() []gguf.TensorData {
	tensors := []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{2, 2, 3, 4}, nil),
		f32Tensor("v.patch_embd.bias", []uint64{4}, nil),
		f32Tensor("v.class_embd", []uint64{4}, nil),
		f32Tensor("v.position_embd.weight", []uint64{4, 5}, nil),
		f32Tensor("mm.model.mlp.1.weight", []uint64{16, 8}, nil),
		f32Tensor("mm.model.mlp.2.weight", []uint64{8, 5}, nil),
		f32Tensor("mm.model.fc.weight", []uint64{5, 6}, nil),
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

func nonzeroTinyLlama4VisionTensors() []gguf.TensorData {
	base := tinyLlama4VisionTensors()
	result := make([]gguf.TensorData, len(base))
	for tensorIndex, item := range base {
		elements := uint64(1)
		for _, dimension := range item.Shape {
			elements *= dimension
		}
		values := make([]float32, int(elements))
		isNorm := strings.Contains(item.Name, "ln1.weight") || strings.Contains(item.Name, "ln2.weight")
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
