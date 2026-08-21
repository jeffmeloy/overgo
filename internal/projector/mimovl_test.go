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

type mimoVLPromptTokenizer struct{ texts []string }

func (t *mimoVLPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	t.texts = append(t.texts, text)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for len(text) > 0 {
		if strings.HasPrefix(text, Qwen3VLImagePad) {
			ids, text = append(ids, 8), text[len(Qwen3VLImagePad):]
		} else {
			ids, text = append(ids, 1), text[1:]
		}
	}
	return ids, nil
}

func TestMiMoVLRunnerTinyFixture(t *testing.T) {
	runner, err := openImageProjectorAs[*MiMoVLRunner](writeTinyMiMoVL(t, tinyMiMoVLTensors(false)), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 4, 4)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(4, 4)) || !slices.Equal(output.Embeddings.Data, make([]float32, 16)) {
		t.Fatalf("output = shape %v values %v", output.Embeddings.Shape, output.Embeddings.Data)
	}
	if output.GridH != 4 || output.GridW != 4 || output.MergeSize != fixtureSpatialMerge {
		t.Fatalf("grid=%d,%d merge=%d", output.GridH, output.GridW, output.MergeSize)
	}
}

func TestOpenImageProjectorDispatchesMiMoVL(t *testing.T) {
	projector, err := OpenAs[Projector](context.Background(), writeTinyMiMoVL(t, tinyMiMoVLTensors(false)), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*MiMoVLRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestMiMoVLPromptContract(t *testing.T) {
	runner, err := openImageProjectorAs[*MiMoVLRunner](writeTinyMiMoVL(t, tinyMiMoVLTensors(false)), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	tok := &mimoVLPromptTokenizer{}
	prompt, err := testSession(t, runner).BuildImagePrompt(
		context.Background(), tok, image.NewRGBA(image.Rect(0, 0, 4, 4)), "before", "after", false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(tok.texts) < 2 || !strings.Contains(tok.texts[0], "<|im_start|>system\n"+MiMoVLSystemPrompt+"<|im_end|>") ||
		!strings.Contains(tok.texts[0], "<|vision_start|>"+strings.Repeat(Qwen3VLImagePad, 4)+"<|vision_end|>") {
		t.Fatalf("prompt text = %q", tok.texts)
	}
	if prompt.EmbeddingWidth != 4 || len(prompt.Embeddings) != 16 || len(prompt.EmbeddingTokenIndices) != 4 ||
		prompt.EmbeddingTokenIndices[0] != uint32(prompt.EmbeddingStart) {
		t.Fatalf("prompt contract = width %d embeddings %d indices %v start %d", prompt.EmbeddingWidth, len(prompt.Embeddings), prompt.EmbeddingTokenIndices, prompt.EmbeddingStart)
	}
	if len(prompt.MultiAxisPositions[0]) != 0 {
		t.Fatalf("unexpected multi-axis positions: %v", prompt.MultiAxisPositions)
	}
}

func TestMiMoVLColumnOrderRoundTrip(t *testing.T) {
	order := columnMajorPatchOrder(2, 3, 2)
	inverse := inversePermutation(order)
	for destination, source := range order {
		if inverse[source] != destination {
			t.Fatalf("inverse[%d]=%d want=%d", source, inverse[source], destination)
		}
	}
}

func TestMiMoVLCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyMiMoVL(t, tinyMiMoVLTensors(true))
	cpu, err := openImageProjectorAs[*MiMoVLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := openImageProjectorAs[*MiMoVLRunner](path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 51), G: uint8(y * 47), B: uint8((x + y) * 29), A: fixtureOpaqueAlpha})
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
	compareFloat32Tolerance(t, "MiMo-VL", got.Embeddings.Data, want.Embeddings.Data, 3e-3)
}

func writeTinyMiMoVL(t *testing.T, tensors []gguf.TensorData) string {
	return writeProjectorFixture(t, "mimovl-mmproj.gguf", tinyMiMoVLMetadata(), tensors)
}

func tinyMiMoVLMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: mimoVLProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.use_silu", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(8)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(6)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(3)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.attention.head_count_kv", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: visionSpatialMergeKey, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(fixtureSpatialMerge)}},
		{Key: visionRopeFrequencyKey, Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: fixtureRopeFrequency}},
		{Key: "clip.vision.window_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: visionMinPixelsKey, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(fixtureSmallPixelBudget)}},
		{Key: visionMaxPixelsKey, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(fixtureSmallPixelBudget)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: visionProjectorNormKey, Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
		{Key: "clip.vision.wa_pattern_mode", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{0, 1, -1}}},
	}
}

func tinyMiMoVLTensors(patterned bool) []gguf.TensorData {
	tensors := []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{1, 1, 3, 8}, nil),
		f32Tensor("v.patch_embd.weight.1", []uint64{1, 1, 3, 8}, nil),
		f32Tensor("v.post_ln.weight", []uint64{8}, slices.Repeat([]float32{1}, 8)),
		f32Tensor("v.post_ln.bias", []uint64{8}, nil),
		f32Tensor("mm.0.weight", []uint64{32, 7}, nil),
		f32Tensor("mm.0.bias", []uint64{7}, nil),
		f32Tensor("mm.2.weight", []uint64{7, 4}, nil),
		f32Tensor("mm.2.bias", []uint64{4}, nil),
	}
	for layer, mode := range []int{0, 1, -1} {
		prefix := "v.blk." + string(rune('0'+layer)) + "."
		for name, shape := range map[string][]uint64{
			"attn_qkv.weight": {8, 16}, "attn_qkv.bias": {16},
			"attn_out.weight": {8, 8}, "attn_out.bias": {8},
			"ffn_up.weight": {8, 6}, "ffn_up.bias": {6},
			"ffn_gate.weight": {8, 6}, "ffn_gate.bias": {6},
			"ffn_down.weight": {6, 8}, "ffn_down.bias": {8},
			"ln1.weight": {8}, "ln1.bias": {8}, "ln2.weight": {8}, "ln2.bias": {8},
		} {
			values := []float32(nil)
			if name == "ln1.weight" || name == "ln2.weight" {
				values = slices.Repeat([]float32{1}, 8)
			}
			tensors = append(tensors, f32Tensor(prefix+name, shape, values))
		}
		if mode != -1 {
			tensors = append(tensors, f32Tensor(prefix+"attn_sinks", []uint64{2}, nil))
		}
	}
	if !patterned {
		return tensors
	}
	output := make([]gguf.TensorData, len(tensors))
	for tensorIndex, item := range tensors {
		elements := uint64(1)
		for _, dimension := range item.Shape {
			elements *= dimension
		}
		values := make([]float32, int(elements))
		isNorm := item.Name == "v.post_ln.weight" ||
			strings.HasSuffix(item.Name, "ln1.weight") || strings.HasSuffix(item.Name, "ln2.weight")
		for index := range values {
			values[index] = float32((index+tensorIndex*7)%19-9) * 0.006
			if isNorm {
				values[index] += 1
			}
		}
		output[tensorIndex] = f32Tensor(item.Name, item.Shape, values)
	}
	return output
}
