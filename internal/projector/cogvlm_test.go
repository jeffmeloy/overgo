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

type cogVLMPromptTokenizer struct{ text string }

func (t *cogVLMPromptTokenizer) TokenizeText(text string, addSpecial, _ bool) ([]tokenizer.TokenID, error) {
	t.text = text
	ids := make([]tokenizer.TokenID, len(text))
	for index := range ids {
		ids[index] = 2
	}
	if addSpecial {
		ids = append([]tokenizer.TokenID{1}, ids...)
	}
	return ids, nil
}

func TestCogVLMVisionRunnerTinyFixture(t *testing.T) {
	runner, err := OpenCogVLMVision(writeTinyCogVLM(t, false))
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 4, 4)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(mustShape(4, 6)) {
		t.Fatalf("output shape = %v", output.Shape)
	}
	want := make([]float32, 24)
	copy(want[:4], []float32{1, 2, 3, 4})
	copy(want[20:], []float32{5, 6, 7, 8})
	if !slices.Equal(output.Data, want) {
		t.Fatalf("output = %v", output.Data)
	}
}

func TestCogVLMPromptMarksVisualExpertBlock(t *testing.T) {
	runner, err := OpenCogVLMVision(writeTinyCogVLM(t, false))
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	tok := &cogVLMPromptTokenizer{}
	prompt, err := runner.BuildImagePrompt(context.Background(), tok, image.NewRGBA(image.Rect(0, 0, 4, 4)), "before", "after", false)
	if err != nil {
		t.Fatal(err)
	}
	if tok.text != "Question: beforeafter Answer:" || !strings.HasPrefix(tok.text, "Question:") {
		t.Fatalf("prompt text = %q", tok.text)
	}
	if len(prompt.VisualBlocks) != 1 || prompt.VisualBlocks[0] != (AttentionBlock{Start: 1, End: 7}) ||
		len(prompt.EmbeddingTokenIndices) != 6 || prompt.EmbeddingWidth != 4 {
		t.Fatalf("prompt contract = blocks %v indices %v width %d", prompt.VisualBlocks, prompt.EmbeddingTokenIndices, prompt.EmbeddingWidth)
	}
}

func TestOpenImageProjectorDispatchesCogVLM(t *testing.T) {
	projector, err := OpenImageProjector(context.Background(), writeTinyCogVLM(t, false))
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*CogVLMVisionRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestCogVLMCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyCogVLM(t, true)
	cpu, err := OpenCogVLMVision(path)
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := OpenCogVLMVisionWithOptions(path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 45), G: uint8(y * 51), B: uint8((x + y) * 27), A: fixtureOpaqueAlpha})
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
	compareFloat32Tolerance(t, "CogVLM", got.Data, want.Data, 3e-3)
}

func writeTinyCogVLM(t *testing.T, patterned bool) string {
	return writeProjectorFixture(t, "cogvlm-mmproj.gguf", tinyCogVLMMetadata(), tinyCogVLMTensors(patterned))
}

func tinyCogVLMMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: cogVLMProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(8)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(6)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-5)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
	}
}

func tinyCogVLMTensors(patterned bool) []gguf.TensorData {
	values := func(count int, scale float32) []float32 {
		if !patterned {
			return nil
		}
		result := make([]float32, count)
		for index := range result {
			result[index] = float32(index%7-3) * scale
		}
		return result
	}
	ones8, ones4 := slices.Repeat([]float32{1}, 8), slices.Repeat([]float32{1}, 4)
	return []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{2, 2, 3, 8}, values(96, .01)),
		f32Tensor("v.patch_embd.bias", []uint64{8}, values(8, .01)),
		f32Tensor("v.class_embd", []uint64{8, 1}, values(8, .02)),
		f32Tensor("v.position_embd.weight", []uint64{8, 5}, values(40, .01)),
		f32Tensor("v.blk.0.attn_qkv.weight", []uint64{8, 24}, values(192, .01)),
		f32Tensor("v.blk.0.attn_qkv.bias", []uint64{24}, values(24, .01)),
		f32Tensor("v.blk.0.attn_out.weight", []uint64{8, 8}, values(64, .01)),
		f32Tensor("v.blk.0.attn_out.bias", []uint64{8}, values(8, .01)),
		f32Tensor("v.blk.0.ffn_up.weight", []uint64{8, 6}, values(48, .01)),
		f32Tensor("v.blk.0.ffn_down.weight", []uint64{6, 8}, values(48, .01)),
		f32Tensor("v.blk.0.ln1.weight", []uint64{8}, ones8), f32Tensor("v.blk.0.ln1.bias", []uint64{8}, nil),
		f32Tensor("v.blk.0.ln2.weight", []uint64{8}, ones8), f32Tensor("v.blk.0.ln2.bias", []uint64{8}, nil),
		f32Tensor("mm.model.fc.weight", []uint64{8, 4}, values(32, .01)),
		f32Tensor("mm.post_fc_norm.weight", []uint64{4}, ones4), f32Tensor("mm.post_fc_norm.bias", []uint64{4}, nil),
		f32Tensor("mm.up.weight", []uint64{4, 7}, values(28, .01)),
		f32Tensor("mm.gate.weight", []uint64{4, 7}, values(28, .01)),
		f32Tensor("mm.down.weight", []uint64{7, 4}, values(28, .01)),
		f32Tensor("v.boi", []uint64{4, 1, 1}, []float32{1, 2, 3, 4}),
		f32Tensor("v.eoi", []uint64{4, 1, 1}, []float32{5, 6, 7, 8}),
	}
}
