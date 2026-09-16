package projector

import (
	"image"
	"image/color"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type gemma3nPromptTokenizer struct{ texts []string }

func (t *gemma3nPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	t.texts = append(t.texts, text)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for len(text) > 0 {
		if strings.HasPrefix(text, Gemma3nImagePad) {
			ids, text = append(ids, 8), text[len(Gemma3nImagePad):]
		} else {
			ids, text = append(ids, 1), text[1:]
		}
	}
	return ids, nil
}

func TestGemma3nVisionTinyFixture(t *testing.T) {
	runner, err := openImageProjectorAs[*Gemma3nVisionRunner](writeTinyGemma3nVision(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(t.Context(), image.NewRGBA(image.Rect(0, 0, 32, 32)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(mustShape(4, 256)) || !slices.Equal(output.Data, make([]float32, 1024)) {
		t.Fatalf("output = shape %v values %v", output.Shape, output.Data[:min(16, len(output.Data))])
	}
}

func TestGemma3nVisionPromptContract(t *testing.T) {
	runner, err := openImageProjectorAs[*Gemma3nVisionRunner](writeTinyGemma3nVision(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	tok := &gemma3nPromptTokenizer{}
	prompt, err := testSession(t, runner).BuildImagePrompt(t.Context(), tok, image.NewRGBA(image.Rect(0, 0, 32, 32)), "before", "after", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(tok.texts) < 2 || !strings.Contains(tok.texts[0], "<start_of_image>"+strings.Repeat(Gemma3nImagePad, 256)+"<end_of_image>") ||
		!strings.HasPrefix(tok.texts[0], "<bos><start_of_turn>user\n") {
		t.Fatalf("prompt text = %q", tok.texts)
	}
	if prompt.EmbeddingWidth != 4 || len(prompt.EmbeddingTokenIndices) != 256 || len(prompt.Embeddings) != 1024 {
		t.Fatalf("prompt = width %d indices %d embeddings %d", prompt.EmbeddingWidth, len(prompt.EmbeddingTokenIndices), len(prompt.Embeddings))
	}
}

func TestOpenImageProjectorDispatchesGemma3nVision(t *testing.T) {
	projector, err := OpenAs[Projector](t.Context(), writeTinyGemma3nVision(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*Gemma3nVisionRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestGemma3nVisionCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyGemma3nVision(t, true)
	input := patternedRGBA(32, 32, func(x, y int) color.RGBA {
		return color.RGBA{R: uint8(x * 7), G: uint8(y * 5), B: uint8((x + y) * 3), A: fixtureOpaqueAlpha}
	})
	referenceParityCase[*Gemma3nVisionRunner](t, "Gemma 3n", path, input, 3e-3)
}

func TestGemma3nAveragePool(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, mustShape(1, 4, 4))
	output := averagePoolSpatial(builder, input, 2, 2)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	results, err := reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		input: {Shape: input.Shape, Data: []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := results[output]
	want := []float32{3.5, 5.5, 11.5, 13.5}
	if !slices.Equal(got.Data, want) {
		t.Fatalf("pool = %v, want %v", got.Data, want)
	}
}

func writeTinyGemma3nVision(t *testing.T, patterned bool) string {
	return writeProjectorFixture(t, "gemma3n-mmproj.gguf", tinyGemma3nVisionMetadata(), tinyGemma3nVisionTensors(patterned))
}

func tinyGemma3nVisionMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: gemma3nVisionProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(256)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(16)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: visionProjectorNormKey, Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
	}
}

func tinyGemma3nVisionTensors(patterned bool) []gguf.TensorData {
	values := func(count, modulus int, scale float32) []float32 {
		if !patterned {
			return nil
		}
		result := make([]float32, count)
		for index := range result {
			result[index] = float32(index%modulus-modulus/2) * scale
		}
		return result
	}
	ones4 := slices.Repeat([]float32{1}, 4)
	return []gguf.TensorData{
		f32Tensor("v.conv_stem.conv.weight", []uint64{3, 3, 3, 4}, values(108, 7, .01)),
		f32Tensor("v.conv_stem.conv.bias", []uint64{4, 1, 1}, values(4, 5, .01)),
		f32Tensor("v.conv_stem.bn.weight", []uint64{4}, ones4),
		f32Tensor("v.blk.0.0.conv_exp.weight", []uint64{3, 3, 4, 6}, values(216, 7, .01)),
		f32Tensor("v.blk.0.0.bn1.weight", []uint64{6}, slices.Repeat([]float32{1}, 6)),
		f32Tensor("v.blk.0.0.conv_pwl.weight", []uint64{1, 1, 6, 4}, values(24, 5, .02)),
		f32Tensor("v.blk.0.0.bn2.weight", []uint64{4}, ones4),
		f32Tensor("v.blk.1.0.pw_exp.conv.weight", []uint64{1, 1, 4, 6}, values(24, 7, .02)),
		f32Tensor("v.blk.1.0.pw_exp.bn.weight", []uint64{6}, slices.Repeat([]float32{1}, 6)),
		f32Tensor("v.blk.1.0.dw_mid.conv.weight", []uint64{3, 3, 1, 6}, values(54, 5, .02)),
		f32Tensor("v.blk.1.0.dw_mid.bn.weight", []uint64{6}, slices.Repeat([]float32{1}, 6)),
		f32Tensor("v.blk.1.0.pw_proj.conv.weight", []uint64{1, 1, 6, 4}, values(24, 7, .02)),
		f32Tensor("v.blk.1.0.pw_proj.bn.weight", []uint64{4}, ones4),
		f32Tensor("v.blk.2.0.pw_exp.conv.weight", []uint64{1, 1, 4, 6}, values(24, 5, .02)),
		f32Tensor("v.blk.2.0.dw_mid.conv.weight", []uint64{3, 3, 1, 6}, values(54, 7, .02)),
		f32Tensor("v.blk.2.0.pw_proj.conv.weight", []uint64{1, 1, 6, 4}, values(24, 5, .02)),
		f32Tensor("v.blk.3.0.pw_exp.conv.weight", []uint64{1, 1, 4, 6}, values(24, 7, .02)),
		f32Tensor("v.blk.3.0.dw_mid.conv.weight", []uint64{3, 3, 1, 6}, values(54, 5, .02)),
		f32Tensor("v.blk.3.0.pw_proj.conv.weight", []uint64{1, 1, 6, 4}, values(24, 7, .02)),
		f32Tensor("v.blk.3.1.attn.query.proj.weight", []uint64{1, 1, 4, 4}, values(16, 7, .02)),
		f32Tensor("v.blk.3.1.attn.key.down_conv.weight", []uint64{3, 3, 1, 4}, values(36, 5, .02)),
		f32Tensor("v.blk.3.1.attn.key.norm.weight", []uint64{4}, ones4),
		f32Tensor("v.blk.3.1.attn.key.proj.weight", []uint64{1, 1, 4, 2}, values(8, 5, .02)),
		f32Tensor("v.blk.3.1.attn.value.down_conv.weight", []uint64{3, 3, 1, 4}, values(36, 7, .02)),
		f32Tensor("v.blk.3.1.attn.value.norm.weight", []uint64{4}, ones4),
		f32Tensor("v.blk.3.1.attn.value.proj.weight", []uint64{1, 1, 4, 2}, values(8, 7, .02)),
		f32Tensor("v.blk.3.1.attn.output.proj.weight", []uint64{1, 1, 4, 4}, values(16, 5, .02)),
		f32Tensor("v.blk.3.1.norm.weight", []uint64{4}, ones4),
		f32Tensor("v.msfa.ffn.pw_exp.conv.weight", []uint64{1, 1, 8, 6}, values(48, 7, .02)),
		f32Tensor("v.msfa.ffn.pw_proj.conv.weight", []uint64{1, 1, 6, 4}, values(24, 5, .02)),
		f32Tensor("v.msfa.norm.weight", []uint64{4}, ones4),
		f32Tensor("mm.soft_emb_norm.weight", []uint64{4}, ones4),
		f32Tensor("mm.input_projection.weight", []uint64{4, 4}, values(16, 7, .02)),
	}
}
