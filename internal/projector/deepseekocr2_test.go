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
)

func TestDeepSeekOCR2TinyFixture(t *testing.T) {
	runner, err := openImageProjectorAs[*DeepSeekOCR2Runner](writeTinyDeepSeekOCR2(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 32, 32)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(mustShape(3, 2)) || !slices.Equal(output.Data, make([]float32, 6)) {
		t.Fatalf("output = shape %v values %v", output.Shape, output.Data)
	}
}

func TestDeepSeekOCR2LocalTilesRemainIndependent(t *testing.T) {
	runner, err := openImageProjectorAs[*DeepSeekOCR2Runner](writeTinyDeepSeekOCR2(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 128, 64)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(mustShape(3, 4)) {
		t.Fatalf("output shape = %v", output.Shape)
	}
}

func TestDeepSeekOCR2PromptContract(t *testing.T) {
	runner, err := openImageProjectorAs[*DeepSeekOCR2Runner](writeTinyDeepSeekOCR2(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	tok := &deepSeekOCRPromptTokenizer{}
	prompt, err := testSession(t, runner).BuildImagePrompt(context.Background(), tok, image.NewRGBA(image.Rect(0, 0, 32, 32)), "before", "after", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(tok.texts) < 2 || tok.texts[0] != "before"+strings.Repeat(DeepSeekOCRImagePad, 2)+"after" {
		t.Fatalf("prompt text = %q", tok.texts)
	}
	if prompt.EmbeddingWidth != 3 || len(prompt.EmbeddingTokenIndices) != 2 || len(prompt.Embeddings) != 6 {
		t.Fatalf("prompt = width %d indices %d embeddings %d", prompt.EmbeddingWidth, len(prompt.EmbeddingTokenIndices), len(prompt.Embeddings))
	}
}

func TestOpenImageProjectorDispatchesDeepSeekOCR2(t *testing.T) {
	projector, err := OpenAs[Projector](context.Background(), writeTinyDeepSeekOCR2(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*DeepSeekOCR2Runner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestDeepSeekOCR2CUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyDeepSeekOCR2(t, true)
	cpu, err := openImageProjectorAs[*DeepSeekOCR2Runner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := openImageProjectorAs[*DeepSeekOCR2Runner](path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 7), G: uint8(y * 5), B: uint8((x + y) * 3), A: fixtureOpaqueAlpha})
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
	compareFloat32Tolerance(t, "DeepSeek-OCR-2", got.Data, want.Data, 4e-3)
}

func writeTinyDeepSeekOCR2(t *testing.T, patterned bool) string {
	return writeProjectorFixture(t, "deepseekocr2-mmproj.gguf", tinyDeepSeekOCR2Metadata(), tinyDeepSeekOCR2Tensors(patterned))
}

func tinyDeepSeekOCR2Metadata() []gguf.Metadata {
	metadata := tinyDeepSeekOCRMetadata()
	for index := range metadata {
		if metadata[index].Key == "clip.projector_type" {
			metadata[index].Value.Data = deepSeekOCR2ProjectorType
		}
	}
	return append(metadata,
		gguf.Metadata{Key: "clip.vision.attention.head_count_kv", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		gguf.Metadata{Key: visionRopeFrequencyKey, Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: fixtureExtendedRopeFrequency}},
	)
}

func tinyDeepSeekOCR2Tensors(patterned bool) []gguf.TensorData {
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
	var tensors []gguf.TensorData
	for _, item := range tinyDeepSeekOCRTensors(patterned) {
		if strings.HasPrefix(item.Name, "v.sam.") {
			tensors = append(tensors, item)
		}
	}
	ones2 := []float32{1, 1}
	return append(tensors,
		f32Tensor("v.resample_query_768.weight", []uint64{2, 1}, values(2, 5, .01)),
		f32Tensor("v.resample_query_1024.weight", []uint64{2, 1}, values(2, 5, .01)),
		f32Tensor("v.blk.0.ln1.weight", []uint64{2}, ones2),
		f32Tensor("v.blk.0.ln2.weight", []uint64{2}, ones2),
		f32Tensor("v.blk.0.attn_q.weight", []uint64{2, 2}, values(4, 5, .01)),
		f32Tensor("v.blk.0.attn_q.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.blk.0.attn_k.weight", []uint64{2, 2}, values(4, 7, .01)),
		f32Tensor("v.blk.0.attn_k.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.blk.0.attn_v.weight", []uint64{2, 2}, values(4, 5, .01)),
		f32Tensor("v.blk.0.attn_v.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.blk.0.attn_out.weight", []uint64{2, 2}, values(4, 7, .01)),
		f32Tensor("v.blk.0.ffn_up.weight", []uint64{2, 3}, values(6, 7, .01)),
		f32Tensor("v.blk.0.ffn_gate.weight", []uint64{2, 3}, values(6, 5, .01)),
		f32Tensor("v.blk.0.ffn_down.weight", []uint64{3, 2}, values(6, 7, .01)),
		f32Tensor("v.post_ln.weight", []uint64{2}, ones2),
		f32Tensor("mm.model.fc.weight", []uint64{2, 3}, values(6, 7, .01)),
		f32Tensor("mm.model.fc.bias", []uint64{3}, values(3, 5, .01)),
		f32Tensor("v.view_seperator", []uint64{3}, nil),
	)
}
