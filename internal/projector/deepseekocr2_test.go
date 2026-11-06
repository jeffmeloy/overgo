package projector

import (
	"image"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/testcheck"
)

func TestDeepSeekOCR2TinyFixture(t *testing.T) {
	runner := testcheck.Must(openImageProjectorAs[*deepSeekOCR2Runner](writeTinyDeepSeekOCR2(t, false), OpenOptions{}))(t)
	defer runner.Close()
	output := testcheck.Must(runner.EncodeImage(t.Context(), image.NewRGBA(image.Rect(0, 0, 32, 32))))(t)
	if !output.Shape.Equal(mustShape(3, 2)) || !slices.Equal(output.Data, make([]float32, 6)) {
		t.Fatalf("output = shape %v values %v", output.Shape, output.Data)
	}
}

func TestDeepSeekOCR2LocalTilesRemainIndependent(t *testing.T) {
	runner := testcheck.Must(openImageProjectorAs[*deepSeekOCR2Runner](writeTinyDeepSeekOCR2(t, false), OpenOptions{}))(t)
	defer runner.Close()
	output := testcheck.Must(runner.EncodeImage(t.Context(), image.NewRGBA(image.Rect(0, 0, 128, 64))))(t)
	if !output.Shape.Equal(mustShape(3, 4)) {
		t.Fatalf("output shape = %v", output.Shape)
	}
}

func TestDeepSeekOCR2PromptContract(t *testing.T) {
	runner := testcheck.Must(openImageProjectorAs[*deepSeekOCR2Runner](writeTinyDeepSeekOCR2(t, false), OpenOptions{}))(t)
	defer runner.Close()
	tok := &recordingPromptTokenizer{pad: DeepSeekOCRImagePad}
	prompt := testcheck.Must(testSession(t, runner).BuildImagePrompt(t.Context(), tok, image.NewRGBA(image.Rect(0, 0, 32, 32)), "before", "after", false))(t)
	if len(tok.texts) < 2 || tok.texts[0] != "before"+strings.Repeat(DeepSeekOCRImagePad, 2)+"after" {
		t.Fatalf("prompt text = %q", tok.texts)
	}
	if prompt.EmbeddingWidth != 3 || len(prompt.EmbeddingTokenIndices) != 2 || len(prompt.Embeddings) != 6 {
		t.Fatalf("prompt = width %d indices %d embeddings %d", prompt.EmbeddingWidth, len(prompt.EmbeddingTokenIndices), len(prompt.Embeddings))
	}
}

func TestOpenImageProjectorDispatchesDeepSeekOCR2(t *testing.T) {
	requireOpensAs[*deepSeekOCR2Runner](t, writeTinyDeepSeekOCR2(t, false))
}

func TestDeepSeekOCR2CUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyDeepSeekOCR2(t, true)
	input := patternedRGBA(32, 32, fixtureColor)
	referenceParityCase[*deepSeekOCR2Runner](t, "DeepSeek-OCR-2", path, input, 4e-3)
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
	values := patternedValues(patterned)
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
