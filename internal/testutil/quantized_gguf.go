package testutil

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// The shape of the quantized fixture: Q8_0 stores a row in blocks of 32, so every
// projection's input width is a multiple of 32.
const (
	quantizedHidden = 32
	quantizedFFN    = 64
	quantizedKV     = 16
	// quantizedVocab is the fixture's vocabulary: the letters a to d and four markers.
	quantizedVocab = 8
	// quantizedSpread is the standard deviation of the fixture's weights, wide enough for the
	// quantization to be a real rounding.
	quantizedSpread = 0.4
	// quantizedEmbedding scales the embedding, so an adapter over frozen weights can lower the loss.
	quantizedEmbedding = 8
	quantizedContext   = 32
	quantizedSeed      = 7
	// quantizedHeads is the fixture's query heads, over one key-value head.
	quantizedHeads = 2
	// quantizedRopeBase and quantizedRMSEpsilon are the rotary base and norm epsilon of the
	// fixture's metadata.
	quantizedRopeBase   = 10_000
	quantizedRMSEpsilon = 1e-5
	// quantizedEOS is the id of the end-of-sequence token, the third of the fixture's vocabulary.
	quantizedEOS = 2
)

// GGUF token types: ordinary text, the unknown token, and a control token.
const (
	tokenText    int32 = 1
	tokenUnknown int32 = 2
	tokenControl int32 = 3
)

// quantizedTokenTypes classify the fixture's eight tokens as GGUF does: the unknown
// token, the two control tokens, and the five of text.
var quantizedTokenTypes = []int32{tokenUnknown, tokenControl, tokenControl, tokenText, tokenText, tokenText, tokenText, tokenText}

// QuantizedDenseGGUF writes a dense qwen2 model of the given number of layers
// whose projections are stored as Q8_0 and whose embedding, head and norms are
// stored in full precision, with a tokenizer of the letters a to d, and returns
// its path.
func QuantizedDenseGGUF(t *testing.T, layers int) string {
	t.Helper()
	return quantizedDenseGGUF(t, layers, "qwen2")
}

// QuantizedLlamaGGUF is QuantizedDenseGGUF for the llama architecture, whose
// conversion permutes the query and key rows (the fixture stores its random
// rows as given, so a load that does not reorder them differs from one that
// does).
func QuantizedLlamaGGUF(t *testing.T, layers int) string {
	t.Helper()
	return quantizedDenseGGUF(t, layers, "llama")
}

func quantizedDenseGGUF(t *testing.T, layers int, architecture string) string {
	t.Helper()
	random := rand.New(rand.NewSource(quantizedSeed))
	values := func(extents []uint64, scale float64) []float32 {
		count := 1
		for _, extent := range extents {
			count *= int(extent)
		}
		out := make([]float32, count)
		for index := range out {
			out[index] = float32(random.NormFloat64() * scale)
		}
		return out
	}
	full := func(name string, extents []uint64, scale float64) gguf.TensorData {
		var storage bytes.Buffer
		if err := binary.Write(&storage, binary.LittleEndian, values(extents, scale)); err != nil {
			t.Fatal(err)
		}
		return gguf.TensorData{Name: name, Shape: extents, Type: gguf.DTypeF32, Data: bytes.NewReader(storage.Bytes())}
	}
	stored := func(name string, extents []uint64) gguf.TensorData {
		data, err := quant.Quantize(dtype.Q8_0, values(extents, quantizedSpread))
		if err != nil {
			t.Fatal(err)
		}
		return gguf.TensorData{Name: name, Shape: extents, Type: dtype.Q8_0, Data: bytes.NewReader(data)}
	}
	metadata := []gguf.Metadata{
		GGUFScalar("general.architecture", gguf.ValueTypeString, architecture),
		GGUFScalar("general.name", gguf.ValueTypeString, "quantized-fixture"),
		GGUFScalar(architecture+".block_count", gguf.ValueTypeUint32, uint32(layers)),
		GGUFScalar(architecture+".context_length", gguf.ValueTypeUint32, uint32(quantizedContext)),
		GGUFScalar(architecture+".embedding_length", gguf.ValueTypeUint32, uint32(quantizedHidden)),
		GGUFScalar(architecture+".feed_forward_length", gguf.ValueTypeUint32, uint32(quantizedFFN)),
		GGUFScalar(architecture+".attention.head_count", gguf.ValueTypeUint32, uint32(quantizedHeads)),
		GGUFScalar(architecture+".attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		GGUFScalar(architecture+".rope.freq_base", gguf.ValueTypeFloat32, float32(quantizedRopeBase)),
		GGUFScalar(architecture+".attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(quantizedRMSEpsilon)),
		GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{"<unk>", "<s>", "</s>", "▁", "a", "b", "c", "d"}),
		GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, quantizedVocab)),
		GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32, quantizedTokenTypes),
		GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(quantizedEOS)),
	}
	tensors := []gguf.TensorData{
		full("token_embd.weight", []uint64{quantizedHidden, quantizedVocab}, quantizedEmbedding),
		full("output_norm.weight", []uint64{quantizedHidden}, 1),
		full("output.weight", []uint64{quantizedHidden, quantizedVocab}, quantizedEmbedding),
	}
	for layer := range layers {
		prefix := fmt.Sprintf("blk.%d.", layer)
		tensors = append(tensors,
			full(prefix+"attn_norm.weight", []uint64{quantizedHidden}, 1),
			stored(prefix+"attn_q.weight", []uint64{quantizedHidden, quantizedHidden}),
			stored(prefix+"attn_k.weight", []uint64{quantizedHidden, quantizedKV}),
			stored(prefix+"attn_v.weight", []uint64{quantizedHidden, quantizedKV}),
			stored(prefix+"attn_output.weight", []uint64{quantizedHidden, quantizedHidden}),
			full(prefix+"ffn_norm.weight", []uint64{quantizedHidden}, 1),
			stored(prefix+"ffn_gate.weight", []uint64{quantizedHidden, quantizedFFN}),
			stored(prefix+"ffn_up.weight", []uint64{quantizedHidden, quantizedFFN}),
			stored(prefix+"ffn_down.weight", []uint64{quantizedFFN, quantizedHidden}),
		)
	}
	path := filepath.Join(t.TempDir(), "quantized.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}
