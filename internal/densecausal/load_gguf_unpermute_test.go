package densecausal

import (
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/testutil"
)

// permuteRows is the permutation llama.cpp's conversion applies to a query or
// key projection, written from its definition: reshape the output rows to
// (heads, 2, half), swap the last two axes, flatten again.
func permuteRows(values []float32, rows, columns, heads int) []float32 {
	half := rows / heads / 2
	permuted := make([]float32, len(values))
	for head := range heads {
		for pair := range 2 {
			for index := range half {
				from := (head*2*half + pair*half + index) * columns
				to := (head*2*half + index*2 + pair) * columns
				copy(permuted[to:to+columns], values[from:from+columns])
			}
		}
	}
	return permuted
}

// TestLlamaGGUFLoadsItsAttentionRowsInCheckpointOrder holds the training load of
// a llama GGUF, whose conversion permuted the query and key rows for the
// interleaved rotary layout, to the checkpoint's order: the loaded rows, put
// through the conversion's permutation, are the file's stored rows, for the
// query's heads and the key's fewer ones; and an interleaved-rotary
// architecture whose permutation is not established still refuses.
func TestLlamaGGUFLoadsItsAttentionRowsInCheckpointOrder(t *testing.T) {
	t.Parallel()
	path := testutil.HermeticLlamaGGUF(t, 64)
	model, err := LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	stored := func(name string) []float32 {
		t.Helper()
		index := slices.IndexFunc(file.Tensors, func(info gguf.TensorInfo) bool { return info.Name == name })
		info := file.Tensors[index]
		raw := make([]byte, info.Size)
		if err := file.ReadTensorData(info, raw); err != nil {
			t.Fatal(err)
		}
		elements, err := info.ElementCount()
		if err != nil {
			t.Fatal(err)
		}
		values, err := quant.Dequantize(info.Type, raw, elements)
		if err != nil {
			t.Fatal(err)
		}
		return values
	}
	for _, projection := range []struct {
		stored, loaded string
		heads          int
	}{
		{"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight", int(model.Dims.Heads)},
		{"blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight", int(model.Dims.KVHeads)},
	} {
		shape := model.Shapes[projection.loaded]
		loaded := model.Weights[projection.loaded]
		if got := permuteRows(loaded, shape[0], shape[1], projection.heads); !slices.Equal(got, stored(projection.stored)) {
			t.Errorf("%s: the loaded rows, permuted as the conversion permutes them, are not the file's", projection.loaded)
		}
		if slices.Equal(loaded, stored(projection.stored)) {
			t.Errorf("%s: the rows load as stored, still permuted", projection.loaded)
		}
		if got := unpermuteRows(permuteRows(loaded, shape[0], shape[1], projection.heads), shape[0], shape[1], projection.heads); !slices.Equal(got, loaded) {
			t.Errorf("%s: unpermuting does not undo the permutation", projection.loaded)
		}
	}
}
