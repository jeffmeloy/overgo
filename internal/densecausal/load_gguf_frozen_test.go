package densecausal

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestFrozenLlamaLoadReadsQueryAndKeyRowsInCheckpointOrder holds a frozen load
// of a quantized llama GGUF to the order the model computes with: its query and
// key projections, whose stored rows the conversion permuted, are read in full
// and reordered exactly as an ordinary load reorders them, while the other
// projections stay held in the quantization the file stores.
func TestFrozenLlamaLoadReadsQueryAndKeyRowsInCheckpointOrder(t *testing.T) {
	t.Parallel()
	path := testutil.QuantizedLlamaGGUF(t, 1)
	dense, err := LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	frozen, err := LoadGGUFFrozen(path, func(string) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	reordered := 0
	for name, values := range dense.Weights {
		attention := strings.HasSuffix(name, queryProjection) || strings.HasSuffix(name, keyProjection)
		switch {
		case attention:
			reordered++
			if frozen.Frozen(name) || !slices.Equal(frozen.Weights[name], values) {
				t.Fatalf("%s: a frozen load left its permuted rows held or in file order (held %t)", name, frozen.Frozen(name))
			}
		case strings.HasSuffix(name, ".self_attn.v_proj.weight"):
			if !frozen.Frozen(name) {
				t.Fatalf("%s: a projection the file stores quantized was not held", name)
			}
		}
	}
	if reordered != 2 {
		t.Fatalf("the fixture has %d query and key projections, want 2", reordered)
	}
}

// TestFrozenLoadRefusesAHeldTensorTheForwardCannotRead holds a frozen load to the
// tensors a layer dequantizes as it reads them: a predicate that names a tensor stored
// quantized that is no layer's projection, such as the embedding, is refused at load,
// where the model would otherwise read it as the zeros of its scratch; the projections
// of a layer are held.
func TestFrozenLoadRefusesAHeldTensorTheForwardCannotRead(t *testing.T) {
	t.Parallel()
	path := testutil.QuantizedEmbeddingGGUF(t, 1)
	_, err := LoadGGUFFrozen(path, func(string) bool { return true })
	if err == nil || !strings.Contains(err.Error(), "model.embed_tokens.weight") {
		t.Fatalf("holding every tensor: err = %v, want the stored embedding refused", err)
	}
	frozen, err := LoadGGUFFrozen(path, func(name string) bool { return strings.Contains(name, ".layers.") })
	if err != nil {
		t.Fatal(err)
	}
	if !frozen.Frozen("model.layers.0.self_attn.v_proj.weight") || frozen.Frozen("model.embed_tokens.weight") {
		t.Error("holding the layers' tensors did not hold their projections and leave the embedding in full precision")
	}
	for name, want := range map[string]bool{
		"model.layers.3.self_attn.q_proj.weight": true, "model.layers.0.mlp.down_proj.weight": true,
		"model.embed_tokens.weight": false, "model.layers.0.input_layernorm.weight": false, "lm_head.weight": false,
		"model.layers.0.mlp.experts.0.up_proj.weight": false,
	} {
		if got := isLayerProjection(name); got != want {
			t.Errorf("isLayerProjection(%q) = %t, want %t", name, got, want)
		}
	}
}

// TestMaskedLossWithoutAMaskIsTheLossOfEveryToken holds the contract of a nil
// mask, that it learns every token as LossAndGradsOf does, to the bit: the
// same loss and the same gradients, so a run that masks nothing keeps the
// retained-logits path and its numbers.
func TestMaskedLossWithoutAMaskIsTheLossOfEveryToken(t *testing.T) {
	t.Parallel()
	model, err := LoadGGUF(testutil.QuantizedDenseGGUF(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{3, 4, 5, 6, 7, 4, 5, 6}
	train := func(name string) bool { return strings.Contains(name, "layers.0.") }
	want, wantGrads, err := model.LossAndGradsOf(tokens, train)
	if err != nil {
		t.Fatal(err)
	}
	got, gotGrads, err := model.MaskedLossAndGradsOf(tokens, nil, train)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || len(gotGrads) != len(wantGrads) {
		t.Fatalf("loss %v with %d gradients, want %v with %d", got, len(gotGrads), want, len(wantGrads))
	}
	for name, gradient := range wantGrads {
		if !slices.Equal(gotGrads[name], gradient) {
			t.Fatalf("%s: the gradient of an unmasked run differs from LossAndGradsOf", name)
		}
	}
}

// TestAttentionRowsAreRefusedBeforeTheyAreReordered holds the reordering of a llama
// query or key projection to a shape it can read: a rank-one tensor, a tensor whose
// rows are not a whole number of rotary pairs per head and a head count of zero are
// refused with an error, where each would have panicked in the row arithmetic.
func TestAttentionRowsAreRefusedBeforeTheyAreReordered(t *testing.T) {
	t.Parallel()
	for name, test := range map[string]struct {
		shape []int
		heads int
		ok    bool
	}{
		"a projection":             {[]int{32, 16}, 2, true},
		"a key projection":         {[]int{16, 32}, 1, true},
		"rank one":                 {[]int{32}, 2, false},
		"rank three":               {[]int{2, 16, 16}, 2, false},
		"rows not in rotary pairs": {[]int{30, 16}, 2, false},
		"rows not in whole heads":  {[]int{34, 16}, 4, false},
		"no heads":                 {[]int{32, 16}, 0, false},
		"a negative head count":    {[]int{32, 16}, -2, false},
	} {
		if err := checkRotaryRows("blk.0.attn_q.weight", test.shape, test.heads); (err == nil) != test.ok {
			t.Errorf("%s: checkRotaryRows(%v, %d heads) = %v, want ok=%t", name, test.shape, test.heads, err, test.ok)
		}
	}
}
