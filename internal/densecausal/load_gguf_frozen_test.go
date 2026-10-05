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
