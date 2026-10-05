package densecausal

import (
	"slices"
	"testing"
)

// TestLossAndGradsOfKeepsOnlyTheNamedGradients holds a run that trains some of
// the weights to the loss of the full backward and, for each weight it names,
// to that backward's gradient bit for bit, with no gradient at all for the
// weights it leaves out: the lexical table and the final norm here.
func TestLossAndGradsOfKeepsOnlyTheNamedGradients(t *testing.T) {
	g := readTinyGolden(t)
	model := modelFromGolden(t, g)
	wantLoss, _, want, err := model.LossAndGrads(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	embedding, finalNorm, head := model.LexicalTensorNames()
	train := func(name string) bool { return name != embedding && name != finalNorm && name != head }
	loss, got, err := model.LossAndGradsOf(g.Tokens, train)
	if err != nil {
		t.Fatal(err)
	}
	if loss != wantLoss {
		t.Fatalf("loss %v, the full backward's is %v", loss, wantLoss)
	}
	for name, gradient := range want {
		held, isHeld := got[name]
		switch {
		case train(name) && !slices.Equal(held, gradient):
			t.Errorf("%s: the gradient differs from the full backward's", name)
		case !train(name) && isHeld:
			t.Errorf("%s: a weight left out has a gradient of %d values", name, len(held))
		}
	}
	if len(got) == 0 || len(got) >= len(want) {
		t.Errorf("%d gradients of the full backward's %d: want some, and fewer", len(got), len(want))
	}
}
