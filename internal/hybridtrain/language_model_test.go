package hybridtrain

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/testcheck"
)

// TestTheHybridStackGivesItsInputGradient holds a frozen hybrid language model's gradient with respect to rows placed
// at the prompt's positions to the loss's central differences, over a stack of a gated-delta layer and a gated
// attention layer with partial rotary positions, as Qwen3.5 holds them; placing the tokens' own embeddings gives the
// unplaced loss, and the weights stay as they were.
func TestTheHybridStackGivesItsInputGradient(t *testing.T) {
	t.Parallel()
	const step, tolerance = 1e-3, 2e-2
	model, draw := smallHybridModel(20261106)
	hidden := model.Geometry.Hidden
	tokens := []int{3, 0, 0, 7, 1, 9, 4}
	trained := []bool{false, false, false, false, true, true, true}
	positions := []int{1, 2}
	rows := draw(len(positions) * hidden)
	head := slices.Clone(model.Head)
	frozen := func(placed []float32) (float64, []float32) {
		t.Helper()
		loss, gradient, err := model.maskedLossAndInputGradient(tokens, trained, positions, placed)
		testcheck.NoError(t, err)
		return loss, gradient
	}
	loss, gradient := frozen(rows)
	for element := range rows {
		shifted := slices.Clone(rows)
		shifted[element] += step
		above, _ := frozen(shifted)
		shifted[element] -= 2 * step
		below, _ := frozen(shifted)
		want := (above - below) / (2 * step)
		if math.Abs(float64(gradient[element])-want) > tolerance*max(1, math.Abs(want)) {
			t.Errorf("row value %d: gradient %v, central difference %v", element, gradient[element], want)
		}
	}
	own := make([]float32, len(rows))
	hostmath.GatherRows(own, model.Embedding, []int{tokens[1], tokens[2]}, hidden)
	placed, _ := frozen(own)
	unplaced, _, err := model.maskedLossAndInputGradient(tokens, trained, nil, nil)
	testcheck.NoError(t, err)
	if placed != unplaced || loss == placed {
		t.Errorf("the tokens' own embeddings placed give %v, the unplaced loss %v; other rows gave %v", placed, unplaced, loss)
	}
	if !slices.Equal(head, model.Head) {
		t.Error("the frozen backward changed the model's weights")
	}
}
