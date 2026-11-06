package densecausal

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestInputGradientMatchesFiniteDifferences holds the frozen model's gradient with respect to rows placed at the
// prompt's positions to the loss's central differences, the rows off zero where a norm's backward is steep; placing
// the token's own embeddings gives the masked loss itself, and the weights take no gradient and stay as they were.
func TestInputGradientMatchesFiniteDifferences(t *testing.T) {
	t.Parallel()
	const step, tolerance = 1e-3, 2e-2
	model := tinyMuonModel(t)
	hidden := model.Dims.Hidden
	positions := []int{1, 2}
	draws := rand.New(rand.NewPCG(5, 9))
	rows := make([]float32, len(positions)*hidden)
	for index := range rows {
		rows[index] = float32(draws.NormFloat64() * 0.5)
	}
	before := slices.Clone(model.tensors.embedding.values)
	frozen := func(placed []float32) (float64, []float32) {
		t.Helper()
		loss, gradient, err := model.MaskedLossAndInputGradient(maskedTokens, maskedTrained, positions, placed)
		testcheck.NoError(t, err)
		return loss, gradient
	}
	loss, gradient := frozen(rows)
	for _, element := range []int{0, 3, hidden + 1, 2*hidden - 1} {
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
	for index, position := range positions {
		token := maskedTokens[position]
		copy(own[index*hidden:(index+1)*hidden], model.tensors.embedding.values[token*hidden:(token+1)*hidden])
	}
	placed, _ := frozen(own)
	masked, _, err := model.MaskedLossAndLayerGradsOf(maskedTokens, maskedTrained, everyWeight, nil)
	testcheck.NoError(t, err)
	if math.Abs(placed-masked) > 1e-6 || loss == placed {
		t.Errorf("the tokens' own embeddings placed give %v, the masked loss %v; other rows gave %v", placed, masked, loss)
	}
	if !slices.Equal(before, model.tensors.embedding.values) {
		t.Error("the frozen backward changed the model's weights")
	}
}
