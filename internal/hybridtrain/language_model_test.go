package hybridtrain

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheHybridStackGivesItsInputGradient holds a frozen hybrid language model's gradient with respect to rows placed
// at the prompt's positions to the loss's central differences, over a stack of a gated-delta layer and a gated
// attention layer with partial rotary positions, as Qwen3.5 holds them; placing the tokens' own embeddings gives the
// unplaced loss, and the weights stay as they were.
func TestTheHybridStackGivesItsInputGradient(t *testing.T) {
	t.Parallel()
	const hidden, inter, headDim, vocab, step, tolerance = 8, 16, 4, 12, 1e-3, 2e-2
	rng := rand.New(rand.NewSource(20261106))
	draw := func(count int) []float32 { return testutil.NormalVector(rng, count, 0.4) }
	mlp := func() hostmath.QwenMLPWeights {
		return hostmath.QwenMLPWeights{Gate: draw(inter * hidden), Up: draw(inter * hidden), Down: draw(hidden * inter)}
	}
	geometry := StackConfig{
		Types: []LayerKind{LinearAttention, FullAttention}, Hidden: hidden, Inter: inter, Eps: 1e-6,
		Heads: 2, KVHeads: 1, HeadDim: headDim, RopeDim: 2, RopeTheta: 10000,
		GDNKeyHeads: 2, GDNValueHeads: 2, GDNHeadDim: headDim, GDNConvK: 3,
	}
	keyDim, valueDim, queryDim, kvDim := 2*headDim, 2*headDim, 2*headDim, headDim
	model := &languageModel{
		Geometry: geometry, Vocab: vocab, Embedding: draw(vocab * hidden), FinalNorm: draw(hidden), Head: draw(vocab * hidden),
		Layers: []hostmath.HybridLayerWeights{
			{InputNorm: draw(hidden), PostNorm: draw(hidden), IsLinear: true, MLP: mlp(), GDN: hostmath.GatedDeltaMixWeights{
				Wq: draw(keyDim * hidden), Wk: draw(keyDim * hidden), Wv: draw(valueDim * hidden),
				ConvQ: draw(keyDim * 3), ConvK: draw(keyDim * 3), ConvV: draw(valueDim * 3),
				ConvBiasQ: draw(keyDim), ConvBiasK: draw(keyDim), ConvBiasV: draw(valueDim),
				Wbeta: draw(2 * hidden), Walpha: draw(2 * hidden), TimeStep: draw(2), A: draw(2),
				Wz: draw(valueDim * hidden), Norm: draw(headDim), Wout: draw(hidden * valueDim),
			}},
			{InputNorm: draw(hidden), PostNorm: draw(hidden), MLP: mlp(), Attn: hostmath.AttentionMixWeights{
				Wq: draw(queryDim * hidden), Wk: draw(kvDim * hidden), Wv: draw(kvDim * hidden), Wo: draw(hidden * queryDim),
				QNorm: draw(headDim), KNorm: draw(headDim), Gate: draw(queryDim * hidden),
			}},
		},
	}
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
