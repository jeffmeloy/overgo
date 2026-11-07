package hybridtrain

import (
	"errors"
	"fmt"
	"math/rand"

	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

// languageModel is a hybrid stack read as a language model, the host reference a device walk of its frozen backward is held to: the embedding rows its tokens take, its decoder layers,
// the final norm, and the head that scores the vocabulary. Geometry gives the layers' shape, its Tokens set by each
// sequence.
type languageModel struct {
	Geometry  StackConfig
	Vocab     int
	Embedding []float32
	Layers    []hostmath.HybridLayerWeights
	FinalNorm []float32
	Head      []float32
}

// maskedLossAndInputGradient is the model's mean next-token loss over the trained tokens, each predicted from the
// token before it, with rows put in place of the embeddings of the tokens at positions, every weight frozen, and the
// loss's gradient with respect to those rows: a frozen hybrid model's backward to an adapter that feeds its input, as
// densecausal's is for a dense model.
func (m *languageModel) maskedLossAndInputGradient(tokens []int, trained []bool, positions []int, rows []float32) (float64, []float32, error) {
	hidden, length := m.Geometry.Hidden, len(tokens)
	if len(trained) != length {
		return 0, nil, fmt.Errorf("hybridtrain: %d trained flags for %d tokens", len(trained), length)
	}
	// Row r predicts token r+1, so a trained token's row is the one before it.
	var predictors, targets []int
	for at := 1; at < length; at++ {
		if trained[at] {
			predictors, targets = append(predictors, at-1), append(targets, tokens[at])
		}
	}
	if len(predictors) == 0 {
		return 0, nil, errors.New("hybridtrain: no token after the first is trained")
	}
	pass, err := m.forward(tokens, positions, rows)
	if err != nil {
		return 0, nil, err
	}
	out, trace, dims, states, eps := pass.out, pass.trace, pass.dims, pass.states, m.Geometry.Eps
	normed := make([]float32, length*hidden)
	hostmath.RMSNormInto(normed, out, m.FinalNorm, length, hidden, eps)
	// Only the rows that predict a trained token reach the head.
	scored := make([]float32, len(predictors)*hidden)
	hostmath.GatherRows(scored, normed, predictors, hidden)
	logits := hostmath.Linear2(scored, m.Head, len(predictors), hidden, m.Vocab)
	dLogits := make([]float32, len(logits))
	loss := hostmath.SoftmaxCrossEntropy(dLogits, logits, targets, len(predictors), m.Vocab)
	dScored := make([]float32, len(scored))
	hostmath.LinearBackward(dScored, nil, nil, scored, m.Head, dLogits, len(predictors), hidden, m.Vocab, false)
	dNormed := make([]float32, length*hidden)
	hostmath.ScatterRows(dNormed, dScored, predictors, hidden)
	dOut := make([]float32, length*hidden)
	hostmath.RMSNormBackward(dOut, make([]float32, hidden), out, m.FinalNorm, dNormed, length, hidden, eps, false)
	_, dX := hostmath.HybridStackBackward(trace, m.Layers, dims, states, dOut, nil)
	gradient := make([]float32, len(rows))
	hostmath.GatherRows(gradient, dX, positions, hidden)
	return loss, gradient, nil
}

// stackPass is a forward through the stack: its output, and the trace, the layers' shapes and the recurrent states its
// backward reads.
type stackPass struct {
	out    []float32
	trace  hostmath.HybridStackTrace
	dims   []hostmath.HybridLayerDims
	states [][]float32
}

// forward runs the stack over the tokens' embeddings with rows in place of the embeddings of the tokens at positions.
func (m *languageModel) forward(tokens []int, positions []int, rows []float32) (stackPass, error) {
	hidden, length := m.Geometry.Hidden, len(tokens)
	if len(m.Layers) != len(m.Geometry.Types) || len(rows) != len(positions)*hidden {
		return stackPass{}, fmt.Errorf("hybridtrain: %d layers for %d kinds and %d values for %d placed rows", len(m.Layers), len(m.Geometry.Types), len(rows), len(positions))
	}
	x := make([]float32, length*hidden)
	for at, token := range tokens {
		if token < 0 || token >= m.Vocab {
			return stackPass{}, fmt.Errorf("hybridtrain: token %d outside the vocabulary of %d", token, m.Vocab)
		}
		copy(x[at*hidden:(at+1)*hidden], m.Embedding[token*hidden:(token+1)*hidden])
	}
	for _, position := range positions {
		if position < 0 || position >= length {
			return stackPass{}, errors.New("hybridtrain: a placed row stands outside the tokens")
		}
	}
	hostmath.ScatterRows(x, rows, positions, hidden)
	geometry := m.Geometry
	geometry.Tokens = length
	pass := stackPass{dims: make([]hostmath.HybridLayerDims, len(m.Layers)), states: make([][]float32, len(m.Layers))}
	for layer, kind := range geometry.Types {
		pass.dims[layer] = geometry.layerDims(kind)
		if kind == LinearAttention {
			pass.states[layer] = make([]float32, geometry.GDNValueHeads*geometry.GDNHeadDim*geometry.GDNHeadDim)
		}
	}
	pass.out, pass.trace = hostmath.HybridStackForward(x, m.Layers, pass.dims, pass.states)
	return pass, nil
}

// smallHybridModel is a small hybrid language model of a gated-delta layer and a gated attention layer with partial
// rotary positions, as Qwen3.5 holds them, its weights drawn from seed, and the draw that made them.
func smallHybridModel(seed int64) (*languageModel, func(int) []float32) {
	const hidden, inter, headDim, vocab = 8, 16, 4, 12
	rng := rand.New(rand.NewSource(seed))
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
	return &languageModel{
		Geometry: geometry, Vocab: vocab, Embedding: draw(vocab * hidden), FinalNorm: draw(hidden), Head: draw(vocab * hidden),
		Layers: []hostmath.HybridLayerWeights{
			{InputNorm: draw(hidden), PostNorm: draw(hidden), IsLinear: true, MLP: mlp(), GDN: hostmath.GatedDeltaMixWeights{
				Wq: draw(keyDim * hidden), Wk: draw(keyDim * hidden), Wv: draw(valueDim * hidden),
				ConvQ: draw(keyDim * 3), ConvK: draw(keyDim * 3), ConvV: draw(valueDim * 3),
				Wbeta: draw(2 * hidden), Walpha: draw(2 * hidden), TimeStep: draw(2), A: draw(2),
				Wz: draw(valueDim * hidden), Norm: draw(headDim), Wout: draw(hidden * valueDim),
			}},
			{InputNorm: draw(hidden), PostNorm: draw(hidden), MLP: mlp(), Attn: hostmath.AttentionMixWeights{
				Wq: draw(queryDim * hidden), Wk: draw(kvDim * hidden), Wv: draw(kvDim * hidden), Wo: draw(hidden * queryDim),
				QNorm: draw(headDim), KNorm: draw(headDim), Gate: draw(queryDim * hidden),
			}},
		},
	}, draw
}
