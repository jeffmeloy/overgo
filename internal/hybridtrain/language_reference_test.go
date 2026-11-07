package hybridtrain

import (
	"errors"
	"fmt"

	"overgo/internal/hostmath"
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
	if len(trained) != length || len(m.Layers) != len(m.Geometry.Types) || len(rows) != len(positions)*hidden {
		return 0, nil, fmt.Errorf("hybridtrain: %d tokens, %d trained flags, %d layers and %d values for %d placed rows", length, len(trained), len(m.Layers), len(rows), len(positions))
	}
	x := make([]float32, length*hidden)
	for at, token := range tokens {
		if token < 0 || token >= m.Vocab {
			return 0, nil, fmt.Errorf("hybridtrain: token %d outside the vocabulary of %d", token, m.Vocab)
		}
		copy(x[at*hidden:(at+1)*hidden], m.Embedding[token*hidden:(token+1)*hidden])
	}
	for _, position := range positions {
		if position < 0 || position >= length {
			return 0, nil, errors.New("hybridtrain: a placed row stands outside the tokens")
		}
	}
	hostmath.ScatterRows(x, rows, positions, hidden)
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
	geometry := m.Geometry
	geometry.Tokens = length
	dims, states := make([]hostmath.HybridLayerDims, len(m.Layers)), make([][]float32, len(m.Layers))
	for layer, kind := range geometry.Types {
		dims[layer] = geometry.layerDims(kind)
		if kind == LinearAttention {
			states[layer] = make([]float32, geometry.GDNValueHeads*geometry.GDNHeadDim*geometry.GDNHeadDim)
		}
	}
	out, trace := hostmath.HybridStackForward(x, m.Layers, dims, states)
	normed := make([]float32, length*hidden)
	hostmath.RMSNormInto(normed, out, m.FinalNorm, length, hidden, geometry.Eps)
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
	hostmath.RMSNormBackward(dOut, make([]float32, hidden), out, m.FinalNorm, dNormed, length, hidden, geometry.Eps, false)
	_, dX := hostmath.HybridStackBackward(trace, m.Layers, dims, states, dOut, nil)
	gradient := make([]float32, len(rows))
	hostmath.GatherRows(gradient, dX, positions, hidden)
	return loss, gradient, nil
}
