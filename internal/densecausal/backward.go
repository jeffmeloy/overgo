//overgo:runtime-inputs caller

// Backward for the dense causal-LM capability, composed from hostmath VJPs
// in checkpoint posture: only the per-layer residual-stream inputs are
// retained; every trace is recomputed from them. Gradients are keyed by the
// checkpoint tensor names an optimizer consumes. Head grads use the compiled binding:
package densecausal

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/hostmath"
)

// Grads accumulates named weight gradients, allocated on first touch.
type Grads map[string][]float32

// LossAndGrads runs forward + full backward on one token batch: causal-LM
// mean CE (positions 0..n-2 predict tokens 1..n-1), the full logits, and
// every parameter gradient.
func (m *Model) LossAndGrads(tokens []int) (float64, []float32, Grads, error) {
	return m.lossAndGrads(tokens, nil, true, nil, Grads{})
}

// LossAndGradsOf is LossAndGrads for a run that trains some of the weights,
// and returns the loss and the gradients of the weights train names alone: the
// others cost neither their memory nor their weight-gradient arithmetic.
func (m *Model) LossAndGradsOf(tokens []int, train func(name string) bool) (float64, Grads, error) {
	skipped := m.skippedGradients(train)
	loss, _, gradients, err := m.lossAndGrads(tokens, nil, true, nil, skipped)
	maps.DeleteFunc(gradients, func(_ string, gradient []float32) bool { return gradient == nil })
	return loss, gradients, err
}

// MaskedLossAndGrads runs forward and full backward on one token batch as
// LossAndGrads does, but learns only some tokens: trained says, token by
// token, whether the model is judged on predicting it, so the loss is the
// mean cross-entropy over the trained tokens alone, a token that is not
// trained adds nothing to it or to any gradient, and a window of no
// trained token is refused. The first token is predicted by nothing, so
// its flag is not read. A nil trained learns every token LossAndGrads
// does.
func (m *Model) MaskedLossAndGrads(tokens []int, trained []bool) (float64, Grads, error) {
	return m.MaskedLossAndGradsOf(tokens, trained, func(string) bool { return true })
}

// MaskedLossAndGradsOf is MaskedLossAndGrads for a run that trains some of the
// weights, as LossAndGradsOf is for LossAndGrads: it returns the gradients of the
// weights train names alone. A nil trained is LossAndGradsOf itself, which keeps
// the logits through the backward, the faster path and the one whose loss and
// gradients are the same bits as ever; a mask runs the windowed head, which the
// retained logits cannot serve.
func (m *Model) MaskedLossAndGradsOf(tokens []int, trained []bool, train func(name string) bool) (float64, Grads, error) {
	if trained == nil {
		return m.LossAndGradsOf(tokens, train)
	}
	skipped := m.skippedGradients(train)
	loss, _, gradients, err := m.lossAndGrads(tokens, trained, false, nil, skipped)
	maps.DeleteFunc(gradients, func(_ string, gradient []float32) bool { return gradient == nil })
	return loss, gradients, err
}

// MaskedLossAndLayerGradsOf is MaskedLossAndGradsOf for a run that takes the
// gradients of each decoder layer's weights as that layer's backward finishes: settle
// is called with them, layers last to first, and they are released as it returns, so a
// step holds one layer's gradients at a time, not every layer's. The gradients returned
// are those of the weights outside the layers (the embedding, the head and the final
// norm) that train names. The loss and every gradient are the bits MaskedLossAndGradsOf
// gives.
func (m *Model) MaskedLossAndLayerGradsOf(tokens []int, trained []bool, train func(name string) bool, settle LayerSettler) (float64, Grads, error) {
	skipped := m.skippedGradients(train)
	loss, _, gradients, err := m.lossAndGradsSettled(tokens, trained, trained == nil, nil, skipped, settle)
	maps.DeleteFunc(gradients, func(_ string, gradient []float32) bool { return gradient == nil })
	return loss, gradients, err
}

// skippedGradients holds a nil gradient for each weight train does not name, the
// form lossAndGrads reads as a gradient to leave uncomputed.
func (m *Model) skippedGradients(train func(name string) bool) Grads {
	skipped := Grads{}
	for name := range m.Weights {
		if !train(name) {
			skipped[name] = nil
		}
	}
	return skipped
}

func (m *Model) trainingLossAndGrads(tokens []int, trained []bool, step int, observeRouter MoERouterObserver) (float64, Grads, error) {
	var observation *routerObservationContext
	if observeRouter != nil {
		observation = &routerObservationContext{step: step, observer: observeRouter}
	}
	loss, _, gradients, err := m.lossAndGrads(tokens, trained, false, observation, Grads{})
	return loss, gradients, err
}

type routerObservationContext struct {
	step     int
	observer MoERouterObserver
}

// lossAndGrads accumulates into g, where a name held with a nil gradient is a
// weight whose gradient is skipped.
func (m *Model) lossAndGrads(tokens []int, trained []bool, retainLogits bool, observation *routerObservationContext, g Grads) (float64, []float32, Grads, error) {
	return m.lossAndGradsSettled(tokens, trained, retainLogits, observation, g, nil)
}

// LayerSettler takes the gradients of one decoder layer's weights as that layer's
// backward finishes. It owns them only for the call: they are released as it returns.
type LayerSettler func(layer int, gradients Grads) error

// releaseLayerGradients drops the layer's gradients, which no later layer's backward
// reads, so that a pass that settles each layer holds one layer's at a time.
func (m *Model) releaseLayerGradients(index int, gradients Grads) {
	names := m.layers[index].names
	for _, name := range []string{names.InLN, names.Q, names.K, names.V, names.O, names.QB, names.KB, names.VB, names.PostLN, names.Gate, names.Up, names.Down} {
		delete(gradients, name)
	}
}

// lossAndGradsSettled is lossAndGrads that, when settle is set, hands it each layer's
// gradients as that layer's backward finishes and releases them once it returns, so
// the gradients held at once are one layer's and not every layer's.
func (m *Model) lossAndGradsSettled(tokens []int, trained []bool, retainLogits bool, observation *routerObservationContext, g Grads, settle LayerSettler) (float64, []float32, Grads, error) {
	if len(tokens) < 2 {
		return 0, nil, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	if trained != nil && (len(trained) != len(tokens) || retainLogits) {
		return 0, nil, nil, fmt.Errorf("densecausal: %d trained flags for %d tokens, or flags asked with retained logits", len(trained), len(tokens))
	}
	// The host trainer runs full causal attention; within the artifact's
	// declared serving window the two are identical, beyond it they diverge,
	// so longer training sequences are refused rather than silently wrong.
	if m.Dims.AttentionWindow > 0 && len(tokens) > m.Dims.AttentionWindow {
		return 0, nil, nil, fmt.Errorf("densecausal: training sequence %d exceeds the declared attention window %d", len(tokens), m.Dims.AttentionWindow)
	}
	states, err := m.forwardStates(tokens, true)
	if err != nil {
		return 0, nil, nil, err
	}
	invFreq := hostmath.RopeInvFreq(m.Dims.RopeTheta, m.Dims.HeadDim)
	return m.lossAndGradsFromStates(tokens, trained, states, retainLogits, g, func(
		index int, input, outputGradient []float32, sequence int, gradients Grads,
	) ([]float32, error) {
		dx, err := m.layerBackwardObserved(index, input, outputGradient, invFreq, sequence, gradients, observation)
		if err != nil || settle == nil {
			return dx, err
		}
		if err := settle(index, gradients); err != nil {
			return nil, err
		}
		m.releaseLayerGradients(index, gradients)
		return dx, nil
	})
}

type layerGradient func(
	index int,
	input, outputGradient []float32,
	sequence int,
	gradients Grads,
) ([]float32, error)

func (m *Model) lossAndGradsFromStates(
	tokens []int,
	trained []bool,
	states [][]float32,
	retainLogits bool,
	g Grads,
	backward layerGradient,
) (float64, []float32, Grads, error) {
	d := m.Dims
	seq := len(tokens)
	final := states[d.Layers]
	normed := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(normed, final, m.tensors.finalNorm.values, seq, d.Hidden, d.RMSEps)
	head := m.tensors.head.values
	gradHead := hostmath.GradientSlot(g, m.tensors.head.name, len(head))
	dNormed := make([]float32, seq*d.Hidden)
	var loss float64
	var logits []float32
	if retainLogits {
		logits = make([]float32, seq*d.Vocab)
		hostmath.Linear(logits, normed, head, seq, d.Hidden, d.Vocab)
		dLogits := make([]float32, seq*d.Vocab)
		loss = hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)
		hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
	} else {
		// A row predicts the token after it, so the flags of the tokens from the second on select the rows.
		var selected []bool
		if trained != nil {
			selected = trained[1:]
			if !slices.Contains(selected, true) {
				return 0, nil, nil, errors.New("densecausal: no token of the window is trained")
			}
		}
		loss = headBackward(normed[:(seq-1)*d.Hidden], tokens[1:], selected, head, gradHead, dNormed[:(seq-1)*d.Hidden], d.Hidden, d.Vocab, headWindowRows(d.Vocab))
	}
	dx := make([]float32, seq*d.Hidden)
	hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, m.tensors.finalNorm.name, d.Hidden), final, m.tensors.finalNorm.values, dNormed, seq, d.Hidden, d.RMSEps, false)

	for index := d.Layers - 1; index >= 0; index-- {
		var err error
		dx, err = backward(index, states[index], dx, seq, g)
		if err != nil {
			return 0, nil, nil, err
		}
	}
	// Input-embedding scatter (the second contribution when tied; the only
	// embedding contribution when untied).
	embed := m.tensors.embedding.values
	gradEmbed := hostmath.GradientSlot(g, m.tensors.embedding.name, len(embed))
	if gradEmbed != nil {
		scatterEmbeddingGradient(gradEmbed, dx, tokens, d.Hidden)
	}
	return loss, logits, g, nil
}

// headWindowBytes bounds the logits and their gradient a logit-free backward
// holds at once, so the memory a long sequence over a large vocabulary needs
// does not grow with the sequence.
const headWindowBytes = 1 << 28

// headWindowBuffers is how many matrices of a window's rows by the vocabulary
// the head holds at once: the logits and their gradient.
const headWindowBuffers = 2

// headWindowRows is how many rows the head is applied to at once: as many as
// their logits and gradient fit in the window, never none.
func headWindowRows(vocab int) int {
	return max(headWindowBytes/(headWindowBuffers*vocab*binaryschema.Uint32Bytes), 1)
}

// headBackward is the mean cross-entropy of the head over the selected rows of
// normed (every row when selected is nil) against their targets, and its
// gradient: dNormed is set, a row not selected keeping the zero it holds, and
// gradHead (when not nil) accumulated. The selected rows are gathered and the
// head is applied to window of them at a time, each window's logits on the
// worker pool and its gradient through the pooled linear backward, so the
// logits of the whole sequence are never held and a row the run does not train
// costs the head nothing. Each row's gradient is its own, so the windows add up
// to the selected rows'.
func headBackward(normed []float32, targets []int, selected []bool, head, gradHead, dNormed []float32, hidden, vocab, window int) float64 {
	chosen := make([]int, 0, len(targets))
	for row := range targets {
		if selected == nil || selected[row] {
			chosen = append(chosen, row)
		}
	}
	width := min(window, len(chosen))
	logits, dLogits := make([]float32, width*vocab), make([]float32, width*vocab)
	gathered, dGathered := make([]float32, width*hidden), make([]float32, width*hidden)
	picked := make([]int, width)
	var loss float64
	for start := 0; start < len(chosen); start += window {
		members := chosen[start:min(start+window, len(chosen))]
		count := len(members)
		for index, row := range members {
			copy(gathered[index*hidden:(index+1)*hidden], normed[row*hidden:(row+1)*hidden])
			picked[index] = targets[row]
		}
		hostmath.Linear(logits[:count*vocab], gathered[:count*hidden], head, count, hidden, vocab)
		mean := hostmath.SoftmaxCrossEntropy(dLogits[:count*vocab], logits[:count*vocab], picked[:count], count, vocab)
		// The window's mean and gradient are over its own rows; the selected rows' are over all of them.
		share := float64(count) / float64(len(chosen))
		loss += mean * share
		scale := float32(share)
		for index := range dLogits[:count*vocab] {
			dLogits[index] *= scale
		}
		hostmath.LinearBackward(dGathered[:count*hidden], gradHead, nil, gathered[:count*hidden], head, dLogits[:count*vocab], count, hidden, vocab, false)
		for index, row := range members {
			copy(dNormed[row*hidden:(row+1)*hidden], dGathered[index*hidden:(index+1)*hidden])
		}
	}
	return loss
}

// scatterEmbeddingGradient adds each token's input gradient row onto its
// embedding row's gradient.
func scatterEmbeddingGradient(embedding, gradient []float32, tokens []int, hidden int) {
	for token, id := range tokens {
		row := embedding[id*hidden : (id+1)*hidden]
		addInPlace(row, gradient[token*hidden:(token+1)*hidden])
	}
}

// layerBackward: VJP of layerForward. x is the layer input residual stream
// (retained); dOut arrives at the layer output; dx returns at the input.
func (m *Model) layerBackward(index int, x, dOut []float32, invFreq []float64, seq int, g Grads) ([]float32, error) {
	return m.layerBackwardObserved(index, x, dOut, invFreq, seq, g, nil)
}

func (m *Model) layerBackwardObserved(
	index int,
	x, dOut []float32,
	invFreq []float64,
	seq int,
	g Grads,
	observationContext *routerObservationContext,
) ([]float32, error) {
	d := m.Dims
	l, err := m.useLayer(index)
	if err != nil {
		return nil, err
	}
	width := d.Heads * d.HeadDim
	kvWidth := d.KVHeads * d.HeadDim

	// Recompute the trace through the attention residual.
	xn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(xn, x, l.inLN, seq, d.Hidden, d.RMSEps)
	tr := m.attnSubForward(l, xn, invFreq, seq)
	attnOut := make([]float32, seq*d.Hidden)
	hostmath.Linear(attnOut, tr.attnCore, l.o, seq, width, d.Hidden)
	h2 := make([]float32, seq*d.Hidden)
	for i := range h2 {
		h2[i] = x[i] + attnOut[i]
	}
	hn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(hn, h2, l.postLN, seq, d.Hidden, d.RMSEps)

	// FFN branch backward; dh2 carries the residual plus the norm path.
	dh2 := append([]float32(nil), dOut...)
	dHn := make([]float32, seq*d.Hidden)
	if l.moe != nil {
		_, route, scores, err := moeForward(hn, *l.moe, seq, d.Hidden, d.MoE)
		if err != nil {
			return nil, err
		}
		if observationContext != nil {
			observation, err := newMoERouterObservation(index, route, scores, seq, len(l.moe.experts), d.MoE.TopK)
			if err != nil {
				return nil, err
			}
			if err := observationContext.observer(observationContext.step, observation); err != nil {
				return nil, err
			}
		}
		dMix, mixGrads, err := moeBackward(hn, *l.moe, seq, d.Hidden, d.MoE, route, dOut, g != nil)
		if err != nil {
			return nil, err
		}
		copy(dHn, dMix)
		accumulateMoEGrads(g, l.moe.names, mixGrads)
	} else {
		gate := make([]float32, seq*d.Intermediate)
		up := make([]float32, seq*d.Intermediate)
		hostmath.Linear(gate, hn, l.gate, seq, d.Hidden, d.Intermediate)
		hostmath.Linear(up, hn, l.up, seq, d.Hidden, d.Intermediate)
		h := make([]float32, seq*d.Intermediate)
		hostmath.SiLUGate(h, gate, up)
		dH := make([]float32, seq*d.Intermediate)
		hostmath.LinearBackward(dH, hostmath.GradientSlot(g, l.names.Down, d.Hidden*d.Intermediate), nil,
			h, l.down, dOut, seq, d.Intermediate, d.Hidden, false)
		dGate := make([]float32, seq*d.Intermediate)
		dUp := make([]float32, seq*d.Intermediate)
		hostmath.SiLUGateBackward(dGate, dUp, gate, up, dH)
		hostmath.LinearBackward(dHn, hostmath.GradientSlot(g, l.names.Gate, d.Intermediate*d.Hidden), nil,
			hn, l.gate, dGate, seq, d.Hidden, d.Intermediate, false)
		hostmath.LinearBackward(dHn, hostmath.GradientSlot(g, l.names.Up, d.Intermediate*d.Hidden), nil,
			hn, l.up, dUp, seq, d.Hidden, d.Intermediate, true)
	}
	hostmath.RMSNormBackward(dh2, hostmath.GradientSlot(g, l.names.PostLN, d.Hidden), h2, l.postLN, dHn, seq, d.Hidden, d.RMSEps, true)

	// Attention branch backward.
	dAttnCore := make([]float32, seq*width)
	hostmath.LinearBackward(dAttnCore, hostmath.GradientSlot(g, l.names.O, d.Hidden*width), nil,
		tr.attnCore, l.o, dh2, seq, width, d.Hidden, false)
	dq := make([]float32, seq*width)
	dk := make([]float32, seq*kvWidth)
	dv := make([]float32, seq*kvWidth)
	hostmath.CausalAttentionBackward(dq, dk, dv, tr.qScaled, tr.kRoped, tr.v, dAttnCore, seq, d.Heads, d.KVHeads, d.HeadDim)
	scale := float32(1 / math.Sqrt(float64(d.HeadDim)))
	for i := range dq {
		dq[i] *= scale
	}
	for p := range seq {
		for h := 0; h < d.Heads; h++ {
			hostmath.RotaryHalfBackward(dq[(p*d.Heads+h)*d.HeadDim:(p*d.Heads+h+1)*d.HeadDim], invFreq, p)
		}
		for h := 0; h < d.KVHeads; h++ {
			hostmath.RotaryHalfBackward(dk[(p*d.KVHeads+h)*d.HeadDim:(p*d.KVHeads+h+1)*d.HeadDim], invFreq, p)
		}
	}
	// Optional q/k/v bias grads (qwen2): dB sums dy rows; dx path unchanged.
	var dbQ, dbK, dbV []float32
	if l.qb != nil {
		dbQ = hostmath.GradientSlot(g, l.names.QB, width)
		dbK = hostmath.GradientSlot(g, l.names.KB, kvWidth)
		dbV = hostmath.GradientSlot(g, l.names.VB, kvWidth)
	}
	dXn := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dXn, hostmath.GradientSlot(g, l.names.Q, width*d.Hidden), dbQ,
		xn, l.q, dq, seq, d.Hidden, width, false)
	hostmath.LinearBackward(dXn, hostmath.GradientSlot(g, l.names.K, kvWidth*d.Hidden), dbK,
		xn, l.k, dk, seq, d.Hidden, kvWidth, true)
	hostmath.LinearBackward(dXn, hostmath.GradientSlot(g, l.names.V, kvWidth*d.Hidden), dbV,
		xn, l.v, dv, seq, d.Hidden, kvWidth, true)
	dx := dh2
	hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, l.names.InLN, d.Hidden), x, l.inLN, dXn, seq, d.Hidden, d.RMSEps, true)
	return dx, nil
}
