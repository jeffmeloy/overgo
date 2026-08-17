package densecausal

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// LayerStates exposes the per-layer hidden streams of one forward pass:
// states[0] is the embedded input and states[Layers] the final hidden. The
// residual-injection connector reads donor knowledge from and measures
// context gaps against these streams.
func (m *Model) LayerStates(tokens []int) ([][]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("densecausal: layer states need tokens")
	}
	return m.forwardStates(tokens)
}

// LossRangeInjected computes the mean cross-entropy over positions >= from
// with the hidden stream shifted at one layer boundary by magnitude-rescaled
// injection: h_i' = (1-alpha)*h_i + alpha*normalize(vector)*|h_i|. The
// injection preserves each position's activation scale, so a poorly aligned
// vector degrades gracefully instead of exploding the stream.
func (m *Model) LossRangeInjected(
	tokens []int, from, layer int, vector []float32, alpha float32,
) (float64, error) {
	d := m.Dims
	if len(tokens) < 2 || from < 1 || from >= len(tokens) {
		return 0, fmt.Errorf("densecausal: injected loss needs tokens and a target range")
	}
	if layer < 0 || layer >= d.Layers {
		return 0, fmt.Errorf("densecausal: injection layer %d outside %d layers", layer, d.Layers)
	}
	if len(vector) != d.Hidden {
		return 0, fmt.Errorf("densecausal: injection vector length %d, want %d", len(vector), d.Hidden)
	}
	norm := float64(0)
	for _, value := range vector {
		norm += float64(value) * float64(value)
	}
	norm = math.Sqrt(norm)
	if norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
		return 0, fmt.Errorf("densecausal: injection vector is degenerate")
	}
	unit := make([]float32, d.Hidden)
	for index, value := range vector {
		unit[index] = float32(float64(value) / norm)
	}
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	states, err := m.retainedForwardStates(tokens, func(x []float32, index, seq int) error {
		if index == layer {
			// Inject only at the scored positions: causal attention keeps
			// earlier positions independent of the elided information, so the
			// connector's signal belongs exactly where the teacher measured
			// the with-context/without-context difference.
			injectRescaled(x[from*d.Hidden:], unit, seq-from, d.Hidden, alpha)
		}
		l, err := m.layerWeights(index)
		if err != nil {
			return err
		}
		m.layerForward(x, l, invFreq, seq)
		return nil
	})
	if err != nil {
		return 0, err
	}
	seq := len(tokens)
	logits := m.logits(states[d.Layers], seq)
	rows := seq - from
	start := from - 1
	scratch := make([]float32, rows*d.Vocab)
	return hostmath.SoftmaxCrossEntropy(
		scratch, logits[start*d.Vocab:(start+rows)*d.Vocab], tokens[from:], rows, d.Vocab,
	), nil
}

func injectRescaled(x, unit []float32, seq, hidden int, alpha float32) {
	for position := 0; position < seq; position++ {
		row := x[position*hidden : (position+1)*hidden]
		magnitude := float64(0)
		for _, value := range row {
			magnitude += float64(value) * float64(value)
		}
		scale := alpha * float32(math.Sqrt(magnitude))
		for index := range row {
			row[index] = (1-alpha)*row[index] + scale*unit[index]
		}
	}
}
