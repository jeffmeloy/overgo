//go:build windows

package densecausal

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
)

// deviceLossAndGrads is the device counterpart to LossAndGrads: it runs the
// per-layer backward on the GPU via deviceLayerBackward (the dominant cost) while
// the head projection, final RMSNorm, softmax-CE and embedding scatter stay on
// host. Loss, logits and every parameter gradient match LossAndGrads within fp32
// tolerance. Attention bias is not yet supported (AttnBias must be false).
func (m *Model) deviceLossAndGrads(worker *device.Worker, tokens []int) (float64, []float32, Grads, error) {
	if len(tokens) < 2 {
		return 0, nil, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	d := m.Dims
	seq := len(tokens)
	states, err := m.forwardStates(tokens)
	if err != nil {
		return 0, nil, nil, err
	}
	final := states[d.Layers]
	normed := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], seq, d.Hidden, d.RMSEps)
	head := m.head()
	logits := make([]float32, seq*d.Vocab)
	hostmath.Linear(logits, normed, head, seq, d.Hidden, d.Vocab)

	g := Grads{}
	dLogits := make([]float32, seq*d.Vocab)
	loss := hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)

	gradHead := g.slot(m.headName(), len(head))
	dNormed := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
	dx := make([]float32, seq*d.Hidden)
	hostmath.RMSNormBackward(dx, g.slot("model.norm.weight", d.Hidden), final, m.Weights["model.norm.weight"], dNormed, seq, d.Hidden, d.RMSEps, false)

	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	for index := d.Layers - 1; index >= 0; index-- {
		dx, err = m.deviceLayerBackward(worker, index, states[index], dx, invFreq, seq, g)
		if err != nil {
			return 0, nil, nil, err
		}
	}
	embed := m.Weights["model.embed_tokens.weight"]
	gradEmbed := g.slot("model.embed_tokens.weight", len(embed))
	for t, id := range tokens {
		row := gradEmbed[id*d.Hidden : (id+1)*d.Hidden]
		dRow := dx[t*d.Hidden : (t+1)*d.Hidden]
		for i := range row {
			row[i] += dRow[i]
		}
	}
	return loss, logits, g, nil
}
