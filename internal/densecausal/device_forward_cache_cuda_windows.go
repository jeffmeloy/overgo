//go:build windows

package densecausal

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// layerCache holds the per-layer forward intermediates the device backward
// consumes, so it need not recompute the forward on host. Saved by
// layerForwardCached during forwardStatesCached.
type layerCache struct {
	xn       []float32 // RMSNorm(x, inLN)
	tr       attnTrace // qScaled, kRoped, v, attnCore
	h2       []float32 // residual stream after the attention add (MLP input residual)
	hn       []float32 // RMSNorm(h2, postLN)
	gate, up []float32 // pre-activation gate/up projections
	a, hMLP  []float32 // silu(gate), silu(gate)*up
}

// layerForwardCached is layerForward that also returns the intermediates the
// backward needs. x advances in place, identically to layerForward.
func (m *Model) layerForwardCached(x []float32, l layer, invFreq []float64, seq int) layerCache {
	d := m.Dims
	width := d.Heads * d.HeadDim
	c := layerCache{}
	c.xn = make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(c.xn, x, l.inLN, seq, d.Hidden, d.RMSEps)
	c.tr = m.attnSubForward(l, c.xn, invFreq, seq)
	attnOut := make([]float32, seq*d.Hidden)
	hostmath.Linear(attnOut, c.tr.attnCore, l.o, seq, width, d.Hidden)
	for i := range x {
		x[i] += attnOut[i]
	}
	c.h2 = append([]float32(nil), x...)
	c.hn = make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(c.hn, x, l.postLN, seq, d.Hidden, d.RMSEps)
	c.gate = make([]float32, seq*d.Intermediate)
	c.up = make([]float32, seq*d.Intermediate)
	hostmath.Linear(c.gate, c.hn, l.gate, seq, d.Hidden, d.Intermediate)
	hostmath.Linear(c.up, c.hn, l.up, seq, d.Hidden, d.Intermediate)
	c.a = make([]float32, seq*d.Intermediate)
	c.hMLP = make([]float32, seq*d.Intermediate)
	for i := range c.gate {
		gi := float64(c.gate[i])
		c.a[i] = float32(gi / (1 + math.Exp(-gi)))
		c.hMLP[i] = c.a[i] * c.up[i]
	}
	mlp := make([]float32, seq*d.Hidden)
	hostmath.Linear(mlp, c.hMLP, l.down, seq, d.Intermediate, d.Hidden)
	for i := range x {
		x[i] += mlp[i]
	}
	return c
}

// forwardStatesCached runs the stack retaining each layer's input residual
// stream AND its forward intermediates, so the device backward consumes the
// cache instead of recomputing the forward per layer.
func (m *Model) forwardStatesCached(tokens []int) ([][]float32, []layerCache, error) {
	d := m.Dims
	seq := len(tokens)
	embed := m.Weights["model.embed_tokens.weight"]
	x := make([]float32, seq*d.Hidden)
	for t, id := range tokens {
		if id < 0 || id >= d.Vocab {
			return nil, nil, fmt.Errorf("densecausal: token %d out of vocab %d", id, d.Vocab)
		}
		copy(x[t*d.Hidden:(t+1)*d.Hidden], embed[id*d.Hidden:(id+1)*d.Hidden])
	}
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	states := make([][]float32, d.Layers+1)
	caches := make([]layerCache, d.Layers)
	for index := 0; index < d.Layers; index++ {
		states[index] = append([]float32(nil), x...)
		l, err := m.layerWeights(index)
		if err != nil {
			return nil, nil, err
		}
		caches[index] = m.layerForwardCached(x, l, invFreq, seq)
	}
	states[d.Layers] = x
	return states, caches, nil
}
