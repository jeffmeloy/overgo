//go:build windows

package densecausal

import (
	"math"

	"overgo/internal/hostmath"
)

// layerCache holds device-backward intermediates.
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
	addInPlace(x, attnOut)
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
	addInPlace(x, mlp)
	return c
}
