//go:build windows

package densecausal

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
)

// deviceLayerForwardCached is the device counterpart to layerForwardCached: it
// advances the residual stream x in place through one pre-norm layer on the GPU
// and returns the same forward intermediates (the layerCache the device backward
// consumes). Each step is a devicemath forward primitive (correctness-first,
// per-op); a resident fused version is a later rung. Attention bias is not
// supported (matching deviceLayerBackward). Parity target: host layerForwardCached.
func (m *Model) deviceLayerForwardCached(worker *device.Worker, x []float32, l layer, invFreq []float32, seq int) (layerCache, error) {
	d := m.Dims
	if d.AttnBias {
		return layerCache{}, fmt.Errorf("deviceLayerForwardCached: attention bias not supported yet")
	}
	width := d.Heads * d.HeadDim
	kvWidth := d.KVHeads * d.HeadDim
	scale := float32(1 / math.Sqrt(float64(d.HeadDim)))
	c := layerCache{}
	var err error

	// xn = RMSNorm(x, inLN)
	if c.xn, err = devicemath.RMSNormForward(worker, x, l.inLN, seq, d.Hidden, d.RMSEps); err != nil {
		return layerCache{}, err
	}

	// Attention sub-forward: projections, split-half rope on q/k, scale fold,
	// causal GQA attention.
	tr := attnTrace{}
	if tr.qPost, err = devicemath.LinearForwardT(worker, c.xn, l.q, seq, d.Hidden, width); err != nil {
		return layerCache{}, err
	}
	if tr.kPost, err = devicemath.LinearForwardT(worker, c.xn, l.k, seq, d.Hidden, kvWidth); err != nil {
		return layerCache{}, err
	}
	if tr.v, err = devicemath.LinearForwardT(worker, c.xn, l.v, seq, d.Hidden, kvWidth); err != nil {
		return layerCache{}, err
	}
	if tr.qScaled, err = devicemath.RoPEHalfForward(worker, tr.qPost, invFreq, seq, d.Heads, d.HeadDim); err != nil {
		return layerCache{}, err
	}
	if tr.kRoped, err = devicemath.RoPEHalfForward(worker, tr.kPost, invFreq, seq, d.KVHeads, d.HeadDim); err != nil {
		return layerCache{}, err
	}
	for i := range tr.qScaled {
		tr.qScaled[i] *= scale
	}
	if tr.attnCore, err = devicemath.MultiHeadAttentionForwardResident(worker, tr.qScaled, tr.kRoped, tr.v, seq, d.Heads, d.KVHeads, d.HeadDim); err != nil {
		return layerCache{}, err
	}
	c.tr = tr

	// x += o(attnCore)
	attnOut, err := devicemath.LinearForwardT(worker, tr.attnCore, l.o, seq, width, d.Hidden)
	if err != nil {
		return layerCache{}, err
	}
	addInPlace(x, attnOut)
	c.h2 = append([]float32(nil), x...)

	// hn = RMSNorm(x, postLN); MLP branch
	if c.hn, err = devicemath.RMSNormForward(worker, x, l.postLN, seq, d.Hidden, d.RMSEps); err != nil {
		return layerCache{}, err
	}
	if c.gate, err = devicemath.LinearForwardT(worker, c.hn, l.gate, seq, d.Hidden, d.Intermediate); err != nil {
		return layerCache{}, err
	}
	if c.up, err = devicemath.LinearForwardT(worker, c.hn, l.up, seq, d.Hidden, d.Intermediate); err != nil {
		return layerCache{}, err
	}
	if c.a, c.hMLP, err = devicemath.SiLUGateForward(worker, c.gate, c.up); err != nil {
		return layerCache{}, err
	}
	mlp, err := devicemath.LinearForwardT(worker, c.hMLP, l.down, seq, d.Intermediate, d.Hidden)
	if err != nil {
		return layerCache{}, err
	}
	addInPlace(x, mlp)
	return c, nil
}

// deviceForwardStatesCached is the device counterpart to forwardStatesCached: it
// runs the whole stack forward on the GPU (embedding lookup on host, then each
// layer via deviceLayerForwardCached), retaining every layer's input residual
// stream and its cache. Replaces the host forward inside deviceLossAndGrads, so
// the per-layer forward no longer runs on host. states[l] is layer l's input;
// states[Layers] is the final pre-norm stream.
func (m *Model) deviceForwardStatesCached(worker *device.Worker, tokens []int) ([][]float32, []layerCache, error) {
	d := m.Dims
	invF32 := ropeInvF32(hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim))
	return m.cachedForwardStates(tokens, func(x []float32, index, seq int) (layerCache, error) {
		l, err := m.layerWeights(index)
		if err != nil {
			return layerCache{}, err
		}
		return m.deviceLayerForwardCached(worker, x, l, invF32, seq)
	})
}
