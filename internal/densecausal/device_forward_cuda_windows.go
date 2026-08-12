//go:build windows

package densecausal

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
)

// deviceLayerForwardCached is the device counterpart to layerForwardCached: it
// advances the residual stream x in place through one pre-norm layer on the GPU
// and returns the same forward intermediates (the layerCache the device backward
// consumes). The whole layer runs in ONE resident cudaBLAS session
// (devicemath.LayerForwardResident) -- weights + activations upload once and
// intermediates stay device-resident, no per-op round-trips. Attention bias is
// not supported (matching deviceLayerBackward). Parity target: host layerForwardCached.
func (m *Model) deviceLayerForwardCached(worker *device.Worker, x []float32, l layer, invFreq []float32, seq int) (layerCache, error) {
	d := m.Dims
	if d.AttnBias {
		return layerCache{}, fmt.Errorf("deviceLayerForwardCached: attention bias not supported yet")
	}
	fc, xOut, err := devicemath.LayerForwardResident(worker, x, devicemath.LayerForwardWeights{
		InLN: l.inLN, PostLN: l.postLN,
		Q: l.q, K: l.k, V: l.v, O: l.o,
		Gate: l.gate, Up: l.up, Down: l.down,
	}, invFreq, seq, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps)
	if err != nil {
		return layerCache{}, err
	}
	copy(x, xOut) // advance the residual stream in place
	return layerCache{
		xn: fc.Xn,
		tr: attnTrace{qScaled: fc.QScaled, kRoped: fc.KRoped, v: fc.V, attnCore: fc.AttnCore},
		h2: fc.H2, hn: fc.Hn, gate: fc.Gate, up: fc.Up, a: fc.A, hMLP: fc.HMLP,
	}, nil
}

// deviceForwardStatesCached is the device counterpart to forwardStatesCached: it
// runs the whole stack forward on the GPU (embedding lookup on host, then each
// layer via deviceLayerForwardCached), retaining every layer's input residual
// stream and its cache. Replaces the host forward inside deviceLossAndGrads, so
// the per-layer forward no longer runs on host. states[l] is layer l's input;
// states[Layers] is the final pre-norm stream.
func (m *Model) deviceForwardStatesCached(worker *device.Worker, tokens []int) ([][]float32, []layerCache, error) {
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
	invF32 := ropeInvF32(hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim))
	states := make([][]float32, d.Layers+1)
	caches := make([]layerCache, d.Layers)
	for index := 0; index < d.Layers; index++ {
		states[index] = append([]float32(nil), x...)
		l, err := m.layerWeights(index)
		if err != nil {
			return nil, nil, err
		}
		caches[index], err = m.deviceLayerForwardCached(worker, x, l, invF32, seq)
		if err != nil {
			return nil, nil, err
		}
	}
	states[d.Layers] = x
	return states, caches, nil
}
