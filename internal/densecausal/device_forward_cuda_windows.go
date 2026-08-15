//go:build windows

package densecausal

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
)

// deviceLayerForwardCached is the device counterpart to layerForwardCached: it
// advances the residual stream x in place through one pre-norm layer on the GPU
// and returns the same forward intermediates (the layerCache the device backward
// consumes). The whole layer runs in ONE resident cudaBLAS session
// (devicemath.LayerForwardResident) -- weights + activations upload once and
// intermediates stay device-resident, no per-op round-trips. Parity target:
// host layerForwardCached.
func (m *Model) deviceLayerForwardCached(worker *device.Worker, x []float32, l layer, invFreq []float32, seq int) (layerCache, error) {
	d := m.Dims
	if ok, reason := DeviceTrainingSupported(d); !ok {
		return layerCache{}, fmt.Errorf("deviceLayerForwardCached: %s", reason)
	}
	fc, xOut, err := devicemath.LayerForwardResident(worker, x, devicemath.LayerForwardWeights{
		InLN: l.inLN, PostLN: l.postLN,
		Q: l.q, K: l.k, V: l.v, O: l.o,
		QBias: l.qb, KBias: l.kb, VBias: l.vb,
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
