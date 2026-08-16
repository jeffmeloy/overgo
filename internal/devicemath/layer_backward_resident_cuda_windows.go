//go:build windows

package devicemath

import (
	"overgo/internal/cuda/driver"
)

// LayerBackwardResult bundles a layer's input-gradient and weight-gradients
// (densecausal layout), returned by LayerBackwardResident.
type LayerBackwardResult struct {
	DX                     []float32
	DWInLN, DWPostLN       []float32
	DWQ, DWK, DWV, DWO     []float32
	DQBias, DKBias, DVBias []float32
	DWGate, DWUp, DWDown   []float32
}

// uploadLayerCache uploads a host forward cache into resident device buffers, so
// the standalone backward wrapper can consume the same forwardDevice/backwardDevice
// contract the whole-stack driver uses (which keeps the cache resident instead).
func uploadLayerCache(s *cudaScope, c LayerForwardCache) (layerCachePtrs, error) {
	var p layerCachePtrs
	for _, spec := range []struct {
		p *driver.DevicePtr
		d []float32
	}{
		{&p.xn, c.Xn}, {&p.qScaled, c.QScaled}, {&p.kRoped, c.KRoped}, {&p.v, c.V},
		{&p.attnCore, c.AttnCore}, {&p.h2, c.H2}, {&p.hn, c.Hn}, {&p.gate, c.Gate},
		{&p.up, c.Up}, {&p.a, c.A}, {&p.hMLP, c.HMLP},
	} {
		ptr, err := s.upload(spec.d)
		if err != nil {
			return layerCachePtrs{}, err
		}
		*spec.p = ptr
	}
	return p, nil
}
