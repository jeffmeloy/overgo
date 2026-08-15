//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
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

// LayerBackwardResident runs a whole pre-norm layer backward in ONE cudaBLAS
// session: x, dOut, the forward cache and every weight upload once, and all
// intermediate gradients stay device-resident (no per-op round-trips). Only dX +
// the nine weight grads come back. The device compute is backwardDevice (shared
// with the whole-stack driver). Same math as densecausal.layerBackward.
func LayerBackwardResident(worker *device.Worker, x, dOut []float32, c LayerForwardCache, w LayerForwardWeights, invFreq []float32, seq, hidden, heads, kvHeads, hd, inter int, rmsEps float64) (LayerBackwardResult, error) {
	d := newLayerDims(seq, hidden, heads, kvHeads, hd, inter, rmsEps)
	if heads%kvHeads != 0 || len(x) != seq*hidden || len(dOut) != seq*hidden || len(invFreq) != hd/2 {
		return LayerBackwardResult{}, fmt.Errorf("LayerBackwardResident: shape mismatch (seq=%d hidden=%d heads=%d kv=%d hd=%d)", seq, hidden, heads, kvHeads, hd)
	}

	var r LayerBackwardResult
	r.DX = make([]float32, seq*hidden)
	r.DWInLN = make([]float32, hidden)
	r.DWPostLN = make([]float32, hidden)
	r.DWQ = make([]float32, d.width*hidden)
	r.DWK = make([]float32, d.kvWidth*hidden)
	r.DWV = make([]float32, d.kvWidth*hidden)
	if w.QBias != nil {
		r.DQBias = make([]float32, d.width)
		r.DKBias = make([]float32, d.kvWidth)
		r.DVBias = make([]float32, d.kvWidth)
	}
	r.DWO = make([]float32, hidden*d.width)
	r.DWGate = make([]float32, inter*hidden)
	r.DWUp = make([]float32, inter*hidden)
	r.DWDown = make([]float32, hidden*inter)

	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		ops, err := newLayerOps(s, layerBackwardFnNames, d, invFreq)
		if err != nil {
			return err
		}
		dOutP, err := s.upload(dOut)
		if err != nil {
			return err
		}
		xP, err := s.upload(x)
		if err != nil {
			return err
		}
		cp, err := uploadLayerCache(s.cudaScope, c)
		if err != nil {
			return err
		}
		wp, err := uploadLayerWeights(s.cudaScope, w)
		if err != nil {
			return err
		}
		gp, err := allocLayerGrads(s.cudaScope, d)
		if err != nil {
			return err
		}
		if err := allocLayerBiasGrads(s.cudaScope, &gp, d, w.QBias != nil); err != nil {
			return err
		}
		dXP, err := s.alloc(seq * hidden)
		if err != nil {
			return err
		}
		if err := ops.backwardDevice(xP, dOutP, cp, wp, gp, dXP); err != nil {
			return err
		}
		downloads := []cudaDownload{
			cudaDownload{r.DX, dXP},
			cudaDownload{r.DWDown, gp.dDown}, cudaDownload{r.DWGate, gp.dGate}, cudaDownload{r.DWUp, gp.dUp},
			cudaDownload{r.DWPostLN, gp.dPostLN}, cudaDownload{r.DWO, gp.dO},
			cudaDownload{r.DWQ, gp.dQ}, cudaDownload{r.DWK, gp.dK}, cudaDownload{r.DWV, gp.dV},
			cudaDownload{r.DWInLN, gp.dInLN},
		}
		if w.QBias != nil {
			downloads = append(downloads, cudaDownload{r.DQBias, gp.dQBias}, cudaDownload{r.DKBias, gp.dKBias}, cudaDownload{r.DVBias, gp.dVBias})
		}
		return s.finish(downloads...)
	})
	if err != nil {
		return LayerBackwardResult{}, err
	}
	return r, nil
}
