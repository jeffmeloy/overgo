//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// LayerForwardWeights bundles one pre-norm transformer layer's weights (host
// slices, densecausal layout: q/o are [heads*hd, hidden], k/v are [kv*hd, hidden],
// gate/up are [inter, hidden], down is [hidden, inter], norms are [hidden]).
type LayerForwardWeights struct {
	InLN, PostLN   []float32
	Q, K, V, O     []float32
	Gate, Up, Down []float32
}

// LayerForwardCache is the resident forward's downloaded intermediates, matching
// densecausal.layerCache (position-major q/k/v/attnCore).
type LayerForwardCache struct {
	Xn, QScaled, KRoped, V, AttnCore []float32
	H2, Hn, Gate, Up, A, HMLP        []float32
}

// uploadLayerWeights uploads one layer's weights into the session, returning the
// resident pointers shared by the forward and the backward.
func uploadLayerWeights(s *cudaScope, w LayerForwardWeights) (layerWeightPtrs, error) {
	var p layerWeightPtrs
	for _, spec := range []struct {
		p *driver.DevicePtr
		d []float32
	}{
		{&p.inLN, w.InLN}, {&p.postLN, w.PostLN}, {&p.q, w.Q}, {&p.k, w.K}, {&p.v, w.V},
		{&p.o, w.O}, {&p.gate, w.Gate}, {&p.up, w.Up}, {&p.down, w.Down},
	} {
		ptr, err := s.upload(spec.d)
		if err != nil {
			return layerWeightPtrs{}, err
		}
		*spec.p = ptr
	}
	return p, nil
}

// LayerForwardResident runs a whole pre-norm layer forward in ONE cudaBLAS
// session: x and every weight upload once, all intermediates stay on resident
// device buffers (no per-op round-trips), and only the cache + advanced residual
// stream come back. Same math as densecausal.layerForwardCached. The device
// compute is forwardDevice (shared with the whole-stack driver).
func LayerForwardResident(worker *device.Worker, x []float32, w LayerForwardWeights, invFreq []float32, seq, hidden, heads, kvHeads, hd, inter int, rmsEps float64) (LayerForwardCache, []float32, error) {
	d := newLayerDims(seq, hidden, heads, kvHeads, hd, inter, rmsEps)
	if heads%kvHeads != 0 || len(x) != seq*hidden || len(w.InLN) != hidden || len(w.PostLN) != hidden ||
		len(w.Q) != d.width*hidden || len(w.K) != d.kvWidth*hidden || len(w.V) != d.kvWidth*hidden || len(w.O) != hidden*d.width ||
		len(w.Gate) != inter*hidden || len(w.Up) != inter*hidden || len(w.Down) != hidden*inter || len(invFreq) != hd/2 {
		return LayerForwardCache{}, nil, fmt.Errorf("LayerForwardResident: shape mismatch (seq=%d hidden=%d heads=%d kv=%d hd=%d inter=%d)", seq, hidden, heads, kvHeads, hd, inter)
	}

	var c LayerForwardCache
	c.Xn = make([]float32, seq*hidden)
	c.QScaled = make([]float32, seq*d.width)
	c.KRoped = make([]float32, seq*d.kvWidth)
	c.V = make([]float32, seq*d.kvWidth)
	c.AttnCore = make([]float32, seq*d.width)
	c.H2 = make([]float32, seq*hidden)
	c.Hn = make([]float32, seq*hidden)
	c.Gate = make([]float32, seq*inter)
	c.Up = make([]float32, seq*inter)
	c.A = make([]float32, seq*inter)
	c.HMLP = make([]float32, seq*inter)
	xOut := make([]float32, seq*hidden)

	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		ops, err := newLayerOps(s, layerForwardFnNames, d, invFreq)
		if err != nil {
			return err
		}
		xP, err := s.upload(x)
		if err != nil {
			return err
		}
		wp, err := uploadLayerWeights(s.cudaScope, w)
		if err != nil {
			return err
		}
		cp, err := allocLayerCache(s.cudaScope, d)
		if err != nil {
			return err
		}
		xOutP, err := s.alloc(seq * hidden)
		if err != nil {
			return err
		}
		if err := ops.forwardDevice(xP, wp, cp, xOutP); err != nil {
			return err
		}
		return s.finish(
			cudaDownload{c.Xn, cp.xn}, cudaDownload{c.QScaled, cp.qScaled}, cudaDownload{c.KRoped, cp.kRoped},
			cudaDownload{c.V, cp.v}, cudaDownload{c.AttnCore, cp.attnCore}, cudaDownload{c.H2, cp.h2},
			cudaDownload{c.Hn, cp.hn}, cudaDownload{c.Gate, cp.gate}, cudaDownload{c.Up, cp.up},
			cudaDownload{c.A, cp.a}, cudaDownload{c.HMLP, cp.hMLP}, cudaDownload{xOut, xOutP},
		)
	})
	if err != nil {
		return LayerForwardCache{}, nil, err
	}
	return c, xOut, nil
}
