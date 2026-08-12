//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

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

// LayerForwardResident runs a whole pre-norm layer forward in ONE cudaBLAS
// session: x and every weight upload once, all intermediates stay on resident
// device buffers (no per-op round-trips), and only the cache + advanced residual
// stream come back. Same math as densecausal.layerForwardCached. This is the
// resident fused counterpart to the per-op device forward.
func LayerForwardResident(worker *device.Worker, x []float32, w LayerForwardWeights, invFreq []float32, seq, hidden, heads, kvHeads, hd, inter int, rmsEps float64) (LayerForwardCache, []float32, error) {
	width := heads * hd
	kvWidth := kvHeads * hd
	if heads%kvHeads != 0 || len(x) != seq*hidden || len(w.InLN) != hidden || len(w.PostLN) != hidden ||
		len(w.Q) != width*hidden || len(w.K) != kvWidth*hidden || len(w.V) != kvWidth*hidden || len(w.O) != hidden*width ||
		len(w.Gate) != inter*hidden || len(w.Up) != inter*hidden || len(w.Down) != hidden*inter || len(invFreq) != hd/2 {
		return LayerForwardCache{}, nil, fmt.Errorf("LayerForwardResident: shape mismatch (seq=%d hidden=%d heads=%d kv=%d hd=%d inter=%d)", seq, hidden, heads, kvHeads, hd, inter)
	}
	group := heads / kvHeads
	scaleQ := float32(1 / math.Sqrt(float64(hd)))

	var c LayerForwardCache
	c.Xn = make([]float32, seq*hidden)
	c.QScaled = make([]float32, seq*width)
	c.KRoped = make([]float32, seq*kvWidth)
	c.V = make([]float32, seq*kvWidth)
	c.AttnCore = make([]float32, seq*width)
	c.H2 = make([]float32, seq*hidden)
	c.Hn = make([]float32, seq*hidden)
	c.Gate = make([]float32, seq*inter)
	c.Up = make([]float32, seq*inter)
	c.A = make([]float32, seq*inter)
	c.HMLP = make([]float32, seq*inter)
	xOut := make([]float32, seq*hidden)

	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		fns := map[string]driver.Function{}
		for _, name := range []string{"weighted_rms_norm_f32", "rope_half_f32", "scale_f32", "add_f32", "causal_softmax_f32", "silu_f32", "multiply_f32", "head_major_f32", "head_major_inverse_f32"} {
			fn, err := s.function(name)
			if err != nil {
				return err
			}
			fns[name] = fn
		}
		up := func(d []float32) (driver.DevicePtr, error) { return s.upload(d) }

		xP, err := up(x)
		if err != nil {
			return err
		}
		inLNP, err := up(w.InLN)
		if err != nil {
			return err
		}
		postLNP, err := up(w.PostLN)
		if err != nil {
			return err
		}
		qWP, err := up(w.Q)
		if err != nil {
			return err
		}
		kWP, err := up(w.K)
		if err != nil {
			return err
		}
		vWP, err := up(w.V)
		if err != nil {
			return err
		}
		oWP, err := up(w.O)
		if err != nil {
			return err
		}
		gateWP, err := up(w.Gate)
		if err != nil {
			return err
		}
		upWP, err := up(w.Up)
		if err != nil {
			return err
		}
		downWP, err := up(w.Down)
		if err != nil {
			return err
		}
		invP, err := up(invFreq)
		if err != nil {
			return err
		}

		alloc := func(n int) (driver.DevicePtr, error) { return s.alloc(n) }
		xnP, err := alloc(seq * hidden)
		if err != nil {
			return err
		}
		qP, err := alloc(seq * width)
		if err != nil {
			return err
		}
		kP, err := alloc(seq * kvWidth)
		if err != nil {
			return err
		}
		vP, err := alloc(seq * kvWidth)
		if err != nil {
			return err
		}
		qHM, err := alloc(seq * width)
		if err != nil {
			return err
		}
		kHM, err := alloc(seq * kvWidth)
		if err != nil {
			return err
		}
		vHM, err := alloc(seq * kvWidth)
		if err != nil {
			return err
		}
		attnHM, err := alloc(seq * width)
		if err != nil {
			return err
		}
		attnCoreP, err := alloc(seq * width)
		if err != nil {
			return err
		}
		attnOutP, err := alloc(seq * hidden)
		if err != nil {
			return err
		}
		h2P, err := alloc(seq * hidden)
		if err != nil {
			return err
		}
		hnP, err := alloc(seq * hidden)
		if err != nil {
			return err
		}
		gateP, err := alloc(seq * inter)
		if err != nil {
			return err
		}
		upP, err := alloc(seq * inter)
		if err != nil {
			return err
		}
		aP, err := alloc(seq * inter)
		if err != nil {
			return err
		}
		hMLPP, err := alloc(seq * inter)
		if err != nil {
			return err
		}
		mlpP, err := alloc(seq * hidden)
		if err != nil {
			return err
		}
		scoresP, err := alloc(seq * seq)
		if err != nil {
			return err
		}
		pP, err := alloc(seq * seq)
		if err != nil {
			return err
		}

		off := func(base driver.DevicePtr, elems int) driver.DevicePtr {
			return base + driver.DevicePtr(uint64(elems)*f32Bytes)
		}
		rms := func(in, weight, out driver.DevicePtr) error {
			widthU, rowsU, epsF := uint32(hidden), uint32(seq), float32(rmsEps)
			return s.launch1D(fns["weighted_rms_norm_f32"], uint32(seq)*deviceBlockThreads,
				unsafe.Pointer(&in), unsafe.Pointer(&weight), unsafe.Pointer(&out),
				unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&epsF))
		}
		rope := func(t driver.DevicePtr, nHeads int) error {
			seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
			total := uint32(seq * nHeads * (hd / 2))
			return s.launch1D(fns["rope_half_f32"], total,
				unsafe.Pointer(&t), unsafe.Pointer(&invP), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
		}
		scale := func(in, out driver.DevicePtr, sc float32, n int) error {
			cU := uint32(n)
			return s.launch1D(fns["scale_f32"], cU,
				unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&sc), unsafe.Pointer(&cU))
		}
		toHM := func(in, out driver.DevicePtr, nHeads int) error {
			seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
			return s.launch1D(fns["head_major_f32"], uint32(seq*nHeads*hd),
				unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
		}
		fromHM := func(in, out driver.DevicePtr, nHeads int) error {
			seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
			return s.launch1D(fns["head_major_inverse_f32"], uint32(seq*nHeads*hd),
				unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
		}
		add := func(a, b, out driver.DevicePtr, n int) error { return s.launchVector3(fns["add_f32"], a, b, out, n) }

		// xn = RMSNorm(x, inLN)
		if err := rms(xP, inLNP, xnP); err != nil {
			return err
		}
		// q/k/v projections (Y = xn * Wᵀ)
		if err := s.gemm(false, true, seq, hidden, width, xnP, qWP, qP); err != nil {
			return err
		}
		if err := s.gemm(false, true, seq, hidden, kvWidth, xnP, kWP, kP); err != nil {
			return err
		}
		if err := s.gemm(false, true, seq, hidden, kvWidth, xnP, vWP, vP); err != nil {
			return err
		}
		// rope on q/k, then scale q (score scale folded into q)
		if err := rope(qP, heads); err != nil {
			return err
		}
		if err := rope(kP, kvHeads); err != nil {
			return err
		}
		if err := scale(qP, qP, scaleQ, seq*width); err != nil {
			return err
		}
		// attention core: head-major, per-head scores -> causal softmax -> p·v
		if err := toHM(qP, qHM, heads); err != nil {
			return err
		}
		if err := toHM(kP, kHM, kvHeads); err != nil {
			return err
		}
		if err := toHM(vP, vHM, kvHeads); err != nil {
			return err
		}
		hs := seq * hd
		for h := 0; h < heads; h++ {
			kv := h / group
			qh := off(qHM, h*hs)
			kh, vh := off(kHM, kv*hs), off(vHM, kv*hs)
			attnh := off(attnHM, h*hs)
			if err := s.gemm(false, true, seq, hd, seq, qh, kh, scoresP); err != nil {
				return err
			}
			rowsCausal := uint32(seq)
			if err := s.launch1D(fns["causal_softmax_f32"], rowsCausal,
				unsafe.Pointer(&scoresP), unsafe.Pointer(&pP), unsafe.Pointer(&rowsCausal)); err != nil {
				return err
			}
			if err := s.gemm(false, false, seq, seq, hd, pP, vh, attnh); err != nil {
				return err
			}
		}
		if err := fromHM(attnHM, attnCoreP, heads); err != nil {
			return err
		}
		// x += o(attnCore)
		if err := s.gemm(false, true, seq, width, hidden, attnCoreP, oWP, attnOutP); err != nil {
			return err
		}
		if err := add(xP, attnOutP, xP, seq*hidden); err != nil {
			return err
		}
		if err := scale(xP, h2P, 1, seq*hidden); err != nil { // h2 = x (copy)
			return err
		}
		// MLP branch
		if err := rms(xP, postLNP, hnP); err != nil {
			return err
		}
		if err := s.gemm(false, true, seq, hidden, inter, hnP, gateWP, gateP); err != nil {
			return err
		}
		if err := s.gemm(false, true, seq, hidden, inter, hnP, upWP, upP); err != nil {
			return err
		}
		cU := uint32(seq * inter)
		if err := s.launch1D(fns["silu_f32"], cU, unsafe.Pointer(&gateP), unsafe.Pointer(&aP), unsafe.Pointer(&cU)); err != nil {
			return err
		}
		if err := s.launchVector3(fns["multiply_f32"], aP, upP, hMLPP, seq*inter); err != nil {
			return err
		}
		if err := s.gemm(false, true, seq, inter, hidden, hMLPP, downWP, mlpP); err != nil {
			return err
		}
		if err := add(xP, mlpP, xP, seq*hidden); err != nil {
			return err
		}

		return s.finish(
			cudaDownload{c.Xn, xnP}, cudaDownload{c.QScaled, qP}, cudaDownload{c.KRoped, kP},
			cudaDownload{c.V, vP}, cudaDownload{c.AttnCore, attnCoreP}, cudaDownload{c.H2, h2P},
			cudaDownload{c.Hn, hnP}, cudaDownload{c.Gate, gateP}, cudaDownload{c.Up, upP},
			cudaDownload{c.A, aP}, cudaDownload{c.HMLP, hMLPP}, cudaDownload{xOut, xP},
		)
	})
	if err != nil {
		return LayerForwardCache{}, nil, err
	}
	return c, xOut, nil
}
