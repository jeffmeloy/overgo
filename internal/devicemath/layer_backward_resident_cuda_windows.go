//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// LayerBackwardResult bundles a layer's input-gradient and weight-gradients
// (densecausal layout), returned by LayerBackwardResident.
type LayerBackwardResult struct {
	DX                   []float32
	DWInLN, DWPostLN     []float32
	DWQ, DWK, DWV, DWO   []float32
	DWGate, DWUp, DWDown []float32
}

// LayerBackwardResident runs a whole pre-norm layer backward in ONE cudaBLAS
// session: x, dOut, the forward cache and every weight upload once, and all
// intermediate gradients stay device-resident (no per-op round-trips). Only dX +
// the nine weight grads come back. Same math as densecausal.deviceLayerBackward
// (which composed per-op resident sub-ops). Attention bias unsupported.
func LayerBackwardResident(worker *device.Worker, x, dOut []float32, c LayerForwardCache, w LayerForwardWeights, invFreq []float32, seq, hidden, heads, kvHeads, hd, inter int, rmsEps float64) (LayerBackwardResult, error) {
	width := heads * hd
	kvWidth := kvHeads * hd
	if heads%kvHeads != 0 || len(x) != seq*hidden || len(dOut) != seq*hidden || len(invFreq) != hd/2 {
		return LayerBackwardResult{}, fmt.Errorf("LayerBackwardResident: shape mismatch (seq=%d hidden=%d heads=%d kv=%d hd=%d)", seq, hidden, heads, kvHeads, hd)
	}
	group := heads / kvHeads
	scaleQ := float32(1 / math.Sqrt(float64(hd)))

	var r LayerBackwardResult
	r.DX = make([]float32, seq*hidden)
	r.DWInLN = make([]float32, hidden)
	r.DWPostLN = make([]float32, hidden)
	r.DWQ = make([]float32, width*hidden)
	r.DWK = make([]float32, kvWidth*hidden)
	r.DWV = make([]float32, kvWidth*hidden)
	r.DWO = make([]float32, hidden*width)
	r.DWGate = make([]float32, inter*hidden)
	r.DWUp = make([]float32, inter*hidden)
	r.DWDown = make([]float32, hidden*inter)

	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		fns := map[string]driver.Function{}
		for _, name := range []string{"multiply_f32", "add_f32", "scale_f32", "silu_backward_f32", "rms_norm_backward_f32", "rope_half_backward_f32", "softmax_backward_f32", "causal_softmax_f32", "head_major_f32", "head_major_inverse_f32"} {
			fn, err := s.function(name)
			if err != nil {
				return err
			}
			fns[name] = fn
		}
		up := func(d []float32) (driver.DevicePtr, error) { return s.upload(d) }
		al := func(n int) (driver.DevicePtr, error) { return s.alloc(n) }

		// Uploads.
		dOutP, err := up(dOut)
		if err != nil {
			return err
		}
		xP, err := up(x)
		if err != nil {
			return err
		}
		xnP, err := up(c.Xn)
		if err != nil {
			return err
		}
		hnP, err := up(c.Hn)
		if err != nil {
			return err
		}
		h2P, err := up(c.H2)
		if err != nil {
			return err
		}
		gateP, err := up(c.Gate)
		if err != nil {
			return err
		}
		upCP, err := up(c.Up)
		if err != nil {
			return err
		}
		aP, err := up(c.A)
		if err != nil {
			return err
		}
		hMLPP, err := up(c.HMLP)
		if err != nil {
			return err
		}
		qScP, err := up(c.QScaled)
		if err != nil {
			return err
		}
		kRoP, err := up(c.KRoped)
		if err != nil {
			return err
		}
		vP, err := up(c.V)
		if err != nil {
			return err
		}
		attnCoreP, err := up(c.AttnCore)
		if err != nil {
			return err
		}
		invP, err := up(invFreq)
		if err != nil {
			return err
		}
		wGateP, err := up(w.Gate)
		if err != nil {
			return err
		}
		wUpP, err := up(w.Up)
		if err != nil {
			return err
		}
		wDownP, err := up(w.Down)
		if err != nil {
			return err
		}
		wOP, err := up(w.O)
		if err != nil {
			return err
		}
		wQP, err := up(w.Q)
		if err != nil {
			return err
		}
		wKP, err := up(w.K)
		if err != nil {
			return err
		}
		wVP, err := up(w.V)
		if err != nil {
			return err
		}
		postLNP, err := up(w.PostLN)
		if err != nil {
			return err
		}
		inLNP, err := up(w.InLN)
		if err != nil {
			return err
		}

		mul := func(a, b, out driver.DevicePtr, n int) error {
			return s.launchVector3(fns["multiply_f32"], a, b, out, n)
		}
		add := func(a, b, out driver.DevicePtr, n int) error { return s.launchVector3(fns["add_f32"], a, b, out, n) }
		siluB := func(x, dy, out driver.DevicePtr, n int) error {
			return s.launchVector3(fns["silu_backward_f32"], dy, x, out, n)
		}
		off := func(base driver.DevicePtr, elems int) driver.DevicePtr {
			return base + driver.DevicePtr(uint64(elems)*f32Bytes)
		}
		rmsB := func(xIn, wIn, dy, dxOut, dscaleOut driver.DevicePtr) error {
			if err := s.state.Driver.MemsetD32Async(dscaleOut, 0, uint64(hidden), s.state.Stream); err != nil {
				return err
			}
			rowsU, dU, epsF := uint32(seq), uint32(hidden), float32(rmsEps)
			return s.launch1D(fns["rms_norm_backward_f32"], rowsU,
				unsafe.Pointer(&dy), unsafe.Pointer(&xIn), unsafe.Pointer(&wIn),
				unsafe.Pointer(&dxOut), unsafe.Pointer(&dscaleOut),
				unsafe.Pointer(&rowsU), unsafe.Pointer(&dU), unsafe.Pointer(&epsF))
		}
		ropeB := func(t driver.DevicePtr, nHeads int) error {
			seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
			total := uint32(seq * nHeads * (hd / 2))
			return s.launch1D(fns["rope_half_backward_f32"], total,
				unsafe.Pointer(&t), unsafe.Pointer(&invP), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
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

		// Scratch + outputs.
		mk := func(n int) driver.DevicePtr { p, e := al(n); err = e; return p }
		dhP, daP, duP, dgP := mk(seq*inter), mk(seq*inter), mk(seq*inter), mk(seq*inter)
		dXgP, dXuP := mk(seq*hidden), mk(seq*hidden)
		dXmlpP := mk(seq * hidden)
		dPostP, dscP := mk(seq*hidden), mk(hidden)
		dh2P := mk(seq * hidden)
		dAttnCoreP := mk(seq * width)
		dwDownP, dwGateP, dwUpP := mk(inter*hidden), mk(inter*hidden), mk(inter*hidden)
		dwOP := mk(hidden * width)
		// Attention head-major buffers.
		qHM, kHM, vHM := mk(seq*width), mk(seq*kvWidth), mk(seq*kvWidth)
		dAttnHM := mk(seq * width)
		dqHM, dkHM, dvHM := mk(seq*width), mk(seq*kvWidth), mk(seq*kvWidth)
		scoresP, pP, dpP, dsP, tmpHd := mk(seq*seq), mk(seq*seq), mk(seq*seq), mk(seq*seq), mk(seq*hd)
		dqP, dkP, dvP := mk(seq*width), mk(seq*kvWidth), mk(seq*kvWidth)
		dXnQ, dXnK, dXnV, dXnP := mk(seq*hidden), mk(seq*hidden), mk(seq*hidden), mk(seq*hidden)
		dwQP, dwKP, dwVP := mk(width*hidden), mk(kvWidth*hidden), mk(kvWidth*hidden)
		dInnP, dscInP := mk(seq*hidden), mk(hidden)
		if err != nil {
			return err
		}

		// --- MLP branch backward (SwiGLU) ---
		// dh = dOut·Wdown ; dWdown = dOutᵀ·hMLP
		if err := s.gemm(false, false, seq, hidden, inter, dOutP, wDownP, dhP); err != nil {
			return err
		}
		if err := s.gemm(true, false, hidden, seq, inter, dOutP, hMLPP, dwDownP); err != nil {
			return err
		}
		if err := mul(dhP, upCP, daP, seq*inter); err != nil {
			return err
		}
		if err := mul(dhP, aP, duP, seq*inter); err != nil {
			return err
		}
		if err := siluB(gateP, daP, dgP, seq*inter); err != nil {
			return err
		}
		if err := s.gemm(false, false, seq, inter, hidden, dgP, wGateP, dXgP); err != nil {
			return err
		}
		if err := s.gemm(true, false, inter, seq, hidden, dgP, hnP, dwGateP); err != nil {
			return err
		}
		if err := s.gemm(false, false, seq, inter, hidden, duP, wUpP, dXuP); err != nil {
			return err
		}
		if err := s.gemm(true, false, inter, seq, hidden, duP, hnP, dwUpP); err != nil {
			return err
		}
		if err := add(dXgP, dXuP, dXmlpP, seq*hidden); err != nil {
			return err
		}
		// RMSNorm backward (postLN): dPost ; dh2 = dOut + dPost
		if err := rmsB(h2P, postLNP, dXmlpP, dPostP, dscP); err != nil {
			return err
		}
		if err := add(dOutP, dPostP, dh2P, seq*hidden); err != nil {
			return err
		}

		// --- Attention branch backward ---
		// o linear: dAttnCore = dh2·O ; dWo = dh2ᵀ·attnCore
		if err := s.gemm(false, false, seq, hidden, width, dh2P, wOP, dAttnCoreP); err != nil {
			return err
		}
		if err := s.gemm(true, false, hidden, seq, width, dh2P, attnCoreP, dwOP); err != nil {
			return err
		}
		// MHA backward (GQA, causal), everything head-major on device.
		if err := toHM(qScP, qHM, heads); err != nil {
			return err
		}
		if err := toHM(kRoP, kHM, kvHeads); err != nil {
			return err
		}
		if err := toHM(vP, vHM, kvHeads); err != nil {
			return err
		}
		if err := toHM(dAttnCoreP, dAttnHM, heads); err != nil {
			return err
		}
		if err := s.state.Driver.MemsetD32Async(dkHM, 0, uint64(seq*kvWidth), s.state.Stream); err != nil {
			return err
		}
		if err := s.state.Driver.MemsetD32Async(dvHM, 0, uint64(seq*kvWidth), s.state.Stream); err != nil {
			return err
		}
		hs := seq * hd
		for h := 0; h < heads; h++ {
			kv := h / group
			qh, dOuth, dqh := off(qHM, h*hs), off(dAttnHM, h*hs), off(dqHM, h*hs)
			kh, vh, dkh, dvh := off(kHM, kv*hs), off(vHM, kv*hs), off(dkHM, kv*hs), off(dvHM, kv*hs)
			// p_h = causal_softmax(qh·khᵀ)
			if err := s.gemm(false, true, seq, hd, seq, qh, kh, scoresP); err != nil {
				return err
			}
			rc := uint32(seq)
			if err := s.launch1D(fns["causal_softmax_f32"], rc, unsafe.Pointer(&scoresP), unsafe.Pointer(&pP), unsafe.Pointer(&rc)); err != nil {
				return err
			}
			// dp = dOuth·vhᵀ ; ds = softmax_backward(p, dp)
			if err := s.gemm(false, true, seq, hd, seq, dOuth, vh, dpP); err != nil {
				return err
			}
			rowsU, dU := uint32(seq), uint32(seq)
			if err := s.launch1D(fns["softmax_backward_f32"], rowsU,
				unsafe.Pointer(&pP), unsafe.Pointer(&dpP), unsafe.Pointer(&dsP), unsafe.Pointer(&rowsU), unsafe.Pointer(&dU)); err != nil {
				return err
			}
			// dV_h += pᵀ·dOuth
			if err := s.gemm(true, false, seq, seq, hd, pP, dOuth, tmpHd); err != nil {
				return err
			}
			if err := add(dvh, tmpHd, dvh, hs); err != nil {
				return err
			}
			// dQ_h = ds·kh (unique head, write)
			if err := s.gemm(false, false, seq, seq, hd, dsP, kh, dqh); err != nil {
				return err
			}
			// dK_h += dsᵀ·qh
			if err := s.gemm(true, false, seq, seq, hd, dsP, qh, tmpHd); err != nil {
				return err
			}
			if err := add(dkh, tmpHd, dkh, hs); err != nil {
				return err
			}
		}
		if err := fromHM(dqHM, dqP, heads); err != nil {
			return err
		}
		if err := fromHM(dkHM, dkP, kvHeads); err != nil {
			return err
		}
		if err := fromHM(dvHM, dvP, kvHeads); err != nil {
			return err
		}
		// The 1/sqrt(hd) score scale is folded into qScaled, so by the chain rule
		// ONLY dq gets the extra scale (k is unscaled; the backward reads the
		// scale-baked qScaled). dk and dv are not scaled here.
		if err := s.launch1D(fns["scale_f32"], uint32(seq*width), unsafe.Pointer(&dqP), unsafe.Pointer(&dqP), unsafe.Pointer(&scaleQ), func() unsafe.Pointer { u := uint32(seq * width); return unsafe.Pointer(&u) }()); err != nil {
			return err
		}
		// rope backward on dq/dk
		if err := ropeB(dqP, heads); err != nil {
			return err
		}
		if err := ropeB(dkP, kvHeads); err != nil {
			return err
		}
		// q/k/v linear backward: dXn* = d*·W ; dW* = d*ᵀ·xn
		if err := s.gemm(false, false, seq, width, hidden, dqP, wQP, dXnQ); err != nil {
			return err
		}
		if err := s.gemm(true, false, width, seq, hidden, dqP, xnP, dwQP); err != nil {
			return err
		}
		if err := s.gemm(false, false, seq, kvWidth, hidden, dkP, wKP, dXnK); err != nil {
			return err
		}
		if err := s.gemm(true, false, kvWidth, seq, hidden, dkP, xnP, dwKP); err != nil {
			return err
		}
		if err := s.gemm(false, false, seq, kvWidth, hidden, dvP, wVP, dXnV); err != nil {
			return err
		}
		if err := s.gemm(true, false, kvWidth, seq, hidden, dvP, xnP, dwVP); err != nil {
			return err
		}
		if err := add(dXnQ, dXnK, dXnP, seq*hidden); err != nil {
			return err
		}
		if err := add(dXnP, dXnV, dXnP, seq*hidden); err != nil {
			return err
		}
		// RMSNorm backward (inLN): dInn ; dx = dh2 + dInn
		if err := rmsB(xP, inLNP, dXnP, dInnP, dscInP); err != nil {
			return err
		}
		if err := add(dh2P, dInnP, dh2P, seq*hidden); err != nil {
			return err
		}

		return s.finish(
			cudaDownload{r.DX, dh2P},
			cudaDownload{r.DWDown, dwDownP}, cudaDownload{r.DWGate, dwGateP}, cudaDownload{r.DWUp, dwUpP},
			cudaDownload{r.DWPostLN, dscP}, cudaDownload{r.DWO, dwOP},
			cudaDownload{r.DWQ, dwQP}, cudaDownload{r.DWK, dwKP}, cudaDownload{r.DWV, dwVP},
			cudaDownload{r.DWInLN, dscInP},
		)
	})
	if err != nil {
		return LayerBackwardResult{}, err
	}
	return r, nil
}
