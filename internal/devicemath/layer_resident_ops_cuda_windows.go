//go:build windows

package devicemath

import (
	"math"
	"unsafe"

	"overgo/internal/cuda/driver"
)

// Kernel names the resident layer forward / backward device ops launch. Kept as
// package vars so the single-layer wrappers and the whole-stack driver load the
// same set once per session.
var (
	layerForwardFnNames  = []string{"weighted_rms_norm_f32", "rope_half_f32", "scale_f32", "add_f32", "causal_softmax_f32", "silu_f32", "multiply_f32", "head_major_f32", "head_major_inverse_f32"}
	layerBackwardFnNames = []string{"multiply_f32", "add_f32", "scale_f32", "silu_backward_f32", "rms_norm_backward_f32", "rope_half_backward_f32", "softmax_backward_f32", "causal_softmax_f32", "head_major_f32", "head_major_inverse_f32"}
)

// layerDims carries one pre-norm transformer layer's shape (derived widths and
// the folded 1/sqrt(hd) score scale) so the device ops need no per-op recompute.
type layerDims struct {
	seq, hidden, heads, kvHeads, hd, inter int
	group, width, kvWidth                  int
	rmsEps                                 float64
	scaleQ                                 float32
}

func newLayerDims(seq, hidden, heads, kvHeads, hd, inter int, rmsEps float64) layerDims {
	return layerDims{
		seq: seq, hidden: hidden, heads: heads, kvHeads: kvHeads, hd: hd, inter: inter,
		group: heads / kvHeads, width: heads * hd, kvWidth: kvHeads * hd,
		rmsEps: rmsEps, scaleQ: float32(1 / math.Sqrt(float64(hd))),
	}
}

// layerOps binds a resident cudaBLAS session, the loaded kernel functions, the
// layer shape and the device-resident rope inverse-frequency table. Its methods
// are the exact kernel launches the per-op resident forward/backward composed,
// now shared by the single-layer wrappers and the whole-stack driver (one owner
// for the layer math, no drift).
type layerOps struct {
	s    *cudaBLAS
	fns  map[string]driver.Function
	d    layerDims
	invP driver.DevicePtr
	// arena, when non-nil, pools every layer's transient forward/backward scratch
	// into one reused region (set by runStack). nil for the single-layer wrappers,
	// which keep the legacy per-call session allocations.
	arena *scratchArena
}

// newLayerOps loads the named kernels and uploads invFreq once for the session.
func newLayerOps(s *cudaBLAS, names []string, d layerDims, invFreq []float32) (*layerOps, error) {
	fns := map[string]driver.Function{}
	for _, name := range names {
		fn, err := s.function(name)
		if err != nil {
			return nil, err
		}
		fns[name] = fn
	}
	invP, err := s.upload(invFreq)
	if err != nil {
		return nil, err
	}
	return &layerOps{s: s, fns: fns, d: d, invP: invP}, nil
}

func (o *layerOps) off(base driver.DevicePtr, elems int) driver.DevicePtr {
	return base + driver.DevicePtr(uint64(elems)*f32Bytes)
}

// scratch returns transient per-op device memory: from the reused arena when
// pooling (peak O(1 layer)) or from the session scope otherwise.
func (o *layerOps) scratch(n int) (driver.DevicePtr, error) {
	if o.arena != nil {
		return o.arena.alloc(n)
	}
	return o.s.alloc(n)
}

func (o *layerOps) rms(in, weight, out driver.DevicePtr) error {
	widthU, rowsU, epsF := uint32(o.d.hidden), uint32(o.d.seq), float32(o.d.rmsEps)
	return o.s.launch1D(o.fns["weighted_rms_norm_f32"], uint32(o.d.seq)*deviceBlockThreads,
		unsafe.Pointer(&in), unsafe.Pointer(&weight), unsafe.Pointer(&out),
		unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&epsF))
}

func (o *layerOps) rope(t driver.DevicePtr, nHeads int) error {
	seqU, nhU, hdU := uint32(o.d.seq), uint32(nHeads), uint32(o.d.hd)
	total := uint32(o.d.seq * nHeads * (o.d.hd / 2))
	return o.s.launch1D(o.fns["rope_half_f32"], total,
		unsafe.Pointer(&t), unsafe.Pointer(&o.invP), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
}

func (o *layerOps) ropeBackward(t driver.DevicePtr, nHeads int) error {
	seqU, nhU, hdU := uint32(o.d.seq), uint32(nHeads), uint32(o.d.hd)
	total := uint32(o.d.seq * nHeads * (o.d.hd / 2))
	return o.s.launch1D(o.fns["rope_half_backward_f32"], total,
		unsafe.Pointer(&t), unsafe.Pointer(&o.invP), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
}

func (o *layerOps) scale(in, out driver.DevicePtr, sc float32, n int) error {
	cU := uint32(n)
	return o.s.launch1D(o.fns["scale_f32"], cU,
		unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&sc), unsafe.Pointer(&cU))
}

func (o *layerOps) toHM(in, out driver.DevicePtr, nHeads int) error {
	seqU, nhU, hdU := uint32(o.d.seq), uint32(nHeads), uint32(o.d.hd)
	return o.s.launch1D(o.fns["head_major_f32"], uint32(o.d.seq*nHeads*o.d.hd),
		unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
}

func (o *layerOps) fromHM(in, out driver.DevicePtr, nHeads int) error {
	seqU, nhU, hdU := uint32(o.d.seq), uint32(nHeads), uint32(o.d.hd)
	return o.s.launch1D(o.fns["head_major_inverse_f32"], uint32(o.d.seq*nHeads*o.d.hd),
		unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU))
}

func (o *layerOps) add(a, b, out driver.DevicePtr, n int) error {
	return o.s.launchVector3(o.fns["add_f32"], a, b, out, n)
}

func (o *layerOps) mul(a, b, out driver.DevicePtr, n int) error {
	return o.s.launchVector3(o.fns["multiply_f32"], a, b, out, n)
}

func (o *layerOps) siluBackward(x, dy, out driver.DevicePtr, n int) error {
	return o.s.launchVector3(o.fns["silu_backward_f32"], dy, x, out, n)
}

// rmsBackward launches rms_norm_backward_f32; the norm-weight grad accumulator is
// zeroed first (the kernel adds into it).
func (o *layerOps) rmsBackward(xIn, wIn, dy, dxOut, dscaleOut driver.DevicePtr) error {
	if err := o.s.state.Driver.MemsetD32Async(dscaleOut, 0, uint64(o.d.hidden), o.s.state.Stream); err != nil {
		return err
	}
	rowsU, dU, epsF := uint32(o.d.seq), uint32(o.d.hidden), float32(o.d.rmsEps)
	return o.s.launch1D(o.fns["rms_norm_backward_f32"], rowsU,
		unsafe.Pointer(&dy), unsafe.Pointer(&xIn), unsafe.Pointer(&wIn),
		unsafe.Pointer(&dxOut), unsafe.Pointer(&dscaleOut),
		unsafe.Pointer(&rowsU), unsafe.Pointer(&dU), unsafe.Pointer(&epsF))
}

// layerWeightPtrs are one layer's weights, device-resident (uploaded once and
// read by both the forward and the backward within a session).
type layerWeightPtrs struct {
	inLN, postLN, q, k, v, o, gate, up, down driver.DevicePtr
}

// layerCachePtrs are the forward's activation intermediates, device-resident. The
// forward writes them; the backward reads them without any host round-trip.
type layerCachePtrs struct {
	xn, qScaled, kRoped, v, attnCore, h2, hn, gate, up, a, hMLP driver.DevicePtr
}

// layerGradPtrs are the backward's weight-gradient outputs (densecausal layout).
type layerGradPtrs struct {
	dInLN, dPostLN, dQ, dK, dV, dO, dGate, dUp, dDown driver.DevicePtr
}

// layerCacheElemSpec is the element count of each of the 11 resident activation
// buffers for one layer, in field order (xn,qScaled,kRoped,v,attnCore,h2,hn,
// gate,up,a,hMLP). Single owner: allocLayerCache assigns from it and the resident
// capacity plan sums it -- no drift.
func layerCacheElemSpec(d layerDims) []int {
	return []int{
		d.seq * d.hidden, d.seq * d.width, d.seq * d.kvWidth,
		d.seq * d.kvWidth, d.seq * d.width, d.seq * d.hidden,
		d.seq * d.hidden, d.seq * d.inter, d.seq * d.inter,
		d.seq * d.inter, d.seq * d.inter,
	}
}

// allocLayerCache allocates the 11 resident activation buffers for one layer.
func allocLayerCache(s *cudaScope, d layerDims) (layerCachePtrs, error) {
	var c layerCachePtrs
	dst := []*driver.DevicePtr{
		&c.xn, &c.qScaled, &c.kRoped, &c.v, &c.attnCore, &c.h2,
		&c.hn, &c.gate, &c.up, &c.a, &c.hMLP,
	}
	for i, n := range layerCacheElemSpec(d) {
		p, err := s.alloc(n)
		if err != nil {
			return layerCachePtrs{}, err
		}
		*dst[i] = p
	}
	return c, nil
}

// allocLayerGrads allocates the 9 resident weight-gradient buffers for one layer.
func allocLayerGrads(s *cudaScope, d layerDims) (layerGradPtrs, error) {
	var g layerGradPtrs
	for _, spec := range []struct {
		p *driver.DevicePtr
		n int
	}{
		{&g.dInLN, d.hidden}, {&g.dPostLN, d.hidden}, {&g.dQ, d.width * d.hidden}, {&g.dK, d.kvWidth * d.hidden},
		{&g.dV, d.kvWidth * d.hidden}, {&g.dO, d.hidden * d.width}, {&g.dGate, d.inter * d.hidden},
		{&g.dUp, d.inter * d.hidden}, {&g.dDown, d.hidden * d.inter},
	} {
		p, err := s.alloc(spec.n)
		if err != nil {
			return layerGradPtrs{}, err
		}
		*spec.p = p
	}
	return g, nil
}

// forwardDevice runs one pre-norm layer forward entirely on device: reads the
// input residual `in` (never mutated) and the resident weights `w`, writes the
// activation cache `c` and the advanced residual `xOut`. Transient attention/MLP
// scratch is allocated in the session scope. Same math as the host
// layerForwardCached. Nothing is uploaded or downloaded here.
func (o *layerOps) forwardDevice(in driver.DevicePtr, w layerWeightPtrs, c layerCachePtrs, xOut driver.DevicePtr) error {
	d := o.d
	al := func(n int) (driver.DevicePtr, error) { return o.scratch(n) }
	qHM, err := al(d.seq * d.width)
	if err != nil {
		return err
	}
	kHM, err := al(d.seq * d.kvWidth)
	if err != nil {
		return err
	}
	vHM, err := al(d.seq * d.kvWidth)
	if err != nil {
		return err
	}
	attnHM, err := al(d.seq * d.width)
	if err != nil {
		return err
	}
	attnOut, err := al(d.seq * d.hidden)
	if err != nil {
		return err
	}
	mlp, err := al(d.seq * d.hidden)
	if err != nil {
		return err
	}
	scores, err := al(d.seq * d.seq)
	if err != nil {
		return err
	}
	pBuf, err := al(d.seq * d.seq)
	if err != nil {
		return err
	}

	// xn = RMSNorm(in, inLN)
	if err := o.rms(in, w.inLN, c.xn); err != nil {
		return err
	}
	// q/k/v projections (Y = xn * Wᵀ) straight into the resident cache buffers.
	if err := o.s.gemm(false, true, d.seq, d.hidden, d.width, c.xn, w.q, c.qScaled); err != nil {
		return err
	}
	if err := o.s.gemm(false, true, d.seq, d.hidden, d.kvWidth, c.xn, w.k, c.kRoped); err != nil {
		return err
	}
	if err := o.s.gemm(false, true, d.seq, d.hidden, d.kvWidth, c.xn, w.v, c.v); err != nil {
		return err
	}
	// rope on q/k, then fold the score scale into q (only q is scaled).
	if err := o.rope(c.qScaled, d.heads); err != nil {
		return err
	}
	if err := o.rope(c.kRoped, d.kvHeads); err != nil {
		return err
	}
	if err := o.scale(c.qScaled, c.qScaled, d.scaleQ, d.seq*d.width); err != nil {
		return err
	}
	// attention core: head-major, per-head causal-softmax scores, p·v.
	if err := o.toHM(c.qScaled, qHM, d.heads); err != nil {
		return err
	}
	if err := o.toHM(c.kRoped, kHM, d.kvHeads); err != nil {
		return err
	}
	if err := o.toHM(c.v, vHM, d.kvHeads); err != nil {
		return err
	}
	hs := d.seq * d.hd
	for h := 0; h < d.heads; h++ {
		kv := h / d.group
		qh := o.off(qHM, h*hs)
		kh, vh := o.off(kHM, kv*hs), o.off(vHM, kv*hs)
		attnh := o.off(attnHM, h*hs)
		if err := o.s.gemm(false, true, d.seq, d.hd, d.seq, qh, kh, scores); err != nil {
			return err
		}
		rowsCausal := uint32(d.seq)
		if err := o.s.launch1D(o.fns["causal_softmax_f32"], rowsCausal,
			unsafe.Pointer(&scores), unsafe.Pointer(&pBuf), unsafe.Pointer(&rowsCausal)); err != nil {
			return err
		}
		if err := o.s.gemm(false, false, d.seq, d.seq, d.hd, pBuf, vh, attnh); err != nil {
			return err
		}
	}
	if err := o.fromHM(attnHM, c.attnCore, d.heads); err != nil {
		return err
	}
	// xOut = in + o(attnCore); h2 = xOut (residual after attention).
	if err := o.s.gemm(false, true, d.seq, d.width, d.hidden, c.attnCore, w.o, attnOut); err != nil {
		return err
	}
	if err := o.add(in, attnOut, xOut, d.seq*d.hidden); err != nil {
		return err
	}
	if err := o.scale(xOut, c.h2, 1, d.seq*d.hidden); err != nil {
		return err
	}
	// MLP (SwiGLU) branch.
	if err := o.rms(xOut, w.postLN, c.hn); err != nil {
		return err
	}
	if err := o.s.gemm(false, true, d.seq, d.hidden, d.inter, c.hn, w.gate, c.gate); err != nil {
		return err
	}
	if err := o.s.gemm(false, true, d.seq, d.hidden, d.inter, c.hn, w.up, c.up); err != nil {
		return err
	}
	cU := uint32(d.seq * d.inter)
	if err := o.s.launch1D(o.fns["silu_f32"], cU, unsafe.Pointer(&c.gate), unsafe.Pointer(&c.a), unsafe.Pointer(&cU)); err != nil {
		return err
	}
	if err := o.mul(c.a, c.up, c.hMLP, d.seq*d.inter); err != nil {
		return err
	}
	if err := o.s.gemm(false, true, d.seq, d.inter, d.hidden, c.hMLP, w.down, mlp); err != nil {
		return err
	}
	if err := o.add(xOut, mlp, xOut, d.seq*d.hidden); err != nil {
		return err
	}
	return nil
}

// backwardDevice runs one pre-norm layer backward entirely on device: reads the
// layer input `x`, the incoming output gradient `dOut`, the resident forward
// cache `c` and the resident weights `w`; writes the weight grads `g` and the
// input gradient `dX` (which becomes the previous layer's dOut). Transient VJP
// scratch is allocated in the session scope. Same math as the host layerBackward.
// Nothing is uploaded or downloaded here.
func (o *layerOps) backwardDevice(x, dOut driver.DevicePtr, c layerCachePtrs, w layerWeightPtrs, g layerGradPtrs, dX driver.DevicePtr) error {
	d := o.d
	var err error
	al := func(n int) driver.DevicePtr {
		var p driver.DevicePtr
		if err == nil {
			p, err = o.scratch(n)
		}
		return p
	}
	dh, da, du, dg := al(d.seq*d.inter), al(d.seq*d.inter), al(d.seq*d.inter), al(d.seq*d.inter)
	dXg, dXu := al(d.seq*d.hidden), al(d.seq*d.hidden)
	dXmlp := al(d.seq * d.hidden)
	dPost := al(d.seq * d.hidden)
	dAttnCore := al(d.seq * d.width)
	qHM, kHM, vHM := al(d.seq*d.width), al(d.seq*d.kvWidth), al(d.seq*d.kvWidth)
	dAttnHM := al(d.seq * d.width)
	dqHM, dkHM, dvHM := al(d.seq*d.width), al(d.seq*d.kvWidth), al(d.seq*d.kvWidth)
	scores, pBuf, dp, ds, tmpHd := al(d.seq*d.seq), al(d.seq*d.seq), al(d.seq*d.seq), al(d.seq*d.seq), al(d.seq*d.hd)
	dq, dk, dv := al(d.seq*d.width), al(d.seq*d.kvWidth), al(d.seq*d.kvWidth)
	dXnQ, dXnK, dXnV, dXn := al(d.seq*d.hidden), al(d.seq*d.hidden), al(d.seq*d.hidden), al(d.seq*d.hidden)
	dInn := al(d.seq * d.hidden)
	if err != nil {
		return err
	}

	// --- MLP branch backward (SwiGLU) ---
	// dh = dOut·Wdown ; dWdown = dOutᵀ·hMLP
	if err := o.s.gemm(false, false, d.seq, d.hidden, d.inter, dOut, w.down, dh); err != nil {
		return err
	}
	if err := o.s.gemm(true, false, d.hidden, d.seq, d.inter, dOut, c.hMLP, g.dDown); err != nil {
		return err
	}
	if err := o.mul(dh, c.up, da, d.seq*d.inter); err != nil {
		return err
	}
	if err := o.mul(dh, c.a, du, d.seq*d.inter); err != nil {
		return err
	}
	if err := o.siluBackward(c.gate, da, dg, d.seq*d.inter); err != nil {
		return err
	}
	if err := o.s.gemm(false, false, d.seq, d.inter, d.hidden, dg, w.gate, dXg); err != nil {
		return err
	}
	if err := o.s.gemm(true, false, d.inter, d.seq, d.hidden, dg, c.hn, g.dGate); err != nil {
		return err
	}
	if err := o.s.gemm(false, false, d.seq, d.inter, d.hidden, du, w.up, dXu); err != nil {
		return err
	}
	if err := o.s.gemm(true, false, d.inter, d.seq, d.hidden, du, c.hn, g.dUp); err != nil {
		return err
	}
	if err := o.add(dXg, dXu, dXmlp, d.seq*d.hidden); err != nil {
		return err
	}
	// RMSNorm backward (postLN): dPost ; dh2 = dOut + dPost (dh2 lives in dX).
	if err := o.rmsBackward(c.h2, w.postLN, dXmlp, dPost, g.dPostLN); err != nil {
		return err
	}
	if err := o.add(dOut, dPost, dX, d.seq*d.hidden); err != nil {
		return err
	}

	// --- Attention branch backward ---
	// o linear: dAttnCore = dh2·O ; dWo = dh2ᵀ·attnCore
	if err := o.s.gemm(false, false, d.seq, d.hidden, d.width, dX, w.o, dAttnCore); err != nil {
		return err
	}
	if err := o.s.gemm(true, false, d.hidden, d.seq, d.width, dX, c.attnCore, g.dO); err != nil {
		return err
	}
	// MHA backward (GQA, causal), everything head-major on device.
	if err := o.toHM(c.qScaled, qHM, d.heads); err != nil {
		return err
	}
	if err := o.toHM(c.kRoped, kHM, d.kvHeads); err != nil {
		return err
	}
	if err := o.toHM(c.v, vHM, d.kvHeads); err != nil {
		return err
	}
	if err := o.toHM(dAttnCore, dAttnHM, d.heads); err != nil {
		return err
	}
	if err := o.s.state.Driver.MemsetD32Async(dkHM, 0, uint64(d.seq*d.kvWidth), o.s.state.Stream); err != nil {
		return err
	}
	if err := o.s.state.Driver.MemsetD32Async(dvHM, 0, uint64(d.seq*d.kvWidth), o.s.state.Stream); err != nil {
		return err
	}
	hs := d.seq * d.hd
	for h := 0; h < d.heads; h++ {
		kv := h / d.group
		qh, dOuth, dqh := o.off(qHM, h*hs), o.off(dAttnHM, h*hs), o.off(dqHM, h*hs)
		kh, vh, dkh, dvh := o.off(kHM, kv*hs), o.off(vHM, kv*hs), o.off(dkHM, kv*hs), o.off(dvHM, kv*hs)
		// p_h = causal_softmax(qh·khᵀ)
		if err := o.s.gemm(false, true, d.seq, d.hd, d.seq, qh, kh, scores); err != nil {
			return err
		}
		rc := uint32(d.seq)
		if err := o.s.launch1D(o.fns["causal_softmax_f32"], rc, unsafe.Pointer(&scores), unsafe.Pointer(&pBuf), unsafe.Pointer(&rc)); err != nil {
			return err
		}
		// dp = dOuth·vhᵀ ; ds = softmax_backward(p, dp)
		if err := o.s.gemm(false, true, d.seq, d.hd, d.seq, dOuth, vh, dp); err != nil {
			return err
		}
		rowsU, dU := uint32(d.seq), uint32(d.seq)
		if err := o.s.launch1D(o.fns["softmax_backward_f32"], rowsU,
			unsafe.Pointer(&pBuf), unsafe.Pointer(&dp), unsafe.Pointer(&ds), unsafe.Pointer(&rowsU), unsafe.Pointer(&dU)); err != nil {
			return err
		}
		// dV_h += pᵀ·dOuth
		if err := o.s.gemm(true, false, d.seq, d.seq, d.hd, pBuf, dOuth, tmpHd); err != nil {
			return err
		}
		if err := o.add(dvh, tmpHd, dvh, hs); err != nil {
			return err
		}
		// dQ_h = ds·kh (unique head, write)
		if err := o.s.gemm(false, false, d.seq, d.seq, d.hd, ds, kh, dqh); err != nil {
			return err
		}
		// dK_h += dsᵀ·qh
		if err := o.s.gemm(true, false, d.seq, d.seq, d.hd, ds, qh, tmpHd); err != nil {
			return err
		}
		if err := o.add(dkh, tmpHd, dkh, hs); err != nil {
			return err
		}
	}
	if err := o.fromHM(dqHM, dq, d.heads); err != nil {
		return err
	}
	if err := o.fromHM(dkHM, dk, d.kvHeads); err != nil {
		return err
	}
	if err := o.fromHM(dvHM, dv, d.kvHeads); err != nil {
		return err
	}
	// The 1/sqrt(hd) score scale is folded into qScaled, so by the chain rule ONLY
	// dq gets the extra scale (k is unscaled; the backward reads the scale-baked
	// qScaled). dk and dv are not scaled here.
	nq := uint32(d.seq * d.width)
	if err := o.s.launch1D(o.fns["scale_f32"], nq, unsafe.Pointer(&dq), unsafe.Pointer(&dq), unsafe.Pointer(&o.d.scaleQ), unsafe.Pointer(&nq)); err != nil {
		return err
	}
	// rope backward on dq/dk.
	if err := o.ropeBackward(dq, d.heads); err != nil {
		return err
	}
	if err := o.ropeBackward(dk, d.kvHeads); err != nil {
		return err
	}
	// q/k/v linear backward: dXn* = d*·W ; dW* = d*ᵀ·xn
	if err := o.s.gemm(false, false, d.seq, d.width, d.hidden, dq, w.q, dXnQ); err != nil {
		return err
	}
	if err := o.s.gemm(true, false, d.width, d.seq, d.hidden, dq, c.xn, g.dQ); err != nil {
		return err
	}
	if err := o.s.gemm(false, false, d.seq, d.kvWidth, d.hidden, dk, w.k, dXnK); err != nil {
		return err
	}
	if err := o.s.gemm(true, false, d.kvWidth, d.seq, d.hidden, dk, c.xn, g.dK); err != nil {
		return err
	}
	if err := o.s.gemm(false, false, d.seq, d.kvWidth, d.hidden, dv, w.v, dXnV); err != nil {
		return err
	}
	if err := o.s.gemm(true, false, d.kvWidth, d.seq, d.hidden, dv, c.xn, g.dV); err != nil {
		return err
	}
	if err := o.add(dXnQ, dXnK, dXn, d.seq*d.hidden); err != nil {
		return err
	}
	if err := o.add(dXn, dXnV, dXn, d.seq*d.hidden); err != nil {
		return err
	}
	// RMSNorm backward (inLN): dInn ; dx = dh2 + dInn (accumulates into dX).
	if err := o.rmsBackward(x, w.inLN, dXn, dInn, g.dInLN); err != nil {
		return err
	}
	if err := o.add(dX, dInn, dX, d.seq*d.hidden); err != nil {
		return err
	}
	return nil
}
