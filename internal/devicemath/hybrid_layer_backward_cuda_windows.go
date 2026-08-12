//go:build windows

package devicemath

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
)

// HybridDecoderLayerBackwardDevice is the device VJP of hostmath's qwen3.5 hybrid
// decoder layer (hostmath.HybridDecoderLayerForward). It composes the existing,
// individually parity-verified device ops -- LinearForwardT/LinearBackwardT,
// RMSNormForward/RMSNormBackward, RoPEHalfForward/RoPEHalfBackward (partial rope
// via a per-head gather/scatter of the first RopeWidth dims), the resident causal
// GQA attention forward/backward, and the SwiGLU (GatedMLPBackwardT) block --
// wired with the two pre-norms and two residuals exactly as
// hostmath.HybridDecoderLayerBackward.
//
// Because the host forward cache (hybridLayerCache) is package-private, the
// device path recomputes the forward intermediates it needs on device (from the
// same verified forward ops) rather than receiving the cache; the recomputed
// activations match the host cache within the ops' parity, so the composed grads
// match hostmath.HybridDecoderLayerBackward to the ~1e-4 kernel-parity class.
//
// MILESTONE 1b: both mix variants. full_attention routes through the resident
// causal GQA attention forward/backward; linear_attention (GDN mix) routes
// through GatedDeltaMixBackwardDevice (device L2Norm/ShortConv/GDN backward).
func HybridDecoderLayerBackwardDevice(worker *device.Worker, x []float32, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state, dOut []float32) (hostmath.HybridDecoderLayerGrads, error) {
	T, H := d.Tokens, d.Hidden
	if len(x) != T*H || len(dOut) != T*H {
		return hostmath.HybridDecoderLayerGrads{}, fmt.Errorf("HybridDecoderLayerBackwardDevice: shape mismatch (T=%d H=%d x=%d dOut=%d)", T, H, len(x), len(dOut))
	}

	var g hostmath.HybridDecoderLayerGrads
	g.IsLinear = w.IsLinear
	eps := d.Eps

	// --- forward recompute (device): only the intermediates the VJP consumes ---
	xn, err := RMSNormForward(worker, x, w.InputNorm, T, H, eps)
	if err != nil {
		return g, err
	}
	// mix forward: attention recomputes on device (returns its VJP cache); the GDN
	// mix reuses the proven host forward for the residual (its VJP recomputes its
	// own intermediates in GatedDeltaMixBackwardDevice).
	var mixOut []float32
	var ac attnMixDeviceCache
	if w.IsLinear {
		mixOut, _ = hostmath.GatedDeltaMixForward(xn, w.GDN, d.GDN, state)
	} else {
		mixOut, ac, err = attentionMixForwardDevice(worker, xn, w.Attn, d.Attn)
		if err != nil {
			return g, err
		}
	}
	h := make([]float32, T*H)
	for i := range h {
		h[i] = x[i] + mixOut[i]
	}
	hn, err := RMSNormForward(worker, h, w.PostNorm, T, H, eps)
	if err != nil {
		return g, err
	}
	gateP, err := LinearForwardT(worker, hn, w.MLP.Gate, T, H, d.Inter)
	if err != nil {
		return g, err
	}
	upP, err := LinearForwardT(worker, hn, w.MLP.Up, T, H, d.Inter)
	if err != nil {
		return g, err
	}
	aP, hMLP, err := SiLUGateForward(worker, gateP, upP)
	if err != nil {
		return g, err
	}

	// --- MLP branch backward (SwiGLU): out = h + MLP(RMSNorm(h,PostNorm)) ---
	mlp, err := GatedMLPBackwardT(worker, hn, w.MLP.Gate, w.MLP.Up, w.MLP.Down, gateP, aP, upP, hMLP, dOut, T, H, d.Inter)
	if err != nil {
		return g, err
	}
	g.DMLP = hostmath.QwenMLPWeights{Gate: mlp.DWGate, Up: mlp.DWUp, Down: mlp.DWDown}
	dhNorm, dPostNorm, err := RMSNormBackward(worker, h, w.PostNorm, mlp.DX, T, H, eps)
	if err != nil {
		return g, err
	}
	g.DPostNorm = dPostNorm
	// out = h + mlpOut -> dh gets dOut directly plus the norm/MLP path.
	dh := make([]float32, T*H)
	for i := range dh {
		dh[i] = dOut[i] + dhNorm[i]
	}

	// --- mix branch backward: h = x + Mix(RMSNorm(x,InputNorm)) ---
	var dXn []float32
	if w.IsLinear {
		mg, err := GatedDeltaMixBackwardDevice(worker, xn, w.GDN, d.GDN, state, dh)
		if err != nil {
			return g, err
		}
		g.DGDN = mg
		g.DState = mg.DState
		dXn = mg.DX
	} else {
		dxn, dAttn, err := attentionMixBackwardDevice(worker, xn, w.Attn, d.Attn, dh, ac)
		if err != nil {
			return g, err
		}
		g.DAttn = dAttn
		dXn = dxn
	}
	dxNorm, dInputNorm, err := RMSNormBackward(worker, x, w.InputNorm, dXn, T, H, eps)
	if err != nil {
		return g, err
	}
	g.DInputNorm = dInputNorm
	// dx gets dh directly plus the input-norm/mix path.
	dx := make([]float32, T*H)
	for i := range dx {
		dx[i] = dh[i] + dxNorm[i]
	}
	g.DX = dx
	return g, nil
}

// attnMixDeviceCache holds the forward intermediates the attention-mix VJP reads,
// mirroring hostmath.attnMixCache (attn output pre-Wo, scaled+rotated q, rotated
// k, value proj, and the pre-norm q/k projections).
type attnMixDeviceCache struct {
	attn, qs, kr, v []float32
	qProj, kProj    []float32
	invFreq         []float32
	rd              int
	scale           float32
	heads, kv, hd   int
	qDim, kvDim     int
}

// attentionMixForwardDevice recomputes hostmath.attentionMixForward on device:
// q/k/v proj -> per-head RMSNorm(q,k) -> partial rotary(q,k) -> scale q by
// 1/sqrt(hd) -> causal GQA attention -> Wo. Returns the mix output and the cache.
func attentionMixForwardDevice(worker *device.Worker, xn []float32, w hostmath.AttentionMixWeights, ad hostmath.AttentionMixDims) ([]float32, attnMixDeviceCache, error) {
	var c attnMixDeviceCache
	T, H := ad.Tokens, ad.Hidden
	c.heads, c.kv, c.hd = ad.Heads, ad.KVHeads, ad.HeadDim
	c.qDim, c.kvDim = c.heads*c.hd, c.kv*c.hd
	c.rd = hostmath.RopeWidth(ad.RopeDim, c.hd)
	c.invFreq = f64To32(hostmath.RopeInvFreq(ad.RopeTheta, c.rd))
	c.scale = float32(1.0 / math.Sqrt(float64(c.hd)))

	qProj, err := LinearForwardT(worker, xn, w.Wq, T, H, c.qDim)
	if err != nil {
		return nil, c, err
	}
	kProj, err := LinearForwardT(worker, xn, w.Wk, T, H, c.kvDim)
	if err != nil {
		return nil, c, err
	}
	v, err := LinearForwardT(worker, xn, w.Wv, T, H, c.kvDim)
	if err != nil {
		return nil, c, err
	}
	c.qProj, c.kProj, c.v = qProj, kProj, v

	qn, err := RMSNormForward(worker, qProj, w.QNorm, T*c.heads, c.hd, ad.Eps)
	if err != nil {
		return nil, c, err
	}
	kn, err := RMSNormForward(worker, kProj, w.KNorm, T*c.kv, c.hd, ad.Eps)
	if err != nil {
		return nil, c, err
	}
	// partial rope then fold the score scale into q (only q is scaled).
	qs, err := ropePartialForward(worker, qn, c.invFreq, T, c.heads, c.hd, c.rd)
	if err != nil {
		return nil, c, err
	}
	for i := range qs {
		qs[i] *= c.scale
	}
	kr, err := ropePartialForward(worker, kn, c.invFreq, T, c.kv, c.hd, c.rd)
	if err != nil {
		return nil, c, err
	}
	c.qs, c.kr = qs, kr

	attn, err := MultiHeadAttentionForwardResident(worker, qs, kr, v, T, c.heads, c.kv, c.hd)
	if err != nil {
		return nil, c, err
	}
	c.attn = attn
	mixOut, err := LinearForwardT(worker, attn, w.Wo, T, c.qDim, H)
	if err != nil {
		return nil, c, err
	}
	return mixOut, c, nil
}

// attentionMixBackwardDevice is the device VJP of attentionMixForwardDevice,
// mirroring hostmath.attentionMixBackward op for op.
func attentionMixBackwardDevice(worker *device.Worker, xn []float32, w hostmath.AttentionMixWeights, ad hostmath.AttentionMixDims, dOut []float32, c attnMixDeviceCache) ([]float32, hostmath.AttentionMixWeights, error) {
	T, H := ad.Tokens, ad.Hidden
	var dw hostmath.AttentionMixWeights

	// Wo: dAttn = dOut·Wo ; dWo = dOutᵀ·attn
	dAttn, dWo, err := LinearBackwardT(worker, c.attn, w.Wo, dOut, T, c.qDim, H)
	if err != nil {
		return nil, dw, err
	}
	dw.Wo = dWo
	// causal GQA attention backward (scale folded into q upstream -> scale 1).
	dqs, dkr, dv, err := MultiHeadAttentionBackwardResident(worker, c.qs, c.kr, c.v, dAttn, T, c.heads, c.kv, c.hd, 1.0)
	if err != nil {
		return nil, dw, err
	}
	// qs = scale·rope(qn): undo the scale, then rope backward (partial, per head).
	for i := range dqs {
		dqs[i] *= c.scale
	}
	dqs, err = ropePartialBackward(worker, dqs, c.invFreq, T, c.heads, c.hd, c.rd)
	if err != nil {
		return nil, dw, err
	}
	dkr, err = ropePartialBackward(worker, dkr, c.invFreq, T, c.kv, c.hd, c.rd)
	if err != nil {
		return nil, dw, err
	}
	// per-head q/k RMSNorm backward.
	dqProj, dQNorm, err := RMSNormBackward(worker, c.qProj, w.QNorm, dqs, T*c.heads, c.hd, ad.Eps)
	if err != nil {
		return nil, dw, err
	}
	dkProj, dKNorm, err := RMSNormBackward(worker, c.kProj, w.KNorm, dkr, T*c.kv, c.hd, ad.Eps)
	if err != nil {
		return nil, dw, err
	}
	dw.QNorm, dw.KNorm = dQNorm, dKNorm
	// q/k/v projection backward; dXn accumulates the three input-grad paths.
	dxQ, dWq, err := LinearBackwardT(worker, xn, w.Wq, dqProj, T, H, c.qDim)
	if err != nil {
		return nil, dw, err
	}
	dxK, dWk, err := LinearBackwardT(worker, xn, w.Wk, dkProj, T, H, c.kvDim)
	if err != nil {
		return nil, dw, err
	}
	dxV, dWv, err := LinearBackwardT(worker, xn, w.Wv, dv, T, H, c.kvDim)
	if err != nil {
		return nil, dw, err
	}
	dw.Wq, dw.Wk, dw.Wv = dWq, dWk, dWv
	dXn := make([]float32, T*H)
	for i := range dXn {
		dXn[i] = dxQ[i] + dxK[i] + dxV[i]
	}
	return dXn, dw, nil
}

// ropePartialForward applies split-half rotary to only the first rd dims of each
// hd-wide head (the serving partial-rope convention, hostmath.RopeWidth), leaving
// [rd:hd] untouched. It reuses the full-width RoPEHalfForward kernel by gathering
// the rotary sub-block into a compact [rows, nHeads*rd] tensor, rotating, and
// scattering back. rd == hd is the full-rope fast path.
func ropePartialForward(worker *device.Worker, x, invFreq []float32, T, nHeads, hd, rd int) ([]float32, error) {
	if rd == hd {
		return RoPEHalfForward(worker, x, invFreq, T, nHeads, hd)
	}
	packed := gatherRotary(x, T*nHeads, hd, rd)
	rot, err := RoPEHalfForward(worker, packed, invFreq, T, nHeads, rd)
	if err != nil {
		return nil, err
	}
	return scatterRotary(x, rot, T*nHeads, hd, rd), nil
}

// ropePartialBackward is the VJP of ropePartialForward: the pass-through dims are
// identity, the rotary sub-block reverses through RoPEHalfBackward.
func ropePartialBackward(worker *device.Worker, dx, invFreq []float32, T, nHeads, hd, rd int) ([]float32, error) {
	if rd == hd {
		return RoPEHalfBackward(worker, dx, invFreq, T, nHeads, hd)
	}
	packed := gatherRotary(dx, T*nHeads, hd, rd)
	rot, err := RoPEHalfBackward(worker, packed, invFreq, T, nHeads, rd)
	if err != nil {
		return nil, err
	}
	return scatterRotary(dx, rot, T*nHeads, hd, rd), nil
}

// gatherRotary packs the first rd dims of each hd-wide row into [rows, rd].
func gatherRotary(x []float32, rows, hd, rd int) []float32 {
	out := make([]float32, rows*rd)
	for r := 0; r < rows; r++ {
		copy(out[r*rd:(r+1)*rd], x[r*hd:r*hd+rd])
	}
	return out
}

// scatterRotary writes the rotated [rows, rd] block back over the first rd dims of
// each hd-wide row of a copy of base (base supplies the untouched [rd:hd] tail).
func scatterRotary(base, rot []float32, rows, hd, rd int) []float32 {
	out := append([]float32(nil), base...)
	for r := 0; r < rows; r++ {
		copy(out[r*hd:r*hd+rd], rot[r*rd:(r+1)*rd])
	}
	return out
}

// f64To32 narrows a f64 slice (RopeInvFreq output) to f32 for the device kernels.
func f64To32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}
