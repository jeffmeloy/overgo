//go:build windows

package devicemath

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
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
	return hybridLayerBackwardW(worker, x, hybridHostMatW(w), w, d, state, dOut)
}

// hybridLayerBackwardW is HybridDecoderLayerBackwardDevice over resident-or-host
// matrix weights (mw); VECTOR weights stay host-owned via w. Unlike the original
// host shortcut, the GDN residual mixOut is recomputed on DEVICE from mw (via
// gatedDeltaMixForwardDeviceW + Wout) rather than from the host GDN forward -- so
// with resident mw it reads the CURRENT resident weights, never the (now-unrefreshed)
// host slices. Matrix gradients follow their linWeight binding: a borrowed
// destination keeps them resident, otherwise they return as host slices.
func hybridLayerBackwardW(worker *device.Worker, x []float32, mw hybridMatW, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state, dOut []float32) (hostmath.HybridDecoderLayerGrads, error) {
	T, H := d.Tokens, d.Hidden
	if len(x) != T*H || len(dOut) != T*H {
		return hostmath.HybridDecoderLayerGrads{}, fmt.Errorf("hybridLayerBackwardW: shape mismatch (T=%d H=%d x=%d dOut=%d)", T, H, len(x), len(dOut))
	}

	var g hostmath.HybridDecoderLayerGrads
	g.IsLinear = w.IsLinear
	eps := d.Eps

	// --- forward recompute (device): only the intermediates the VJP consumes ---
	xn, err := RMSNormForward(worker, x, w.InputNorm, T, H, eps)
	if err != nil {
		return g, err
	}
	// mix forward on device from mw (resident-safe): attention returns its VJP cache;
	// the GDN residual mixOut is gated·Woutᵀ recomputed from mw (the backward's own
	// GatedDeltaMixBackwardDeviceW recomputes the rest of its intermediates).
	var mixOut []float32
	var ac attnMixDeviceCache
	if w.IsLinear {
		gc, err := gatedDeltaMixForwardDeviceW(worker, xn, mw.gdn, w.GDN, d.GDN, state)
		if err != nil {
			return g, err
		}
		valDim := d.GDN.ValueHeads * d.GDN.HeadDim
		mixOut, err = linearForwardTW(worker, gc.gated, mw.gdn.wout, T, valDim, d.GDN.OutDim)
		if err != nil {
			return g, err
		}
	} else {
		mixOut, ac, err = attentionMixForwardDeviceW(worker, xn, mw.attn, w.Attn, d.Attn)
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
	gateP, err := linearForwardTW(worker, hn, mw.mlp.gate, T, H, d.Inter)
	if err != nil {
		return g, err
	}
	upP, err := linearForwardTW(worker, hn, mw.mlp.up, T, H, d.Inter)
	if err != nil {
		return g, err
	}
	aP, hMLP, err := SiLUGateForward(worker, gateP, upP)
	if err != nil {
		return g, err
	}

	// --- MLP branch backward (SwiGLU): out = h + MLP(RMSNorm(h,PostNorm)) ---
	mlp, err := gatedMLPBackwardTW(worker, hn, mw.mlp, gateP, aP, upP, hMLP, dOut, T, H, d.Inter)
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
		mg, err := gatedDeltaMixBackwardDeviceW(worker, xn, mw.gdn, w.GDN, d.GDN, state, dh)
		if err != nil {
			return g, err
		}
		g.DGDN = mg
		g.DState = mg.DState
		dXn = mg.DX
	} else {
		dxn, dAttn, err := attentionMixBackwardDeviceW(worker, xn, mw.attn, w.Attn, d.Attn, dh, ac)
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

// attentionMixForwardDeviceW is attentionMixForwardDevice over resident-or-host
// matrix weights (mw: Wq/Wk/Wv/Wo); the per-head q/k RMSNorm weights stay
// host-owned via w. A resident-weight attention forward re-uploads no matrix
// weight.
func attentionMixForwardDeviceW(worker *device.Worker, xn []float32, mw attnMatW, w hostmath.AttentionMixWeights, ad hostmath.AttentionMixDims) ([]float32, attnMixDeviceCache, error) {
	var c attnMixDeviceCache
	T, H := ad.Tokens, ad.Hidden
	c.heads, c.kv, c.hd = ad.Heads, ad.KVHeads, ad.HeadDim
	c.qDim, c.kvDim = c.heads*c.hd, c.kv*c.hd
	c.rd = hostmath.RopeWidth(ad.RopeDim, c.hd)
	c.invFreq = dtype.Float64SliceToFloat32(hostmath.RopeInvFreq(ad.RopeTheta, c.rd))
	c.scale = float32(1.0 / math.Sqrt(float64(c.hd)))

	qProj, err := linearForwardTW(worker, xn, mw.wq, T, H, c.qDim)
	if err != nil {
		return nil, c, err
	}
	kProj, err := linearForwardTW(worker, xn, mw.wk, T, H, c.kvDim)
	if err != nil {
		return nil, c, err
	}
	v, err := linearForwardTW(worker, xn, mw.wv, T, H, c.kvDim)
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
	qs, err := ropePartial(worker, qn, c.invFreq, T, c.heads, c.hd, c.rd, ropeForward)
	if err != nil {
		return nil, c, err
	}
	for i := range qs {
		qs[i] *= c.scale
	}
	kr, err := ropePartial(worker, kn, c.invFreq, T, c.kv, c.hd, c.rd, ropeForward)
	if err != nil {
		return nil, c, err
	}
	c.qs, c.kr = qs, kr

	attn, err := MultiHeadAttentionForwardResident(worker, qs, kr, v, T, c.heads, c.kv, c.hd)
	if err != nil {
		return nil, c, err
	}
	c.attn = attn
	mixOut, err := linearForwardTW(worker, attn, mw.wo, T, c.qDim, H)
	if err != nil {
		return nil, c, err
	}
	return mixOut, c, nil
}

// attentionMixBackwardDeviceW is attentionMixBackwardDevice over resident-or-host
// matrix weights (mw); q/k RMSNorm weights stay host-owned via w. Matrix VJPs use
// linearBackwardTW over mw (no weight read-back); weight grads still return host.
func attentionMixBackwardDeviceW(worker *device.Worker, xn []float32, mw attnMatW, w hostmath.AttentionMixWeights, ad hostmath.AttentionMixDims, dOut []float32, c attnMixDeviceCache) ([]float32, hostmath.AttentionMixWeights, error) {
	T, H := ad.Tokens, ad.Hidden
	var dw hostmath.AttentionMixWeights

	// Wo: dAttn = dOut·Wo ; dWo = dOutᵀ·attn
	dAttn, dWo, err := linearBackwardTW(worker, c.attn, mw.wo, dOut, T, c.qDim, H)
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
	dqs, err = ropePartial(worker, dqs, c.invFreq, T, c.heads, c.hd, c.rd, ropeGradient)
	if err != nil {
		return nil, dw, err
	}
	dkr, err = ropePartial(worker, dkr, c.invFreq, T, c.kv, c.hd, c.rd, ropeGradient)
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
	dxQ, dWq, err := linearBackwardTW(worker, xn, mw.wq, dqProj, T, H, c.qDim)
	if err != nil {
		return nil, dw, err
	}
	dxK, dWk, err := linearBackwardTW(worker, xn, mw.wk, dkProj, T, H, c.kvDim)
	if err != nil {
		return nil, dw, err
	}
	dxV, dWv, err := linearBackwardTW(worker, xn, mw.wv, dv, T, H, c.kvDim)
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

// ropePartial: split-half rotation over each head prefix.
func ropePartial(worker *device.Worker, values, invFreq []float32, tokens, heads, width, rotaryWidth int, direction ropeDirection) ([]float32, error) {
	transform := RoPEHalfForward
	if direction == ropeGradient {
		transform = RoPEHalfBackward
	}
	if rotaryWidth == width {
		return transform(worker, values, invFreq, tokens, heads, width)
	}
	rows := tokens * heads
	packed := gatherRotary(values, rows, width, rotaryWidth)
	rotated, err := transform(worker, packed, invFreq, tokens, heads, rotaryWidth)
	if err != nil {
		return nil, err
	}
	return scatterRotary(values, rotated, rows, width, rotaryWidth), nil
}

// gatherRotary: compact row prefixes.
func gatherRotary(values []float32, rows, width, rotaryWidth int) []float32 {
	out := make([]float32, rows*rotaryWidth)
	for row := range rows {
		copy(out[row*rotaryWidth:(row+1)*rotaryWidth], values[row*width:row*width+rotaryWidth])
	}
	return out
}

// scatterRotary: replace row prefixes; preserve tails.
func scatterRotary(base, rotated []float32, rows, width, rotaryWidth int) []float32 {
	out := append([]float32(nil), base...)
	for row := range rows {
		copy(out[row*width:row*width+rotaryWidth], rotated[row*rotaryWidth:(row+1)*rotaryWidth])
	}
	return out
}
