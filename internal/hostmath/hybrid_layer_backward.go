package hostmath

// VJP of the qwen3.5 hybrid decoder layer (hybrid_layer.go). Each op is
// reversed with the neutral backward primitives; FD-checked as one layer for
// both mix variants.

// HybridDecoderLayerGrads: gradients of every layer input. Exactly one of
// DGDN/DAttn is populated per IsLinear; DState is set for linear layers.
type HybridDecoderLayerGrads struct {
	DX                    []float32
	DInputNorm, DPostNorm []float32
	IsLinear              bool
	DGDN                  GatedDeltaMixGrads
	DAttn                 AttentionMixWeights // grads reuse the weight struct
	DMLP                  QwenMLPWeights
	DState                []float32
}

// rmsBack wraps RMSNormBackward into (dx, dscale) with a fresh dx.
func rmsBack(x, weight, dy []float32, rows, d int, eps float64) (dx, dscale []float32) {
	dx = make([]float32, len(x))
	dscale = make([]float32, d)
	RMSNormBackward(dx, dscale, x, weight, dy, rows, d, eps, false)
	return dx, dscale
}

// mlpBackward: VJP of mlpForward.
func mlpBackward(x []float32, w QwenMLPWeights, dOut []float32, c mlpCache, T, hidden, inter int) (dx []float32, dw QwenMLPWeights) {
	dg, dDown := linearBackward2(c.g, w.Down, dOut, T, inter, hidden)
	dGate := make([]float32, T*inter)
	dUp := make([]float32, T*inter)
	SiLUGateBackward(dGate, dUp, c.gate, c.up, dg)
	dxG, dWgate := linearBackward2(x, w.Gate, dGate, T, hidden, inter)
	dxU, dWup := linearBackward2(x, w.Up, dUp, T, hidden, inter)
	dx = make([]float32, T*hidden)
	for i := range dx {
		dx[i] = dxG[i] + dxU[i]
	}
	return dx, QwenMLPWeights{Gate: dWgate, Up: dWup, Down: dDown}
}

// attentionMixBackward: VJP of attentionMixForward.
func attentionMixBackward(x []float32, w AttentionMixWeights, d AttentionMixDims, dOut []float32, c attnMixCache) (dx []float32, dw AttentionMixWeights) {
	T, H, heads, kv, hd := d.Tokens, d.Hidden, d.Heads, d.KVHeads, d.HeadDim
	qDim, kvDim := heads*hd, kv*hd
	dAttn, dWo := linearBackward2(c.attn, w.Wo, dOut, T, qDim, H)
	dqs := make([]float32, T*qDim)
	dkr := make([]float32, T*kvDim)
	dv := make([]float32, T*kvDim)
	CausalAttentionBackward(dqs, dkr, dv, c.qs, c.kr, c.v, dAttn, T, heads, kv, hd)
	// qs = scale * rope(qn): undo scale, then rope (in place, per head).
	for i := range dqs {
		dqs[i] = float32(float64(dqs[i]) * c.scale)
	}
	rd := RopeWidth(d.RopeDim, hd)
	for t := range T {
		for h := range heads {
			base := (t*heads + h) * hd
			RotaryHalfBackward(dqs[base:base+rd], c.invFreq, t)
		}
		for h := range kv {
			base := (t*kv + h) * hd
			RotaryHalfBackward(dkr[base:base+rd], c.invFreq, t)
		}
	}
	dqProj, dQNorm := rmsBack(c.qProj, w.QNorm, dqs, T*heads, hd, d.Eps)
	dkProj, dKNorm := rmsBack(c.kProj, w.KNorm, dkr, T*kv, hd, d.Eps)
	dxQ, dWq := linearBackward2(x, w.Wq, dqProj, T, H, qDim)
	dxK, dWk := linearBackward2(x, w.Wk, dkProj, T, H, kvDim)
	dxV, dWv := linearBackward2(x, w.Wv, dv, T, H, kvDim)
	dx = make([]float32, T*H)
	for i := range dx {
		dx[i] = dxQ[i] + dxK[i] + dxV[i]
	}
	return dx, AttentionMixWeights{Wq: dWq, Wk: dWk, Wv: dWv, Wo: dWo, QNorm: dQNorm, KNorm: dKNorm}
}

// HybridDecoderLayerBackward: VJP of HybridDecoderLayerForward. Both residuals
// route the cotangent to two paths; each pre-norm and its mix/MLP reverse with
// the primitives above.
func HybridDecoderLayerBackward(x []float32, w HybridLayerWeights, d HybridLayerDims, state, dOut []float32, c hybridLayerCache) HybridDecoderLayerGrads {
	T, H := d.Tokens, d.Hidden
	var g HybridDecoderLayerGrads
	g.IsLinear = w.IsLinear
	// out = h + mlpOut  -> dh gets dOut directly plus the norm/MLP path.
	dh := make([]float32, T*H)
	copy(dh, dOut)
	dHn, dMLP := mlpBackward(c.hn, w.MLP, dOut, c.mlp, T, H, d.Inter)
	g.DMLP = dMLP
	dhNorm, dPostNorm := rmsBack(c.h, w.PostNorm, dHn, T, H, d.Eps)
	g.DPostNorm = dPostNorm
	for i := range dh {
		dh[i] += dhNorm[i]
	}
	// h = x + mixOut  -> dx gets dh directly plus the input-norm/mix path.
	dx := make([]float32, T*H)
	copy(dx, dh)
	var dXn []float32
	if w.IsLinear {
		mg := GatedDeltaMixBackward(c.xn, w.GDN, d.GDN, state, dh, c.mix)
		g.DGDN = mg
		g.DState = mg.DState
		dXn = mg.DX
	} else {
		dxn, daw := attentionMixBackward(c.xn, w.Attn, d.Attn, dh, c.attnMix)
		g.DAttn = daw
		dXn = dxn
	}
	dxNorm, dInputNorm := rmsBack(x, w.InputNorm, dXn, T, H, d.Eps)
	g.DInputNorm = dInputNorm
	for i := range dx {
		dx[i] += dxNorm[i]
	}
	g.DX = dx
	return g
}
