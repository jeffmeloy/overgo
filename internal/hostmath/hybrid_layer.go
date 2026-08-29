package hostmath

import "math"

// This file composes a qwen3.5 hybrid decoder layer on host from the neutral
// primitives: RMSNorm, the gated-delta mix (linear_attention), a standard
// full_attention mix (q/k RMSNorm + rotary + causal GQA), and a SwiGLU MLP,
// wired with the two pre-norms and two residuals. Forward here, VJP in
// hybrid_layer_backward.go, FD-checked as one composed layer.

// QwenMLPWeights: SwiGLU MLP. Gate/Up are [inter,hidden], Down is [hidden,inter].
type QwenMLPWeights struct{ Gate, Up, Down []float32 }

// AttentionMixWeights: full_attention projections and per-head q/k RMSNorm.
// Wq [heads*hd,hidden]; Wk/Wv [kvHeads*hd,hidden]; Wo [hidden,heads*hd];
// QNorm/KNorm [hd].
type AttentionMixWeights struct {
	Wq, Wk, Wv, Wo []float32
	QNorm, KNorm   []float32
}

// AttentionMixDims: full_attention geometry. RopeDim is the rotary width from
// the model config (qwen3.5: head_dim*partial_rotary_factor); 0 or >=HeadDim
// means full-head rope. Only the first RopeWidth(RopeDim,HeadDim) dims rotate.
type AttentionMixDims struct {
	Tokens, Hidden, Heads, KVHeads, HeadDim int
	RopeDim                                 int
	RopeTheta, Eps                          float64
}

// HybridLayerWeights: one decoder layer. IsLinear picks the mix (GDN vs
// attention); the unused mix's weights are ignored.
type HybridLayerWeights struct {
	InputNorm, PostNorm []float32
	IsLinear            bool
	GDN                 GatedDeltaMixWeights
	Attn                AttentionMixWeights
	MLP                 QwenMLPWeights
}

// HybridLayerDims: dims for both mix variants + the MLP (Inter). The active mix
// uses GDN* or Attn* per IsLinear.
type HybridLayerDims struct {
	Tokens, Hidden, Inter int
	Eps                   float64
	GDN                   GatedDeltaMixDims
	Attn                  AttentionMixDims
}

// attnMixCache holds full_attention forward intermediates for the VJP.
type attnMixCache struct {
	qProj, kProj, v []float32 // projections (v post-proj is used directly)
	qs, kr          []float32 // scaled+rotated q, rotated k
	attn            []float32 // attention output pre Wo
	invFreq         []float64
	scale           float64
}

// mlpCache holds SwiGLU forward intermediates for the VJP.
type mlpCache struct {
	gate, up, g []float32 // gate=Wg.x, up=Wu.x, g=silu(gate)*up
}

// hybridLayerCache holds every intermediate the layer VJP consumes.
type hybridLayerCache struct {
	xn, h, hn []float32 // pre-norms and the mid residual
	mix       gatedDeltaMixCache
	attnMix   attnMixCache
	mlp       mlpCache
}

// mlpForward: out = Down·(SiLU(Gate·x)⊙(Up·x)).
func mlpForward(x []float32, w QwenMLPWeights, T, hidden, inter int) (out []float32, c mlpCache) {
	c.gate = Linear2(x, w.Gate, T, hidden, inter)
	c.up = Linear2(x, w.Up, T, hidden, inter)
	c.g = make([]float32, T*inter)
	SiLUGate(c.g, c.gate, c.up)
	out = Linear2(c.g, w.Down, T, inter, hidden)
	return out, c
}

// attentionMixForward: q/k/v proj -> per-head RMSNorm(q,k) -> rotary(q,k) ->
// scale q by 1/sqrt(hd) -> causal GQA attention -> Wo.
func attentionMixForward(x []float32, w AttentionMixWeights, d AttentionMixDims) (out []float32, c attnMixCache) {
	T, H, heads, kv, hd := d.Tokens, d.Hidden, d.Heads, d.KVHeads, d.HeadDim
	qDim, kvDim := heads*hd, kv*hd
	c.qProj = Linear2(x, w.Wq, T, H, qDim)
	c.kProj = Linear2(x, w.Wk, T, H, kvDim)
	c.v = Linear2(x, w.Wv, T, H, kvDim)
	qn := make([]float32, T*qDim)
	kn := make([]float32, T*kvDim)
	RMSNormInto(qn, c.qProj, w.QNorm, T*heads, hd, d.Eps)
	RMSNormInto(kn, c.kProj, w.KNorm, T*kv, hd, d.Eps)
	rd := RopeWidth(d.RopeDim, hd)
	c.invFreq = RopeInvFreq(d.RopeTheta, rd)
	c.scale = 1.0 / math.Sqrt(float64(hd))
	c.qs = make([]float32, T*qDim)
	c.kr = make([]float32, T*kvDim)
	copy(c.qs, qn)
	copy(c.kr, kn)
	for t := range T {
		for h := range heads {
			row := c.qs[(t*heads+h)*hd : (t*heads+h+1)*hd]
			ApplyRotaryHalf(row[:rd], c.invFreq, t)
			for i := range row {
				row[i] *= float32(c.scale)
			}
		}
		for h := range kv {
			base := (t*kv + h) * hd
			ApplyRotaryHalf(c.kr[base:base+rd], c.invFreq, t)
		}
	}
	c.attn = make([]float32, T*qDim)
	CausalAttention(c.attn, c.qs, c.kr, c.v, T, heads, kv, hd)
	out = Linear2(c.attn, w.Wo, T, qDim, d.Hidden)
	return out, c
}

// HybridDecoderLayerForward: h = x + Mix(RMSNorm(x,InputNorm)); out = h +
// MLP(RMSNorm(h,PostNorm)). Mix is the gated-delta or attention mix per
// w.IsLinear. state is the GDN input state (nil/ignored for attention layers).
func HybridDecoderLayerForward(x []float32, w HybridLayerWeights, d HybridLayerDims, state []float32) (out []float32, c hybridLayerCache) {
	T, H := d.Tokens, d.Hidden
	c.xn = make([]float32, T*H)
	RMSNormInto(c.xn, x, w.InputNorm, T, H, d.Eps)
	var mixOut []float32
	if w.IsLinear {
		mixOut, c.mix = GatedDeltaMixForward(c.xn, w.GDN, d.GDN, state)
	} else {
		mixOut, c.attnMix = attentionMixForward(c.xn, w.Attn, d.Attn)
	}
	c.h = make([]float32, T*H)
	for i := range c.h {
		c.h[i] = x[i] + mixOut[i]
	}
	c.hn = make([]float32, T*H)
	RMSNormInto(c.hn, c.h, w.PostNorm, T, H, d.Eps)
	mlpOut, mlp := mlpForward(c.hn, w.MLP, T, H, d.Inter)
	c.mlp = mlp
	out = make([]float32, T*H)
	for i := range out {
		out[i] = c.h[i] + mlpOut[i]
	}
	return out, c
}
