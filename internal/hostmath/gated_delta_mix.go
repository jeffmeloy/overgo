package hostmath

import "math"

// GatedDeltaMixWeights holds the qwen3.5 linear_attention (gated-delta) mix
// weights, densecausal-style row-major [out,in] projections. Shapes use head dim
// d, key heads hk, value heads hv, hidden H, conv kernel K:
//
//	Wq/Wk: [hk*d, H]   Wv: [hv*d, H]         (projections from the normed input)
//	ConvQ/ConvK: [hk*d, K]  ConvV: [hv*d, K]  ConvBias: per channel (nil ok)
//	Wbeta/Walpha: [hv, H]   TimeStep/A: [hv]  Wz: [hv*d, H]
//	Norm: [d] (per-value-head SSMNorm)  Wout: [Hout, hv*d]
type GatedDeltaMixWeights struct {
	Wq, Wk, Wv                      []float32
	ConvQ, ConvK, ConvV             []float32
	ConvBiasQ, ConvBiasK, ConvBiasV []float32
	Wbeta, Walpha                   []float32
	TimeStep, A                     []float32
	Wz                              []float32
	Norm                            []float32
	Wout                            []float32
}

// GatedDeltaMixDims: geometry of the mix.
type GatedDeltaMixDims struct {
	Tokens, Hidden, KeyHeads, ValueHeads, HeadDim, ConvK, OutDim int
	Eps                                                          float64
}

// gatedDeltaMixCache holds forward intermediates the VJP consumes.
type gatedDeltaMixCache struct {
	qProj, kProj, vProj  []float32 // [T, h*d] projections
	qConv, kConv, vConv  []float32 // after ShortConv (SiLU(conv))
	qL2, kL2             []float32 // after L2Norm
	beta, alpha, gate, z []float32 // beta[T,hv], alpha[T,hv], gate[T,hv], z[T,hv*d]
	gdnOut, gdnState     []float32 // GDN output [T,hv,d] and final state
	normed, siluZ        []float32 // WeightedRMSNorm(gdnOut) and SiLU(z), both [T,hv*d]
	gated                []float32 // normed*siluZ [T,hv*d]
}

// GatedDeltaMixForward runs the qwen3.5 gated-delta mix on host: from the normed
// input x [T,H] and state, produces the layer output [T,OutDim] and the cache the
// VJP consumes. state is [hv, d, d]. This composes the pieces built for this rung
// (ShortConv, L2Norm, GatedDeltaNet) with the projection/gate/norm path.
func GatedDeltaMixForward(x []float32, w GatedDeltaMixWeights, d GatedDeltaMixDims, state []float32) (out []float32, c gatedDeltaMixCache) {
	T, H, hk, hv, hd, K := d.Tokens, d.Hidden, d.KeyHeads, d.ValueHeads, d.HeadDim, d.ConvK
	keyDim, valDim := hk*hd, hv*hd

	c.qProj = Linear2(x, w.Wq, T, H, keyDim)
	c.kProj = Linear2(x, w.Wk, T, H, keyDim)
	c.vProj = Linear2(x, w.Wv, T, H, valDim)
	// ShortConv is channel-major [ch,T]; projections are token-major [T,ch].
	c.qConv = Transpose2D(ShortConvForward(Transpose2D(c.qProj, T, keyDim), keyDim, T, w.ConvQ, w.ConvBiasQ, K), keyDim, T)
	c.kConv = Transpose2D(ShortConvForward(Transpose2D(c.kProj, T, keyDim), keyDim, T, w.ConvK, w.ConvBiasK, K), keyDim, T)
	c.vConv = Transpose2D(ShortConvForward(Transpose2D(c.vProj, T, valDim), valDim, T, w.ConvV, w.ConvBiasV, K), valDim, T)
	// L2Norm q,k per (token,head) over hd.
	c.qL2 = L2NormForward(c.qConv, T*hk, hd, d.Eps)
	c.kL2 = L2NormForward(c.kConv, T*hk, hd, d.Eps)
	// beta = sigmoid(Wbeta.x) [T,hv]; alpha = Walpha.x [T,hv]; gate = softplus(alpha+ts)*A.
	c.beta = Linear2(x, w.Wbeta, T, H, hv)
	for i, v := range c.beta {
		c.beta[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
	c.alpha = Linear2(x, w.Walpha, T, H, hv)
	c.gate = make([]float32, T*hv)
	for t := 0; t < T; t++ {
		for h := 0; h < hv; h++ {
			sp := softplus(float64(c.alpha[t*hv+h]) + float64(w.TimeStep[h]))
			c.gate[t*hv+h] = float32(sp * float64(w.A[h]))
		}
	}
	c.z = Linear2(x, w.Wz, T, H, valDim)
	// GDN: query/key [T,hk,hd], value [T,hv,hd], gate [T,hv] (gateWidth 1), beta [T,hv].
	c.gdnOut, c.gdnState = GatedDeltaNetForward(c.qL2, c.kL2, c.vConv, c.gate, c.beta, state,
		hd, hk, hk, hv, T, 1, 1, false)
	// output gate: WeightedRMSNorm(gdnOut, Norm) * SiLU(z), then Wout.
	c.normed = weightedRMSNorm(c.gdnOut, w.Norm, T*hv, hd, d.Eps)
	c.siluZ = make([]float32, len(c.z))
	for i, v := range c.z {
		c.siluZ[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
	}
	c.gated = make([]float32, valDim*T)
	for i := range c.gated {
		c.gated[i] = float32(float64(c.normed[i]) * float64(c.siluZ[i]))
	}
	out = Linear2(c.gated, w.Wout, T, valDim, d.OutDim)
	return out, c
}

// softplus in f64.
func softplus(x float64) float64 {
	if x > 30 {
		return x
	}
	return math.Log1p(math.Exp(x))
}

// weightedRMSNorm: per row (length width) y = x/sqrt(mean(x^2)+eps) * weight.
func weightedRMSNorm(x, weight []float32, rows, width int, eps float64) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		row := x[r*width : r*width+width]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(ss/float64(width)+eps)
		for c, v := range row {
			out[r*width+c] = float32(float64(v) * inv * float64(weight[c]))
		}
	}
	return out
}

// Linear2: row-major Y = X·Wᵀ, X[rows,in], W[out,in], Y[rows,out] (hostmath.Linear
// convention), returned as a fresh slice.
func Linear2(x, w []float32, rows, in, out int) []float32 {
	y := make([]float32, rows*out)
	Linear(y, x, w, rows, in, out)
	return y
}
