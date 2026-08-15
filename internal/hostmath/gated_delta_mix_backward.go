package hostmath

import "math"

// GatedDeltaMixGrads holds the gradients of every GatedDeltaMixForward input.
type GatedDeltaMixGrads struct {
	DX                                 []float32
	DWq, DWk, DWv                      []float32
	DConvQ, DConvK, DConvV             []float32
	DConvBiasQ, DConvBiasK, DConvBiasV []float32
	DWbeta, DWalpha                    []float32
	DTimeStep, DA                      []float32
	DWz                                []float32
	DNorm                              []float32
	DWout                              []float32
	DState                             []float32
}

// GatedDeltaMixBackward is the VJP of GatedDeltaMixForward: given the output
// cotangent dOut [T,OutDim] and the forward cache, returns every input gradient.
// It reverses the composition op by op, reusing the pieces built for this rung
// (GatedDeltaNetBackward, ShortConvBackward, L2NormBackward) plus the projection/
// gate/norm VJPs.
func GatedDeltaMixBackward(x []float32, w GatedDeltaMixWeights, d GatedDeltaMixDims, state, dOut []float32, c gatedDeltaMixCache) GatedDeltaMixGrads {
	T, H, hk, hv, hd, K := d.Tokens, d.Hidden, d.KeyHeads, d.ValueHeads, d.HeadDim, d.ConvK
	keyDim, valDim := hk*hd, hv*hd
	var g GatedDeltaMixGrads
	g.DX = make([]float32, T*H)
	addDX := func(part []float32) {
		for i := range part {
			g.DX[i] += part[i]
		}
	}

	// out = gated·Woutᵀ
	dGated, dWout := linearBackward2(c.gated, w.Wout, dOut, T, valDim, d.OutDim)
	g.DWout = dWout
	// gated = normed ⊙ siluZ
	dNormed := make([]float32, valDim*T)
	dSiluZ := make([]float32, valDim*T)
	for i := range dGated {
		dNormed[i] = float32(float64(dGated[i]) * float64(c.siluZ[i]))
		dSiluZ[i] = float32(float64(dGated[i]) * float64(c.normed[i]))
	}
	// siluZ = SiLU(z) -> dz
	dz := make([]float32, valDim*T)
	for i := range c.z {
		v := float64(c.z[i])
		s := 1 / (1 + math.Exp(-v))
		dz[i] = float32(float64(dSiluZ[i]) * s * (1 + v*(1-s)))
	}
	// normed = weightedRMSNorm(gdnOut, Norm) -> dGdnOut, dNorm
	dGdnOut, dNorm := weightedRMSNormBackward(c.gdnOut, w.Norm, dNormed, T, valDim, d.Eps)
	g.DNorm = dNorm
	// z = Wz·x
	dxZ, dWz := linearBackward2(x, w.Wz, dz, T, H, valDim)
	g.DWz = dWz
	addDX(dxZ)
	// GDN backward
	dqL2, dkL2, dvConv, dGate, dBeta, dState := GatedDeltaNetBackward(
		c.qL2, c.kL2, c.vConv, c.gate, c.beta, state, dGdnOut,
		hd, hk, hk, hv, T, 1, 1, false)
	g.DState = dState
	// gate = softplus(alpha+ts)*A
	dAlpha := make([]float32, T*hv)
	g.DTimeStep = make([]float32, hv)
	g.DA = make([]float32, hv)
	for t := 0; t < T; t++ {
		for h := 0; h < hv; h++ {
			pre := float64(c.alpha[t*hv+h]) + float64(w.TimeStep[h])
			sp := softplus(pre)
			dg := float64(dGate[t*hv+h])
			g.DA[h] += float32(dg * sp)
			dPre := dg * float64(w.A[h]) * (1 / (1 + math.Exp(-pre))) // *softplus'(pre)=sigmoid
			dAlpha[t*hv+h] = float32(dPre)
			g.DTimeStep[h] += float32(dPre)
		}
	}
	// alpha = Walpha·x
	dxA, dWalpha := linearBackward2(x, w.Walpha, dAlpha, T, H, hv)
	g.DWalpha = dWalpha
	addDX(dxA)
	// beta = sigmoid(Wbeta·x)
	dBetaPre := make([]float32, T*hv)
	for i := range c.beta {
		b := float64(c.beta[i])
		dBetaPre[i] = float32(float64(dBeta[i]) * b * (1 - b))
	}
	dxB, dWbeta := linearBackward2(x, w.Wbeta, dBetaPre, T, H, hv)
	g.DWbeta = dWbeta
	addDX(dxB)
	// qL2/kL2 = L2Norm(qConv/kConv)
	dqConv := L2NormBackward(c.qConv, dqL2, T*hk, hd, d.Eps)
	dkConv := L2NormBackward(c.kConv, dkL2, T*hk, hd, d.Eps)
	// {q,k,v}Conv = ShortConv({q,k,v}Proj)  (channel-major inside)
	dqProj, dConvQ, dConvBiasQ := shortConvBackTokenMajor(c.qProj, dqConv, T, keyDim, w.ConvQ, w.ConvBiasQ, K)
	dkProj, dConvK, dConvBiasK := shortConvBackTokenMajor(c.kProj, dkConv, T, keyDim, w.ConvK, w.ConvBiasK, K)
	dvProj, dConvV, dConvBiasV := shortConvBackTokenMajor(c.vProj, dvConv, T, valDim, w.ConvV, w.ConvBiasV, K)
	g.DConvQ, g.DConvBiasQ = dConvQ, dConvBiasQ
	g.DConvK, g.DConvBiasK = dConvK, dConvBiasK
	g.DConvV, g.DConvBiasV = dConvV, dConvBiasV
	// {q,k,v}Proj = W{q,k,v}·x
	dxQ, dWq := linearBackward2(x, w.Wq, dqProj, T, H, keyDim)
	dxK, dWk := linearBackward2(x, w.Wk, dkProj, T, H, keyDim)
	dxV, dWv := linearBackward2(x, w.Wv, dvProj, T, H, valDim)
	g.DWq, g.DWk, g.DWv = dWq, dWk, dWv
	addDX(dxQ)
	addDX(dxK)
	addDX(dxV)
	return g
}

// shortConvBackTokenMajor wraps ShortConvBackward with the token-major<->channel-
// major transposes the mix uses.
func shortConvBackTokenMajor(proj, dConv []float32, T, ch int, w, bias []float32, K int) (dProj, dW, dBias []float32) {
	dcm := Transpose2D(dConv, T, ch)
	xcm := Transpose2D(proj, T, ch)
	dxcm, dW, dBias := ShortConvBackward(xcm, dcm, ch, T, w, bias, K)
	return Transpose2D(dxcm, ch, T), dW, dBias
}

// weightedRMSNormBackward: VJP of y = x/sqrt(mean(x^2)+eps)*weight. Returns dx and
// dweight (summed over rows).
func weightedRMSNormBackward(x, weight, dY []float32, rows, width int, eps float64) (dx, dWeight []float32) {
	dx = make([]float32, len(x))
	dWeight = make([]float32, width)
	for r := 0; r < rows; r++ {
		row := x[r*width : r*width+width]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(ss/float64(width)+eps)
		var dot float64 // sum_c dY_c * w_c * x_c
		for c, v := range row {
			dot += float64(dY[r*width+c]) * float64(weight[c]) * float64(v)
			dWeight[c] += float32(float64(dY[r*width+c]) * float64(v) * inv)
		}
		inv3 := inv * inv * inv / float64(width)
		for c, v := range row {
			g := inv*float64(weight[c])*float64(dY[r*width+c]) - inv3*float64(v)*dot
			dx[r*width+c] = float32(g)
		}
	}
	return dx, dWeight
}

// linearBackward2: VJP of Y = X·Wᵀ (W[out,in]). dX = dY·W ; dW = dYᵀ·X.
func linearBackward2(x, w, dY []float32, rows, in, out int) (dX, dW []float32) {
	dX = make([]float32, rows*in)
	dW = make([]float32, out*in)
	for r := 0; r < rows; r++ {
		for i := 0; i < in; i++ {
			var acc float64
			for o := 0; o < out; o++ {
				acc += float64(dY[r*out+o]) * float64(w[o*in+i])
			}
			dX[r*in+i] = float32(acc)
		}
	}
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			var acc float64
			for r := 0; r < rows; r++ {
				acc += float64(dY[r*out+o]) * float64(x[r*in+i])
			}
			dW[o*in+i] = float32(acc)
		}
	}
	return dX, dW
}
