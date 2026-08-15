//go:build windows

package devicemath

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
)

// gatedDeltaMixDeviceCache holds the GDN-mix forward intermediates the VJP
// consumes (mirrors hostmath.gatedDeltaMixCache). It is the single owner of the
// device forward-recompute: gatedDeltaMixForwardDevice produces it, and both the
// hybrid layer forward and GatedDeltaMixBackwardDevice consume it.
type gatedDeltaMixDeviceCache struct {
	qProj, kProj, vProj  []float32 // [T,h*d] projections
	qConv, kConv, vConv  []float32 // after ShortConv (SiLU(conv))
	qL2, kL2             []float32 // after L2Norm
	beta, alpha, gate, z []float32 // beta[T,hv], alpha[T,hv], gate[T,hv], z[T,valDim]
	gdnOut               []float32 // GDN output [T,hv*d]
	normed, siluZ        []float32 // WeightedRMSNorm(gdnOut) and SiLU(z)
	gated                []float32 // normed*siluZ [T,valDim]
}

// gatedDeltaMixForwardDevice recomputes hostmath.GatedDeltaMixForward's
// intermediates on device (through `gated`, one Wout matmul short of the mix
// output): device projections/norms via LinearForwardT/RMSNormForward, host
// ShortConv/L2Norm/GDN forward for the pieces without a device forward kernel,
// and the sigmoid/softplus/SiLU elementwise glue (exactly as
// hostmath.GatedDeltaMixForward). x is the normed mix input [T,Hidden]; state is
// the GDN input state [hv,hd,hd]. The mix output is
// LinearForwardT(c.gated, Wout); callers that need it (the layer forward) apply
// it, callers that reverse from `gated` (the backward) do not.
func gatedDeltaMixForwardDevice(worker *device.Worker, x []float32, w hostmath.GatedDeltaMixWeights, d hostmath.GatedDeltaMixDims, state []float32) (gatedDeltaMixDeviceCache, error) {
	return gatedDeltaMixForwardDeviceW(worker, x, gdnHostMatW(w), w, d, state)
}

// gatedDeltaMixForwardDeviceW is gatedDeltaMixForwardDevice over resident-or-host
// matrix weights (mw). The vector weights (conv kernels/biases, per-head scalars,
// the output RMSNorm weight) stay host-owned (Sign-updated) and come from w; only
// the projection matrices Wq/Wk/Wv/Wbeta/Walpha/Wz are read through mw, so a
// resident-weight GDN forward re-uploads no matrix weights.
func gatedDeltaMixForwardDeviceW(worker *device.Worker, x []float32, mw gdnMatW, w hostmath.GatedDeltaMixWeights, d hostmath.GatedDeltaMixDims, state []float32) (gatedDeltaMixDeviceCache, error) {
	T, H, hk, hv, hd, K := d.Tokens, d.Hidden, d.KeyHeads, d.ValueHeads, d.HeadDim, d.ConvK
	keyDim, valDim := hk*hd, hv*hd
	eps := d.Eps
	var c gatedDeltaMixDeviceCache

	qProj, err := linearForwardTW(worker, x, mw.wq, T, H, keyDim)
	if err != nil {
		return c, err
	}
	kProj, err := linearForwardTW(worker, x, mw.wk, T, H, keyDim)
	if err != nil {
		return c, err
	}
	vProj, err := linearForwardTW(worker, x, mw.wv, T, H, valDim)
	if err != nil {
		return c, err
	}
	c.qProj, c.kProj, c.vProj = qProj, kProj, vProj
	// ShortConv is channel-major [ch,T]; projections are token-major [T,ch].
	c.qConv = hostmath.Transpose2D(hostmath.ShortConvForward(hostmath.Transpose2D(qProj, T, keyDim), keyDim, T, w.ConvQ, w.ConvBiasQ, K), keyDim, T)
	c.kConv = hostmath.Transpose2D(hostmath.ShortConvForward(hostmath.Transpose2D(kProj, T, keyDim), keyDim, T, w.ConvK, w.ConvBiasK, K), keyDim, T)
	c.vConv = hostmath.Transpose2D(hostmath.ShortConvForward(hostmath.Transpose2D(vProj, T, valDim), valDim, T, w.ConvV, w.ConvBiasV, K), valDim, T)
	c.qL2 = hostmath.L2NormForward(c.qConv, T*hk, hd, eps)
	c.kL2 = hostmath.L2NormForward(c.kConv, T*hk, hd, eps)
	// beta = sigmoid(Wbeta·x); alpha = Walpha·x; gate = softplus(alpha+ts)*A.
	betaPre, err := linearForwardTW(worker, x, mw.wbeta, T, H, hv)
	if err != nil {
		return c, err
	}
	c.beta = make([]float32, T*hv)
	for i, v := range betaPre {
		c.beta[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
	c.alpha, err = linearForwardTW(worker, x, mw.walpha, T, H, hv)
	if err != nil {
		return c, err
	}
	c.gate = make([]float32, T*hv)
	for t := 0; t < T; t++ {
		for h := 0; h < hv; h++ {
			sp := mixSoftplus(float64(c.alpha[t*hv+h]) + float64(w.TimeStep[h]))
			c.gate[t*hv+h] = float32(sp * float64(w.A[h]))
		}
	}
	c.z, err = linearForwardTW(worker, x, mw.wz, T, H, valDim)
	if err != nil {
		return c, err
	}
	c.siluZ = make([]float32, len(c.z))
	for i, v := range c.z {
		c.siluZ[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
	}
	c.gdnOut, _ = hostmath.GatedDeltaNetForward(c.qL2, c.kL2, c.vConv, c.gate, c.beta, state, hd, hk, hk, hv, T, 1, 1, false)
	c.normed, err = RMSNormForward(worker, c.gdnOut, w.Norm, T*hv, hd, eps)
	if err != nil {
		return c, err
	}
	c.gated = make([]float32, valDim*T)
	for i := range c.gated {
		c.gated[i] = float32(float64(c.normed[i]) * float64(c.siluZ[i]))
	}
	return c, nil
}

// GatedDeltaMixBackwardDevice is the device VJP of hostmath's qwen3.5
// linear_attention (gated-delta) mix -- the device counterpart of
// hostmath.GatedDeltaMixBackward, composed op-for-op from the individually
// parity-verified device ops (LinearForwardT/LinearBackwardT, ShortConvBackward
// Device, L2NormBackwardDevice, GatedDeltaNetBackwardDevice, RMSNormForward/
// RMSNormBackward) plus the host-side gate/beta/SiLU elementwise glue (exactly as
// hostmath.GatedDeltaMixBackward and GatedMLPBackward do).
//
// The host forward cache (gatedDeltaMixCache) is package-private, so the device
// path recomputes the forward intermediates the VJP consumes from the proven
// forward primitives (device projections/norms, host ShortConv/L2Norm/GDN forward
// for the pieces without a device forward wrapper); the recomputed activations
// match the host cache within the ops' parity, so the composed grads match
// hostmath.GatedDeltaMixBackward to the kernel-parity class. x is the normed mix
// input [T,Hidden]; state is the GDN input state [hv,hd,hd]; dOut is [T,OutDim].
func GatedDeltaMixBackwardDevice(worker *device.Worker, x []float32, w hostmath.GatedDeltaMixWeights, d hostmath.GatedDeltaMixDims, state, dOut []float32) (hostmath.GatedDeltaMixGrads, error) {
	return gatedDeltaMixBackwardDeviceW(worker, x, gdnHostMatW(w), w, d, state, dOut)
}

// gatedDeltaMixBackwardDeviceW is GatedDeltaMixBackwardDevice over resident-or-host
// matrix weights (mw); vector weights stay host-owned via w. It reads no matrix
// weight back to host: the projection VJPs use linearBackwardTW over mw and the
// forward recompute uses gatedDeltaMixForwardDeviceW over the same mw. The weight
// GRADIENTS (DWq..DWout) still return as host slices -- grads are recomputed each
// step and uploaded into the resident grad buffer by the caller.
func gatedDeltaMixBackwardDeviceW(worker *device.Worker, x []float32, mw gdnMatW, w hostmath.GatedDeltaMixWeights, d hostmath.GatedDeltaMixDims, state, dOut []float32) (hostmath.GatedDeltaMixGrads, error) {
	T, H, hk, hv, hd, K := d.Tokens, d.Hidden, d.KeyHeads, d.ValueHeads, d.HeadDim, d.ConvK
	keyDim, valDim := hk*hd, hv*hd
	eps := d.Eps
	var g hostmath.GatedDeltaMixGrads
	if len(x) != T*H || len(dOut) != T*d.OutDim {
		return g, fmt.Errorf("gatedDeltaMixBackwardDeviceW: shape mismatch (T=%d H=%d OutDim=%d x=%d dOut=%d)", T, H, d.OutDim, len(x), len(dOut))
	}

	// --- forward recompute: only the intermediates the VJP consumes ---
	fc, err := gatedDeltaMixForwardDeviceW(worker, x, mw, w, d, state)
	if err != nil {
		return g, err
	}
	qProj, kProj, vProj := fc.qProj, fc.kProj, fc.vProj
	qConv, kConv, vConv := fc.qConv, fc.kConv, fc.vConv
	qL2, kL2 := fc.qL2, fc.kL2
	beta, alpha, gate, z := fc.beta, fc.alpha, fc.gate, fc.z
	siluZ, gdnOut, normed, gated := fc.siluZ, fc.gdnOut, fc.normed, fc.gated

	// --- backward: reverse the composition op by op ---
	g.DX = make([]float32, T*H)
	addDX := func(part []float32) {
		for i := range part {
			g.DX[i] += part[i]
		}
	}

	// out = gated·Woutᵀ
	dGated, dWout, err := linearBackwardTW(worker, gated, mw.wout, dOut, T, valDim, d.OutDim)
	if err != nil {
		return g, err
	}
	g.DWout = dWout
	// gated = normed ⊙ siluZ
	dNormed := make([]float32, valDim*T)
	dSiluZ := make([]float32, valDim*T)
	for i := range dGated {
		dNormed[i] = float32(float64(dGated[i]) * float64(siluZ[i]))
		dSiluZ[i] = float32(float64(dGated[i]) * float64(normed[i]))
	}
	// siluZ = SiLU(z) -> dz
	dz := make([]float32, valDim*T)
	for i := range z {
		v := float64(z[i])
		s := 1 / (1 + math.Exp(-v))
		dz[i] = float32(float64(dSiluZ[i]) * s * (1 + v*(1-s)))
	}
	// normed = weightedRMSNorm(gdnOut, Norm) -> dGdnOut, dNorm.
	dGdnOut, dNorm, err := RMSNormBackward(worker, gdnOut, w.Norm, dNormed, T*hv, hd, eps)
	if err != nil {
		return g, err
	}
	g.DNorm = dNorm
	// z = Wz·x
	dxZ, dWz, err := linearBackwardTW(worker, x, mw.wz, dz, T, H, valDim)
	if err != nil {
		return g, err
	}
	g.DWz = dWz
	addDX(dxZ)
	// GDN backward
	dqL2, dkL2, dvConv, dGate, dBeta, dState, err := GatedDeltaNetBackwardDevice(
		worker, qL2, kL2, vConv, gate, beta, state, dGdnOut,
		hd, hk, hk, hv, T, 1, 1, false)
	if err != nil {
		return g, err
	}
	g.DState = dState
	// gate = softplus(alpha+ts)*A
	dAlpha := make([]float32, T*hv)
	g.DTimeStep = make([]float32, hv)
	g.DA = make([]float32, hv)
	for t := 0; t < T; t++ {
		for h := 0; h < hv; h++ {
			pre := float64(alpha[t*hv+h]) + float64(w.TimeStep[h])
			sp := mixSoftplus(pre)
			dgate := float64(dGate[t*hv+h])
			g.DA[h] += float32(dgate * sp)
			dPre := dgate * float64(w.A[h]) * (1 / (1 + math.Exp(-pre))) // *softplus'(pre)=sigmoid
			dAlpha[t*hv+h] = float32(dPre)
			g.DTimeStep[h] += float32(dPre)
		}
	}
	// alpha = Walpha·x
	dxA, dWalpha, err := linearBackwardTW(worker, x, mw.walpha, dAlpha, T, H, hv)
	if err != nil {
		return g, err
	}
	g.DWalpha = dWalpha
	addDX(dxA)
	// beta = sigmoid(Wbeta·x)
	dBetaPre := make([]float32, T*hv)
	for i := range beta {
		b := float64(beta[i])
		dBetaPre[i] = float32(float64(dBeta[i]) * b * (1 - b))
	}
	dxB, dWbeta, err := linearBackwardTW(worker, x, mw.wbeta, dBetaPre, T, H, hv)
	if err != nil {
		return g, err
	}
	g.DWbeta = dWbeta
	addDX(dxB)
	// qL2/kL2 = L2Norm(qConv/kConv)
	dqConv, err := L2NormBackwardDevice(worker, qConv, dqL2, T*hk, hd, eps)
	if err != nil {
		return g, err
	}
	dkConv, err := L2NormBackwardDevice(worker, kConv, dkL2, T*hk, hd, eps)
	if err != nil {
		return g, err
	}
	// {q,k,v}Conv = ShortConv({q,k,v}Proj) (channel-major inside)
	dqProj, dConvQ, dConvBiasQ, err := mixShortConvBackTokenMajor(worker, qProj, dqConv, T, keyDim, w.ConvQ, w.ConvBiasQ, K)
	if err != nil {
		return g, err
	}
	dkProj, dConvK, dConvBiasK, err := mixShortConvBackTokenMajor(worker, kProj, dkConv, T, keyDim, w.ConvK, w.ConvBiasK, K)
	if err != nil {
		return g, err
	}
	dvProj, dConvV, dConvBiasV, err := mixShortConvBackTokenMajor(worker, vProj, dvConv, T, valDim, w.ConvV, w.ConvBiasV, K)
	if err != nil {
		return g, err
	}
	g.DConvQ, g.DConvBiasQ = dConvQ, dConvBiasQ
	g.DConvK, g.DConvBiasK = dConvK, dConvBiasK
	g.DConvV, g.DConvBiasV = dConvV, dConvBiasV
	// {q,k,v}Proj = W{q,k,v}·x
	dxQ, dWq, err := linearBackwardTW(worker, x, mw.wq, dqProj, T, H, keyDim)
	if err != nil {
		return g, err
	}
	dxK, dWk, err := linearBackwardTW(worker, x, mw.wk, dkProj, T, H, keyDim)
	if err != nil {
		return g, err
	}
	dxV, dWv, err := linearBackwardTW(worker, x, mw.wv, dvProj, T, H, valDim)
	if err != nil {
		return g, err
	}
	g.DWq, g.DWk, g.DWv = dWq, dWk, dWv
	addDX(dxQ)
	addDX(dxK)
	addDX(dxV)
	return g, nil
}

// mixShortConvBackTokenMajor wraps ShortConvBackwardDevice with the token-major
// <-> channel-major transposes the mix uses (mirrors host
// shortConvBackTokenMajor).
func mixShortConvBackTokenMajor(worker *device.Worker, proj, dConv []float32, T, ch int, w, bias []float32, K int) (dProj, dW, dBias []float32, err error) {
	dcm := hostmath.Transpose2D(dConv, T, ch)
	xcm := hostmath.Transpose2D(proj, T, ch)
	dxcm, dW, dBias, err := ShortConvBackwardDevice(worker, xcm, dcm, ch, T, w, bias, K)
	if err != nil {
		return nil, nil, nil, err
	}
	return hostmath.Transpose2D(dxcm, ch, T), dW, dBias, nil
}

// mixSoftplus mirrors hostmath's package-private softplus (f64, overflow guard).
func mixSoftplus(x float64) float64 {
	if x > 30 {
		return x
	}
	return math.Log1p(math.Exp(x))
}
