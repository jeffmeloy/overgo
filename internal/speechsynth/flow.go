// Conditional flow head contract (vendor SimpleMLPAdaLN): timestep embedder
// normalizes with UNBIASED (N-1) variance; block/final LayerNorm use biased
// variance; the final LayerNorm has no affine; modulate = x*(1+scale)+shift;
// one-step decode = x0 + F(cond, 0, 1, x0).
package speechsynth

import (
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

// forwardInto: out[flowDim] = timeEmbed(t) — the alpha-scaled RMS quirk.
func (te *timeEmbed) forwardInto(out []float32, t float64, flowDim int, rmsEpsilon float64) {
	half := len(te.freqs)
	emb := make([]float32, tensor.PairedExtent*half)
	for i, f := range te.freqs {
		arg := t * float64(f)
		emb[i] = float32(math.Cos(arg))
		emb[half+i] = float32(math.Sin(arg))
	}
	h0 := make([]float32, flowDim)
	hostmath.LinearF64(h0, emb, te.l0w, te.l0b, tensor.SingletonExtent, tensor.PairedExtent*half, flowDim)
	hostmath.SiLUInPlace(h0)
	h := make([]float32, flowDim)
	hostmath.LinearF64(h, h0, te.l2w, te.l2b, tensor.SingletonExtent, flowDim, flowDim)
	te.normalizeTime(out, h, rmsEpsilon)
}

// normalizeTime preserves the reference's FP32 variance, reciprocal and scale boundaries.
func (te *timeEmbed) normalizeTime(out, h []float32, rmsEpsilon float64) (mean, inv float64) {
	for _, v := range h {
		mean += float64(v)
	}
	mean /= float64(len(h))
	var varsum float64
	for _, v := range h {
		delta := float64(v) - mean
		varsum += delta * delta
	}
	variance := float32(varsum / float64(len(h)-tensor.SingletonExtent))
	inv = float64(float32(tensor.SingletonExtent) / float32(math.Sqrt(float64(variance+float32(rmsEpsilon)))))
	for i := range out {
		out[i] = h[i] * float32(te.alpha[i]*float32(inv))
	}
	return mean, inv
}

func (m *Model) flowForwardInto(out, cond []float32, s, t float64, x []float32) {
	fn := &m.flow
	dim, latent := m.Dims.FlowDim, m.Dims.LatentDim

	cur := make([]float32, dim)
	hostmath.LinearF64(cur, x, fn.inputProjW, fn.inputProjB, tensor.SingletonExtent, latent, dim)

	y := make([]float32, dim)
	hostmath.LinearF64(y, cond, fn.condEmbedW, fn.condEmbedB, tensor.SingletonExtent, len(cond), dim)
	ySiLU, timeValue := make([]float32, dim), make([]float32, dim)
	fn.timeEmbeds[0].forwardInto(ySiLU, s, dim, m.Normalization.TimeEmbeddingRMS)
	fn.timeEmbeds[1].forwardInto(timeValue, t, dim, m.Normalization.TimeEmbeddingRMS)
	for j := range y {
		y[j] += (ySiLU[j] + timeValue[j]) / tensor.PairedExtent
		ySiLU[j] = y[j]
	}
	hostmath.SiLUInPlace(ySiLU)

	ada := make([]float32, tensor.TripleExtent*dim)
	hln := make([]float32, dim)
	h1 := make([]float32, dim)
	h2 := make([]float32, dim)
	for bi := range fn.blocks {
		b := &fn.blocks[bi]
		hostmath.LinearF64(ada, ySiLU, b.adaW, b.adaB, tensor.SingletonExtent, dim, tensor.TripleExtent*dim)
		shift, scale, gate := ada[:dim], ada[dim:tensor.PairedExtent*dim], ada[tensor.PairedExtent*dim:]
		hostmath.LayerNormF32UnfusedInto(hln, cur, b.inLnW, b.inLnB, tensor.SingletonExtent, dim, m.Normalization.FlowLayer)
		for i := range hln {
			hln[i] = hln[i]*(tensor.SingletonExtent+scale[i]) + shift[i]
		}
		hostmath.LinearF64(h1, hln, b.mlp0w, b.mlp0b, tensor.SingletonExtent, dim, dim)
		hostmath.SiLUInPlace(h1)
		hostmath.LinearF64(h2, h1, b.mlp2w, b.mlp2b, tensor.SingletonExtent, dim, dim)
		for i := range cur {
			cur[i] += gate[i] * h2[i]
		}
	}

	adaF := ada[:tensor.PairedExtent*dim]
	hostmath.LinearF64(adaF, ySiLU, fn.finalAdaW, fn.finalAdaB, tensor.SingletonExtent, dim, tensor.PairedExtent*dim)
	shift, scale := adaF[:dim], adaF[dim:]
	nf := hln
	hostmath.LayerNormF32UnfusedInto(nf, cur, nil, nil, tensor.SingletonExtent, dim, m.Normalization.FlowLayer)
	for i := range nf {
		nf[i] = nf[i]*(tensor.SingletonExtent+scale[i]) + shift[i]
	}
	hostmath.LinearF64(out, nf, fn.finalLinW, fn.finalLinB, tensor.SingletonExtent, dim, latent)
}

// OneStepLatentInto: the one-step decode x0 + F(cond, 0, 1, x0).
func (m *Model) OneStepLatentInto(out, cond, noise []float32) {
	m.flowForwardInto(out, cond, tensor.FirstOffset, tensor.SingletonExtent, noise)
	for i := range out {
		out[i] += noise[i]
	}
}
