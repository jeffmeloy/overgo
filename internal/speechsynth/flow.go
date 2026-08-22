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
	hostmath.Linear(h0, emb, te.l0w, tensor.SingletonExtent, tensor.PairedExtent*half, flowDim)
	hostmath.AddBias(h0, te.l0b)
	hostmath.SiLUInPlace(h0)
	h := make([]float32, flowDim)
	hostmath.Linear(h, h0, te.l2w, tensor.SingletonExtent, flowDim, flowDim)
	hostmath.AddBias(h, te.l2b)
	var mean float64
	for _, v := range h {
		mean += float64(v)
	}
	mean /= float64(flowDim)
	var varsum float64
	for _, v := range h {
		delta := float64(v) - mean
		varsum += delta * delta
	}
	inv := float64(tensor.SingletonExtent) / math.Sqrt(rmsEpsilon+varsum/float64(flowDim-tensor.SingletonExtent))
	for i := range out {
		out[i] = float32(float64(h[i]) * float64(te.alpha[i]) * inv)
	}
}

// FlowForward computes F(cond, s, t, x): cond [d], x [latent] -> [latent].
func (m *Model) FlowForward(cond []float32, s, t float64, x []float32) []float32 {
	out := make([]float32, m.Dims.LatentDim)
	m.flowForwardInto(out, cond, s, t, x)
	return out
}

func (m *Model) flowForwardInto(out, cond []float32, s, t float64, x []float32) {
	fn := &m.flow
	dim, latent := m.Dims.FlowDim, m.Dims.LatentDim

	cur := make([]float32, dim)
	hostmath.Linear(cur, x, fn.inputProjW, tensor.SingletonExtent, latent, dim)
	hostmath.AddBias(cur, fn.inputProjB)

	y := make([]float32, dim)
	hostmath.Linear(y, cond, fn.condEmbedW, tensor.SingletonExtent, len(cond), dim)
	hostmath.AddBias(y, fn.condEmbedB)
	timeValue := make([]float32, dim)
	for i, time := range [...]float64{s, t} {
		fn.timeEmbeds[i].forwardInto(timeValue, time, dim, m.Normalization.TimeEmbeddingRMS)
		for j := range y {
			y[j] += timeValue[j] / tensor.PairedExtent
		}
	}
	ySiLU := append([]float32(nil), y...)
	hostmath.SiLUInPlace(ySiLU)

	ada := make([]float32, tensor.TripleExtent*dim)
	hln := make([]float32, dim)
	h1 := make([]float32, dim)
	h2 := make([]float32, dim)
	for bi := range fn.blocks {
		b := &fn.blocks[bi]
		hostmath.Linear(ada, ySiLU, b.adaW, tensor.SingletonExtent, dim, tensor.TripleExtent*dim)
		hostmath.AddBias(ada, b.adaB)
		shift, scale, gate := ada[:dim], ada[dim:tensor.PairedExtent*dim], ada[tensor.PairedExtent*dim:]
		hostmath.LayerNormInto(hln, cur, b.inLnW, b.inLnB, tensor.SingletonExtent, dim, m.Normalization.FlowLayer)
		for i := range hln {
			hln[i] = hln[i]*(tensor.SingletonExtent+scale[i]) + shift[i]
		}
		hostmath.Linear(h1, hln, b.mlp0w, tensor.SingletonExtent, dim, dim)
		hostmath.AddBias(h1, b.mlp0b)
		hostmath.SiLUInPlace(h1)
		hostmath.Linear(h2, h1, b.mlp2w, tensor.SingletonExtent, dim, dim)
		hostmath.AddBias(h2, b.mlp2b)
		for i := range cur {
			cur[i] += gate[i] * h2[i]
		}
	}

	adaF := ada[:tensor.PairedExtent*dim]
	hostmath.Linear(adaF, ySiLU, fn.finalAdaW, tensor.SingletonExtent, dim, tensor.PairedExtent*dim)
	hostmath.AddBias(adaF, fn.finalAdaB)
	shift, scale := adaF[:dim], adaF[dim:]
	nf := hln
	hostmath.LayerNormInto(nf, cur, nil, nil, tensor.SingletonExtent, dim, m.Normalization.FlowLayer)
	for i := range nf {
		nf[i] = nf[i]*(tensor.SingletonExtent+scale[i]) + shift[i]
	}
	hostmath.Linear(out, nf, fn.finalLinW, tensor.SingletonExtent, dim, latent)
	hostmath.AddBias(out, fn.finalLinB)
}

// OneStepLatentInto: the one-step decode x0 + F(cond, 0, 1, x0).
func (m *Model) OneStepLatentInto(out, cond, noise []float32) {
	m.flowForwardInto(out, cond, tensor.FirstOffset, tensor.SingletonExtent, noise)
	for i := range out {
		out[i] += noise[i]
	}
}
