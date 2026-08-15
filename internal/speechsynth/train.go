// Training leg for the speech-flow capability (reference:
// adaptive FineTuneAudioFlowClips): joint fine-tune of the flow head AND the
// conditioner backbone on the flow-matching latent-L2 objective — the
// one-step decode anchor loss = mean((x0 + F(cond,0,1,x0) - z)^2) over
// teacher-forced frames. Trained set = all flow_net tensors + backbone
// attn/mlp matrices + norms + out_norm; conditioner embed table,
// input_linear, EOS head, bos/emb stats, and the codec stay FROZEN.
// Backward is recompute-based in checkpoint posture (densecausal pattern):
// only per-layer residual-stream inputs are retained; traces recompute.
package speechsynth

import (
	"fmt"
	"math"
	"math/rand"

	"overgo/internal/hostmath"
)

// Grads accumulates named weight gradients, allocated on first touch.
type Grads map[string][]float32

// forwardStates: full-sequence teacher-forced backbone pass. states[i] is
// layer i's input residual stream [T*d]; states[Layers] is the final stream
// (pre-out_norm). Per-row math is identical to AppendForward (the full
// CausalAttention row equals CausalAttentionStep at that position).
func (m *Model) forwardStates(stream []float32, T int) [][]float32 {
	states := make([][]float32, len(m.layers)+1)
	states[0] = append([]float32(nil), stream...)
	for li := range m.layers {
		out, _ := m.layerForwardTrace(&m.layers[li], states[li], T)
		states[li+1] = out
	}
	return states
}

// layerTrace: recomputable layer intermediates. q is post-RoPE AND
// score-scaled (hostmath attention runs scale 1); k is post-RoPE.
type layerTrace struct {
	xn1, q, k, v, attn, mid, xn2, h1pre, h1post []float32
}

// layerForwardTrace recomputes one layer from its input x [T*d].
func (m *Model) layerForwardTrace(l *attnLayer, x []float32, T int) ([]float32, layerTrace) {
	d, heads, hd, ff := m.Dims.DModel, m.Dims.Heads, m.Dims.HeadDim, m.Dims.FF
	var tr layerTrace
	tr.xn1 = make([]float32, T*d)
	hostmath.LayerNormInto(tr.xn1, x, l.norm1W, l.norm1B, T, d, transformerLayerNormEps)
	qkv := make([]float32, T*3*d)
	hostmath.Linear(qkv, tr.xn1, l.inProj, T, d, 3*d)
	tr.q, tr.k, tr.v = make([]float32, T*d), make([]float32, T*d), make([]float32, T*d)
	for t := 0; t < T; t++ {
		row := qkv[t*3*d : (t+1)*3*d]
		copy(tr.q[t*d:(t+1)*d], row[:d])
		copy(tr.k[t*d:(t+1)*d], row[d:2*d])
		copy(tr.v[t*d:(t+1)*d], row[2*d:])
	}
	for t := 0; t < T; t++ {
		for h := 0; h < heads; h++ {
			hostmath.ApplyRotaryInterleaved(tr.q[(t*heads+h)*hd:(t*heads+h+1)*hd], m.invFreq, t)
			hostmath.ApplyRotaryInterleaved(tr.k[(t*heads+h)*hd:(t*heads+h+1)*hd], m.invFreq, t)
		}
	}
	for i := range tr.q {
		tr.q[i] *= m.scoreScale
	}
	tr.attn = make([]float32, T*d)
	hostmath.CausalAttention(tr.attn, tr.q, tr.k, tr.v, T, heads, heads, hd)
	attnOut := make([]float32, T*d)
	hostmath.Linear(attnOut, tr.attn, l.outProj, T, d, d)
	tr.mid = make([]float32, T*d)
	for i := range tr.mid {
		tr.mid[i] = x[i] + attnOut[i]
	}
	tr.xn2 = make([]float32, T*d)
	hostmath.LayerNormInto(tr.xn2, tr.mid, l.norm2W, l.norm2B, T, d, transformerLayerNormEps)
	tr.h1pre = make([]float32, T*ff)
	hostmath.Linear(tr.h1pre, tr.xn2, l.lin1, T, d, ff)
	tr.h1post = append([]float32(nil), tr.h1pre...)
	hostmath.GELUErfInPlace(tr.h1post)
	h2 := make([]float32, T*d)
	hostmath.Linear(h2, tr.h1post, l.lin2, T, ff, d)
	out := make([]float32, T*d)
	for i := range out {
		out[i] = tr.mid[i] + h2[i]
	}
	return out, tr
}

// layerBackward: VJP of one backbone layer. x is the retained layer input;
// dOut arrives at the layer output; returns dx at the input. Gradients key
// by the checkpoint tensor names.
func (m *Model) layerBackward(li int, x, dOut []float32, T int, g Grads) []float32 {
	l := &m.layers[li]
	_, tr := m.layerForwardTrace(l, x, T)
	d, heads, hd, ff := m.Dims.DModel, m.Dims.Heads, m.Dims.HeadDim, m.Dims.FF
	prefix := fmt.Sprintf("%s%d.", layerPrefix, li)

	// FFN residual: out = mid + lin2(gelu(lin1(LN2(mid)))).
	dMid := append([]float32(nil), dOut...)
	dH1 := make([]float32, T*ff)
	hostmath.LinearBackward(dH1, hostmath.GradientSlot(g, prefix+"linear2.weight", d*ff), nil, tr.h1post, l.lin2, dOut, T, ff, d, false)
	hostmath.GELUErfBackward(dH1, tr.h1pre, dH1)
	dXn2 := make([]float32, T*d)
	hostmath.LinearBackward(dXn2, hostmath.GradientSlot(g, prefix+"linear1.weight", ff*d), nil, tr.xn2, l.lin1, dH1, T, d, ff, false)
	hostmath.LayerNormBackward(dMid, hostmath.GradientSlot(g, prefix+"norm2.weight", d), hostmath.GradientSlot(g, prefix+"norm2.bias", d), tr.mid, l.norm2W, dXn2, T, d, transformerLayerNormEps, true)

	// Attention residual: mid = x + outProj(attn(rope(qkv(LN1(x))))).
	dAttn := make([]float32, T*d)
	hostmath.LinearBackward(dAttn, hostmath.GradientSlot(g, prefix+"self_attn.out_proj.weight", d*d), nil, tr.attn, l.outProj, dMid, T, d, d, false)
	dq := make([]float32, T*d)
	dk := make([]float32, T*d)
	dv := make([]float32, T*d)
	hostmath.CausalAttentionBackward(dq, dk, dv, tr.q, tr.k, tr.v, dAttn, T, heads, heads, hd)
	for i := range dq { // chain through the folded score scale
		dq[i] *= m.scoreScale
	}
	for t := 0; t < T; t++ {
		for h := 0; h < heads; h++ {
			hostmath.RotaryInterleavedBackward(dq[(t*heads+h)*hd:(t*heads+h+1)*hd], m.invFreq, t)
			hostmath.RotaryInterleavedBackward(dk[(t*heads+h)*hd:(t*heads+h+1)*hd], m.invFreq, t)
		}
	}
	dqkv := make([]float32, T*3*d)
	for t := 0; t < T; t++ {
		row := dqkv[t*3*d : (t+1)*3*d]
		copy(row[:d], dq[t*d:(t+1)*d])
		copy(row[d:2*d], dk[t*d:(t+1)*d])
		copy(row[2*d:], dv[t*d:(t+1)*d])
	}
	dXn1 := dXn2 // fully overwritten below
	hostmath.LinearBackward(dXn1, hostmath.GradientSlot(g, prefix+"self_attn.in_proj.weight", 3*d*d), nil, tr.xn1, l.inProj, dqkv, T, d, 3*d, false)
	dx := dMid // residual skip: dx = dMid + LN1 path
	hostmath.LayerNormBackward(dx, hostmath.GradientSlot(g, prefix+"norm1.weight", d), hostmath.GradientSlot(g, prefix+"norm1.bias", d), x, l.norm1W, dXn1, T, d, transformerLayerNormEps, true)
	return dx
}

// flowTrace: recomputed flow-head intermediates for one (cond, s, t, x).
type flowTrace struct {
	y, ySiLU      []float32
	te            [2]timeEmbedTrace
	cur           [][]float32 // block input streams; cur[depth] feeds the final layer
	blocks        []flowBlockTrace
	nf, adaF, nfm []float32
}

type timeEmbedTrace struct {
	emb, h0pre, h0post, h []float32
	mean, inv             float64
}

type flowBlockTrace struct {
	ada, hln, hmod, h1pre, h1post, h2 []float32
}

// forwardTrace mirrors timeEmbed.forwardInto, retaining intermediates.
func (te *timeEmbed) forwardTrace(out []float32, t float64, flowDim int) timeEmbedTrace {
	half := len(te.freqs)
	tr := timeEmbedTrace{
		emb: make([]float32, 2*half), h0pre: make([]float32, flowDim),
		h0post: make([]float32, flowDim), h: make([]float32, flowDim),
	}
	for i, f := range te.freqs {
		arg := t * float64(f)
		tr.emb[i] = float32(math.Cos(arg))
		tr.emb[half+i] = float32(math.Sin(arg))
	}
	hostmath.Linear(tr.h0pre, tr.emb, te.l0w, 1, 2*half, flowDim)
	hostmath.AddBias(tr.h0pre, te.l0b)
	copy(tr.h0post, tr.h0pre)
	hostmath.SiLUInPlace(tr.h0post)
	hostmath.Linear(tr.h, tr.h0post, te.l2w, 1, flowDim, flowDim)
	hostmath.AddBias(tr.h, te.l2b)
	var mean float64
	for _, v := range tr.h {
		mean += float64(v)
	}
	mean /= float64(flowDim)
	var varsum float64
	for _, v := range tr.h {
		delta := float64(v) - mean
		varsum += delta * delta
	}
	tr.mean = mean
	tr.inv = 1 / math.Sqrt(timeEmbedRMSEps+varsum/float64(flowDim-1))
	for i := range out {
		out[i] = float32(float64(tr.h[i]) * float64(te.alpha[i]) * tr.inv)
	}
	return tr
}

// backward: VJP of the timeEmbed tail — y_i = h_i*alpha_i*inv with inv from
// the UNBIASED mean-subtracted variance of h (the vendor quirk). emb is
// constant w.r.t. parameters, so l0 takes parameter grads only.
func (te *timeEmbed) backward(tr *timeEmbedTrace, dY []float32, flowDim int, g Grads, prefix string) {
	alpha := hostmath.GradientSlot(g, prefix+"mlp.3.alpha", flowDim)
	var dInv float64
	for i := range dY {
		alpha[i] += float32(float64(dY[i]) * float64(tr.h[i]) * tr.inv)
		dInv += float64(dY[i]) * float64(te.alpha[i]) * float64(tr.h[i])
	}
	dVar := dInv * (-0.5) * tr.inv * tr.inv * tr.inv
	dH := make([]float32, flowDim)
	for i := range dH {
		dH[i] = float32(float64(dY[i])*float64(te.alpha[i])*tr.inv +
			dVar*2*(float64(tr.h[i])-tr.mean)/float64(flowDim-1))
	}
	dH0 := make([]float32, flowDim)
	hostmath.LinearBackward(dH0, hostmath.GradientSlot(g, prefix+"mlp.2.weight", flowDim*flowDim), hostmath.GradientSlot(g, prefix+"mlp.2.bias", flowDim), tr.h0post, te.l2w, dH, 1, flowDim, flowDim, false)
	hostmath.SiLUBackward(dH0, tr.h0pre, dH0)
	hostmath.LinearBackward(nil, hostmath.GradientSlot(g, prefix+"mlp.0.weight", flowDim*2*len(te.freqs)), hostmath.GradientSlot(g, prefix+"mlp.0.bias", flowDim), tr.emb, te.l0w, dH0, 1, 2*len(te.freqs), flowDim, false)
}

// flowForwardTrace mirrors flowForwardInto, retaining every intermediate the
// VJP consumes. Same operation sequence, so outputs match the serving path.
func (m *Model) flowForwardTrace(out, cond []float32, s, t float64, x []float32) *flowTrace {
	fn := &m.flow
	dim, latent := m.Dims.FlowDim, m.Dims.LatentDim
	tr := &flowTrace{cur: make([][]float32, len(fn.blocks)+1), blocks: make([]flowBlockTrace, len(fn.blocks))}

	cur := make([]float32, dim)
	hostmath.Linear(cur, x, fn.inputProjW, 1, latent, dim)
	hostmath.AddBias(cur, fn.inputProjB)
	tr.cur[0] = cur

	tr.y = make([]float32, dim)
	hostmath.Linear(tr.y, cond, fn.condEmbedW, 1, len(cond), dim)
	hostmath.AddBias(tr.y, fn.condEmbedB)
	timeValue := make([]float32, dim)
	for i, tv := range [...]float64{s, t} {
		tr.te[i] = fn.timeEmbeds[i].forwardTrace(timeValue, tv, dim)
		for j := range tr.y {
			tr.y[j] += timeValue[j] / 2
		}
	}
	tr.ySiLU = append([]float32(nil), tr.y...)
	hostmath.SiLUInPlace(tr.ySiLU)

	for bi := range fn.blocks {
		b := &fn.blocks[bi]
		bt := &tr.blocks[bi]
		bt.ada = make([]float32, 3*dim)
		hostmath.Linear(bt.ada, tr.ySiLU, b.adaW, 1, dim, 3*dim)
		hostmath.AddBias(bt.ada, b.adaB)
		shift, scale, gate := bt.ada[:dim], bt.ada[dim:2*dim], bt.ada[2*dim:]
		bt.hln = make([]float32, dim)
		hostmath.LayerNormInto(bt.hln, cur, b.inLnW, b.inLnB, 1, dim, flowLayerNormEps)
		bt.hmod = make([]float32, dim)
		for i := range bt.hmod {
			bt.hmod[i] = bt.hln[i]*(1+scale[i]) + shift[i]
		}
		bt.h1pre = make([]float32, dim)
		hostmath.Linear(bt.h1pre, bt.hmod, b.mlp0w, 1, dim, dim)
		hostmath.AddBias(bt.h1pre, b.mlp0b)
		bt.h1post = append([]float32(nil), bt.h1pre...)
		hostmath.SiLUInPlace(bt.h1post)
		bt.h2 = make([]float32, dim)
		hostmath.Linear(bt.h2, bt.h1post, b.mlp2w, 1, dim, dim)
		hostmath.AddBias(bt.h2, b.mlp2b)
		next := make([]float32, dim)
		for i := range next {
			next[i] = cur[i] + gate[i]*bt.h2[i]
		}
		tr.cur[bi+1] = next
		cur = next
	}

	tr.adaF = make([]float32, 2*dim)
	hostmath.Linear(tr.adaF, tr.ySiLU, fn.finalAdaW, 1, dim, 2*dim)
	hostmath.AddBias(tr.adaF, fn.finalAdaB)
	tr.nf = make([]float32, dim)
	hostmath.LayerNormInto(tr.nf, cur, nil, nil, 1, dim, flowLayerNormEps)
	shift, scale := tr.adaF[:dim], tr.adaF[dim:]
	tr.nfm = make([]float32, dim)
	for i := range tr.nfm {
		tr.nfm[i] = tr.nf[i]*(1+scale[i]) + shift[i]
	}
	hostmath.Linear(out, tr.nfm, fn.finalLinW, 1, dim, latent)
	hostmath.AddBias(out, fn.finalLinB)
	return tr
}

// flowBackward: VJP of flowForwardTrace. Accumulates every flow_net
// parameter gradient into g; dX and dCond are nil-able input grads (set).
func (m *Model) flowBackward(tr *flowTrace, cond, x, dOut []float32, g Grads, dX, dCond []float32) {
	fn := &m.flow
	dim, latent := m.Dims.FlowDim, m.Dims.LatentDim
	const p = "flow_lm.flow_net."

	dNfm := make([]float32, dim)
	hostmath.LinearBackward(dNfm, hostmath.GradientSlot(g, p+"final_layer.linear.weight", latent*dim), hostmath.GradientSlot(g, p+"final_layer.linear.bias", latent), tr.nfm, fn.finalLinW, dOut, 1, dim, latent, false)
	scaleF := tr.adaF[dim:]
	dNf := make([]float32, dim)
	dAdaF := make([]float32, 2*dim)
	for i := 0; i < dim; i++ {
		dNf[i] = dNfm[i] * (1 + scaleF[i])
		dAdaF[i] = dNfm[i]                // dShift
		dAdaF[dim+i] = dNfm[i] * tr.nf[i] // dScale
	}
	dYSiLU := make([]float32, dim)
	hostmath.LinearBackward(dYSiLU, hostmath.GradientSlot(g, p+"final_layer.adaLN_modulation.1.weight", 2*dim*dim), hostmath.GradientSlot(g, p+"final_layer.adaLN_modulation.1.bias", 2*dim), tr.ySiLU, fn.finalAdaW, dAdaF, 1, dim, 2*dim, false)
	dCur := make([]float32, dim)
	hostmath.LayerNormBackward(dCur, nil, nil, tr.cur[len(fn.blocks)], nil, dNf, 1, dim, flowLayerNormEps, false)

	dAda := make([]float32, 3*dim)
	dH2 := make([]float32, dim)
	dH1 := make([]float32, dim)
	dHmod := make([]float32, dim)
	dHln := make([]float32, dim)
	dCurLn := make([]float32, dim)
	for bi := len(fn.blocks) - 1; bi >= 0; bi-- {
		b := &fn.blocks[bi]
		bt := &tr.blocks[bi]
		bp := fmt.Sprintf("%s%d.", resBlockPrefix, bi)
		scale, gate := bt.ada[dim:2*dim], bt.ada[2*dim:]
		for i := 0; i < dim; i++ {
			dH2[i] = dCur[i] * gate[i]
			dAda[2*dim+i] = dCur[i] * bt.h2[i] // dGate
		}
		hostmath.LinearBackward(dH1, hostmath.GradientSlot(g, bp+"mlp.2.weight", dim*dim), hostmath.GradientSlot(g, bp+"mlp.2.bias", dim), bt.h1post, b.mlp2w, dH2, 1, dim, dim, false)
		hostmath.SiLUBackward(dH1, bt.h1pre, dH1)
		hostmath.LinearBackward(dHmod, hostmath.GradientSlot(g, bp+"mlp.0.weight", dim*dim), hostmath.GradientSlot(g, bp+"mlp.0.bias", dim), bt.hmod, b.mlp0w, dH1, 1, dim, dim, false)
		for i := 0; i < dim; i++ {
			dHln[i] = dHmod[i] * (1 + scale[i])
			dAda[i] = dHmod[i]                 // dShift
			dAda[dim+i] = dHmod[i] * bt.hln[i] // dScale
		}
		hostmath.LinearBackward(dYSiLU, hostmath.GradientSlot(g, bp+"adaLN_modulation.1.weight", 3*dim*dim), hostmath.GradientSlot(g, bp+"adaLN_modulation.1.bias", 3*dim), tr.ySiLU, b.adaW, dAda, 1, dim, 3*dim, true)
		hostmath.LayerNormBackward(dCurLn, hostmath.GradientSlot(g, bp+"in_ln.weight", dim), hostmath.GradientSlot(g, bp+"in_ln.bias", dim), tr.cur[bi], b.inLnW, dHln, 1, dim, flowLayerNormEps, false)
		for i := range dCur { // block input feeds residual and LN
			dCur[i] += dCurLn[i]
		}
	}

	hostmath.LinearBackward(dX, hostmath.GradientSlot(g, p+"input_proj.weight", dim*latent), hostmath.GradientSlot(g, p+"input_proj.bias", dim), x, fn.inputProjW, dCur, 1, latent, dim, false)
	dY := make([]float32, dim)
	hostmath.SiLUBackward(dY, tr.y, dYSiLU)
	dTE := make([]float32, dim)
	for i := range dY { // both embedders contribute timeValue/2
		dTE[i] = dY[i] / 2
	}
	fn.timeEmbeds[0].backward(&tr.te[0], dTE, dim, g, p+"time_embed.0.")
	fn.timeEmbeds[1].backward(&tr.te[1], dTE, dim, g, p+"time_embed.1.")
	hostmath.LinearBackward(dCond, hostmath.GradientSlot(g, p+"cond_embed.weight", dim*len(cond)), hostmath.GradientSlot(g, p+"cond_embed.bias", dim), cond, fn.condEmbedW, dY, 1, len(cond), dim, false)
}

// LossAndGrads: joint teacher-forced fine-tune step math. Stream =
// [textEmb(ids); input_linear(bos); input_linear(z_0..z_{F-2})]; per frame
// the one-step latent-L2 anchor with standard-normal x0 from noiseSeed:
// loss = sum((x0 + F(cond_t,0,1,x0) - z_t)^2) / (latent*frames).
// g nil = loss only (heldout posture). Prompt rows carry zero cond gradient.
func (m *Model) LossAndGrads(textIDs []int, z []float32, frames int, noiseSeed int64, g Grads) (float64, error) {
	d, latent := m.Dims.DModel, m.Dims.LatentDim
	if frames <= 0 || len(z) != frames*latent {
		return 0, fmt.Errorf("speechsynth: latent batch %d values != %d frames x %d", len(z), frames, latent)
	}
	if len(textIDs) == 0 {
		return 0, fmt.Errorf("speechsynth: training needs a text prompt")
	}
	rng := rand.New(rand.NewSource(noiseSeed))
	T := len(textIDs) + frames
	stream := make([]float32, T*d)
	if err := m.TextEmbedInto(stream, textIDs); err != nil {
		return 0, err
	}
	m.LatentInputInto(stream[len(textIDs)*d:(len(textIDs)+1)*d], m.BosEmb)
	for t := 1; t < frames; t++ {
		m.LatentInputInto(stream[(len(textIDs)+t)*d:(len(textIDs)+t+1)*d], z[(t-1)*latent:t*latent])
	}
	states := m.forwardStates(stream, T)
	final := states[len(m.layers)]

	cond := make([]float32, d)
	out := make([]float32, latent)
	x0 := make([]float32, latent)
	dOut := make([]float32, latent)
	var dCondFull []float32
	if g != nil {
		dCondFull = make([]float32, T*d)
	}
	var loss float64
	norm := float64(latent * frames)
	for t := 0; t < frames; t++ {
		pos := len(textIDs) + t
		m.OutNormInto(cond, final[pos*d:(pos+1)*d], 1)
		for i := range x0 {
			x0[i] = float32(rng.NormFloat64())
		}
		zt := z[t*latent : (t+1)*latent]
		var tr *flowTrace
		if g != nil {
			tr = m.flowForwardTrace(out, cond, 0, 1, x0)
		} else {
			m.flowForwardInto(out, cond, 0, 1, x0)
		}
		for i := range out {
			diff := float64(x0[i]) + float64(out[i]) - float64(zt[i])
			loss += diff * diff
			dOut[i] = float32(2 * diff / norm)
		}
		if g != nil {
			m.flowBackward(tr, cond, x0, dOut, g, nil, dCondFull[pos*d:(pos+1)*d])
		}
	}
	loss /= norm
	if g == nil {
		return loss, nil
	}
	// out_norm VJP over the full stream, then the backbone walk.
	dStream := make([]float32, T*d)
	hostmath.LayerNormBackward(dStream, hostmath.GradientSlot(g, "flow_lm.out_norm.weight", d), hostmath.GradientSlot(g, "flow_lm.out_norm.bias", d), final, m.outNormW, dCondFull, T, d, transformerLayerNormEps, false)
	for li := len(m.layers) - 1; li >= 0; li-- {
		dStream = m.layerBackward(li, states[li], dStream, T, g)
	}
	// Stream inputs (embed table, input_linear) are frozen: dStream discarded.
	return loss, nil
}

// TrainedTensors: the joint trained set as live weight views plus Muon
// matrix shapes ([n,1] for vectors), mirroring the reference
// audioFlowTrainingTensorNames. includeBackbone=false is the flow-only set
// (the lrJoint numerator).
func (m *Model) TrainedTensors(includeBackbone bool) (map[string][]float32, map[string][2]int) {
	dim, latent, d, ff := m.Dims.FlowDim, m.Dims.LatentDim, m.Dims.DModel, m.Dims.FF
	tensors := map[string][]float32{}
	shapes := map[string][2]int{}
	add := func(name string, values []float32, rows, cols int) {
		tensors[name] = values
		shapes[name] = [2]int{rows, cols}
	}
	fn := &m.flow
	const p = "flow_lm.flow_net."
	add(p+"input_proj.weight", fn.inputProjW, dim, latent)
	add(p+"input_proj.bias", fn.inputProjB, dim, 1)
	add(p+"cond_embed.weight", fn.condEmbedW, dim, d)
	add(p+"cond_embed.bias", fn.condEmbedB, dim, 1)
	add(p+"final_layer.adaLN_modulation.1.weight", fn.finalAdaW, 2*dim, dim)
	add(p+"final_layer.adaLN_modulation.1.bias", fn.finalAdaB, 2*dim, 1)
	add(p+"final_layer.linear.weight", fn.finalLinW, latent, dim)
	add(p+"final_layer.linear.bias", fn.finalLinB, latent, 1)
	for i := range fn.timeEmbeds {
		te := &fn.timeEmbeds[i]
		tp := fmt.Sprintf("%stime_embed.%d.", p, i)
		add(tp+"mlp.0.weight", te.l0w, dim, 2*len(te.freqs))
		add(tp+"mlp.0.bias", te.l0b, dim, 1)
		add(tp+"mlp.2.weight", te.l2w, dim, dim)
		add(tp+"mlp.2.bias", te.l2b, dim, 1)
		add(tp+"mlp.3.alpha", te.alpha, dim, 1)
	}
	for i := range fn.blocks {
		b := &fn.blocks[i]
		bp := fmt.Sprintf("%s%d.", resBlockPrefix, i)
		add(bp+"in_ln.weight", b.inLnW, dim, 1)
		add(bp+"in_ln.bias", b.inLnB, dim, 1)
		add(bp+"mlp.0.weight", b.mlp0w, dim, dim)
		add(bp+"mlp.0.bias", b.mlp0b, dim, 1)
		add(bp+"mlp.2.weight", b.mlp2w, dim, dim)
		add(bp+"mlp.2.bias", b.mlp2b, dim, 1)
		add(bp+"adaLN_modulation.1.weight", b.adaW, 3*dim, dim)
		add(bp+"adaLN_modulation.1.bias", b.adaB, 3*dim, 1)
	}
	if includeBackbone {
		add("flow_lm.out_norm.weight", m.outNormW, d, 1)
		add("flow_lm.out_norm.bias", m.outNormB, d, 1)
		for i := range m.layers {
			l := &m.layers[i]
			lp := fmt.Sprintf("%s%d.", layerPrefix, i)
			add(lp+"norm1.weight", l.norm1W, d, 1)
			add(lp+"norm1.bias", l.norm1B, d, 1)
			add(lp+"norm2.weight", l.norm2W, d, 1)
			add(lp+"norm2.bias", l.norm2B, d, 1)
			add(lp+"self_attn.in_proj.weight", l.inProj, 3*d, d)
			add(lp+"self_attn.out_proj.weight", l.outProj, d, d)
			add(lp+"linear1.weight", l.lin1, ff, d)
			add(lp+"linear2.weight", l.lin2, d, ff)
		}
	}
	return tensors, shapes
}

// DerivedJointLR ports the reference scale law: a base rate tuned for the
// flow head alone dilutes by the joint parameter surface,
// lrJoint = baseLR * sqrt(nFlow/nJoint). Counts derive from live tensors.
func (m *Model) DerivedJointLR(baseLR float64) (lrJoint float64, nFlow, nJoint int) {
	count := func(includeBackbone bool) int {
		tensors, _ := m.TrainedTensors(includeBackbone)
		total := 0
		for _, values := range tensors {
			total += len(values)
		}
		return total
	}
	nFlow, nJoint = count(false), count(true)
	return baseLR * math.Sqrt(float64(nFlow)/float64(nJoint)), nFlow, nJoint
}
