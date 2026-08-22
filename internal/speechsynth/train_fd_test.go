// Finite-difference gates for the training leg, mirroring the reference
// protocol (adaptive TestSpeechFlowFlowNetGradCheck /
// TestSpeechFlowBackboneBackwardFiniteDifference): small seeded-random
// synthetic configs — NO artifact and NO mimi encoder; the training-step
// math needs only latents, which are synthetic here exactly as in the
// reference FD tests.
package speechsynth

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/media"
)

// gcStep / gradCheckTol: the reference finite-diff step and tolerance
// contract (fp32 roundoff plus truncation by accumulation depth, safety 50).
const gcStep = 1e-2

func gradCheckTol(step float64, accumDepth int) float64 {
	const epsF32 = 1.1920929e-7
	const safety = 50.0
	return safety * (epsF32*math.Sqrt(float64(accumDepth))/step + step*step/6.0)
}

func normVec(rng *rand.Rand, n int, base, s float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(base + rng.NormFloat64()*s)
	}
	return v
}

func fixtureSpeechProgram() media.NormalizationProgram {
	program, err := loadSpeechProgram()
	if err != nil {
		panic(err)
	}
	return program
}

// tinyFlowModel: synthetic SimpleMLPAdaLN (reference tinyFlowNet geometry).
func tinyFlowModel(rng *rand.Rand, dim, latent, condDim, depth, half int) *Model {
	m := &Model{Normalization: fixtureSpeechProgram()}
	m.Dims.FlowDim, m.Dims.LatentDim, m.Dims.DModel = dim, latent, condDim
	m.Dims.FlowDepth, m.Dims.TimeFreqs = depth, half
	fn := &m.flow
	vec := func(n int) []float32 { return normVec(rng, n, 0, 0.3) }
	fn.inputProjW, fn.inputProjB = vec(dim*latent), vec(dim)
	fn.condEmbedW, fn.condEmbedB = vec(dim*condDim), vec(dim)
	for i := range fn.timeEmbeds {
		te := &fn.timeEmbeds[i]
		te.freqs = vec(half)
		te.l0w, te.l0b = vec(dim*2*half), vec(dim)
		te.l2w, te.l2b = vec(dim*dim), vec(dim)
		te.alpha = vec(dim)
	}
	fn.blocks = make([]flowBlock, depth)
	for i := range fn.blocks {
		b := &fn.blocks[i]
		b.inLnW, b.inLnB = vec(dim), vec(dim)
		b.mlp0w, b.mlp0b = vec(dim*dim), vec(dim)
		b.mlp2w, b.mlp2b = vec(dim*dim), vec(dim)
		b.adaW, b.adaB = vec(3*dim*dim), vec(3*dim)
	}
	fn.finalAdaW, fn.finalAdaB = vec(2*dim*dim), vec(2*dim)
	fn.finalLinW, fn.finalLinB = vec(latent*dim), vec(latent)
	return m
}

// tinyBackboneModel: seeded synthetic conditioner transformer + out_norm
// (reference tinySpeechFlowTransformer geometry, layerScale-free like the
// real backbone).
func tinyBackboneModel(rng *rand.Rand, d, heads, ff, layers int) *Model {
	m := &Model{Normalization: fixtureSpeechProgram()}
	m.Dims.DModel, m.Dims.Heads, m.Dims.HeadDim, m.Dims.FF, m.Dims.Layers = d, heads, d/heads, ff, layers
	m.Dims.MaxPeriod = 10000
	m.invFreq = hostmath.RopeInvFreq(m.Dims.MaxPeriod, m.Dims.HeadDim)
	m.scoreScale = float32(1 / math.Sqrt(float64(m.Dims.HeadDim)))
	m.layers = make([]attnLayer, layers)
	for i := range m.layers {
		l := &m.layers[i]
		l.norm1W, l.norm1B = normVec(rng, d, 1, 0.05), normVec(rng, d, 0, 0.05)
		l.norm2W, l.norm2B = normVec(rng, d, 1, 0.05), normVec(rng, d, 0, 0.05)
		l.inProj, l.outProj = normVec(rng, 3*d*d, 0, 0.3), normVec(rng, d*d, 0, 0.3)
		l.lin1, l.lin2 = normVec(rng, ff*d, 0, 0.3), normVec(rng, d*ff, 0, 0.3)
	}
	m.outNormW = normVec(rng, d, 1, 0.05)
	m.outNormB = normVec(rng, d, 0, 0.05)
	return m
}

// TestFlowNetGradCheck mirrors the reference flow-net FD gate: every trained
// tensor class plus dX and dCond on the one-step latent-L2 objective
// loss = mean((x + F(cond,0,1,x) - z)^2).
func TestFlowNetGradCheck(t *testing.T) {
	const dim, latent, condDim, depth, half = 10, 4, 6, 2, 3
	rng := rand.New(rand.NewSource(7))
	m := tinyFlowModel(rng, dim, latent, condDim, depth, half)
	fn := &m.flow
	x := normVec(rng, latent, 0, 1)
	z := normVec(rng, latent, 0, 1)
	cond := normVec(rng, condDim, 0, 1)

	loss := func() float64 {
		f := m.FlowForward(cond, 0, 1, x)
		var l float64
		for i := range f {
			d := float64(x[i]) + float64(f[i]) - float64(z[i])
			l += d * d
		}
		return l / float64(latent)
	}

	out := make([]float32, latent)
	tr := m.flowForwardTrace(out, cond, 0, 1, x)
	ref := m.FlowForward(cond, 0, 1, x)
	for i := range ref {
		if out[i] != ref[i] {
			t.Fatalf("trace forward[%d]=%g != FlowForward %g", i, out[i], ref[i])
		}
	}
	dOut := make([]float32, latent)
	for i := range dOut {
		dOut[i] = float32(2 * (float64(x[i]) + float64(out[i]) - float64(z[i])) / float64(latent))
	}
	g := Grads{}
	dX := make([]float32, latent)
	dCond := make([]float32, condDim)
	m.flowBackward(tr, cond, x, dOut, g, dX, dCond)
	for i := range dX { // dLoss/dx has BOTH the identity path and the flow path
		dX[i] += dOut[i]
	}

	tol := gradCheckTol(gcStep, dim*depth)
	worstRel := 0.0
	defer func() { t.Logf("flow-net grad check worst rel %.3e (gate %.1e)", worstRel, tol) }()
	checkVec := func(name string, vec, grad []float32, idxs []int) {
		t.Helper()
		for _, i := range idxs {
			orig := vec[i]
			vec[i] = orig + gcStep
			lp := loss()
			vec[i] = orig - gcStep
			lm := loss()
			vec[i] = orig
			fd := (lp - lm) / (2 * gcStep)
			an := float64(grad[i])
			if r := math.Abs(fd-an) / (1 + math.Abs(fd)); r > worstRel {
				worstRel = r
			}
			if math.Abs(fd-an) > tol*(1+math.Abs(fd)) {
				t.Fatalf("%s[%d]: fd %.6g vs analytic %.6g", name, i, fd, an)
			}
		}
	}
	const p = "flow_lm.flow_net."
	gr := func(n string) []float32 { return g[p+n] }
	checkVec("input_proj.weight", fn.inputProjW, gr("input_proj.weight"), []int{0, dim*latent - 1, 7})
	checkVec("input_proj.bias", fn.inputProjB, gr("input_proj.bias"), []int{0, dim - 1})
	checkVec("cond_embed.weight", fn.condEmbedW, gr("cond_embed.weight"), []int{1, dim*condDim - 2})
	checkVec("cond_embed.bias", fn.condEmbedB, gr("cond_embed.bias"), []int{0, dim - 1})
	checkVec("time_embed.0.mlp.0.weight", fn.timeEmbeds[0].l0w, gr("time_embed.0.mlp.0.weight"), []int{0, 5})
	checkVec("time_embed.1.mlp.2.weight", fn.timeEmbeds[1].l2w, gr("time_embed.1.mlp.2.weight"), []int{3, dim*dim - 1})
	checkVec("time_embed.0.mlp.3.alpha", fn.timeEmbeds[0].alpha, gr("time_embed.0.mlp.3.alpha"), []int{0, dim - 1})
	checkVec("res_blocks.0.in_ln.weight", fn.blocks[0].inLnW, gr("res_blocks.0.in_ln.weight"), []int{0, dim - 1})
	checkVec("res_blocks.0.mlp.0.weight", fn.blocks[0].mlp0w, gr("res_blocks.0.mlp.0.weight"), []int{2, dim*dim - 3})
	checkVec("res_blocks.1.mlp.2.bias", fn.blocks[1].mlp2b, gr("res_blocks.1.mlp.2.bias"), []int{1})
	checkVec("res_blocks.1.adaLN_modulation.1.weight", fn.blocks[1].adaW, gr("res_blocks.1.adaLN_modulation.1.weight"), []int{0, 3*dim*dim - 1, dim * dim})
	checkVec("final_layer.adaLN_modulation.1.bias", fn.finalAdaB, gr("final_layer.adaLN_modulation.1.bias"), []int{0, 2*dim - 1})
	checkVec("final_layer.linear.weight", fn.finalLinW, gr("final_layer.linear.weight"), []int{0, latent*dim - 1})
	checkVec("x", x, dX, []int{0, latent - 1})
	checkVec("cond", cond, dCond, []int{0, condDim - 1})
}

// TestBackboneBackwardFiniteDifference mirrors the reference backbone gate:
// gradients through L = sum(OutNorm(backbone(stream)) * target) for every
// transformer + out_norm weight and the input stream must reproduce a
// central finite difference within the reference-pinned bound.
func TestBackboneBackwardFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const d, heads, ff, layers, T = 8, 2, 16, 2, 5
	m := tinyBackboneModel(rng, d, heads, ff, layers)
	stream := normVec(rng, T*d, 0, 0.5)
	target := normVec(rng, T*d, 0, 1)

	loss := func() float64 {
		final := m.forwardStates(stream, T)[layers]
		cond := make([]float32, T*d)
		m.OutNormInto(cond, final, T)
		var s float64
		for i := range cond {
			s += float64(cond[i]) * float64(target[i])
		}
		return s
	}

	// Batched forward must agree with the incremental serving decode.
	states := m.forwardStates(stream, T)
	incremental := append([]float32(nil), stream...)
	m.AppendForward(m.NewDecodeState(T), incremental, T)
	for i := range incremental {
		if diff := math.Abs(float64(states[layers][i]) - float64(incremental[i])); diff > 1e-6 {
			t.Fatalf("batched forward diverges from AppendForward at %d: %g", i, diff)
		}
	}

	g := Grads{}
	dStream := make([]float32, T*d)
	hostmath.LayerNormBackward(dStream, hostmath.GradientSlot(g, "flow_lm.out_norm.weight", d), hostmath.GradientSlot(g, "flow_lm.out_norm.bias", d), states[layers], m.outNormW, target, T, d, m.Normalization.TransformerLayer, false)
	for li := layers - 1; li >= 0; li-- {
		dStream = m.layerBackward(li, states[li], dStream, T, g)
	}

	// Backbone FD step: 1e-2 (the reference step) hits O(h^2) truncation
	// ~3e-3 on this seed's GELU curvature (verified by a step sweep
	// converging to the analytic value); 3e-3 sits at the fp32
	// roundoff/truncation balance for this loss scale.
	const backboneStep = 3e-3
	worst, worstName := 0.0, ""
	check := func(name string, w, dW []float32) {
		t.Helper()
		if dW == nil {
			t.Fatalf("%s: nil analytic grad", name)
		}
		for i := range w {
			orig := w[i]
			w[i] = orig + backboneStep
			lp := loss()
			w[i] = orig - backboneStep
			lm := loss()
			w[i] = orig
			num := (lp - lm) / (2 * backboneStep)
			rel := math.Abs(num-float64(dW[i])) / (1 + math.Abs(float64(dW[i])))
			if rel > worst {
				worst, worstName = rel, fmt.Sprintf("%s[%d]", name, i)
			}
		}
	}
	for li := range m.layers {
		l := &m.layers[li]
		p := fmt.Sprintf("%s%d.", layerPrefix, li)
		check(p+"norm1.weight", l.norm1W, g[p+"norm1.weight"])
		check(p+"norm1.bias", l.norm1B, g[p+"norm1.bias"])
		check(p+"norm2.weight", l.norm2W, g[p+"norm2.weight"])
		check(p+"norm2.bias", l.norm2B, g[p+"norm2.bias"])
		check(p+"self_attn.in_proj.weight", l.inProj, g[p+"self_attn.in_proj.weight"])
		check(p+"self_attn.out_proj.weight", l.outProj, g[p+"self_attn.out_proj.weight"])
		check(p+"linear1.weight", l.lin1, g[p+"linear1.weight"])
		check(p+"linear2.weight", l.lin2, g[p+"linear2.weight"])
	}
	check("flow_lm.out_norm.weight", m.outNormW, g["flow_lm.out_norm.weight"])
	check("flow_lm.out_norm.bias", m.outNormB, g["flow_lm.out_norm.bias"])
	check("input", stream, dStream)

	t.Logf("backbone (through out_norm) VJP: worst rel %.3g at %s", worst, worstName)
	// Reference-pinned bound (adaptive backboneVJPGate).
	const backboneVJPGate = 7.42e-4
	if worst > backboneVJPGate {
		t.Fatalf("backbone backward diverges: worst rel %.3g > %.1e at %s", worst, backboneVJPGate, worstName)
	}
}

// tinyJointModel: backbone + flow + frozen conditioning tables — the full
// trained surface at toy scale.
func tinyJointModel(rng *rand.Rand) *Model {
	const d, heads, ff, layers = 8, 2, 16, 2
	const dim, latent, depth, half = 10, 4, 2, 3
	m := tinyBackboneModel(rng, d, heads, ff, layers)
	fm := tinyFlowModel(rng, dim, latent, d, depth, half)
	m.flow = fm.flow
	m.Dims.FlowDim, m.Dims.LatentDim = dim, latent
	m.Dims.FlowDepth, m.Dims.TimeFreqs = depth, half
	m.Dims.TextVocab = 5
	m.condEmbed = normVec(rng, m.Dims.TextVocab*d, 0, 0.5)
	m.inputLinear = normVec(rng, d*latent, 0, 0.3)
	m.BosEmb = normVec(rng, latent, 0, 1)
	return m
}

// TestJointLossAndGradsFiniteDifference gates the SHIPPED training-step
// math: backbone-through-flow gradients of LossAndGrads (fixed noise seed)
// reproduce central finite differences on sampled indices of every trained
// tensor class.
func TestJointLossAndGradsFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	m := tinyJointModel(rng)
	const frames, seed = 2, 42
	ids := []int{1, 3, 0}
	z := normVec(rng, frames*m.Dims.LatentDim, 0, 1)

	loss := func() float64 {
		l, err := m.LossAndGrads(ids, z, frames, seed, nil)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}

	g := Grads{}
	analytic, err := m.LossAndGrads(ids, z, frames, seed, g)
	if err != nil {
		t.Fatal(err)
	}
	if got := loss(); got != analytic {
		t.Fatalf("loss-only path %.9g != grad path %.9g", got, analytic)
	}
	tensors, _ := m.TrainedTensors(true)
	if len(g) != len(tensors) {
		t.Fatalf("grad tensors %d != trained set %d", len(g), len(tensors))
	}
	for name := range g {
		if _, ok := tensors[name]; !ok {
			t.Fatalf("gradient %q is absent from the trained set", name)
		}
		for i, v := range g[name] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("nonfinite grad %s[%d]", name, i)
			}
		}
	}

	// Accumulation depth: flow stack + backbone stack + frame sum.
	tol := gradCheckTol(gcStep, (m.Dims.FlowDim*m.Dims.FlowDepth+m.Dims.DModel*m.Dims.Layers)*frames)
	worstRel := 0.0
	defer func() { t.Logf("joint FD worst rel %.3e (gate %.1e)", worstRel, tol) }()
	check := func(name string, idxs []int) {
		t.Helper()
		w, grad := tensors[name], g[name]
		for _, i := range idxs {
			orig := w[i]
			w[i] = orig + gcStep
			lp := loss()
			w[i] = orig - gcStep
			lm := loss()
			w[i] = orig
			fd := (lp - lm) / (2 * gcStep)
			an := float64(grad[i])
			if r := math.Abs(fd-an) / (1 + math.Abs(fd)); r > worstRel {
				worstRel = r
			}
			if math.Abs(fd-an) > tol*(1+math.Abs(fd)) {
				t.Fatalf("%s[%d]: fd %.6g vs analytic %.6g (tol %.1e)", name, i, fd, an, tol)
			}
		}
	}
	check("flow_lm.flow_net.input_proj.weight", []int{0, 7})
	check("flow_lm.flow_net.res_blocks.1.adaLN_modulation.1.weight", []int{3, 50})
	check("flow_lm.flow_net.time_embed.1.mlp.2.weight", []int{9})
	check("flow_lm.flow_net.final_layer.linear.bias", []int{0, 3})
	check("flow_lm.out_norm.weight", []int{0, m.Dims.DModel - 1})
	check("flow_lm.out_norm.bias", []int{1})
	check("flow_lm.transformer.layers.0.self_attn.in_proj.weight", []int{0, 63, 191})
	check("flow_lm.transformer.layers.1.linear1.weight", []int{5, 100})
	check("flow_lm.transformer.layers.0.norm1.weight", []int{0, 7})
	check("flow_lm.transformer.layers.1.norm2.bias", []int{2})
	check("flow_lm.transformer.layers.1.self_attn.out_proj.weight", []int{0, 63})
	check("flow_lm.transformer.layers.0.linear2.weight", []int{10})
}
