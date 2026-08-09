package oscillatorimage

import "testing"

// Training constants are the reference bootstrap's execution facts
// (TrainConditionalOscillatorCheckpoint: baseLR 0.02, mu 0.95; the reference
// test gates 60 steps to below 0.8x the starting loss).
const (
	trainSteps      = 60
	trainBaseLR     = 0.02
	trainMu         = 0.95
	trainStrongFrac = 0.8
)

// tinyModel: a small but complete generator, reference buildTinyUn0 geometry
// (two classes; conv weights damped 0.2 so activations stay in range).
func tinyModel() *Model {
	cfg := Config{
		ModelType: "test", N: 6, NCond: 2, NClasses: 2,
		InChannels: 3, InH: 2, InW: 2, OutChannels: 3,
		NumSteps: 4, Dt: 0.25, KScale: 1, KCondScale: 1, KDriveScale: 1,
		Relativization: "ref_oscillator", Encoding: "sin_cos", TanhOut: true,
		BlockChannels: []int{32, 32},
	}
	scale := func(v []float32) []float32 {
		for i := range v {
			v[i] *= 0.2
		}
		return v
	}
	return &Model{
		Cfg: cfg, Namespace: "test", Slope: decoderNegativeSlope,
		Omega:     randF32(cfg.N, 52),
		OmegaCond: randF32(cfg.NCond, 53),
		K:         randF32(cfg.N*cfg.N, 54),
		KCond:     randF32(cfg.NCond*cfg.NCond, 55),
		Drive:     randF32(cfg.NClasses*cfg.N*cfg.NCond, 56),
		Blocks: []DecoderBlock{
			{W1: scale(randF32(32*3*convTaps, 57)), B1: randF32(32, 58), W2: scale(randF32(32*32*convTaps, 59)), B2: randF32(32, 60), Cout: 32},
			{W1: scale(randF32(32*32*convTaps, 61)), B1: randF32(32, 62), W2: scale(randF32(32*32*convTaps, 63)), B2: randF32(32, 64), Cout: 32},
		},
		ToOutW: scale(randF32(3*32*convTaps, 65)),
		ToOutB: randF32(3, 66),
	}
}

// TestGeneratorBackwardFiniteDiff: full generator VJP (decoder + readout +
// Euler BPTT) vs central differences on spot indices.
func TestGeneratorBackwardFiniteDiff(t *testing.T) {
	m := tinyModel()
	cfg := m.Cfg
	classes := cfg.NClasses
	tot := cfg.N + cfg.NCond
	init := randF32(classes*tot, 51)
	dImage := randF32(classes*cfg.OutChannels*cfg.OutH()*cfg.OutW(), 70)
	loss := func() float64 {
		img, _ := m.trainingForwardTrace(init, m.Drive, classes)
		var s float64
		for i := range img {
			s += float64(dImage[i]) * float64(img[i])
		}
		return s
	}
	grads := Grads{}
	_, trace := m.trainingForwardTrace(init, m.Drive, classes)
	m.backwardInto(trace, m.Drive, dImage, grads)
	// Gate 2e-2 relative: the reference generator FD test's committed bound.
	// eps 3e-3 balances f32-forward FD noise against curvature under this
	// conv accumulation order (probe: fd oscillates ~5% at 1e-3, ~0.4% here).
	check := func(name string, params, analytic []float32, index int) {
		save := params[index]
		const eps = 3e-3
		params[index] = save + eps
		lp := loss()
		params[index] = save - eps
		lm := loss()
		params[index] = save
		num := (lp - lm) / (2 * eps)
		den := float64(analytic[index])
		scale := den
		if scale < 0 {
			scale = -scale
		}
		if scale < 1 {
			scale = 1
		}
		if d := (num - den) / scale; d > 2e-2 || d < -2e-2 {
			t.Errorf("%s[%d]: analytic=%.5f finite-diff=%.5f (rel %.2e)", name, index, den, num, d)
		}
	}
	check("dOmega", m.Omega, grads[m.tensorName("omega")], 0)
	check("dK", m.K, grads[m.tensorName("k")], 1)
	check("dToOutW", m.ToOutW, grads[m.tensorName("to_out.weight")], 0)
	check("dBlock0W2", m.Blocks[0].W2, grads[m.blockTensorName(0, "w2")], 5)
}

// TestTrainDriftDecreasesLoss: the ported objective under the ported Muon
// optimizer descends on the tiny generator.
func TestTrainDriftDecreasesLoss(t *testing.T) {
	m := tinyModel()
	trajectory, err := m.TrainDrift(trainSteps, trainBaseLR, trainMu, 7)
	if err != nil {
		t.Fatal(err)
	}
	first, last := trajectory[0], trajectory[len(trajectory)-1]
	t.Logf("drift loss %.5f -> %.5f over %d steps (lr %g mu %g)", first, last, trainSteps, trainBaseLR, trainMu)
	if !(last < first*trainStrongFrac) {
		t.Fatalf("loss did not clearly descend: %.5f -> %.5f", first, last)
	}
}
