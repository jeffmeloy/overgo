package latentvideo

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"overgo/internal/trainingprogram"
)

// ditTestConfig: tiny synthetic diffusion transformer exercising every graph
// piece — 3-axis rope over a multi-frame grid, full-width QK norms, adaptive
// modulation, cross-attention with pad-broadcast context, tanh-GELU FFN,
// source-extended input channels (InDim > OutDim, the reference-edit case),
// and the shared time/text conditioning paths.
func ditTestConfig() (DenoiserConfig, LatentGeometry, int) {
	cfg := DenoiserConfig{
		Dim: 24, FFNDim: 16, FreqDim: 8,
		InDim: 4, OutDim: 2,
		NumHeads: 2, NumLayers: 2,
		TextLen: 5, Eps: 1e-6,
		PatchSize: [3]int{1, 2, 2},
		Policy: DenoiserPolicy{
			NumTrainTimesteps: 1000, SinusoidalPeriod: 10000,
			RotaryFrequencyBase: 10000, VAEStride: [3]int{4, 8, 8},
		},
	}
	geometry := LatentGeometry{
		Channels: cfg.InDim, LatentFrames: 2, LatentHeight: 2, LatentWidth: 4,
		Grid: [3]int{2, 1, 2}, Seq: 4,
	}
	return cfg, geometry, 6
}

func ditTestTrainer(t *testing.T) (*DiTTrainer, DiTTrainBatch) {
	t.Helper()
	cfg, geometry, textDim := ditTestConfig()
	rng := rand.New(rand.NewSource(52))
	tensors := make(map[string][]float32)
	for _, spec := range ditTensorSpecs(cfg, textDim) {
		values := make([]float32, spec.rows*spec.cols)
		norm := strings.HasSuffix(spec.name, "norm_q.weight") ||
			strings.HasSuffix(spec.name, "norm_k.weight") ||
			strings.HasSuffix(spec.name, "norm3.weight")
		for i := range values {
			if norm {
				values[i] = 1 + float32(rng.NormFloat64())*0.1
			} else {
				values[i] = float32(rng.NormFloat64()) * 0.2
			}
		}
		tensors[spec.name] = values
	}
	trainer, err := NewDiTTrainer(cfg, textDim, geometry, tensors)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { trainer.Close() })
	if len(tensors) != 0 {
		t.Fatalf("trainer left %d tensors unconsumed", len(tensors))
	}
	latent := make([]float32, geometry.Elements())
	for i := range latent {
		latent[i] = float32(rng.NormFloat64()) * 0.5
	}
	tokens := 3
	rawText := make([]float32, tokens*textDim)
	for i := range rawText {
		rawText[i] = float32(rng.NormFloat64()) * 0.5
	}
	target := make([]float32, cfg.OutDim*geometry.LatentFrames*geometry.LatentHeight*geometry.LatentWidth)
	for i := range target {
		target[i] = float32(rng.NormFloat64()) * 0.5
	}
	return trainer, DiTTrainBatch{
		Latent: latent, RawText: rawText, TextTokens: tokens,
		Timestep: 500, Target: target,
	}
}

func TestDiTTrainerDerivedHyperparameters(t *testing.T) {
	trainer, _ := ditTestTrainer(t)
	want := trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(trainer.ParameterCount())
	if trainer.Config().BaseLearningRate != want {
		t.Fatalf("base LR %g, want derived %g", trainer.Config().BaseLearningRate, want)
	}
	if trainer.Config().Momentum != 29.0/31.0 {
		t.Fatalf("momentum %g, want CLT-derived 29/31", trainer.Config().Momentum)
	}
}

// TestDiTTrainerGradientMatchesFiniteDifference: full-graph central
// differences on the packed masters against the analytic VJP — whole-tensor
// random directional derivatives on every tensor role including the prologue
// (patch embed, text/time embeddings, time projection, head) and the shared
// conditioning tensors that accumulate from every block.
func TestDiTTrainerGradientMatchesFiniteDifference(t *testing.T) {
	trainer, batch := ditTestTrainer(t)
	if _, _, err := trainer.lossAndGradients(batch); err != nil {
		t.Fatal(err)
	}
	analytic := append([]float32(nil), trainer.gradients...)
	// Directional derivatives per tensor: FD noise on single elements sits at
	// the f32 forward noise floor; a whole-tensor random direction both
	// covers every element and lifts the measured magnitude above it. The
	// step is 2e-2 (vs the cross-entropy trainers' 5e-3) because this MSE
	// objective yields directional derivatives ~1e-3 whose central
	// differences at 5e-3 sit within f32 forward noise (measured: the
	// discrepancy shrinks ∝1/eps, the noise signature, not a gradient error).
	const epsilon = 2e-2
	rng := rand.New(rand.NewSource(30301))
	cfg, _, textDim := ditTestConfig()
	for _, spec := range ditTensorSpecs(cfg, textDim) {
		s := trainer.spans[spec.name]
		size := s.end - s.start
		direction := make([]float64, size)
		var norm float64
		for i := range direction {
			direction[i] = rng.NormFloat64()
			norm += direction[i] * direction[i]
		}
		norm = math.Sqrt(norm)
		saved := append([]float32(nil), trainer.weights[s.start:s.end]...)
		shift := func(sign float64) {
			for i := range direction {
				trainer.weights[s.start+i] = saved[i] + float32(sign*epsilon*direction[i]/norm)
			}
		}
		shift(1)
		plus, err := trainer.Loss(batch)
		if err != nil {
			t.Fatal(err)
		}
		shift(-1)
		minus, err := trainer.Loss(batch)
		if err != nil {
			t.Fatal(err)
		}
		copy(trainer.weights[s.start:s.end], saved)
		numeric := (plus - minus) / (2 * epsilon)
		var got float64
		for i := range direction {
			got += float64(analytic[s.start+i]) * direction[i] / norm
		}
		scale := math.Max(math.Max(math.Abs(numeric), math.Abs(got)), 1e-3)
		if math.Abs(numeric-got)/scale > 2e-2 {
			t.Fatalf("%s: analytic directional %g, finite-difference %g", spec.name, got, numeric)
		}
	}
}

// TestDiTTrainerStepDescends: repeated steps on the fixed batch must reduce
// the flow-matching MSE.
func TestDiTTrainerStepDescends(t *testing.T) {
	trainer, batch := ditTestTrainer(t)
	first, err := trainer.Loss(batch)
	if err != nil {
		t.Fatal(err)
	}
	var last DiTTrainStepResult
	for step := 0; step < 5; step++ {
		last, err = trainer.Step(batch)
		if err != nil {
			t.Fatal(err)
		}
		if last.GradientL2 <= 0 {
			t.Fatalf("step %d: gradient norm %g", step, last.GradientL2)
		}
	}
	final, err := trainer.Loss(batch)
	if err != nil {
		t.Fatal(err)
	}
	if !(final < first) {
		t.Fatalf("loss did not descend: %g -> %g (last step loss %g)", first, final, last.Loss)
	}
}
