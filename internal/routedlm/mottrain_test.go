package routedlm

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/trainingprogram"
)

// motTestConfig: tiny branch-routed stack exercising every graph piece —
// mixed text/vision mask (real segment windows), GQA, rope, QK-norm, both
// branches, two layers, tied head.
func motTestConfig() Config {
	return Config{
		HiddenSize: 8, NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 4,
		IntermediateSize: 12, NumHiddenLayers: 2, VocabSize: 10,
		RMSNormEps: 1e-5, RopeTheta: 10000,
	}
}

func bf16Encode(values []float32) []uint16 {
	out := make([]uint16, len(values))
	for i, v := range values {
		out[i] = uint16(math.Float32bits(v) >> 16)
	}
	return out
}

func motTestMatrix(rng *rand.Rand, out, in int) BF16Matrix {
	values := make([]float32, out*in)
	for i := range values {
		values[i] = float32(rng.NormFloat64()) * 0.2
	}
	return BF16Matrix{Data: bf16Encode(values), In: in, Out: out}
}

func motTestVector(rng *rand.Rand, n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = 1 + float32(rng.NormFloat64())*0.1
	}
	return values
}

func motTestTrainer(t *testing.T) (*ModalityTransformerTrainer, []float32, []int, []MoTTarget) {
	t.Helper()
	cfg := motTestConfig()
	rng := rand.New(rand.NewSource(88675123))
	d, inter := cfg.HiddenSize, cfg.IntermediateSize
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	layers := make([]LayerWeights, cfg.NumHiddenLayers)
	for layer := range layers {
		layers[layer] = LayerWeights{
			InputNorm: InputNormWeights{Text: motTestVector(rng, d), Vision: motTestVector(rng, d)},
			QKV: QKVWeights{
				QText: motTestMatrix(rng, qOut, d), KText: motTestMatrix(rng, kvOut, d),
				VText: motTestMatrix(rng, kvOut, d), OText: motTestMatrix(rng, d, qOut),
				QVision: motTestMatrix(rng, qOut, d), KVision: motTestMatrix(rng, kvOut, d),
				VVision: motTestMatrix(rng, kvOut, d), OVision: motTestMatrix(rng, d, qOut),
				QNorm: [2][]float32{motTestVector(rng, cfg.HeadDim), motTestVector(rng, cfg.HeadDim)},
				KNorm: [2][]float32{motTestVector(rng, cfg.HeadDim), motTestVector(rng, cfg.HeadDim)},
			},
			Output: OutputWeights{
				PostText: motTestVector(rng, d), PostVision: motTestVector(rng, d),
				GateText: motTestMatrix(rng, inter, d), UpText: motTestMatrix(rng, inter, d),
				DownText:   motTestMatrix(rng, d, inter),
				GateVision: motTestMatrix(rng, inter, d), UpVision: motTestMatrix(rng, inter, d),
				DownVision: motTestMatrix(rng, d, inter),
			},
		}
	}
	finalNorm := motTestVector(rng, d)
	headValues := make([]float32, cfg.VocabSize*d)
	for i := range headValues {
		headValues[i] = float32(rng.NormFloat64()) * 0.3
	}
	trainer, err := NewModalityTransformerTrainer(cfg, layers, finalNorm, bf16Encode(headValues), trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { trainer.Close() })
	// Mixed prompt: text, text, vision, vision, vision, text, text.
	mask := []int{0, 0, 1, 1, 1, 0, 0}
	hidden := make([]float32, len(mask)*d)
	for i := range hidden {
		hidden[i] = float32(rng.NormFloat64()) * 0.5
	}
	targets := []MoTTarget{{Position: 0, Token: 3}, {Position: 4, Token: 7}, {Position: 5, Token: 1}}
	return trainer, hidden, mask, targets
}

func TestModalityTransformerDerivedHyperparameters(t *testing.T) {
	trainer, _, _, _ := motTestTrainer(t)
	want := trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(trainer.ParameterCount())
	if trainer.Config().BaseLearningRate != want {
		t.Fatalf("base LR %g, want derived %g", trainer.Config().BaseLearningRate, want)
	}
	if trainer.Config().Momentum != 29.0/31.0 {
		t.Fatalf("momentum %g, want CLT-derived 29/31", trainer.Config().Momentum)
	}
}

// TestModalityTransformerGradientMatchesFiniteDifference: full-graph central
// differences on the packed masters against the analytic VJP, sampled from
// every tensor role in every layer.
func TestModalityTransformerGradientMatchesFiniteDifference(t *testing.T) {
	trainer, hidden, mask, targets := motTestTrainer(t)
	if _, _, err := trainer.lossAndGradients(hidden, mask, targets); err != nil {
		t.Fatal(err)
	}
	analytic := append([]float32(nil), trainer.gradients...)
	// Directional derivatives per tensor: FD noise on single elements sits at
	// the f32 forward noise floor; a whole-tensor random direction both
	// covers every element and lifts the measured magnitude above it.
	const epsilon = 5e-3
	rng := rand.New(rand.NewSource(30301))
	for layer := range trainer.spans {
		sp := trainer.spans[layer]
		for _, section := range []struct {
			name string
			s    span
		}{
			{"inputNorm", sp.inputNorm}, {"q", sp.q}, {"k", sp.k}, {"v", sp.v}, {"o", sp.o},
			{"postNorm", sp.postNorm}, {"gate", sp.gate}, {"up", sp.up}, {"down", sp.down},
		} {
			size := section.s.end - section.s.start
			direction := make([]float64, size)
			var norm float64
			for i := range direction {
				direction[i] = rng.NormFloat64()
				norm += direction[i] * direction[i]
			}
			norm = math.Sqrt(norm)
			saved := append([]float32(nil), trainer.weights[section.s.start:section.s.end]...)
			shift := func(sign float64) {
				for i := range direction {
					trainer.weights[section.s.start+i] = saved[i] + float32(sign*epsilon*direction[i]/norm)
				}
			}
			shift(1)
			plus, err := trainer.Loss(hidden, mask, targets)
			if err != nil {
				t.Fatal(err)
			}
			shift(-1)
			minus, err := trainer.Loss(hidden, mask, targets)
			if err != nil {
				t.Fatal(err)
			}
			copy(trainer.weights[section.s.start:section.s.end], saved)
			numeric := (plus - minus) / (2 * epsilon)
			var got float64
			for i := range direction {
				got += float64(analytic[section.s.start+i]) * direction[i] / norm
			}
			scale := math.Max(math.Max(math.Abs(numeric), math.Abs(got)), 1e-3)
			if math.Abs(numeric-got)/scale > 2e-2 {
				t.Fatalf("layer %d %s: analytic directional %g, finite-difference %g", layer, section.name, got, numeric)
			}
		}
	}
}

// TestModalityTransformerStepDescends: repeated steps on the fixed prompt
// must reduce the pipeline's own cross-entropy.
func TestModalityTransformerStepDescends(t *testing.T) {
	trainer, hidden, mask, targets := motTestTrainer(t)
	first, err := trainer.Loss(hidden, mask, targets)
	if err != nil {
		t.Fatal(err)
	}
	var last MoTTrainStepResult
	for step := 0; step < 5; step++ {
		last, err = trainer.Step(hidden, mask, targets)
		if err != nil {
			t.Fatal(err)
		}
		if last.GradientL2 <= 0 {
			t.Fatalf("step %d: gradient norm %g", step, last.GradientL2)
		}
	}
	final, err := trainer.Loss(hidden, mask, targets)
	if err != nil {
		t.Fatal(err)
	}
	if !(final < first) {
		t.Fatalf("loss did not descend: %g -> %g (last step loss %g)", first, final, last.Loss)
	}
}
