package routedlm

import (
	"math"
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"

	"overgo/internal/tensor/dtype"
)

func tinyBridge(hidden, latent int) LatentBridgeWeights {
	seed := uint32(123456789)
	next := testutil.XorShift(seed)
	bf16 := func(n int) []uint16 {
		values := make([]uint16, n)
		for i := range values {
			values[i] = dtype.Float32ToBF16(next())
		}
		return values
	}
	f32 := func(n int) []float32 {
		values := make([]float32, n)
		for i := range values {
			values[i] = next() * 0.05
		}
		return values
	}
	return LatentBridgeWeights{
		Hidden: hidden, Latent: latent,
		DownW: BF16Matrix{Data: bf16(latent * hidden), In: hidden, Out: latent},
		DownB: f32(latent),
		UpW:   BF16Matrix{Data: bf16(hidden * latent), In: latent, Out: hidden},
		UpB:   f32(hidden),
	}
}

func TestLatentBridgeTrainerDescends(t *testing.T) {
	trainer := testcheck.Must(NewLatentBridgeTrainer(tinyBridge(10, 3), trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer trainer.Close()
	if got, want := trainer.Config().Momentum, 29.0/31.0; math.Abs(got-want) > 1e-12 {
		t.Fatalf("momentum %g, want derived %g", got, want)
	}
	seed := uint32(987654321)
	hidden := make([]float32, 6*10)
	for i := range hidden {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		hidden[i] = float32(seed%2000)/1000 - 1
	}
	before := testcheck.Must(trainer.Loss(hidden))(t)
	var first FlowHeadStepResult
	for step := range 15 {
		result := testcheck.Must(trainer.Step(hidden))(t)
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			t.Fatalf("step %d non-finite loss", step)
		}
		if step == 0 {
			first = result
			if math.Abs(result.Loss-before) > 1e-9 || result.GradientL2 <= 0 {
				t.Fatalf("first step loss %.9g vs evaluation %.9g grad=%g", result.Loss, before, result.GradientL2)
			}
		}
	}
	after := testcheck.Must(trainer.Loss(hidden))(t)
	if !(after < first.Loss) {
		t.Fatalf("bridge reconstruction did not descend: %.6f -> %.6f", first.Loss, after)
	}
}

// TestLatentBridgeTrainerCheckpointRoundTripIsBitExact trains one step,
// publishes the trainer's pair and optimizer state as a checkpoint, and
// reads both back bit for bit under every projection tensor name.
func TestLatentBridgeTrainerCheckpointRoundTripIsBitExact(t *testing.T) {
	trainer := testcheck.Must(NewLatentBridgeTrainer(tinyBridge(10, 3), trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer trainer.Close()
	hidden, _, _ := flowStimulus(6, trainer.hidden, 1)
	if _, err := trainer.Step(hidden); err != nil {
		t.Fatal(err)
	}
	program := testcheck.Must(trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveLatentL2, nil, trainer.OptimizerPlan()))(t)
	view := trainer.pack.Weights
	checkpointtest.RequireRoundTrip(t, program, trainer, map[string][]float32{
		"bridge.down.weight": view(bridgeDownW), "bridge.down.bias": view(bridgeDownB),
		"bridge.up.weight": view(bridgeUpW), "bridge.up.bias": view(bridgeUpB),
	})
}

func TestLatentBridgeGradientMatchesFiniteDifference(t *testing.T) {
	const epsilon, limit = 1e-3, 2e-2
	weights := tinyBridge(7, 2)
	hidden := make([]float32, 3*7)
	seed := uint32(55555)
	for i := range hidden {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		hidden[i] = float32(seed%2000)/1000 - 1
	}
	probe := testcheck.Must(NewLatentBridgeTrainer(weights, trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer probe.Close()

	// Capture analytic gradients by rebuilding a fresh trainer whose Step is
	// intercepted before the optimizer consumes them: replicate via a fresh
	// trainer and numerical comparison against the shared Loss.
	analyticTrainer := testcheck.Must(NewLatentBridgeTrainer(weights, trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer analyticTrainer.Close()
	baseline := testcheck.Must(analyticTrainer.Loss(hidden))(t)
	result := testcheck.Must(analyticTrainer.Step(hidden))(t)
	if math.Abs(result.Loss-baseline) > 1e-9 {
		t.Fatalf("step loss %.9g disagrees with evaluation %.9g", result.Loss, baseline)
	}

	groups := probe.OptimizerPlan().Groups()
	for _, index := range []int{0, 5, groups[bridgeUpW].Start + 3, groups[bridgeDownB].Start, probe.ParameterCount() - 1} {
		fresh := testcheck.Must(NewLatentBridgeTrainer(weights, trainingprogram.BuiltinOptimizerPolicy()))(t)
		freshWeights, _ := fresh.pack.Slabs()
		original := freshWeights[index]
		freshWeights[index] = original + epsilon
		plus := testcheck.Must(fresh.Loss(hidden))(t)
		freshWeights[index] = original - epsilon
		minus := testcheck.Must(fresh.Loss(hidden))(t)
		freshWeights[index] = original
		numeric := (plus - minus) / (2 * epsilon)

		grader := testcheck.Must(NewLatentBridgeTrainer(weights, trainingprogram.BuiltinOptimizerPolicy()))(t)
		lossValue, gradientL2, err := grader.lossAndBridgeGradients(hidden)
		testcheck.NoError(t, err)
		if math.Abs(lossValue-baseline) > 1e-9 || gradientL2 <= 0 {
			t.Fatalf("gradient pass loss %.9g vs %.9g", lossValue, baseline)
		}
		_, gradients := grader.pack.Slabs()
		got := float64(gradients[index])
		_ = grader.Close()
		_ = fresh.Close()
		scale := max(1, max(math.Abs(numeric), math.Abs(got)))
		if math.Abs(numeric-got)/scale > limit {
			t.Fatalf("gradient[%d]: analytic=%.6g numeric=%.6g", index, got, numeric)
		}
	}
}
