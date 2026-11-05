package scratchmodel

import (
	"math"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

func TestScratchSharedHostRuntimeParity(t *testing.T) {
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, testDerivationProfile(t))
	testcheck.NoError(t, err)
	result, err := construction.TrainShared(oracle.Steps, trainingprogram.BuiltinOptimizerPolicy())
	testcheck.NoError(t, err)
	if len(result.Losses) != len(oracle.LossHistory) {
		t.Fatalf("shared trajectory length=%d want=%d", len(result.Losses), len(oracle.LossHistory))
	}
	for step, loss := range result.Losses {
		if math.IsNaN(loss) || math.IsInf(loss, 0) || math.Abs(loss-oracle.LossHistory[step]) > oracle.LossTolerance {
			t.Fatalf("shared loss[%d]=%.9f oracle=%.9f", step, loss, oracle.LossHistory[step])
		}
	}
	if math.IsNaN(result.ValidationLoss) || math.IsInf(result.ValidationLoss, 0) || math.Abs(result.ValidationLoss-oracle.FinalValLoss) > oracle.LossTolerance {
		t.Fatalf("shared validation=%.9f oracle=%.9f", result.ValidationLoss, oracle.FinalValLoss)
	}

	exact, err := construction.Probe(oracle.Train[:1])
	testcheck.NoError(t, err)
	weights, gradients := make([]float32, len(construction.weights)), make([]float32, len(construction.weights))
	for index, value := range construction.weights {
		weights[index] = float32(value)
	}
	model, err := construction.BindMADTransformer(weights)
	testcheck.NoError(t, err)
	gradient, err := construction.BindMADTransformer(gradients)
	testcheck.NoError(t, err)
	tokens, err := construction.Tokens(oracle.Train[0])
	testcheck.NoError(t, err)
	loss, trace, err := hostmath.MADTransformerForward(model, tokens)
	testcheck.NoError(t, err)
	testcheck.NoError(t, hostmath.MADTransformerBackward(model, gradient, tokens, trace))
	if math.Abs(loss-exact.Loss) > 1e-5 {
		t.Fatalf("shared probe loss=%.9f oracle=%.9f", loss, exact.Loss)
	}
	for _, group := range exact.Groups {
		binding := construction.bindings[group.Name]
		got := gradients[binding.start:binding.end]
		worst := 0.0
		for index, want := range group.Gradients {
			worst = max(worst, math.Abs(float64(got[index])-want))
		}
		if worst > 2e-4 {
			t.Fatalf("shared gradient %s worst=%.3e", group.Name, worst)
		}
	}
	t.Logf("shared scratch trajectory=%v validation=%.9f oracle=%v/%.9f", result.Losses, result.ValidationLoss, oracle.LossHistory, oracle.FinalValLoss)
}
