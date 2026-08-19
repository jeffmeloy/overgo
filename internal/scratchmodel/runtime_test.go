package scratchmodel

import (
	"math"
	"testing"

	"overgo/internal/hostmath"
)

func TestScratchSharedHostRuntimeParity(t *testing.T) {
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, AdaptiveDerivationProfile())
	if err != nil {
		t.Fatal(err)
	}
	result, err := construction.TrainShared(oracle.Steps)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	weights, gradients := make([]float32, len(construction.weights)), make([]float32, len(construction.weights))
	for index, value := range construction.weights {
		weights[index] = float32(value)
	}
	model, err := construction.bindMADTransformer(weights)
	if err != nil {
		t.Fatal(err)
	}
	gradient, err := construction.bindMADTransformer(gradients)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := construction.Tokens(oracle.Train[0])
	if err != nil {
		t.Fatal(err)
	}
	loss, trace, err := hostmath.MADTransformerForward(model, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if err := hostmath.MADTransformerBackward(model, gradient, tokens, trace); err != nil {
		t.Fatal(err)
	}
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
