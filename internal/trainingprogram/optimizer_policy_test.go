package trainingprogram

import (
	"bytes"
	"context"
	"io"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

func TestBuiltinOptimizerPolicyDerivesConfig(t *testing.T) {
	policy := BuiltinOptimizerPolicy()
	config, err := policy.Config(policy.MomentumEffectiveSamples)
	if err != nil {
		t.Fatal(err)
	}
	wantRate := math.Pow(float64(policy.MomentumEffectiveSamples), policy.ParameterExponent)
	wantMomentum := float64(policy.MomentumEffectiveSamples-1) / float64(policy.MomentumEffectiveSamples+1)
	if config.BaseLearningRate != wantRate || config.Momentum != wantMomentum {
		t.Fatalf("optimizer config = %+v, policy = %+v", config, policy)
	}
}

// TestDerivedRateFineTunesMeasuredModels holds the policy's derived rate
// inside the band a measurement found to fine-tune each model without
// drifting: progressive held-out loss (each window scored before the model
// trains on it, against the base's own loss on the same windows) improved
// and stayed improved over the whole run. A rate above the band gains for a
// few dozen windows and then walks the model off its optimum.
func TestDerivedRateFineTunesMeasuredModels(t *testing.T) {
	t.Parallel()
	policy := BuiltinOptimizerPolicy()
	for _, measured := range []struct {
		model           string
		parameters      int
		lowest, highest float64
		evidence        string
	}{
		{"Qwen2.5-0.5B", 494032768, 4.5e-6, 1.35e-5,
			"2026-10-02 WikiText-2 sweep, frozen lexical, 512-token windows: 4.5e-5 drifted from -0.15 to +0.22 nats over 200 windows; " +
				"4.5e-6 to 1.35e-5 held -0.21 to -0.26 nats over 200 and 540 windows"},
	} {
		if rate := policy.BaseLearningRate(measured.parameters); rate < measured.lowest || rate > measured.highest {
			t.Errorf("%s: derived rate %.3g outside the measured fine-tuning band [%.3g, %.3g] (%s)",
				measured.model, rate, measured.lowest, measured.highest, measured.evidence)
		}
	}
}

// TestOptimizerPolicyRecordedBeforeInitializationServesBoth reads a policy
// stored before training from initialization had its own exponent: its one
// exponent derives both rates, so recipes that bind it still resolve.
func TestOptimizerPolicyRecordedBeforeInitializationServesBoth(t *testing.T) {
	t.Parallel()
	policy, _, err := optimizerPolicyCodec.Normalize([]byte(`{"parameter_exponent": -0.5, "momentum_effective_samples": 30, "schedule": "constant"}`))
	if err != nil {
		t.Fatal(err)
	}
	const parameters = 494032768
	if base, initialization := policy.BaseLearningRate(parameters), policy.InitializationLearningRate(parameters); base != initialization || base != math.Pow(parameters, -0.5) {
		t.Fatalf("recorded policy rates = %g and %g, want both %g", base, initialization, math.Pow(parameters, -0.5))
	}
	builtin := BuiltinOptimizerPolicy()
	if builtin.InitializationLearningRate(parameters) <= builtin.BaseLearningRate(parameters) {
		t.Fatalf("training from initialization (%g) must step faster than training a model that already learned (%g)",
			builtin.InitializationLearningRate(parameters), builtin.BaseLearningRate(parameters))
	}
}

func TestOptimizerPolicyRequiresRecipeContent(t *testing.T) {
	policy := BuiltinOptimizerPolicy()
	content, err := policy.Content()
	if err != nil {
		t.Fatal(err)
	}
	store := &policyReader{contents: map[artifact.ID]artifact.Content{policy.ID: content}}
	model, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	definition, err := recipe.NewDefinitionWithDependencies(recipe.TaskTraining, []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: model},
		{Role: recipe.DependencyOptimizer, Artifact: policy.ID},
	}, []recipe.Node{{ID: "optimize", Module: "test", Placement: recipe.PlacementHost}}, nil, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}})
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := OptimizerPolicyFromRecipe(t.Context(), store, definition)
	if err != nil || resolved.ID != policy.ID {
		t.Fatalf("resolved = %+v, err = %v", resolved, err)
	}
	delete(store.contents, policy.ID)
	if _, err := OptimizerPolicyFromRecipe(t.Context(), store, definition); err == nil {
		t.Fatal("missing optimizer policy content accepted")
	}
}

type policyReader struct {
	artifact.Reader
	contents map[artifact.ID]artifact.Content
}

func (reader *policyReader) OpenContent(_ context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	content, ok := reader.contents[id]
	return content.Descriptor, bytes.NewReader(content.Data), ok, nil
}
