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
	resolved, err := OptimizerPolicyFromRecipe(context.Background(), store, definition)
	if err != nil || resolved.ID != policy.ID {
		t.Fatalf("resolved = %+v, err = %v", resolved, err)
	}
	delete(store.contents, policy.ID)
	if _, err := OptimizerPolicyFromRecipe(context.Background(), store, definition); err == nil {
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
