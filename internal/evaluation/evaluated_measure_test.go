package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestEvaluatedMeasureNamesAMeasureBuiltAgain evaluates one measure twice,
// its cases asking about other clips the second time, as they do when a model
// draws the clips a judge is asked about again. The two sets have two
// identities, and the stored plans name them as one measure. A set the same
// builder made over other cases, as it does at another size or length, is
// another measure under the same source. A plan the store does not hold names
// nothing.
func TestEvaluatedMeasureNamesAMeasureBuiltAgain(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	published := false
	build := func(drawn string, clips ...string) (artifact.ID, string) {
		t.Helper()
		suite := GeneratedAnswerSuite{Kind: GeneratedAnswerKind, Schema: "fixture/v1", Source: "fixture/edit", Transforms: []string{TransformTrimSpace}}
		for _, clip := range clips {
			suite.Cases = append(suite.Cases, GeneratedAnswerCase{Name: "fixture/edit/" + clip, Prompt: "which is true of " + clip + " as drawn " + drawn, MaxTokens: 1, Answers: []string{"a"}})
		}
		compiled, err := CompileGeneratedAnswer(suite)
		if err != nil {
			t.Fatal(err)
		}
		plan, err := bindGeneratedAnswer(compiled, authorities)
		if err != nil {
			t.Fatal(err)
		}
		// The model, recipe and environment the plans name are the same three,
		// published with the first.
		if !published {
			publishPlanFixtureAuthorities(t, store, plan)
			published = true
		}
		if _, err := evaluateGeneratedAnswer(t.Context(), store, exactGenerator{pieces: []string{"a"}}, compiled, plan); err != nil {
			t.Fatal(err)
		}
		measure, ok := EvaluatedMeasure(t.Context(), store, plan.Identity())
		if !ok {
			t.Fatalf("the plan over %v names no measure", clips)
		}
		return compiled.dataset, measure
	}
	first, measure := build("first", "one", "two")
	again, rebuilt := build("again", "one", "two")
	other, variant := build("first", "one")
	if first == again || first == other || measure != rebuilt || measure == variant {
		t.Fatalf("sets %s, %s and %s name measures %q, %q and %q: want three sets, the first two one measure and the third another", first, again, other, measure, rebuilt, variant)
	}
	if measure, ok := EvaluatedMeasure(t.Context(), store, planID(t, artifact.KindProfile, "no such plan")); ok {
		t.Fatalf("a plan the store does not hold names measure %q", measure)
	}
}
