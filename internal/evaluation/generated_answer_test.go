package evaluation

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestGeneratedAnswerRetainsRawAndIdentifiesScoredView(t *testing.T) {
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: "fixture/v1", Source: "fixture",
		Transforms: []string{TransformTrimSpace, TransformLowercase},
		Cases: []GeneratedAnswerCase{{
			Name: "answer", Prompt: "prompt", MaxTokens: 1, Answers: []string{"ok"},
		}},
	}
	compiled, err := CompileGeneratedAnswer(suite)
	if err != nil {
		t.Fatal(err)
	}
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	plan, err := BindGeneratedAnswer(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	suite.Transforms = []string{TransformTrimSpace}
	other, err := CompileGeneratedAnswer(suite)
	if err != nil {
		t.Fatal(err)
	}
	otherPlan, err := BindGeneratedAnswer(other, authorities)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Identity() == otherPlan.Identity() {
		t.Fatal("normalization profile did not change evaluation identity")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := EvaluateGeneratedAnswer(
		context.Background(), store, exactGenerator{pieces: []string{" OK "}}, compiled, plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	observation := report.Observations[0]
	if report.Accuracy != 1 || observation.Raw != " OK " || observation.Scored != "ok" || !observation.Accepted {
		t.Fatalf("generated-answer report = %+v", report)
	}
}
