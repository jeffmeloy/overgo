package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestGeneratedAnswerRetainsRawAndIdentifiesScoredView(t *testing.T) {
	t.Parallel()
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: "fixture/v1", Source: "fixture",
		Transforms: []string{TransformTrimSpace, transformLowercase},
		Cases: []GeneratedAnswerCase{{
			Name: "answer", Prompt: "prompt", MaxTokens: 1, Answers: []string{"ok"},
		}},
	}
	compiled := testcheck.Must(CompileGeneratedAnswer(suite))(t)
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	plan := testcheck.Must(bindGeneratedAnswer(compiled, authorities))(t)
	suite.Transforms = []string{TransformTrimSpace}
	other := testcheck.Must(CompileGeneratedAnswer(suite))(t)
	otherPlan := testcheck.Must(bindGeneratedAnswer(other, authorities))(t)
	if plan.Identity() == otherPlan.Identity() {
		t.Fatal("normalization profile did not change evaluation identity")
	}
	store := overgodbtest.Open(t, t.TempDir())
	publishPlanFixtureAuthorities(t, store, plan)
	report := testcheck.Must(evaluateGeneratedAnswer(
		t.Context(), store, exactGenerator{pieces: []string{" OK "}}, compiled, plan,
	))(t)
	observation := report.Observations[0]
	if report.Accuracy != 1 || observation.Raw != " OK " || observation.Scored != "ok" || !observation.Accepted {
		t.Fatalf("generated-answer report = %+v", report)
	}
}
