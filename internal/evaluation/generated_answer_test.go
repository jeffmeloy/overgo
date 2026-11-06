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
	compiled, err := CompileGeneratedAnswer(suite)
	testcheck.NoError(t, err)
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	plan, err := bindGeneratedAnswer(compiled, authorities)
	testcheck.NoError(t, err)
	suite.Transforms = []string{TransformTrimSpace}
	other, err := CompileGeneratedAnswer(suite)
	testcheck.NoError(t, err)
	otherPlan, err := bindGeneratedAnswer(other, authorities)
	testcheck.NoError(t, err)
	if plan.Identity() == otherPlan.Identity() {
		t.Fatal("normalization profile did not change evaluation identity")
	}
	store := overgodbtest.Open(t, t.TempDir())
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := evaluateGeneratedAnswer(
		t.Context(), store, exactGenerator{pieces: []string{" OK "}}, compiled, plan,
	)
	testcheck.NoError(t, err)
	observation := report.Observations[0]
	if report.Accuracy != 1 || observation.Raw != " OK " || observation.Scored != "ok" || !observation.Accepted {
		t.Fatalf("generated-answer report = %+v", report)
	}
}
