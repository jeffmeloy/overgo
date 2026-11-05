package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/testcheck"
)

func TestBBHAndMuSRPinnedOracleParity(t *testing.T) {
	t.Parallel()
	const bbhPrompt = "Q: True and not False is\nA:True\n\nQ: not True is\nA:"
	bbh, err := CompileGroupedChoice(GroupedChoiceSuite{
		Kind: GroupedChoiceKind, Schema: "lm-eval/leaderboard-bbh/v1.0", Source: "bbh-fixture",
		Normalization: sequencescore.NormalizationMean,
		Cases: []DemonstratedChoice{{
			Name: "boolean", Group: "boolean-expressions", Prompt: "Q: not True is\nA:", Candidates: []string{"False", "True"}, Answer: 0,
			Demonstrations: []MultipleChoiceCase{{Name: "demo", Prompt: "Q: True and not False is\nA:", Candidates: []string{"False", "True"}, Answer: 1}},
		}},
	})
	testcheck.NoError(t, err)
	if bbh.choice.suite.Cases[0].Prompt != bbhPrompt {
		t.Fatalf("BBH prompt = %q", bbh.choice.suite.Cases[0].Prompt)
	}
	musr, err := CompileGroupedChoice(GroupedChoiceSuite{
		Kind: GroupedChoiceKind, Schema: "lm-eval/leaderboard-musr/v1.0", Source: "musr-fixture",
		Normalization: sequencescore.NormalizationMean,
		Cases: []DemonstratedChoice{{
			Name: "placement", Group: "object-placements",
			Prompt:     "A is left of B.\n\nWhere is A?\n\n1 - left\n2 - right\nAnswer:",
			Candidates: []string{"left", "right"}, Answer: 0,
		}},
	})
	testcheck.NoError(t, err)
	for _, compiled := range []groupedChoicePlan{bbh, musr} {
		plan, err := BindGroupedChoice(compiled, ExactAuthorities{
			ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
			RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
			Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
		})
		testcheck.NoError(t, err)
		store, err := overgodb.Open(t.TempDir())
		testcheck.NoError(t, err)
		publishPlanFixtureAuthorities(t, store, plan)
		report, err := EvaluateGroupedChoice(
			t.Context(), store,
			fixedContinuationScorer{scores: []sequencescore.Score{{LogProbability: -1, Tokens: 1}, {LogProbability: -2, Tokens: 1}}},
			compiled, plan,
		)
		closeErr := store.Close()
		if err != nil || closeErr != nil {
			t.Fatalf("grouped evaluation = %v, close = %v", err, closeErr)
		}
		if report.Accuracy != 1 || len(report.Groups) != 1 || report.Groups[0].Accuracy != 1 || report.Groups[0].Total != 1 {
			t.Fatalf("grouped report = %+v", report)
		}
	}
}
