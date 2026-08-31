package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
)

func TestMMLUProPinnedOracleParity(t *testing.T) {
	const expectedPrompt = "Which number is even?\nA. three\nB. four\nAnswer:B\n\nWhich color is primary?\nA. red\nB. green\nAnswer:"
	compiled, err := CompileMMLUPro(MMLUProSuite{
		Kind: MMLUProKind, Schema: "lm-eval/leaderboard-mmlu-pro/v0.1", Source: "fixture",
		Labels: []string{"A", "B"}, FewShotCount: 1, Normalization: sequencescore.NormalizationSum,
		Demonstrations: []MMLUProCase{
			{Name: "demo-1", Category: "math", Question: "Which number is even?", Options: []string{"three", "four"}, Answer: 1},
			{Name: "demo-2", Category: "math", Question: "unused", Options: []string{"x", "y"}, Answer: 0},
		},
		Cases: []MMLUProCase{
			{Name: "color", Category: "knowledge", Question: "Which color is primary?", Options: []string{"red", "green"}, Answer: 0},
			{Name: "sum", Category: "math", Question: "One plus one?", Options: []string{"one", "two"}, Answer: 1},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if compiled.choice.suite.Cases[0].Prompt != expectedPrompt {
		t.Fatalf("MMLU-Pro prompt = %q", compiled.choice.suite.Cases[0].Prompt)
	}
	plan, err := BindMMLUPro(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	scorer := fixedContinuationScorer{scores: []sequencescore.Score{{LogProbability: -1, Tokens: 1}, {LogProbability: -2, Tokens: 1}}}
	report, err := EvaluateMMLUPro(t.Context(), store, scorer, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Accuracy != 0.5 || len(report.Categories) != 2 || report.Categories[0].Name != "knowledge" ||
		report.Categories[0].Accuracy != 1 || report.Categories[1].Name != "math" || report.Categories[1].Accuracy != 0 {
		t.Fatalf("MMLU-Pro report = %+v", report)
	}
}
