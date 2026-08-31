package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestIFEvalPinnedOracleParity(t *testing.T) {
	const response = "preface\nHELLO WORLD\npostscript"
	compiled, err := CompileInstructionRules(InstructionRulesSuite{
		Kind: InstructionRulesKind, Schema: "lm-eval/ifeval/v4.0", Source: "fixture",
		Cases: []InstructionRulesCase{{
			Name: "uppercase", Prompt: "Reply in uppercase and include HELLO.", MaxTokens: 8,
			Rules: []InstructionRule{
				{Name: "uppercase", Kind: RuleUppercase},
				{Name: "keyword", Kind: RuleContainsAll, Values: []string{"HELLO"}},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindInstructionRules(compiled, ExactAuthorities{
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
	report, err := EvaluateInstructionRules(
		t.Context(), store, exactGenerator{pieces: []string{response}}, compiled, plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	observation := report.Observations[0]
	if report.PromptStrict != 0 || report.InstructionStrict != 0.5 || report.PromptLoose != 1 ||
		report.InstructionLoose != 1 || observation.Raw != response ||
		observation.Strict[0] || !observation.Strict[1] || !allTrue(observation.Loose) {
		t.Fatalf("IFEval report = %+v", report)
	}
}
