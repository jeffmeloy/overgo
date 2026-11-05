package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

func TestIFEvalPinnedOracleParity(t *testing.T) {
	t.Parallel()
	const response = "preface\nHELLO WORLD\npostscript"
	compiled, err := compileInstructionRules(instructionRulesSuite{
		Kind: instructionRulesKind, Schema: "lm-eval/ifeval/v4.0", Source: "fixture",
		Cases: []instructionRulesCase{{
			Name: "uppercase", Prompt: "Reply in uppercase and include HELLO.", MaxTokens: 8,
			Rules: []instructionRule{
				{Name: "uppercase", Kind: ruleUppercase},
				{Name: "keyword", Kind: ruleContainsAll, Values: []string{"HELLO"}},
			},
		}},
	})
	testcheck.NoError(t, err)
	plan, err := bindInstructionRules(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	testcheck.NoError(t, err)
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := evaluateInstructionRules(
		t.Context(), store, exactGenerator{pieces: []string{response}}, compiled, plan,
	)
	testcheck.NoError(t, err)
	observation := report.Observations[0]
	if report.PromptStrict != 0 || report.InstructionStrict != 0.5 || report.PromptLoose != 1 ||
		report.InstructionLoose != 1 || observation.Raw != response ||
		observation.Strict[0] || !observation.Strict[1] || !allTrue(observation.Loose) {
		t.Fatalf("IFEval report = %+v", report)
	}
}
