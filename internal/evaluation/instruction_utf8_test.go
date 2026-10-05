package evaluation

import (
	"testing"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestInstructionGenerationWithAnInvalidByteResumes holds a retained generation to what resuming
// reads back: a generation cut inside a multibyte character, as a token limit cuts one, ends in
// bytes that are no text; JSON writes them as the replacement character's escape and a read
// finds the character itself, which encodes differently, so a generation kept as produced could
// never be read again and every rerun of its case failed on it. The text is kept as the text
// it is, the invalid bytes replaced by the replacement character, and a second evaluation reads
// the retained generation and generates nothing.
func TestInstructionGenerationWithAnInvalidByteResumes(t *testing.T) {
	t.Parallel()
	suite := instructionRulesSuite{Kind: instructionRulesKind, Schema: "fixture/v1", Source: "utf8-fixture", Cases: []instructionRulesCase{
		{Name: "cut", Prompt: "Cut answer.", MaxTokens: 1, Rules: []instructionRule{{Name: "uppercase", Kind: ruleUppercase}}},
	}}
	compiled, err := compileInstructionRules(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindInstructionRules(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
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
	cut := "OK\xe2\x82"
	first := &instructionChatFixture{answer: cut}
	if _, err := evaluateInstructionRules(t.Context(), store, first, compiled, plan); err != nil {
		t.Fatal(err)
	}
	second := &instructionChatFixture{answer: cut}
	report, err := evaluateInstructionRules(t.Context(), store, second, compiled, plan)
	if err != nil {
		t.Fatalf("a second evaluation of a generation that ended inside a character: %v", err)
	}
	if len(second.prompts) != 0 || len(report.Observations) != 1 {
		t.Fatalf("the second evaluation generated %v and reported %d cases, want it to read the retained generation", second.prompts, len(report.Observations))
	}
	if raw := report.Observations[0].Raw; !utf8.ValidString(raw) {
		t.Errorf("the retained text %q is not valid UTF-8", raw)
	}
}
