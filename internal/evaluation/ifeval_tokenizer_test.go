package evaluation

import (
	"bytes"
	"errors"
	"testing"

	"github.com/dlclark/regexp2/v2"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestIFEvalTokenizerRefusals holds the compiled tokenizer to refusing a
// changed profile and an unresolved count, and a tokenizer error to
// refusing publication rather than becoming a verdict. It reads no retained
// evidence, so the landing runs it; the parity against the native oracle is
// the cycle's.
func TestIFEvalTokenizerRefusals(t *testing.T) {
	t.Parallel()
	tokenizer, err := compiledIFEvalTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	t.Run("profile identity and unresolved operands", func(t *testing.T) {
		for _, data := range [][]byte{nil, bytes.Replace(ifevalNLTKProfile, []byte("3.9.2"), []byte("3.9.1"), 1)} {
			if _, err := loadIFEvalTokenizer(data); err == nil {
				t.Fatal("changed profile accepted")
			}
		}
		for _, kind := range []string{ifevalSentences, ifevalCapitalWords} {
			for _, count := range []*countRule{nil, {Relation: relationAtLeast, Value: 0}, {Relation: relationEqual, Value: 2}, {Relation: ifevalLessThan, Value: -1}} {
				if _, err := compileInstructionRule(instructionRule{Name: "invalid", Kind: kind, Count: count}); err == nil {
					t.Fatal("unresolved native count accepted")
				}
			}
		}
	})
	t.Run("tokenizer errors refuse evaluation publication", func(t *testing.T) {
		broken := *tokenizer
		broken.period, err = regexp2.Compile(`(?:^){100}`, regexp2.OptionMaxBacktrackingStackSize(32))
		if err != nil {
			t.Fatal(err)
		}
		rule, err := compileInstructionRule(instructionRule{Name: "sentences", Kind: ifevalSentences, Count: &countRule{Relation: relationAtLeast, Value: 1}})
		if err != nil {
			t.Fatal(err)
		}
		rule.tokenizer = &broken
		strict, loose, err := evaluateInstructionViews("x", []compiledInstructionRule{rule})
		if !errors.Is(err, regexp2.ErrBacktrackingStackLimit) || strict != nil || loose != nil {
			t.Fatalf("tokenizer error became a verdict: %v %v %v", strict, loose, err)
		}
		compiled, err := compileInstructionRules(instructionRulesSuite{Kind: instructionRulesKind, Schema: "tokenizer-error", Source: "fixture", Cases: []instructionRulesCase{{Name: "sentence", Prompt: "Write a sentence.", MaxTokens: 1, Rules: []instructionRule{rule.source}}}})
		if err != nil {
			t.Fatal(err)
		}
		compiled.rules[0][0].tokenizer = &broken
		plan, err := bindInstructionRules(compiled, ExactAuthorities{ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident}})
		if err != nil {
			t.Fatal(err)
		}
		local, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer local.Close()
		publishPlanFixtureAuthorities(t, local, plan)
		report, err := evaluateInstructionRules(t.Context(), local, exactGenerator{pieces: []string{"x"}}, compiled, plan)
		if !errors.Is(err, regexp2.ErrBacktrackingStackLimit) || report.ID.Valid() {
			t.Fatalf("tokenizer error published a report: %s %v", report.ID, err)
		}
		compiled.rules[0][0].tokenizer = tokenizer
		report, err = evaluateInstructionRules(t.Context(), local, exactGenerator{pieces: []string{"One sentence."}}, compiled, plan)
		if err != nil || !report.ID.Valid() || report.PromptStrict != 1 {
			t.Fatalf("evaluation did not recover: %s %v", report.ID, err)
		}
	})
}
