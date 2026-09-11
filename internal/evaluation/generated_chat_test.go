package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
)

func TestGeneratedSuiteUsesDeclaredChatPrompt(t *testing.T) {
	for _, kind := range []string{GeneratedAnswerKind, StructuredGeneratedKind} {
		t.Run(kind, func(t *testing.T) {
			source := map[string]any{"kind": kind, "schema": "fixture/v1", "source": "chat-framing-fixture", "cases": []map[string]any{{"name": "addition", "prompt": "2 + 2?", "max_tokens": 8, "answers": []string{"4"}}}}
			if kind == StructuredGeneratedKind {
				source["extractor"] = ExtractorDollarSpan
				source["equivalence"] = EquivalenceLatexSurface
				source["cases"].([]map[string]any)[0]["group"] = "arithmetic"
			}
			raw, err := json.Marshal(source)
			if err != nil {
				t.Fatal(err)
			}
			authorities := ExactAuthorities{ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate}}
			suite, err := CompileSuite(raw, authorities)
			if err != nil {
				t.Fatal(err)
			}
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			publishPlanFixtureAuthorities(t, store, suite.plan)
			chat := &instructionChatFixture{answer: "4"}
			if _, err := suite.execute(t.Context(), store, chat); err != nil {
				t.Fatal(err)
			}
			if len(chat.prompts) != 1 || chat.prompts[0] != "<user>2 + 2?<model>" {
				t.Fatalf("chat-template plan generated with prompt %q", chat.prompts)
			}
		})
	}
}

type rawInstructionGenerator instructionChatFixture

func (f *rawInstructionGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return (*instructionChatFixture)(f).Generate(ctx, prompt, options)
}

type refusedInstructionChat struct {
	instructionChatFixture
	cause error
}

func (f *refusedInstructionChat) ShapeChatPrompt(string) (string, error) { return "", f.cause }

func TestInstructionGenerationPreservesRawAndRefusesFramingError(t *testing.T) {
	raw := &rawInstructionGenerator{answer: "4"}
	if _, err := recordInstruction(t.Context(), raw, "raw", "2 + 2?", 8); err != nil {
		t.Fatal(err)
	}
	if len(raw.prompts) != 1 || raw.prompts[0] != "2 + 2?" {
		t.Fatalf("raw prompt changed: %q", raw.prompts)
	}
	cause := errors.New("declared template failed")
	chat := &refusedInstructionChat{cause: cause}
	if _, err := recordInstruction(t.Context(), chat, "chat", "2 + 2?", 8); !errors.Is(err, cause) {
		t.Fatalf("framing failure lost: %v", err)
	}
	if len(chat.prompts) != 0 {
		t.Fatal("generation ran after framing failed")
	}
}
