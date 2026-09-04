package evaluation

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// instructionChatFixture is a templated model for the instruction-rules
// suite: it records the prompt it generates from and answers with a
// canned text.
type instructionChatFixture struct {
	prompts []string
	answer  string
}

func (f *instructionChatFixture) ShapeChatPrompt(prompt string) (string, error) {
	return "<user>" + prompt + "<model>", nil
}

func (f *instructionChatFixture) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	f.prompts = append(f.prompts, prompt)
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 1})
	}
	if options.OnToken != nil {
		if err := options.OnToken(inference.TokenEvent{Piece: f.answer}); err != nil {
			return nil, "", err
		}
	}
	return []tokenizer.TokenID{0, 1}, f.answer, nil
}

func (f *instructionChatFixture) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	panic("instruction rules must not score likelihoods")
}

// TestInstructionRulesShapeThroughTheChatTemplate: under the
// chat-template protocol every instruction prompt reaches the model as
// its shaped user turn, and the plain generator still sees the raw
// prompt.
func TestInstructionRulesShapeThroughTheChatTemplate(t *testing.T) {
	compiled, err := CompileInstructionRules(InstructionRulesSuite{
		Kind: InstructionRulesKind, Schema: "lm-eval/ifeval/v4.0", Source: "fixture",
		Cases: []InstructionRulesCase{{
			Name: "uppercase", Prompt: "Reply in uppercase.", MaxTokens: 8,
			Rules: []InstructionRule{{Name: "uppercase", Kind: RuleUppercase}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution:   ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
	}
	plan, err := BindInstructionRules(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	chat := &instructionChatFixture{answer: "HELLO"}
	report, err := EvaluateInstructionRules(t.Context(), store, chat, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.prompts) != 1 || chat.prompts[0] != "<user>Reply in uppercase.<model>" {
		t.Fatalf("shaped prompts = %q", chat.prompts)
	}
	if report.PromptStrict != 1 {
		t.Fatalf("report = %+v", report)
	}
	// The raw protocol binds a different plan for the same suite, so the
	// two protocols' records never collide; the raw generator's raw
	// prompt is the pinned oracle parity test's path.
	authorities.Execution.Prompting = PromptingRawCompletion
	rawPlan, err := BindInstructionRules(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	if rawPlan.identity == plan.identity {
		t.Fatal("the raw and chat-template plans share an identity")
	}
	if !strings.HasPrefix(chat.prompts[0], "<user>") {
		t.Fatal("chat prompt lost its framing")
	}
}
