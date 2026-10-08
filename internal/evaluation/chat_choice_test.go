package evaluation

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// chatFixture answers every shaped prompt with a canned text and
// records what it was asked, standing in for a templated model.
type chatFixture struct {
	answers []string
	shaped  []string
	calls   int
}

func (f *chatFixture) ShapeChatPrompt(prompt string) (string, error) {
	f.shaped = append(f.shaped, prompt)
	return "<user>" + prompt + "<model>", nil
}

func (f *chatFixture) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if !strings.HasPrefix(prompt, "<user>") || !strings.HasSuffix(prompt, "<model>"+chatAnswerOpener) {
		panic("generation received an unshaped prompt or an unopened model turn")
	}
	text := f.answers[f.calls%len(f.answers)]
	f.calls++
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 1})
	}
	if options.OnToken != nil {
		if err := options.OnToken(inference.TokenEvent{Piece: text}); err != nil {
			return nil, "", err
		}
	}
	return []tokenizer.TokenID{0, 1}, text, nil
}

func (f *chatFixture) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	panic("chat-template scoring must not score likelihoods")
}

// askedChat keeps the text each generation started from.
type askedChat struct {
	*chatFixture
	asked []string
}

func (a *askedChat) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	a.asked = append(a.asked, prompt)
	return a.chatFixture.Generate(ctx, prompt, options)
}

// TestChatChoicePromptIsTheTextTheEvaluationGeneratesFrom holds the one
// function that says what a model is asked under the chat-template protocol
// to being what the evaluation generates from: the question with its
// instruction as the user's turn, in the runtime's shaping, and the model's
// turn opened. A case whose candidates are texts is asked as lettered options.
// A reader of prompts that calls it reads what the model was given.
func TestChatChoicePromptIsTheTextTheEvaluationGeneratesFrom(t *testing.T) {
	t.Parallel()
	runtime := &askedChat{chatFixture: &chatFixture{answers: []string{" B"}}}
	for _, testCase := range []MultipleChoiceCase{
		{Name: "letters", Prompt: "Q1\nA. one\nB. two\nAnswer:", Candidates: []string{" A", " B"}, Answer: 1},
		{Name: "texts", Prompt: "Is it so?", Candidates: []string{" False", " True"}, Answer: 1},
	} {
		want, err := ChatChoicePrompt(runtime, testCase)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := answerChoiceByGeneration(t.Context(), runtime, testCase); err != nil {
			t.Fatal(err)
		}
		if asked := runtime.asked[len(runtime.asked)-1]; asked != want {
			t.Errorf("%s: the evaluation generated from %q, and the prompt is said to be %q", testCase.Name, asked, want)
		}
		if !strings.HasPrefix(want, "<user>"+testCase.Prompt) || !strings.HasSuffix(want, chatChoiceInstruction+"<model>"+chatAnswerOpener) {
			t.Errorf("%s: the prompt is %q", testCase.Name, want)
		}
	}
	if listed, _ := ChatChoicePrompt(runtime, MultipleChoiceCase{Name: "texts", Prompt: "Is it so?", Candidates: []string{" False", " True"}}); !strings.Contains(listed, "\nOptions:\nA. False\nB. True") {
		t.Errorf("candidates that are texts are asked as %q", listed)
	}
	if _, err := ChatChoicePrompt(runtime, MultipleChoiceCase{Name: "many", Prompt: "Q", Candidates: make([]string, chatOptionLetters+1)}); err == nil {
		t.Error("a case with more options than letters was given a prompt")
	}
}

// TestCountUnreadCountsGeneratedAnswersThatNamedNoOption holds the count the
// scorecard states beside a generated-letter reading: the answers of a report
// that were generated and in which no option was read. An answer that named
// one, and a case scored by likelihood, which has no text, are not counted;
// what is no report counts none.
func TestCountUnreadCountsGeneratedAnswersThatNamedNoOption(t *testing.T) {
	t.Parallel()
	report := `{"observations":[{"selected":1,"raw":" B"},{"selected":-1,"raw":" not given in the options"},{"selected":-1,"raw":" calculated as follows"},{"selected":-1},{"selected":0}]}`
	if unread := countUnread([]byte(report)); unread != 2 {
		t.Fatalf("%d answers are counted as naming no option, want the two that were generated and named none", unread)
	}
	for _, other := range []string{`{"observations":[]}`, `{"accuracy":0.5}`, `not a report`} {
		if unread := countUnread([]byte(other)); unread != 0 {
			t.Errorf("%d answers are counted in %s", unread, other)
		}
	}
}

func TestMatchChoiceLetterReadsTheFirstStandaloneLetter(t *testing.T) {
	t.Parallel()
	candidates := []string{" A", " B", " C", " D"}
	for _, testCase := range []struct {
		text    string
		index   int
		matched bool
	}{
		{"B", 1, true},
		{" C.", 2, true},
		{"The answer is (D).", 3, true},
		{"Answer: A", 0, true},
		{"BAD choice", 0, false},
		{"", 0, false},
	} {
		index, matched := matchChoiceLetter(testCase.text, candidates)
		if index != testCase.index || matched != testCase.matched {
			t.Fatalf("%q -> %d/%v, want %d/%v", testCase.text, index, matched, testCase.index, testCase.matched)
		}
	}
}

func TestChatTemplateScoringGeneratesTheLetter(t *testing.T) {
	t.Parallel()
	suite := MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "test/mmlu/v1", Source: "store/mmlu",
		Normalization: sequencescore.NormalizationSum, Aggregation: AggregationAccuracy,
		Cases: []MultipleChoiceCase{
			{Name: "one", Prompt: "Q1\nAnswer:", Candidates: []string{" A", " B"}, Answer: 1},
			{Name: "two", Prompt: "Q2\nAnswer:", Candidates: []string{" A", " B"}, Answer: 0},
			{Name: "three", Prompt: "Q3\nAnswer:", Candidates: []string{" A", " B"}, Answer: 0},
		},
	}
	// The opener precedes every answer: a reasoning model's " **B**." and a
	// thinking model's " (A) option" both carry the letter first.
	fixture := &chatFixture{answers: []string{" **B**.", " (B) option", " no idea"}}
	observations, accuracy, err := scoreMultipleChoice(t.Context(), fixture, suite)
	if err != nil {
		t.Fatal(err)
	}
	if len(observations) != 3 || observations[0].Selected != 1 || observations[1].Selected != 1 ||
		observations[2].Selected != unmatchedChoice || observations[2].Raw != " no idea" {
		t.Fatalf("observations = %+v", observations)
	}
	if accuracy != 1.0/3 {
		t.Fatalf("accuracy = %v", accuracy)
	}
	if len(fixture.shaped) != 3 || !strings.HasSuffix(fixture.shaped[0], chatChoiceInstruction) {
		t.Fatalf("shaped prompts = %q", fixture.shaped)
	}
	raw := ListingAuthorities()
	templated := raw
	templated.Execution.Prompting = PromptingChatTemplate
	compiled, err := CompileMultipleChoice(suite)
	if err != nil {
		t.Fatal(err)
	}
	rawPlan, err := BindMultipleChoice(compiled, raw)
	if err != nil {
		t.Fatal(err)
	}
	templatedPlan, err := BindMultipleChoice(compiled, templated)
	if err != nil {
		t.Fatal(err)
	}
	if rawPlan.body.Scorer == templatedPlan.body.Scorer {
		t.Fatal("generated-letter method did not change the scorer authority")
	}
}
