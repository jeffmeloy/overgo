package evaluation

import (
	"fmt"
	"strings"
	"testing"

	"overgo/internal/sequencescore"
)

// A suite whose candidates are answer texts is asked as lettered options;
// the generated letter maps back to the candidate index. A suite whose
// candidates are letters keeps the direct prompt.
func TestChatChoicePromptListsTextCandidatesAsLetteredOptions(t *testing.T) {
	textCase := MultipleChoiceCase{
		Name: "bbh/boolean_expressions/test/0", Prompt: "Q: not ( True ) is\nA:",
		Candidates: []string{" False", " True"}, Answer: 0,
	}
	prompt, letters, err := chatChoicePrompt(textCase)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "\nOptions:\nA. False\nB. True") || !strings.HasSuffix(prompt, chatChoiceInstruction) {
		t.Fatalf("prompt = %q", prompt)
	}
	if len(letters) != 2 || letters[0] != " A" || letters[1] != " B" {
		t.Fatalf("letters = %q", letters)
	}
	if index, matched := matchChoiceLetter("B", letters); !matched || index != 1 {
		t.Fatalf("B -> %d, %v", index, matched)
	}
	letterCase := MultipleChoiceCase{
		Name: "mmlu/x/0", Prompt: "Q\nA. one\nB. two\nAnswer:", Candidates: []string{" A", " B"}, Answer: 1,
	}
	prompt, letters, err = chatChoicePrompt(letterCase)
	if err != nil || prompt != letterCase.Prompt+chatChoiceInstruction || len(letters) != 2 || letters[1] != " B" {
		t.Fatalf("letter case prompt = %q letters = %q err = %v", prompt, letters, err)
	}
	fixture := &chatFixture{answers: []string{"A"}}
	observation, err := answerChoiceByGeneration(t.Context(), fixture, textCase)
	if err != nil || observation.Selected != 0 || !strings.Contains(fixture.shaped[0], "A. False") {
		t.Fatalf("observation = %+v err = %v shaped = %q", observation, err, fixture.shaped)
	}
	wide := MultipleChoiceCase{Name: "wide", Prompt: "Q", Candidates: make([]string, chatOptionLetters+1)}
	for index := range wide.Candidates {
		wide.Candidates[index] = " option"
	}
	if _, _, err := chatChoicePrompt(wide); err == nil {
		t.Fatal("more options than letters was accepted")
	}
}

// A chat protocol that cannot score its suite refuses: at plan time when
// a case cannot be asked, and during the run when the first probe window
// of generated answers names no candidate at all.
func TestChatProtocolRefusesUnscorableSuites(t *testing.T) {
	textCases := func(count int) []MultipleChoiceCase {
		cases := make([]MultipleChoiceCase, count)
		for index := range cases {
			cases[index] = MultipleChoiceCase{
				Name: fmt.Sprintf("bbh/x/%d", index), Prompt: "Q\nA:", Candidates: []string{" False", " True"}, Answer: 1,
			}
		}
		return cases
	}
	suite := MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "test/bbh/v1", Source: "store/bbh",
		Normalization: sequencescore.NormalizationMean, Aggregation: AggregationAccuracy,
		Cases: textCases(chatProbeCases + 4),
	}
	unmatched := &chatFixture{answers: []string{"True"}}
	_, _, err := scoreMultipleChoice(t.Context(), unmatched, suite)
	if err == nil || !strings.Contains(err.Error(), "protocol refused") || !strings.Contains(err.Error(), `"True"`) {
		t.Fatalf("all-unmatched run was not refused: %v", err)
	}
	if unmatched.calls != chatProbeCases {
		t.Fatalf("probe judged after %d cases, want %d", unmatched.calls, chatProbeCases)
	}
	short := suite
	short.Cases = textCases(3)
	if _, _, err := scoreMultipleChoice(t.Context(), &chatFixture{answers: []string{"nope"}}, short); err == nil {
		t.Fatal("a short all-unmatched suite was not refused at its end")
	}
	mixed := &chatFixture{answers: []string{"nope", "nope", "B"}}
	observations, _, err := scoreMultipleChoice(t.Context(), mixed, suite)
	if err != nil || len(observations) != chatProbeCases+4 {
		t.Fatalf("a suite with matches was refused: %v", err)
	}
	var sampled []string
	ctx := WithProgress(t.Context(), func(value Progress) {
		if value.Sample != "" {
			sampled = append(sampled, value.Sample)
		}
	})
	if _, _, err := scoreMultipleChoice(ctx, &chatFixture{answers: []string{"B"}}, short); err != nil {
		t.Fatal(err)
	}
	if len(sampled) != 1 || !strings.Contains(sampled[0], `"B"`) || !strings.Contains(sampled[0], `" A" " B"`) {
		t.Fatalf("first-answer sample = %q", sampled)
	}
	wide := suite
	wide.Cases = []MultipleChoiceCase{{Name: "wide", Prompt: "Q", Candidates: make([]string, chatOptionLetters+1), Answer: 0}}
	for index := range wide.Cases[0].Candidates {
		wide.Cases[0].Candidates[index] = fmt.Sprintf(" option %d", index)
	}
	compiled, err := CompileMultipleChoice(wide)
	if err != nil {
		t.Fatal(err)
	}
	templated := ListingAuthorities()
	templated.Execution.Prompting = PromptingChatTemplate
	if _, err := BindMultipleChoice(compiled, templated); err == nil {
		t.Fatal("plan bound a chat protocol over a case it cannot ask")
	}
	if _, err := BindMultipleChoice(compiled, ListingAuthorities()); err != nil {
		t.Fatalf("raw protocol refused the same suite: %v", err)
	}
}
