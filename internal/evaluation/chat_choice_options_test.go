package evaluation

import (
	"strings"
	"testing"
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
