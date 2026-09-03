package evaluation

import (
	"context"
	"fmt"
	"strings"
	"unicode"
)

// ChatChoiceRuntime is a runtime that scores multiple choice the way an
// instruct model answers: the question rides its declared chat template
// and the model generates its answer, from which the choice letter is
// read. Likelihood over letters at the opening of the model turn is not
// where such a model answers (a template that opens a thought channel
// puts prose there), so the chat-template protocol generates instead.
type ChatChoiceRuntime interface {
	Generator
	// ShapeChatPrompt renders one user message through the model's
	// declared template with the generation prompt appended.
	ShapeChatPrompt(prompt string) (string, error)
}

// chatChoiceInstruction closes the question so the answer is the
// letter itself, which the extractor reads as the first standalone
// candidate letter in the generated text.
const chatChoiceInstruction = "\nAnswer with the letter only."

// chatChoiceMethod names the scorer method the chat-template protocol
// binds into the plan, so a generated-letter record is a different
// plan from a likelihood record of the same suite.
const chatChoiceMethod = "generated-letter"

// chatAnswerTokens bounds the answer generation (owner decision
// 2026-09-02): a letter-only answer is a few tokens; the bound admits a
// short phrase before the letter and records a model that exceeds it as
// unmatched rather than letting it run on. Reopen if a template's
// answer opener is longer than this.
const chatAnswerTokens = 16

// matchChoiceLetter reads the first standalone candidate letter in the
// generated text: a letter that is not part of a longer word. Candidates
// carry the suite's leading space; the comparison ignores it.
func matchChoiceLetter(text string, candidates []string) (int, bool) {
	runes := []rune(text)
	for position, char := range runes {
		if position > 0 && unicode.IsLetter(runes[position-1]) {
			continue
		}
		if position+1 < len(runes) && unicode.IsLetter(runes[position+1]) {
			continue
		}
		for index, candidate := range candidates {
			letter := []rune(strings.TrimSpace(candidate))
			if len(letter) == 1 && letter[0] == char {
				return index, true
			}
		}
	}
	return 0, false
}

// answerChoiceByGeneration scores one case through the template: the
// observation records the generated text and whether a letter matched;
// an unmatched answer selects nothing and counts as incorrect.
func answerChoiceByGeneration(
	ctx context.Context,
	runtime ChatChoiceRuntime,
	testCase MultipleChoiceCase,
) (ChoiceObservation, error) {
	prompt, letters, err := chatChoicePrompt(testCase)
	if err != nil {
		return ChoiceObservation{}, err
	}
	shaped, err := runtime.ShapeChatPrompt(prompt)
	if err != nil {
		return ChoiceObservation{}, err
	}
	result, err := generateText(ctx, runtime, testCase.Name, shaped, chatAnswerTokens)
	if err != nil {
		return ChoiceObservation{}, err
	}
	observation := ChoiceObservation{
		Name: testCase.Name, Answer: testCase.Answer, Raw: result.Text, Selected: unmatchedChoice,
	}
	if index, matched := matchChoiceLetter(result.Text, letters); matched {
		observation.Selected = index
	}
	return observation, nil
}

// chatOptionLetters bounds the lettered option list a chat prompt can
// carry: one letter per option through Z.
const chatOptionLetters = 26

// chatChoicePrompt renders the case for the chat protocol and returns the
// letter candidates the extractor reads. A suite whose candidates are
// already letters (MMLU's " A".." D") asks for the letter directly. A
// suite whose candidates are answer texts (BBH's " False"/" True",
// MuSR's option sentences) lists them as lettered options after the
// prompt and asks for the letter, so the model answers a choice question
// the way an instruct model is prompted to, and the letter maps back to
// the candidate index.
func chatChoicePrompt(testCase MultipleChoiceCase) (string, []string, error) {
	if choiceCandidatesAreLetters(testCase.Candidates) {
		return testCase.Prompt + chatChoiceInstruction, testCase.Candidates, nil
	}
	if len(testCase.Candidates) > chatOptionLetters {
		return "", nil, fmt.Errorf("evaluation: case %s has %d options, more than the %d letters a chat prompt can list",
			testCase.Name, len(testCase.Candidates), chatOptionLetters)
	}
	var prompt strings.Builder
	prompt.WriteString(testCase.Prompt)
	prompt.WriteString("\nOptions:")
	letters := make([]string, len(testCase.Candidates))
	for index, candidate := range testCase.Candidates {
		letter := string(rune('A' + index))
		letters[index] = " " + letter
		prompt.WriteString("\n")
		prompt.WriteString(letter)
		prompt.WriteString(". ")
		prompt.WriteString(strings.TrimSpace(candidate))
	}
	prompt.WriteString(chatChoiceInstruction)
	return prompt.String(), letters, nil
}

// choiceCandidatesAreLetters reports whether every candidate is a single
// letter (with the suite's leading space), the shape the extractor reads
// directly.
func choiceCandidatesAreLetters(candidates []string) bool {
	for _, candidate := range candidates {
		letter := []rune(strings.TrimSpace(candidate))
		if len(letter) != 1 || !unicode.IsLetter(letter[0]) {
			return false
		}
	}
	return len(candidates) > 0
}

// unmatchedChoice is the selection of an answer that named no candidate.
const unmatchedChoice = -1
