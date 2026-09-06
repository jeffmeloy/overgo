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

// chatAnswerOpener opens the model turn so the letter is the next
// thing the model writes (protocol decision 2026-09-03, measured on
// the BBH chat pass): under the instruction alone gemma-4 reasons for
// paragraphs before naming a letter and MiniCPM5 opens its thinking
// channel, both truncated by the answer bound and scored unmatched
// (12 of 27 BBH groups near zero); a system message reached gemma but
// not the thinking model, while this opener drew the letter from both
// as the first generated token. It is appended after the template's
// generation prompt, so it is the model's own turn, not user text.
const chatAnswerOpener = "The answer is"

// chatChoiceMethod names the scorer method the chat-template protocol
// binds into the plan, so a generated-letter record is a different
// plan from a likelihood record of the same suite. The opener changed
// what the protocol measures, so records made under the instruction
// alone stay a different plan under the earlier name.
const chatChoiceMethod = "generated-letter-opener"

// chatAnswerTokens bounds the answer generation after the opener
// (protocol decision 2026-09-03, reopened from the 16 of 2026-09-02):
// the letter follows the opener within a token or two, dressed at most
// as " **B**." or " (B) option", and a model that has not named a
// letter by then is reasoning instead, recorded as unmatched rather
// than run on at a per-case cost the decode rate cannot afford.
const chatAnswerTokens = 8

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
	result, err := Record(ctx, runtime, testCase.Name, shaped+chatAnswerOpener, chatAnswerTokens)
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

// chatChoiceLetters returns the letters the extractor reads for a case,
// the same ones answerChoiceByGeneration asked for.
func chatChoiceLetters(testCase MultipleChoiceCase) []string {
	_, letters, err := chatChoicePrompt(testCase)
	if err != nil {
		return testCase.Candidates
	}
	return letters
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

// validateChatChoiceSuite refuses, at plan time, a suite the chat-template
// protocol cannot ask: every case must render as a prompt whose answer is
// one of the letters the extractor reads. The BBH chat pass scored zero
// over thousands of cases because nothing checked this before the run.
func validateChatChoiceSuite(cases []MultipleChoiceCase) error {
	for _, testCase := range cases {
		if len(testCase.Candidates) == 0 {
			return fmt.Errorf("evaluation: case %s has no candidates for the chat-template protocol", testCase.Name)
		}
		if _, _, err := chatChoicePrompt(testCase); err != nil {
			return err
		}
	}
	return nil
}

// chatProbeCases is how many generated answers the protocol judges before
// it decides the suite is being scored at all (the owner's measure-one-case
// rule applied to correctness): when none of the first eight answers names
// a candidate, the run refuses with the first reply shown beside its
// candidates, seconds into the pass instead of hours.
const chatProbeCases = 8

// chatProbe watches the generated answers of one suite for the
// all-unmatched signature of a protocol that cannot score it.
type chatProbe struct {
	matched  int
	judged   int
	first    ChoiceObservation
	firstSet bool
	letters  []string
}

// judge records one generated answer and refuses once the probe window
// (or the whole suite, when shorter) has produced no match at all.
func (probe *chatProbe) judge(observation ChoiceObservation, letters []string, last bool) error {
	if !probe.firstSet {
		probe.first, probe.letters, probe.firstSet = observation, letters, true
	}
	probe.judged++
	if observation.Selected != unmatchedChoice {
		probe.matched++
	}
	if probe.matched != 0 || (probe.judged < chatProbeCases && !last) {
		return nil
	}
	return fmt.Errorf(
		"evaluation: chat-template protocol refused: none of the first %d generated answers named a candidate; case %s answered %q, candidates %q",
		probe.judged, probe.first.Name, probe.first.Raw, probe.letters,
	)
}
