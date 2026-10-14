package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/sampling"
)

// reasoningPolicy is every setting the reasoning protocol asks a choice question under, beside the model, the cases and the scorer.
//   - Thinking = the template's thinking switch
//   - Wording = the text that follows the question in the user's turn
//   - Temperature, TopP, TopK, MinP, PresencePenalty = the sampler's settings
//   - PenaltyWindow = the most recent tokens the presence penalty counts
//   - Seed = what each case's sampler seed is drawn from with the case's name (caseSeed)
//   - MaxTokens = the most tokens the model writes for one case
//   - Parser = how the choice is read from what the model wrote (reasonedParsers)
//   - Parser -> the scorer authority, every other field -> the execution authority
//   - Parser != part of the execution because a parser reads answers already given, which another parser reads again without asking
type reasoningPolicy struct {
	Thinking        bool    `json:"thinking"`
	Wording         string  `json:"wording"`
	Temperature     float32 `json:"temperature"`
	TopP            float32 `json:"top_p"`
	TopK            int     `json:"top_k"`
	MinP            float32 `json:"min_p"`
	PresencePenalty float32 `json:"presence_penalty"`
	PenaltyWindow   int     `json:"penalty_window"`
	Seed            int64   `json:"seed"`
	MaxTokens       int     `json:"max_tokens"`
	Parser          string  `json:"-"`
}

// The exploratory request protocol = the settings of the 40-question MMLU-Pro sample internal/scorecard/protocol_samples.json records.
//   - scope = every model asked under PromptingReasoning
//   - reconsider each <=> an ablation separates its effect from the others'
const (
	// exploratoryWording is MMLU-Pro's own instruction to a model that reasons (TIGER-Lab/MMLU-Pro, evaluate_from_local.py).
	//   - reconsider when a suite's benchmark words its reasoning instruction otherwise
	exploratoryWording = "\nThink step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice."
	// exploratoryTemperature, exploratoryTopP, exploratoryTopK, exploratoryMinP and exploratoryPresencePenalty are the Qwen3.5 model card's settings for thinking mode on general tasks.
	//   - source = the card's Best Practices (temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)
	//   - reconsider when a model whose card gives other settings is asked
	exploratoryTemperature     = 1
	exploratoryTopP            = 0.95
	exploratoryTopK            = 20
	exploratoryMinP            = 0
	exploratoryPresencePenalty = 1.5
	// exploratoryPenaltyWindow is llama.cpp's default penalty_last_n, in tokens.
	//   - source = llama.cpp common_params_sampling, the sampler overgo ports
	//   - reconsider when answers repeat past the window
	exploratoryPenaltyWindow = 64
	// exploratoryMaxTokens is the output length the Qwen3.5 card recommends for most queries, in tokens.
	//   - no answer of the 40 reached it (internal/scorecard/protocol_samples.json, limit)
	exploratoryMaxTokens = 32768
	// exploratorySeed is the value each case's seed is drawn from.
	//   - any fixed value names the draws
	exploratorySeed = 0
)

// Parsers of what a model wrote under the reasoning protocol, by the name the scorer authority binds.
const (
	// reasonedAnswerIs reads the letter of the last "answer is (X)" said after the thought, and else the last option letter standing alone there.
	reasonedAnswerIs = "after-thought-answer-is"
	// reasonedFirstLetter reads the first option letter standing alone said after the thought, as the direct protocol reads an answer.
	reasonedFirstLetter = "after-thought-first-letter"
	// answerIsPhrase is what the wording asks an answer to finish with before its letter.
	answerIsPhrase = "answer is"
	// answerIsDressing is what may stand between answerIsPhrase and the letter ("is (B)", "is **B**", "is: B").
	answerIsDressing = " \t*:("
	// developmentKey prefixes the key a development part is published under, before the part's identity.
	developmentKey = "evaluation/development/"
)

// reasonedParsers reads a choice from what a model said after its thought, by parser name.
var reasonedParsers = map[string]func(said string, letters []string) (int, bool){
	reasonedAnswerIs: func(said string, letters []string) (int, bool) {
		if index, found := lastAnswerIs(said, letters); found {
			return index, true
		}
		return lastChoiceLetter(said, letters)
	},
	reasonedFirstLetter: matchChoiceLetter,
}

// exploratoryReasoning is the exploratory request protocol as one policy.
func exploratoryReasoning() reasoningPolicy {
	return reasoningPolicy{
		Thinking: true, Wording: exploratoryWording,
		Temperature: exploratoryTemperature, TopP: exploratoryTopP, TopK: exploratoryTopK, MinP: exploratoryMinP,
		PresencePenalty: exploratoryPresencePenalty, PenaltyWindow: exploratoryPenaltyWindow, Seed: exploratorySeed,
		MaxTokens: exploratoryMaxTokens, Parser: reasonedAnswerIs,
	}
}

// valid accepts a policy whose sampler builds, with a wording and a limit.
//   - the parser is checked where the scorer binds it
func (policy reasoningPolicy) valid() bool {
	_, err := policy.sampler("")
	return err == nil && policy.Wording != "" && policy.MaxTokens > 0
}

// sampler is the sampler the case named name is answered with.
//   - repeat penalty = 1 = no multiplicative penalty, as the card gives repetition_penalty=1.0
func (policy reasoningPolicy) sampler(name string) (*sampling.Sampler, error) {
	return sampling.New(sampling.Config{
		Temperature: policy.Temperature, TopP: policy.TopP, TopK: policy.TopK, MinP: policy.MinP,
		PresencePenalty: policy.PresencePenalty, RepeatLastN: policy.PenaltyWindow, RepeatPenalty: 1,
		Seed: caseSeed(policy.Seed, name),
	})
}

// caseSeed draws one case's sampler seed from the policy's seed and the case's name, so a case's draw is the same in any run of the suite and in any order.
func caseSeed(seed int64, name string) int64 {
	digest := sha256.Sum256(fmt.Appendf(nil, "%d/%s", seed, name))
	return int64(binary.BigEndian.Uint64(digest[:]))
}

// reasoningRuntime is a runtime the reasoning protocol asks.
type reasoningRuntime interface {
	Generator
	// ShapeChatTurn renders one user message through the model's declared template with the model's turn opened, thinking switched on or off where the template has the switch.
	ShapeChatTurn(prompt string, thinking bool) (string, error)
	// AfterThought is what a generation said after its thought closed, by the markers the model's template declares.
	//   - closed = false <=> no thought closed in it
	AfterThought(text string) (said string, closed bool)
}

// answerChoiceByReasoning puts one case to the model under the policy and returns all it wrote, its thought included, less the time it took.
func answerChoiceByReasoning(ctx context.Context, runtime reasoningRuntime, policy reasoningPolicy, testCase MultipleChoiceCase) (ExactResult, error) {
	asked, _, err := chatChoicePrompt(testCase, policy.Wording)
	if err != nil {
		return ExactResult{}, err
	}
	shaped, err := runtime.ShapeChatTurn(asked, policy.Thinking)
	if err != nil {
		return ExactResult{}, err
	}
	sampler, err := policy.sampler(testCase.Name)
	if err != nil {
		return ExactResult{}, err
	}
	written, err := record(ctx, runtime, testCase.Name, shaped, inference.GenerateOptions{MaxNewTokens: policy.MaxTokens, Sampler: sampler})
	written.WallNS = 0
	return written, err
}

// readReasonedChoice reads one case's choice from what the model wrote under the policy.
//   - thinking on => the parser reads what was said after the thought closed
//   - thinking on and no thought closed => no choice (the answer ran out inside its thought)
//   - Raw = the text the parser read
func readReasonedChoice(testCase MultipleChoiceCase, answer choiceAnswer, policy reasoningPolicy, runtime reasoningRuntime) (choiceObservation, error) {
	parse, known := reasonedParsers[policy.Parser]
	if answer.Written == nil || !known {
		return choiceObservation{}, fmt.Errorf("evaluation: case %s was answered with written text %t and is read by parser %q; the reasoning protocol reads a written answer by one of %d parsers", testCase.Name, answer.Written != nil, policy.Parser, len(reasonedParsers))
	}
	observation := choiceObservation{Name: testCase.Name, Answer: testCase.Answer, Selected: unmatchedChoice}
	said, closed := answer.Written.Text, true
	if policy.Thinking {
		said, closed = runtime.AfterThought(said)
	}
	if !closed {
		return observation, nil
	}
	observation.Raw = said
	if index, matched := parse(said, chatChoiceLetters(testCase)); matched {
		observation.Selected = index
	}
	return observation, nil
}

// lastAnswerIs reads the letter of the last answerIsPhrase in text, in any case, that some dressing separates from a standalone option letter.
func lastAnswerIs(text string, letters []string) (int, bool) {
	selected, found := 0, false
	for at := 0; at+len(answerIsPhrase) <= len(text); at++ {
		if !strings.EqualFold(text[at:at+len(answerIsPhrase)], answerIsPhrase) {
			continue
		}
		after := text[at+len(answerIsPhrase):]
		rest := strings.TrimLeft(after, answerIsDressing)
		if len(rest) == len(after) {
			continue
		}
		letter, size := utf8.DecodeRuneInString(rest)
		if next, _ := utf8.DecodeRuneInString(rest[size:]); size == 0 || unicode.IsLetter(next) {
			continue
		}
		for index, candidate := range letters {
			if strings.TrimSpace(candidate) == string(letter) {
				selected, found = index, true
			}
		}
	}
	return selected, found
}

// lastChoiceLetter reads the last standalone option letter in text.
func lastChoiceLetter(text string, candidates []string) (int, bool) {
	runes := []rune(text)
	for position := len(runes) - 1; position >= 0; position-- {
		if index, matched := choiceLetterAt(runes, position, candidates); matched {
			return index, true
		}
	}
	return 0, false
}

// choiceContent is a digest of what a choice case asks, its prompt and its candidates, the same wherever the case is registered and under whatever name.
func choiceContent(testCase MultipleChoiceCase) (string, error) {
	asked, err := json.Marshal(askedChoice{Prompt: testCase.Prompt, Candidates: testCase.Candidates})
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", sha256.Sum256(asked)), nil
}

// exposeDevelopment publishes the cases a plan asks under the reasoning protocol as a development part of the plan's dataset, published as containing what each case asks.
//   - a case an exploratory protocol has asked = exposed development evidence
//   - a promotion part that holds what an exposed case asks is refused a confirmation (confirmOnPart) because its reading was not held out
func exposeDevelopment(ctx context.Context, repository artifact.Repository, plan Plan, cases []MultipleChoiceCase) error {
	records := make([]dataset.Record, len(cases))
	contents := make(map[string]string, len(cases))
	for index, testCase := range cases {
		content, err := choiceContent(testCase)
		if err != nil {
			return err
		}
		records[index], contents[testCase.Name] = dataset.Record{ID: testCase.Name, Group: testCase.Name}, content
	}
	membership, err := dataset.NewMembership(plan.body.Dataset, 0, PoolParts[poolDevelopment], records)
	if err != nil {
		return err
	}
	content, err := membership.Content()
	if err != nil {
		return err
	}
	batch := artifact.Batch{Key: developmentKey + membership.ID.String(), Contents: []artifact.Content{content}, Lineage: []artifact.Lineage{membership.Lineage()}}
	if err := contain(&batch, membership, contents); err != nil {
		return err
	}
	if _, err := artifact.Publish(ctx, repository, batch); err != nil {
		return errors.Join(fmt.Errorf("evaluation: the %d cases asked under the %s protocol were not published as development evidence, so none was asked", len(cases), PromptingReasoning.Label()), err)
	}
	return nil
}
