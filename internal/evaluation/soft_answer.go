package evaluation

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/inference"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// SoftMetric names the share a suite of choices scores beside its accuracy: the mean, over what its cases ask about,
// of the geometric mean of the judge's probability of each case's right answer (Soft-TIFA, GenEval 2, arXiv
// 2512.16853). A binary share at its ceiling cannot fall when a picture gets worse; this one falls as the judge grows
// less sure.
const SoftMetric = "soft-accuracy"

// choiceReader is a runtime that names the one token each choice is generated as, so the logits of a generation's
// first token give the probability of each.
type choiceReader interface {
	ChoiceTokens(choices []string) ([]tokenizer.TokenID, error)
}

// choices says the suite's cases are choices among answers, so it scores its soft share beside its accuracy.
func (suite GeneratedAnswerSuite) choices() bool {
	return len(suite.Cases) > 0 && len(suite.Cases[0].Choices) > 0
}

// generatedAnswerMetrics are a generated-answer suite's metrics: its accuracy, and a suite of choices' soft share.
func generatedAnswerMetrics(suite GeneratedAnswerSuite, accuracy, soft float64) []runrecord.Metric {
	metrics := accuracyMetrics(accuracy, nil)
	if suite.choices() {
		metrics = append(metrics, runrecord.Metric{Name: SoftMetric, Value: soft, Direction: runrecord.DirectionMaximize})
	}
	return metrics
}

// rightChoice is the place among a case's choices of the one its answers accept, read under the suite's transforms;
// -1 where none or more than one is.
func rightChoice(testCase GeneratedAnswerCase, transforms []string) int {
	right := -1
	for index, choice := range testCase.Choices {
		if slices.ContainsFunc(testCase.Answers, func(answer string) bool {
			return normalizeAnswer(answer, transforms) == normalizeAnswer(choice, transforms)
		}) {
			if right >= 0 {
				return -1
			}
			right = index
		}
	}
	return right
}

// recordChoice generates a case whose answer is one of its choices as recordInstruction does, on the host's sampler so
// the logits of its first token reach the evaluation, and returns with it the probability of the right choice: the
// softmax of those logits over the choices' tokens.
func recordChoice(ctx context.Context, generator Generator, testCase GeneratedAnswerCase, right int) (ExactResult, float64, error) {
	reader, reads := generator.(choiceReader)
	if !reads {
		return ExactResult{}, 0, fmt.Errorf("evaluation: case %q is a choice, and the runtime names no token of its choices", testCase.Name)
	}
	tokens, err := reader.ChoiceTokens(testCase.Choices)
	if err != nil {
		return ExactResult{}, 0, err
	}
	prompt, err := shapeInstruction(generator, testCase.Prompt)
	if err != nil {
		return ExactResult{}, 0, err
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		return ExactResult{}, 0, err
	}
	var first []float32
	read := false
	result, err := record(ctx, generator, testCase.Name, prompt, inference.GenerateOptions{MaxNewTokens: testCase.MaxTokens, Sampler: greedy,
		OnToken: func(event inference.TokenEvent) error {
			if !read {
				first, read = slices.Clone(event.Logits), true
			}
			return nil
		}})
	if err != nil {
		return ExactResult{}, 0, err
	}
	weights := make([]float64, len(tokens))
	var total float64
	for index, token := range tokens {
		if int(token) >= len(first) {
			return ExactResult{}, 0, fmt.Errorf("evaluation: case %q's first token reached the evaluation with no logit of choice %q", testCase.Name, testCase.Choices[index])
		}
		weights[index] = math.Exp(float64(first[token]))
		total += weights[index]
	}
	probability := weights[right] / total
	if math.IsNaN(probability) || math.IsInf(probability, 0) {
		return ExactResult{}, 0, errors.New("evaluation: a choice's probability is not finite")
	}
	return result, probability, nil
}

// softShare is the mean, over what the cases ask about, of the geometric mean of the probabilities of the right
// answers of the cases that ask about it.
func softShare(cases []GeneratedAnswerCase, probabilities []float64) float64 {
	logs := map[string][]float64{}
	var groups []string
	for index, testCase := range cases {
		group := cmp.Or(testCase.Group, testCase.Name)
		if _, held := logs[group]; !held {
			groups = append(groups, group)
		}
		logs[group] = append(logs[group], math.Log(probabilities[index]))
	}
	var total float64
	for _, group := range groups {
		var sum float64
		for _, value := range logs[group] {
			sum += value
		}
		total += math.Exp(sum / float64(len(logs[group])))
	}
	return total / float64(len(groups))
}
