// Package servingeval owns exact real-model generation evaluation.
package servingeval

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"overgo/internal/inference"
	"overgo/internal/sampling"
)

type Case struct {
	Name            string `json:"name"`
	Prompt          string `json:"prompt"`
	MaxTokens       int    `json:"max_tokens"`
	Text            string `json:"text"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
}

type Suite struct {
	Schema string `json:"schema"`
	Source string `json:"source"`
	Cases  []Case `json:"cases"`
}

type Result struct {
	Name            string `json:"name"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	Text            string `json:"text"`
	WallNS          uint64 `json:"wall_ns"`
}

func EvaluateExact(ctx context.Context, runner *inference.Runner, suite Suite) ([]Result, error) {
	if ctx == nil || runner == nil || len(suite.Cases) == 0 {
		return nil, errors.New("serving evaluation: incomplete suite")
	}
	results := make([]Result, len(suite.Cases))
	for index, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || testCase.MaxTokens <= 0 ||
			testCase.PromptTokens <= 0 || testCase.GeneratedTokens <= 0 {
			return nil, fmt.Errorf("serving evaluation: case %d is invalid", index)
		}
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			return nil, err
		}
		var generated strings.Builder
		promptTokens := 0
		started := time.Now()
		ids, _, err := runner.Generate(ctx, testCase.Prompt, inference.GenerateOptions{
			MaxNewTokens: testCase.MaxTokens,
			Sampler:      greedy,
			DeviceGreedy: true,
			OnToken: func(event inference.TokenEvent) error {
				generated.WriteString(event.Piece)
				return nil
			},
			OnPromptEvaluated: func(evaluation inference.PromptEvaluation) {
				promptTokens = evaluation.Tokens
			},
		})
		if err != nil {
			return nil, fmt.Errorf("serving evaluation: case %q: %w", testCase.Name, err)
		}
		result := Result{
			Name: testCase.Name, PromptTokens: promptTokens,
			GeneratedTokens: len(ids) - promptTokens, Text: generated.String(),
			WallNS: uint64(time.Since(started).Nanoseconds()),
		}
		if result.PromptTokens != testCase.PromptTokens ||
			result.GeneratedTokens != testCase.GeneratedTokens || result.Text != testCase.Text {
			return nil, fmt.Errorf(
				"serving evaluation: case %q got tokens=%d/%d text=%q; want %d/%d %q",
				testCase.Name, result.PromptTokens, result.GeneratedTokens, result.Text,
				testCase.PromptTokens, testCase.GeneratedTokens, testCase.Text,
			)
		}
		results[index] = result
	}
	return results, nil
}
