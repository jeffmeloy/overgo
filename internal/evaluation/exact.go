// Package evaluation owns compiled native evaluation.
package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

type ExactCase struct {
	Name            string `json:"name"`
	Prompt          string `json:"prompt"`
	MaxTokens       int    `json:"max_tokens"`
	Text            string `json:"text"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
}

type ExactSuite struct {
	Schema string      `json:"schema"`
	Source string      `json:"source"`
	Cases  []ExactCase `json:"cases"`
}

type ExactPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    ExactSuite
}

type ExactResult struct {
	Name            string `json:"name"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	Text            string `json:"text"`
	WallNS          uint64 `json:"wall_ns"`
}

type Generator interface {
	Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error)
}

func CompileExact(suite ExactSuite) (ExactPlan, error) {
	if strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return ExactPlan{}, errors.New("evaluation: incomplete exact suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	for index, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || testCase.MaxTokens <= 0 ||
			testCase.PromptTokens <= 0 || testCase.GeneratedTokens <= 0 {
			return ExactPlan{}, fmt.Errorf("evaluation: exact case %d is invalid", index)
		}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return ExactPlan{}, err
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return ExactPlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: dataset})
	if err != nil {
		return ExactPlan{}, err
	}
	return ExactPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func (p ExactPlan) Identity() artifact.ID { return p.identity }

func EvaluateExact(ctx context.Context, generator Generator, plan ExactPlan, observe func(ExactResult) error) error {
	if ctx == nil || generator == nil || !plan.identity.Valid() || len(plan.suite.Cases) == 0 {
		return errors.New("evaluation: incomplete exact plan")
	}
	for _, testCase := range plan.suite.Cases {
		result, err := evaluateExactCase(ctx, generator, testCase)
		if err != nil {
			return err
		}
		if observe != nil {
			if err := observe(result); err != nil {
				return fmt.Errorf("evaluation: exact case %q observation: %w", testCase.Name, err)
			}
		}
	}
	return nil
}

func evaluateExactCase(ctx context.Context, generator Generator, testCase ExactCase) (ExactResult, error) {
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		return ExactResult{}, err
	}
	var generated strings.Builder
	promptTokens := 0
	started := time.Now()
	ids, _, err := generator.Generate(ctx, testCase.Prompt, inference.GenerateOptions{
		MaxNewTokens: testCase.MaxTokens,
		Sampler:      greedy,
		DeviceGreedy: true,
		OnToken: func(event inference.TokenEvent) error {
			generated.WriteString(event.Piece)
			return nil
		},
		OnPromptEvaluated: func(result inference.PromptEvaluation) {
			promptTokens = result.Tokens
		},
	})
	if err != nil {
		return ExactResult{}, fmt.Errorf("evaluation: exact case %q: %w", testCase.Name, err)
	}
	result := ExactResult{
		Name: testCase.Name, PromptTokens: promptTokens,
		GeneratedTokens: len(ids) - promptTokens, Text: generated.String(),
		WallNS: uint64(time.Since(started).Nanoseconds()),
	}
	if result.PromptTokens != testCase.PromptTokens ||
		result.GeneratedTokens != testCase.GeneratedTokens || result.Text != testCase.Text {
		return ExactResult{}, fmt.Errorf(
			"evaluation: exact case %q got tokens=%d/%d text=%q; want %d/%d %q",
			testCase.Name, result.PromptTokens, result.GeneratedTokens, result.Text,
			testCase.PromptTokens, testCase.GeneratedTokens, testCase.Text,
		)
	}
	return result, nil
}
