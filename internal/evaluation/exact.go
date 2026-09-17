// Package evaluation owns compiled native evaluation.
package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/processmeasure"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

type ExactCase struct {
	Name            string  `json:"name"`
	Prompt          string  `json:"prompt"`
	MaxTokens       int     `json:"max_tokens"`
	Text            string  `json:"text"`
	PromptTokens    int     `json:"prompt_tokens"`
	GeneratedTokens int     `json:"generated_tokens"`
	DecodeMedianMS  float64 `json:"decode_median_ms,omitzero"`
	DecodeTokPerSec float64 `json:"decode_tok_per_sec,omitzero"`
	ServingPath     string  `json:"serving_path,omitzero"`
	DecodeMode      string  `json:"decode_mode,omitzero"`
}

type ExactSuite struct {
	Schema   string      `json:"schema"`
	Source   string      `json:"source"`
	ModelDir string      `json:"model_dir,omitzero"`
	Date     string      `json:"date,omitzero"`
	Cases    []ExactCase `json:"cases"`
}

type ExactPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    ExactSuite
}

const (
	exactDatasetMediaType = "application/vnd.overgo.exact-dataset+json"
	exactDatasetSchema    = "overgo/exact-dataset/v1"
	exactSplitMediaType   = "application/vnd.overgo.exact-split+json"
	exactSplitSchema      = "overgo/exact-split/v1"
)

var (
	exactDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: exactDatasetMediaType, Schema: exactDatasetSchema,
	}
	exactSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: exactSplitMediaType, Schema: exactSplitSchema,
	}
)

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

func (p ExactPlan) contents() ([]artifact.Content, error) {
	return datasetContents(p.dataset, p.split, p.suite.Cases, exactDatasetContract, exactSplitContract)
}

// evaluateExactCase is the one exact-case evaluator; the sharded campaign
// (EvaluateExactSharded) is its only driver — the unsharded loop it
// displaced is deleted, not kept as a second authority.
func evaluateExactCase(ctx context.Context, generator Generator, testCase ExactCase) (ExactResult, error) {
	result, err := Record(ctx, generator, testCase.Name, testCase.Prompt, testCase.MaxTokens)
	if err != nil {
		return ExactResult{}, err
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

// Record generates one case greedily and returns its complete text and token
// counts. Exact replay and model intake share this execution path.
func Record(
	ctx context.Context,
	generator Generator,
	name, prompt string,
	maxTokens int,
) (ExactResult, error) {
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		return ExactResult{}, err
	}
	var generated strings.Builder
	promptTokens := 0
	started := processmeasure.NewStopwatch()
	ids, _, err := generator.Generate(ctx, prompt, inference.GenerateOptions{
		MaxNewTokens: maxTokens,
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
		return ExactResult{}, fmt.Errorf("evaluation: generated case %q: %w", name, err)
	}
	wall, err := started.Elapsed()
	if err != nil {
		return ExactResult{}, err
	}
	result := ExactResult{
		Name: name, PromptTokens: promptTokens,
		GeneratedTokens: len(ids) - promptTokens, Text: generated.String(),
		WallNS: wall,
	}
	return result, nil
}

// recordInstruction applies declared conversation framing once before recording
// generated instruction answers. Raw generators retain their original prompt.
func recordInstruction(ctx context.Context, generator Generator, name, prompt string, maxTokens int) (ExactResult, error) {
	if chat, ok := generator.(ChatChoiceRuntime); ok {
		framed, err := chat.ShapeChatPrompt(prompt)
		if err != nil {
			return ExactResult{}, err
		}
		prompt = framed
	}
	return Record(ctx, generator, name, prompt, maxTokens)
}
