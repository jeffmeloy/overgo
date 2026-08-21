package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	GeneratedAnswerKind         = "generated-answer"
	TransformTrimSpace          = "trim-space"
	TransformLowercase          = "lowercase"
	generatedAnswerMediaType    = "application/vnd.overgo.generated-answer-dataset+json"
	generatedAnswerSchema       = "overgo/generated-answer-dataset/v1"
	generatedAnswerSplitMedia   = "application/vnd.overgo.generated-answer-split+json"
	generatedAnswerSplitSchema  = "overgo/generated-answer-split/v1"
	generatedAnswerReportMedia  = "application/vnd.overgo.generated-answer-report+json"
	generatedAnswerReportSchema = "overgo/generated-answer-report/v1"
)

var (
	generatedAnswerDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: generatedAnswerMediaType, Schema: generatedAnswerSchema,
	}
	generatedAnswerSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: generatedAnswerSplitMedia, Schema: generatedAnswerSplitSchema,
	}
	generatedAnswerReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: generatedAnswerReportMedia, Schema: generatedAnswerReportSchema,
	}
)

type GeneratedAnswerCase struct {
	Name      string   `json:"name"`
	Prompt    string   `json:"prompt"`
	MaxTokens int      `json:"max_tokens"`
	Answers   []string `json:"answers"`
}

type GeneratedAnswerSuite struct {
	Kind       string                `json:"kind"`
	Schema     string                `json:"schema"`
	Source     string                `json:"source"`
	Transforms []string              `json:"transforms"`
	Cases      []GeneratedAnswerCase `json:"cases"`
}

type GeneratedAnswerPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    GeneratedAnswerSuite
}

type GeneratedAnswerObservation struct {
	Name      string `json:"name"`
	Raw       string `json:"raw"`
	Scored    string `json:"scored"`
	Accepted  bool   `json:"accepted"`
	Prompt    int    `json:"prompt_tokens"`
	Generated int    `json:"generated_tokens"`
}

type GeneratedAnswerReport struct {
	ID           artifact.ID                  `json:"-"`
	Version      uint16                       `json:"version"`
	Plan         artifact.ID                  `json:"plan"`
	Dataset      artifact.ID                  `json:"dataset"`
	Observations []GeneratedAnswerObservation `json:"observations"`
	Accuracy     float64                      `json:"accuracy"`
}

func CompileGeneratedAnswer(suite GeneratedAnswerSuite) (GeneratedAnswerPlan, error) {
	if suite.Kind != GeneratedAnswerKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return GeneratedAnswerPlan{}, errors.New("evaluation: invalid generated-answer suite")
	}
	suite.Transforms = slices.Clone(suite.Transforms)
	seenTransforms := make(map[string]struct{}, len(suite.Transforms))
	for _, transform := range suite.Transforms {
		if transform != TransformTrimSpace && transform != TransformLowercase {
			return GeneratedAnswerPlan{}, errors.New("evaluation: unknown answer transform")
		}
		if _, duplicate := seenTransforms[transform]; duplicate {
			return GeneratedAnswerPlan{}, errors.New("evaluation: duplicate answer transform")
		}
		seenTransforms[transform] = struct{}{}
	}
	suite.Cases = slices.Clone(suite.Cases)
	names := make(map[string]struct{}, len(suite.Cases))
	for index := range suite.Cases {
		testCase := &suite.Cases[index]
		testCase.Answers = slices.Clone(testCase.Answers)
		if strings.TrimSpace(testCase.Name) == "" || testCase.Prompt == "" ||
			testCase.MaxTokens <= 0 || len(testCase.Answers) == 0 {
			return GeneratedAnswerPlan{}, errors.New("evaluation: invalid generated-answer case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return GeneratedAnswerPlan{}, errors.New("evaluation: duplicate generated-answer case")
		}
		names[testCase.Name] = struct{}{}
		for _, answer := range testCase.Answers {
			if answer == "" {
				return GeneratedAnswerPlan{}, errors.New("evaluation: empty generated answer")
			}
		}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return GeneratedAnswerPlan{}, err
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return GeneratedAnswerPlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: dataset})
	if err != nil {
		return GeneratedAnswerPlan{}, err
	}
	return GeneratedAnswerPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func BindGeneratedAnswer(compiled GeneratedAnswerPlan, authorities ExactAuthorities) (Plan, error) {
	scorer := struct {
		Version    uint16   `json:"version"`
		Kind       string   `json:"kind"`
		Transforms []string `json:"transforms"`
	}{Version: artifact.InitialDocumentVersion, Kind: GeneratedAnswerKind, Transforms: compiled.suite.Transforms}
	return bindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, scorer, authorities)
}

func EvaluateGeneratedAnswer(
	ctx context.Context,
	repository artifact.Repository,
	generator Generator,
	compiled GeneratedAnswerPlan,
	plan Plan,
) (GeneratedAnswerReport, error) {
	if ctx == nil || repository == nil || generator == nil || compiled.identity != plan.body.CaseProfile {
		return GeneratedAnswerReport{}, errors.New("evaluation: generated-answer authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases,
		generatedAnswerDatasetContract, generatedAnswerSplitContract,
	)
	if err != nil {
		return GeneratedAnswerReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return GeneratedAnswerReport{}, err
	}
	report := GeneratedAnswerReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]GeneratedAnswerObservation, len(compiled.suite.Cases)),
	}
	correct := 0
	for index, testCase := range compiled.suite.Cases {
		result, err := generateText(ctx, generator, testCase.Name, testCase.Prompt, testCase.MaxTokens)
		if err != nil {
			return GeneratedAnswerReport{}, err
		}
		scored := transformAnswer(result.Text, compiled.suite.Transforms)
		accepted := false
		for _, answer := range testCase.Answers {
			if scored == transformAnswer(answer, compiled.suite.Transforms) {
				accepted = true
				break
			}
		}
		if accepted {
			correct++
		}
		report.Observations[index] = GeneratedAnswerObservation{
			Name: testCase.Name, Raw: result.Text, Scored: scored, Accepted: accepted,
			Prompt: result.PromptTokens, Generated: result.GeneratedTokens,
		}
	}
	report.Accuracy = float64(correct) / float64(len(report.Observations))
	if math.IsNaN(report.Accuracy) || math.IsInf(report.Accuracy, 0) {
		return GeneratedAnswerReport{}, errors.New("evaluation: generated-answer accuracy is not finite")
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return GeneratedAnswerReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, generatedAnswerReportContract, report,
	); err != nil {
		return GeneratedAnswerReport{}, err
	}
	return report, nil
}

func transformAnswer(value string, transforms []string) string {
	for _, transform := range transforms {
		switch transform {
		case TransformTrimSpace:
			value = strings.TrimSpace(value)
		case TransformLowercase:
			value = strings.ToLower(value)
		}
	}
	return value
}
