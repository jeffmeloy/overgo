package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
)

const (
	GeneratedAnswerKind = "generated-answer"
	TransformTrimSpace  = "trim-space"
	// transformLeadingLetter reads an answer as the one letter it opens with.
	transformLeadingLetter      = "leading-letter"
	transformLowercase          = "lowercase"
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

type generatedAnswerPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    GeneratedAnswerSuite
}

type generatedAnswerObservation struct {
	Name      string `json:"name"`
	Raw       string `json:"raw"`
	Scored    string `json:"scored"`
	Accepted  bool   `json:"accepted"`
	Prompt    int    `json:"prompt_tokens"`
	Generated int    `json:"generated_tokens"`
}

type generatedAnswerReport struct {
	ID           artifact.ID                  `json:"-"`
	Version      uint16                       `json:"version"`
	Plan         artifact.ID                  `json:"plan"`
	Dataset      artifact.ID                  `json:"dataset"`
	Observations []generatedAnswerObservation `json:"observations"`
	Accuracy     float64                      `json:"accuracy"`
}

func CompileGeneratedAnswer(suite GeneratedAnswerSuite) (generatedAnswerPlan, error) {
	if suite.Kind != GeneratedAnswerKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return generatedAnswerPlan{}, errors.New("evaluation: invalid generated-answer suite")
	}
	suite.Transforms = slices.Clone(suite.Transforms)
	seenTransforms := make(map[string]struct{}, len(suite.Transforms))
	for _, transform := range suite.Transforms {
		if transform != TransformTrimSpace && transform != transformLowercase && transform != transformLeadingLetter {
			return generatedAnswerPlan{}, errors.New("evaluation: unknown answer transform")
		}
		if _, duplicate := seenTransforms[transform]; duplicate {
			return generatedAnswerPlan{}, errors.New("evaluation: duplicate answer transform")
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
			return generatedAnswerPlan{}, errors.New("evaluation: invalid generated-answer case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return generatedAnswerPlan{}, errors.New("evaluation: duplicate generated-answer case")
		}
		names[testCase.Name] = struct{}{}
		if slices.Contains(testCase.Answers, "") {
			return generatedAnswerPlan{}, errors.New("evaluation: empty generated answer")
		}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return generatedAnswerPlan{}, err
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return generatedAnswerPlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: dataset})
	if err != nil {
		return generatedAnswerPlan{}, err
	}
	return generatedAnswerPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func bindGeneratedAnswer(compiled generatedAnswerPlan, authorities ExactAuthorities) (Plan, error) {
	scorer := struct {
		Version    uint16   `json:"version"`
		Kind       string   `json:"kind"`
		Transforms []string `json:"transforms"`
	}{Version: artifact.InitialDocumentVersion, Kind: GeneratedAnswerKind, Transforms: compiled.suite.Transforms}
	return bindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, scorer, authorities)
}

func evaluateGeneratedAnswer(
	ctx context.Context,
	repository artifact.Repository,
	generator Generator,
	compiled generatedAnswerPlan,
	plan Plan,
) (generatedAnswerReport, error) {
	if ctx == nil || repository == nil || generator == nil || compiled.identity != plan.body.CaseProfile {
		return generatedAnswerReport{}, errors.New("evaluation: generated-answer authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases,
		generatedAnswerDatasetContract, generatedAnswerSplitContract,
	)
	if err != nil {
		return generatedAnswerReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return generatedAnswerReport{}, err
	}
	report := generatedAnswerReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]generatedAnswerObservation, len(compiled.suite.Cases)),
	}
	correct := 0
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for index, testCase := range compiled.suite.Cases {
		result, err := recordInstruction(ctx, generator, testCase.Name, testCase.Prompt, testCase.MaxTokens)
		if err != nil {
			return generatedAnswerReport{}, err
		}
		scored := normalizeAnswer(result.Text, compiled.suite.Transforms)
		accepted := false
		for _, answer := range testCase.Answers {
			if scored == normalizeAnswer(answer, compiled.suite.Transforms) {
				accepted = true
				break
			}
		}
		if accepted {
			correct++
		}
		report.Observations[index] = generatedAnswerObservation{
			Name: testCase.Name, Raw: result.Text, Scored: scored, Accepted: accepted,
			Prompt: result.PromptTokens, Generated: result.GeneratedTokens,
		}
		progress.hit(accepted)
	}
	report.Accuracy = float64(correct) / float64(len(report.Observations))
	if math.IsNaN(report.Accuracy) || math.IsInf(report.Accuracy, 0) {
		return generatedAnswerReport{}, errors.New("evaluation: generated-answer accuracy is not finite")
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return generatedAnswerReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, generatedAnswerReportContract, report,
	); err != nil {
		return generatedAnswerReport{}, err
	}
	return report, nil
}

// normalizeAnswer applies the ordered normalization operations both
// answer scorers share; the wire names are identical across suites.
func normalizeAnswer[T ~string](value string, operations []T) string {
	for _, operation := range operations {
		switch string(operation) {
		case TransformTrimSpace:
			value = strings.TrimSpace(value)
		case transformLowercase:
			value = strings.ToLower(value)
		case transformLeadingLetter:
			value = leadingLetter(value)
		}
	}
	return value
}

// leadingLetter is the single letter an answer opens with, as "B" opens "B.",
// "B)" and "B because ...", and the answer itself when it opens with a word.
func leadingLetter(value string) string {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) == 0 || !unicode.IsLetter(runes[0]) || len(runes) > 1 && unicode.IsLetter(runes[1]) {
		return value
	}
	return string(runes[0])
}
