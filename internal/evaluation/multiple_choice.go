package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
)

const (
	MultipleChoiceKind          = "multiple-choice"
	AggregationAccuracy         = "accuracy"
	multipleChoiceMediaType     = "application/vnd.overgo.multiple-choice-dataset+json"
	multipleChoiceSchema        = "overgo/multiple-choice-dataset/v1"
	multipleChoiceSplitMedia    = "application/vnd.overgo.multiple-choice-split+json"
	multipleChoiceSplitSchema   = "overgo/multiple-choice-split/v1"
	multipleChoiceReportMedia   = "application/vnd.overgo.multiple-choice-report+json"
	multipleChoiceReportSchema  = "overgo/multiple-choice-report/v1"
	multipleChoiceReportVersion = 1
)

var (
	multipleChoiceDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: multipleChoiceMediaType, Schema: multipleChoiceSchema,
	}
	multipleChoiceSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: multipleChoiceSplitMedia, Schema: multipleChoiceSplitSchema,
	}
	multipleChoiceReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: multipleChoiceReportMedia, Schema: multipleChoiceReportSchema,
	}
)

type MultipleChoiceCase struct {
	Name       string   `json:"name"`
	Prompt     string   `json:"prompt"`
	Candidates []string `json:"candidates"`
	Answer     int      `json:"answer"`
}

type MultipleChoiceSuite struct {
	Kind          string                      `json:"kind"`
	Schema        string                      `json:"schema"`
	Source        string                      `json:"source"`
	Normalization sequencescore.Normalization `json:"normalization"`
	Aggregation   string                      `json:"aggregation"`
	Cases         []MultipleChoiceCase        `json:"cases"`
}

type MultipleChoicePlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    MultipleChoiceSuite
}

type ChoiceObservation struct {
	Name     string    `json:"name"`
	Values   []float64 `json:"values"`
	Selected int       `json:"selected"`
	Answer   int       `json:"answer"`
	Tied     bool      `json:"tied"`
}

type MultipleChoiceReport struct {
	ID           artifact.ID         `json:"-"`
	Version      uint16              `json:"version"`
	Plan         artifact.ID         `json:"plan"`
	Dataset      artifact.ID         `json:"dataset"`
	Observations []ChoiceObservation `json:"observations"`
	Accuracy     float64             `json:"accuracy"`
}

type ContinuationScorer interface {
	ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error)
}

func CompileMultipleChoice(suite MultipleChoiceSuite) (MultipleChoicePlan, error) {
	if suite.Kind != MultipleChoiceKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || suite.Aggregation != AggregationAccuracy ||
		len(suite.Cases) == 0 || suite.Normalization != sequencescore.NormalizationSum &&
		suite.Normalization != sequencescore.NormalizationMean {
		return MultipleChoicePlan{}, errors.New("evaluation: invalid multiple-choice suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	names := make(map[string]struct{}, len(suite.Cases))
	for index := range suite.Cases {
		testCase := &suite.Cases[index]
		testCase.Candidates = slices.Clone(testCase.Candidates)
		if strings.TrimSpace(testCase.Name) == "" || testCase.Prompt == "" || len(testCase.Candidates) < 2 ||
			testCase.Answer < 0 || testCase.Answer >= len(testCase.Candidates) {
			return MultipleChoicePlan{}, errors.New("evaluation: invalid multiple-choice case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return MultipleChoicePlan{}, errors.New("evaluation: duplicate multiple-choice case")
		}
		names[testCase.Name] = struct{}{}
		candidates := make(map[string]struct{}, len(testCase.Candidates))
		for _, candidate := range testCase.Candidates {
			if candidate == "" {
				return MultipleChoicePlan{}, errors.New("evaluation: empty multiple-choice candidate")
			}
			if _, duplicate := candidates[candidate]; duplicate {
				return MultipleChoicePlan{}, errors.New("evaluation: duplicate multiple-choice candidate")
			}
			candidates[candidate] = struct{}{}
		}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return MultipleChoicePlan{}, err
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return MultipleChoicePlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: dataset})
	if err != nil {
		return MultipleChoicePlan{}, err
	}
	return MultipleChoicePlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func BindMultipleChoice(compiled MultipleChoicePlan, authorities ExactAuthorities) (Plan, error) {
	scorer, err := artifact.JSONID(artifact.KindProfile, struct {
		Version       uint16                      `json:"version"`
		Kind          string                      `json:"kind"`
		Normalization sequencescore.Normalization `json:"normalization"`
		Aggregation   string                      `json:"aggregation"`
	}{
		Version: evaluationPlanVersion, Kind: MultipleChoiceKind,
		Normalization: compiled.suite.Normalization, Aggregation: compiled.suite.Aggregation,
	})
	if err != nil {
		return Plan{}, err
	}
	return bindPlan(compiled.dataset, compiled.split, compiled.identity, scorer, authorities)
}

func EvaluateMultipleChoice(
	ctx context.Context,
	repository artifact.Repository,
	scorer ContinuationScorer,
	compiled MultipleChoicePlan,
	plan Plan,
) (MultipleChoiceReport, error) {
	if ctx == nil || repository == nil || scorer == nil || compiled.identity != plan.body.CaseProfile {
		return MultipleChoiceReport{}, errors.New("evaluation: multiple-choice authority differs")
	}
	contents, err := compiled.contents()
	if err != nil {
		return MultipleChoiceReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return MultipleChoiceReport{}, err
	}
	report := MultipleChoiceReport{
		Version: multipleChoiceReportVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]ChoiceObservation, len(compiled.suite.Cases)),
	}
	correct := 0
	for index, testCase := range compiled.suite.Cases {
		scores, err := scorer.ScoreContinuations(ctx, testCase.Prompt, testCase.Candidates)
		if err != nil {
			return MultipleChoiceReport{}, err
		}
		selection, err := sequencescore.Select(scores, compiled.suite.Normalization)
		if err != nil {
			return MultipleChoiceReport{}, err
		}
		report.Observations[index] = ChoiceObservation{
			Name: testCase.Name, Values: selection.Values, Selected: selection.Index,
			Answer: testCase.Answer, Tied: selection.Tied,
		}
		if !selection.Tied && selection.Index == testCase.Answer {
			correct++
		}
	}
	report.Accuracy = float64(correct) / float64(len(report.Observations))
	if math.IsNaN(report.Accuracy) || math.IsInf(report.Accuracy, 0) {
		return MultipleChoiceReport{}, errors.New("evaluation: multiple-choice accuracy is not finite")
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return MultipleChoiceReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, multipleChoiceReportContract, report,
	); err != nil {
		return MultipleChoiceReport{}, err
	}
	return report, nil
}

func (p MultipleChoicePlan) contents() ([]artifact.Content, error) {
	return datasetContents(
		p.dataset, p.split, p.suite.Cases, multipleChoiceDatasetContract, multipleChoiceSplitContract,
	)
}
