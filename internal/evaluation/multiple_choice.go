package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
)

const (
	MultipleChoiceKind         = "multiple-choice"
	AggregationAccuracy        = "accuracy"
	multipleChoiceMediaType    = "application/vnd.overgo.multiple-choice-dataset+json"
	multipleChoiceSchema       = "overgo/multiple-choice-dataset/v1"
	multipleChoiceSplitMedia   = "application/vnd.overgo.multiple-choice-split+json"
	multipleChoiceSplitSchema  = "overgo/multiple-choice-split/v1"
	multipleChoiceReportMedia  = "application/vnd.overgo.multiple-choice-report+json"
	multipleChoiceReportSchema = "overgo/multiple-choice-report/v1"
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
	// Raw is the generated answer text when the case was answered by
	// generation through the chat template; likelihood scoring leaves
	// it empty and Selected below zero means no candidate letter was
	// found in it.
	Raw string `json:"raw,omitzero"`
}

type MultipleChoiceReport struct {
	ID           artifact.ID         `json:"-"`
	Version      uint16              `json:"version"`
	Plan         artifact.ID         `json:"plan"`
	Dataset      artifact.ID         `json:"dataset"`
	Observations []ChoiceObservation `json:"observations"`
	Accuracy     float64             `json:"accuracy"`
}

type AccuracyGroup struct {
	Name     string  `json:"name"`
	Correct  uint64  `json:"correct"`
	Total    uint64  `json:"total"`
	Accuracy float64 `json:"accuracy"`
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
		if err := validateMultipleChoiceCase(*testCase); err != nil {
			return MultipleChoicePlan{}, err
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return MultipleChoicePlan{}, errors.New("evaluation: duplicate multiple-choice case")
		}
		names[testCase.Name] = struct{}{}
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

func validateMultipleChoiceCase(testCase MultipleChoiceCase) error {
	if strings.TrimSpace(testCase.Name) == "" || testCase.Prompt == "" || len(testCase.Candidates) < 2 ||
		testCase.Answer < 0 || testCase.Answer >= len(testCase.Candidates) {
		return errors.New("evaluation: invalid multiple-choice case")
	}
	candidates := make(map[string]struct{}, len(testCase.Candidates))
	for _, candidate := range testCase.Candidates {
		if candidate == "" {
			return errors.New("evaluation: empty multiple-choice candidate")
		}
		if _, duplicate := candidates[candidate]; duplicate {
			return errors.New("evaluation: duplicate multiple-choice candidate")
		}
		candidates[candidate] = struct{}{}
	}
	return nil
}

func BindMultipleChoice(compiled MultipleChoicePlan, authorities ExactAuthorities) (Plan, error) {
	scorer := struct {
		Version       uint16                      `json:"version"`
		Kind          string                      `json:"kind"`
		Normalization sequencescore.Normalization `json:"normalization"`
		Aggregation   string                      `json:"aggregation"`
		// Method is empty for likelihood scoring (every earlier plan)
		// and names the generated-letter method under the chat-template
		// protocol, so the two are distinct scorer authorities.
		Method string `json:"method,omitzero"`
	}{
		Version: artifact.InitialDocumentVersion, Kind: MultipleChoiceKind,
		Normalization: compiled.suite.Normalization, Aggregation: compiled.suite.Aggregation,
	}
	if authorities.Execution.Prompting == PromptingChatTemplate {
		scorer.Method = chatChoiceMethod
	}
	return bindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, scorer, authorities)
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
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
	}
	report.Observations, report.Accuracy, err = scoreMultipleChoice(ctx, scorer, compiled.suite)
	if err != nil {
		return MultipleChoiceReport{}, err
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

func scoreMultipleChoice(
	ctx context.Context,
	scorer ContinuationScorer,
	suite MultipleChoiceSuite,
) ([]ChoiceObservation, float64, error) {
	observations := make([]ChoiceObservation, len(suite.Cases))
	correct := 0
	progress := trackProgress(ctx, suite.Source, len(suite.Cases))
	chat, generative := scorer.(ChatChoiceRuntime)
	for index, testCase := range suite.Cases {
		if generative {
			observation, err := answerChoiceByGeneration(ctx, chat, testCase)
			if err != nil {
				return nil, 0, err
			}
			observations[index] = observation
		} else {
			scores, err := scorer.ScoreContinuations(ctx, testCase.Prompt, testCase.Candidates)
			if err != nil {
				return nil, 0, err
			}
			selection, err := sequencescore.Select(scores, suite.Normalization)
			if err != nil {
				return nil, 0, err
			}
			observations[index] = ChoiceObservation{
				Name: testCase.Name, Values: selection.Values, Selected: selection.Index,
				Answer: testCase.Answer, Tied: selection.Tied,
			}
		}
		hit := !observations[index].Tied && observations[index].Selected == testCase.Answer
		if hit {
			correct++
		}
		progress.hit(hit)
	}
	accuracy := float64(correct) / float64(len(observations))
	if math.IsNaN(accuracy) || math.IsInf(accuracy, 0) {
		return nil, 0, errors.New("evaluation: multiple-choice accuracy is not finite")
	}
	return observations, accuracy, nil
}

func aggregateChoiceAccuracy(groups []string, observations []ChoiceObservation) ([]AccuracyGroup, error) {
	if len(groups) != len(observations) {
		return nil, errors.New("evaluation: choice groups differ from observations")
	}
	correct := make([]bool, len(observations))
	for index, observation := range observations {
		correct[index] = !observation.Tied && observation.Selected == observation.Answer
	}
	return aggregateAccuracy(groups, correct)
}

func aggregateAccuracy(groups []string, correct []bool) ([]AccuracyGroup, error) {
	if len(groups) != len(correct) {
		return nil, errors.New("evaluation: groups differ from outcomes")
	}
	byName := make(map[string]*AccuracyGroup)
	for index, group := range groups {
		metric := byName[group]
		if metric == nil {
			metric = &AccuracyGroup{Name: group}
			byName[group] = metric
		}
		metric.Total++
		if correct[index] {
			metric.Correct++
		}
	}
	result := make([]AccuracyGroup, 0, len(byName))
	for _, metric := range byName {
		metric.Accuracy = float64(metric.Correct) / float64(metric.Total)
		result = append(result, *metric)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result, nil
}

func scoreChoiceGroups(
	ctx context.Context,
	scorer ContinuationScorer,
	suite MultipleChoiceSuite,
	groups []string,
) ([]ChoiceObservation, []AccuracyGroup, float64, error) {
	observations, accuracy, err := scoreMultipleChoice(ctx, scorer, suite)
	if err != nil {
		return nil, nil, 0, err
	}
	metrics, err := aggregateChoiceAccuracy(groups, observations)
	return observations, metrics, accuracy, err
}

func (p MultipleChoicePlan) contents() ([]artifact.Content, error) {
	return datasetContents(
		p.dataset, p.split, p.suite.Cases, multipleChoiceDatasetContract, multipleChoiceSplitContract,
	)
}
