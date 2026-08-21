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
	ProbabilityMassKind          = "probability-mass-choice"
	probabilityMassDatasetMedia  = "application/vnd.overgo.probability-mass-dataset+json"
	probabilityMassDatasetSchema = "overgo/probability-mass-dataset/v1"
	probabilityMassSplitMedia    = "application/vnd.overgo.probability-mass-split+json"
	probabilityMassSplitSchema   = "overgo/probability-mass-split/v1"
	probabilityMassReportMedia   = "application/vnd.overgo.probability-mass-report+json"
	probabilityMassReportSchema  = "overgo/probability-mass-report/v1"
)

var (
	probabilityMassDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: probabilityMassDatasetMedia, Schema: probabilityMassDatasetSchema,
	}
	probabilityMassSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: probabilityMassSplitMedia, Schema: probabilityMassSplitSchema,
	}
	probabilityMassReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: probabilityMassReportMedia, Schema: probabilityMassReportSchema,
	}
)

type ProbabilityMassCase struct {
	Name       string   `json:"name"`
	Prompt     string   `json:"prompt"`
	Candidates []string `json:"candidates"`
	Positive   []bool   `json:"positive"`
}

type ProbabilityMassSuite struct {
	Kind          string                      `json:"kind"`
	Schema        string                      `json:"schema"`
	Source        string                      `json:"source"`
	Normalization sequencescore.Normalization `json:"normalization"`
	Cases         []ProbabilityMassCase       `json:"cases"`
}

type ProbabilityMassPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    ProbabilityMassSuite
}

type ProbabilityMassObservation struct {
	Name   string    `json:"name"`
	Values []float64 `json:"values"`
	Mass   float64   `json:"mass"`
}

type ProbabilityMassReport struct {
	ID           artifact.ID                  `json:"-"`
	Version      uint16                       `json:"version"`
	Plan         artifact.ID                  `json:"plan"`
	Dataset      artifact.ID                  `json:"dataset"`
	Observations []ProbabilityMassObservation `json:"observations"`
	Mean         float64                      `json:"mean"`
}

func CompileProbabilityMass(suite ProbabilityMassSuite) (ProbabilityMassPlan, error) {
	if suite.Kind != ProbabilityMassKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		len(suite.Cases) == 0 || suite.Normalization != sequencescore.NormalizationSum && suite.Normalization != sequencescore.NormalizationMean {
		return ProbabilityMassPlan{}, errors.New("evaluation: invalid probability-mass suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	names := make(map[string]struct{}, len(suite.Cases))
	for index := range suite.Cases {
		testCase := &suite.Cases[index]
		testCase.Candidates = slices.Clone(testCase.Candidates)
		testCase.Positive = slices.Clone(testCase.Positive)
		probe := MultipleChoiceCase{Name: testCase.Name, Prompt: testCase.Prompt, Candidates: testCase.Candidates, Answer: 0}
		if err := validateMultipleChoiceCase(probe); err != nil || len(testCase.Positive) != len(testCase.Candidates) {
			return ProbabilityMassPlan{}, errors.New("evaluation: invalid probability-mass case")
		}
		positive := 0
		for _, value := range testCase.Positive {
			if value {
				positive++
			}
		}
		if positive == 0 || positive == len(testCase.Positive) {
			return ProbabilityMassPlan{}, errors.New("evaluation: probability-mass classes are incomplete")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return ProbabilityMassPlan{}, errors.New("evaluation: duplicate probability-mass case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return ProbabilityMassPlan{}, err
	}
	datasetID, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return ProbabilityMassPlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: datasetID})
	if err != nil {
		return ProbabilityMassPlan{}, err
	}
	return ProbabilityMassPlan{identity: identity, dataset: datasetID, split: split, suite: suite}, nil
}

func BindProbabilityMass(compiled ProbabilityMassPlan, authorities ExactAuthorities) (Plan, error) {
	scorer, err := artifact.JSONID(artifact.KindProfile, struct {
		Version       uint16                      `json:"version"`
		Kind          string                      `json:"kind"`
		Normalization sequencescore.Normalization `json:"normalization"`
	}{Version: artifact.InitialDocumentVersion, Kind: ProbabilityMassKind, Normalization: compiled.suite.Normalization})
	if err != nil {
		return Plan{}, err
	}
	return bindPlan(compiled.dataset, compiled.split, compiled.identity, scorer, authorities)
}

func EvaluateProbabilityMass(
	ctx context.Context,
	repository artifact.Repository,
	scorer ContinuationScorer,
	compiled ProbabilityMassPlan,
	plan Plan,
) (ProbabilityMassReport, error) {
	if ctx == nil || repository == nil || scorer == nil || compiled.identity != plan.body.CaseProfile {
		return ProbabilityMassReport{}, errors.New("evaluation: probability-mass authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases, probabilityMassDatasetContract, probabilityMassSplitContract,
	)
	if err != nil {
		return ProbabilityMassReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return ProbabilityMassReport{}, err
	}
	report := ProbabilityMassReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]ProbabilityMassObservation, len(compiled.suite.Cases)),
	}
	for index, testCase := range compiled.suite.Cases {
		scores, err := scorer.ScoreContinuations(ctx, testCase.Prompt, testCase.Candidates)
		if err != nil {
			return ProbabilityMassReport{}, err
		}
		values, mass, err := positiveProbabilityMass(scores, testCase.Positive, compiled.suite.Normalization)
		if err != nil {
			return ProbabilityMassReport{}, err
		}
		report.Observations[index] = ProbabilityMassObservation{Name: testCase.Name, Values: values, Mass: mass}
		report.Mean += mass
	}
	report.Mean /= float64(len(report.Observations))
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return ProbabilityMassReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(ctx, repository, report.Plan, report.Dataset, report.ID, probabilityMassReportContract, report); err != nil {
		return ProbabilityMassReport{}, err
	}
	return report, nil
}

func positiveProbabilityMass(
	scores []sequencescore.Score,
	positive []bool,
	normalization sequencescore.Normalization,
) ([]float64, float64, error) {
	if len(scores) != len(positive) {
		return nil, 0, errors.New("evaluation: probability labels differ from scores")
	}
	selection, err := sequencescore.Select(scores, normalization)
	if err != nil {
		return nil, 0, err
	}
	maximum := selection.Values[0]
	for _, value := range selection.Values[1:] {
		maximum = max(maximum, value)
	}
	denominator, numerator := 0.0, 0.0
	for index, value := range selection.Values {
		probability := math.Exp(value - maximum)
		denominator += probability
		if positive[index] {
			numerator += probability
		}
	}
	mass := numerator / denominator
	if math.IsNaN(mass) || math.IsInf(mass, 0) {
		return nil, 0, errors.New("evaluation: probability mass is not finite")
	}
	return selection.Values, mass, nil
}
