package evaluation

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
)

const (
	GroupedChoiceKind         = "grouped-multiple-choice"
	groupedChoiceReportMedia  = "application/vnd.overgo.grouped-choice-report+json"
	groupedChoiceReportSchema = "overgo/grouped-choice-report/v1"
	groupedChoiceVersion      = 1
)

var groupedChoiceReportContract = artifact.DocumentContract{
	Kind: artifact.KindEvaluation, MediaType: groupedChoiceReportMedia, Schema: groupedChoiceReportSchema,
}

type DemonstratedChoice struct {
	Name           string               `json:"name"`
	Group          string               `json:"group"`
	Prompt         string               `json:"prompt"`
	Candidates     []string             `json:"candidates"`
	Answer         int                  `json:"answer"`
	Demonstrations []MultipleChoiceCase `json:"demonstrations,omitempty"`
}

type GroupedChoiceSuite struct {
	Kind          string                      `json:"kind"`
	Schema        string                      `json:"schema"`
	Source        string                      `json:"source"`
	Normalization sequencescore.Normalization `json:"normalization"`
	Cases         []DemonstratedChoice        `json:"cases"`
}

type GroupedChoicePlan struct {
	choice MultipleChoicePlan
	groups []string
}

type GroupedChoiceReport struct {
	ID           artifact.ID         `json:"-"`
	Version      uint16              `json:"version"`
	Plan         artifact.ID         `json:"plan"`
	Dataset      artifact.ID         `json:"dataset"`
	Observations []ChoiceObservation `json:"observations"`
	Groups       []AccuracyGroup     `json:"groups"`
	Accuracy     float64             `json:"accuracy"`
}

func CompileGroupedChoice(suite GroupedChoiceSuite) (GroupedChoicePlan, error) {
	if suite.Kind != GroupedChoiceKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		len(suite.Cases) == 0 || suite.Normalization != sequencescore.NormalizationSum && suite.Normalization != sequencescore.NormalizationMean {
		return GroupedChoicePlan{}, errors.New("evaluation: invalid grouped-choice suite")
	}
	cases := make([]MultipleChoiceCase, len(suite.Cases))
	groups := make([]string, len(suite.Cases))
	for index, source := range suite.Cases {
		if strings.TrimSpace(source.Group) != source.Group || source.Group == "" {
			return GroupedChoicePlan{}, errors.New("evaluation: grouped-choice group is absent")
		}
		prompt := strings.Builder{}
		for _, demonstration := range source.Demonstrations {
			if err := validateMultipleChoiceCase(demonstration); err != nil {
				return GroupedChoicePlan{}, errors.New("evaluation: invalid grouped-choice demonstration")
			}
			prompt.WriteString(demonstration.Prompt)
			prompt.WriteString(demonstration.Candidates[demonstration.Answer])
			prompt.WriteString("\n\n")
		}
		prompt.WriteString(source.Prompt)
		cases[index] = MultipleChoiceCase{
			Name: source.Name, Prompt: prompt.String(), Candidates: slices.Clone(source.Candidates), Answer: source.Answer,
		}
		groups[index] = source.Group
	}
	choice, err := CompileMultipleChoice(MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: suite.Schema, Source: suite.Source,
		Normalization: suite.Normalization, Aggregation: AggregationAccuracy, Cases: cases,
	})
	if err != nil {
		return GroupedChoicePlan{}, err
	}
	return GroupedChoicePlan{choice: choice, groups: groups}, nil
}

func BindGroupedChoice(compiled GroupedChoicePlan, authorities ExactAuthorities) (Plan, error) {
	return BindMultipleChoice(compiled.choice, authorities)
}

func EvaluateGroupedChoice(
	ctx context.Context,
	repository artifact.Repository,
	scorer ContinuationScorer,
	compiled GroupedChoicePlan,
	plan Plan,
) (GroupedChoiceReport, error) {
	if ctx == nil || repository == nil || scorer == nil || compiled.choice.identity != plan.body.CaseProfile {
		return GroupedChoiceReport{}, errors.New("evaluation: grouped-choice authority differs")
	}
	contents, err := compiled.choice.contents()
	if err != nil {
		return GroupedChoiceReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return GroupedChoiceReport{}, err
	}
	observations, groups, accuracy, err := scoreChoiceGroups(ctx, scorer, compiled.choice.suite, compiled.groups)
	if err != nil {
		return GroupedChoiceReport{}, err
	}
	report := GroupedChoiceReport{
		Version: groupedChoiceVersion, Plan: plan.identity, Dataset: compiled.choice.dataset,
		Observations: observations, Groups: groups, Accuracy: accuracy,
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return GroupedChoiceReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(ctx, repository, report.Plan, report.Dataset, report.ID, groupedChoiceReportContract, report); err != nil {
		return GroupedChoiceReport{}, err
	}
	return report, nil
}
