package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
)

const (
	GroupedChoiceKind         = "grouped-multiple-choice"
	groupedChoiceReportMedia  = "application/vnd.overgo.grouped-choice-report+json"
	groupedChoiceReportSchema = "overgo/grouped-choice-report/v1"
)

var groupedChoiceReportContract = artifact.DocumentContract{
	Kind: artifact.KindEvaluation, MediaType: groupedChoiceReportMedia, Schema: groupedChoiceReportSchema,
}

type DemonstratedChoice struct {
	Name                string               `json:"name"`
	Group               string               `json:"group"`
	Prompt              string               `json:"prompt"`
	Candidates          []string             `json:"candidates"`
	CandidateCharacters []uint64             `json:"candidate_characters,omitzero"`
	Answer              int                  `json:"answer"`
	Demonstrations      []MultipleChoiceCase `json:"demonstrations,omitempty"`
}

type GroupedChoiceSuite struct {
	Kind          string                      `json:"kind"`
	Schema        string                      `json:"schema"`
	Source        string                      `json:"source"`
	Normalization sequencescore.Normalization `json:"normalization"`
	TieBreak      string                      `json:"tie_break,omitzero"`
	// Delimiter is what a likelihood scoring puts between a case's prompt and each of its candidates.
	//   - accepted = empty or targetDelimiter, because the scorer's method names the target delimiter alone
	//   - empty => each candidate carries its own delimiter
	Delimiter string               `json:"delimiter,omitzero"`
	Cases     []DemonstratedChoice `json:"cases"`
}

type groupedChoicePlan struct {
	choice multipleChoicePlan
	groups []string
}

type GroupedChoiceReport struct {
	ID      artifact.ID `json:"-"`
	Version uint16      `json:"version"`
	Plan    artifact.ID `json:"plan"`
	Dataset artifact.ID `json:"dataset"`
	// Answers names what the plan's execution was asked (Plan.observed): the
	// store holds, under it, each case's answer as the model gave it, which
	// the observations here were scored from. A report written before answers
	// were kept apart from their scores names none.
	Answers      artifact.ID         `json:"answers,omitzero"`
	Observations []choiceObservation `json:"observations"`
	Groups       []accuracyGroup     `json:"groups"`
	Accuracy     float64             `json:"accuracy"`
}

func CompileGroupedChoice(suite GroupedChoiceSuite) (groupedChoicePlan, error) {
	if suite.Kind != GroupedChoiceKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		len(suite.Cases) == 0 || suite.Normalization != sequencescore.NormalizationSum && suite.Normalization != sequencescore.NormalizationMean && suite.Normalization != sequencescore.NormalizationCharacters {
		return groupedChoicePlan{}, errors.New("evaluation: invalid grouped-choice suite")
	}
	if suite.Delimiter != "" && suite.Delimiter != targetDelimiter {
		return groupedChoicePlan{}, fmt.Errorf("evaluation: grouped-choice delimiter %q is refused; a suite names none or %q", suite.Delimiter, targetDelimiter)
	}
	cases := make([]MultipleChoiceCase, len(suite.Cases))
	groups := make([]string, len(suite.Cases))
	for index, source := range suite.Cases {
		if strings.TrimSpace(source.Group) != source.Group || source.Group == "" {
			return groupedChoicePlan{}, errors.New("evaluation: grouped-choice group is absent")
		}
		prompt := strings.Builder{}
		for _, demonstration := range source.Demonstrations {
			if err := validateMultipleChoiceCase(demonstration); err != nil {
				return groupedChoicePlan{}, errors.New("evaluation: invalid grouped-choice demonstration")
			}
			prompt.WriteString(demonstration.Prompt)
			prompt.WriteString(demonstration.Candidates[demonstration.Answer])
			prompt.WriteString("\n\n")
		}
		prompt.WriteString(source.Prompt)
		cases[index] = MultipleChoiceCase{
			Name: source.Name, Prompt: prompt.String(), Candidates: slices.Clone(source.Candidates), CandidateCharacters: slices.Clone(source.CandidateCharacters), Answer: source.Answer,
		}
		groups[index] = source.Group
	}
	choice, err := CompileMultipleChoice(MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: suite.Schema, Source: suite.Source,
		Normalization: suite.Normalization, TieBreak: suite.TieBreak, Aggregation: AggregationAccuracy, Cases: cases,
	})
	if err != nil {
		return groupedChoicePlan{}, err
	}
	choice.delimiter = suite.Delimiter
	return groupedChoicePlan{choice: choice, groups: groups}, nil
}

func BindGroupedChoice(compiled groupedChoicePlan, authorities ExactAuthorities) (Plan, error) {
	return BindMultipleChoice(compiled.choice, authorities)
}

func EvaluateGroupedChoice(
	ctx context.Context,
	repository artifact.Repository,
	scorer continuationScorer,
	compiled groupedChoicePlan,
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
	observations, groups, answers, accuracy, err := scoreChoiceGroups(ctx, repository, plan, scorer, compiled.choice.delimited(), compiled.groups)
	if err != nil {
		return GroupedChoiceReport{}, err
	}
	report := GroupedChoiceReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.choice.dataset, Answers: answers,
		Observations: observations, Groups: groups, Accuracy: accuracy,
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return GroupedChoiceReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(ctx, repository, report.Plan, report.Dataset, report.ID, groupedChoiceReportContract, report, answers); err != nil {
		return GroupedChoiceReport{}, err
	}
	return report, nil
}
