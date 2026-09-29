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
	mmluProKind         = "mmlu-pro"
	mmluProReportMedia  = "application/vnd.overgo.mmlu-pro-report+json"
	mmluProReportSchema = "overgo/mmlu-pro-report/v1"
)

var mmluProReportContract = artifact.DocumentContract{
	Kind: artifact.KindEvaluation, MediaType: mmluProReportMedia, Schema: mmluProReportSchema,
}

type mmluProCase struct {
	Name     string   `json:"name"`
	Category string   `json:"category"`
	Question string   `json:"question"`
	Options  []string `json:"options"`
	Answer   int      `json:"answer"`
}

type mmluProSuite struct {
	Kind           string                      `json:"kind"`
	Schema         string                      `json:"schema"`
	Source         string                      `json:"source"`
	Labels         []string                    `json:"labels"`
	FewShotCount   int                         `json:"few_shot_count"`
	Demonstrations []mmluProCase               `json:"demonstrations"`
	Cases          []mmluProCase               `json:"cases"`
	Normalization  sequencescore.Normalization `json:"normalization"`
}

type mmluProPlan struct {
	choice     multipleChoicePlan
	categories []string
}

type mmluProReport struct {
	ID           artifact.ID         `json:"-"`
	Version      uint16              `json:"version"`
	Plan         artifact.ID         `json:"plan"`
	Dataset      artifact.ID         `json:"dataset"`
	Observations []choiceObservation `json:"observations"`
	Categories   []accuracyGroup     `json:"categories"`
	Accuracy     float64             `json:"accuracy"`
}

func compileMMLUPro(suite mmluProSuite) (mmluProPlan, error) {
	if suite.Kind != mmluProKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		len(suite.Labels) < 2 || suite.FewShotCount < 0 || suite.FewShotCount > len(suite.Demonstrations) || len(suite.Cases) == 0 ||
		suite.Normalization != sequencescore.NormalizationSum && suite.Normalization != sequencescore.NormalizationMean {
		return mmluProPlan{}, errors.New("evaluation: invalid MMLU-Pro suite")
	}
	labels := slices.Clone(suite.Labels)
	seenLabels := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		if strings.TrimSpace(label) != label || label == "" {
			return mmluProPlan{}, errors.New("evaluation: invalid MMLU-Pro label")
		}
		if _, exists := seenLabels[label]; exists {
			return mmluProPlan{}, errors.New("evaluation: duplicate MMLU-Pro label")
		}
		seenLabels[label] = struct{}{}
	}
	selected := suite.Demonstrations[:suite.FewShotCount]
	demonstrations := strings.Builder{}
	for _, example := range selected {
		if err := validateMMLUProCase(example, len(labels)); err != nil {
			return mmluProPlan{}, err
		}
		renderMMLUProCase(&demonstrations, example, labels)
		demonstrations.WriteString(labels[example.Answer])
		demonstrations.WriteString("\n\n")
	}
	cases := make([]MultipleChoiceCase, len(suite.Cases))
	categories := make([]string, len(suite.Cases))
	for index, testCase := range suite.Cases {
		if err := validateMMLUProCase(testCase, len(labels)); err != nil {
			return mmluProPlan{}, err
		}
		prompt := strings.Builder{}
		prompt.Grow(demonstrations.Len() + len(testCase.Question))
		prompt.WriteString(demonstrations.String())
		renderMMLUProCase(&prompt, testCase, labels)
		cases[index] = MultipleChoiceCase{
			Name: testCase.Name, Prompt: prompt.String(), Candidates: slices.Clone(labels[:len(testCase.Options)]), Answer: testCase.Answer,
		}
		categories[index] = testCase.Category
	}
	choice, err := CompileMultipleChoice(MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: suite.Schema, Source: suite.Source,
		Normalization: suite.Normalization, Aggregation: AggregationAccuracy, Cases: cases,
	})
	if err != nil {
		return mmluProPlan{}, err
	}
	return mmluProPlan{choice: choice, categories: categories}, nil
}

func bindMMLUPro(compiled mmluProPlan, authorities ExactAuthorities) (Plan, error) {
	return BindMultipleChoice(compiled.choice, authorities)
}

func evaluateMMLUPro(
	ctx context.Context,
	repository artifact.Repository,
	scorer continuationScorer,
	compiled mmluProPlan,
	plan Plan,
) (mmluProReport, error) {
	if ctx == nil || repository == nil || scorer == nil || compiled.choice.identity != plan.body.CaseProfile {
		return mmluProReport{}, errors.New("evaluation: MMLU-Pro authority differs")
	}
	contents, err := compiled.choice.contents()
	if err != nil {
		return mmluProReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return mmluProReport{}, err
	}
	observations, categories, accuracy, err := scoreChoiceGroups(ctx, scorer, compiled.choice.suite, compiled.categories)
	if err != nil {
		return mmluProReport{}, err
	}
	report := mmluProReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.choice.dataset,
		Observations: observations, Categories: categories, Accuracy: accuracy,
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return mmluProReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(ctx, repository, report.Plan, report.Dataset, report.ID, mmluProReportContract, report); err != nil {
		return mmluProReport{}, err
	}
	return report, nil
}

func validateMMLUProCase(testCase mmluProCase, labels int) error {
	if strings.TrimSpace(testCase.Name) == "" || strings.TrimSpace(testCase.Category) == "" || testCase.Question == "" ||
		len(testCase.Options) < 2 || len(testCase.Options) > labels || testCase.Answer < 0 || testCase.Answer >= len(testCase.Options) {
		return errors.New("evaluation: invalid MMLU-Pro case")
	}
	if slices.Contains(testCase.Options, "") {
		return errors.New("evaluation: empty MMLU-Pro option")
	}
	return nil
}

func renderMMLUProCase(prompt *strings.Builder, testCase mmluProCase, labels []string) {
	prompt.WriteString(testCase.Question)
	prompt.WriteByte('\n')
	for index, option := range testCase.Options {
		prompt.WriteString(labels[index])
		prompt.WriteString(". ")
		prompt.WriteString(option)
		prompt.WriteByte('\n')
	}
	prompt.WriteString("Answer:")
}
