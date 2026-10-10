package evaluation

import (
	"context"
	"encoding/json"
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
	// mmluProSchema names the task the store's MMLU-Pro suite is a port of.
	mmluProSchema = "lm-eval/leaderboard-mmlu-pro/v0.1"
	// gluedLetters is the part of a protocol label that marks a likelihood
	// reading of MMLU-Pro taken before its candidates were given the task's
	// target delimiter: each option letter was scored glued to the answer
	// cue, "Answer:A".
	gluedLetters = "letters glued to the cue"
)

// scoredGlued reports whether a plan is a likelihood scoring of MMLU-Pro
// that glued each option letter to the answer cue: the suite of that task,
// under a scorer that names no method, which is every likelihood plan of it
// bound before the delimiter was. Such a record is not a reading of the
// benchmark as its harness asks it, and its label says so.
func scoredGlued(ctx context.Context, reader artifact.Reader, plan Plan) bool {
	profile, found, err := artifact.ReadContent(ctx, reader, plan.body.CaseProfile)
	if err != nil || !found {
		return false
	}
	var suite struct {
		Kind   string `json:"kind"`
		Schema string `json:"schema"`
	}
	if json.Unmarshal(profile.Data, &suite) != nil || suite.Kind != MultipleChoiceKind || suite.Schema != mmluProSchema {
		return false
	}
	scorer, found, err := artifact.ReadContent(ctx, reader, plan.body.Scorer)
	if err != nil || !found {
		return false
	}
	var method struct {
		Method string `json:"method"`
	}
	return json.Unmarshal(scorer.Data, &method) == nil && method.Method == ""
}

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
		demonstrations.WriteString(targetDelimiter)
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
	// The task this suite is a port of writes its choices as bare letters and
	// leaves the space before each to its target delimiter.
	choice.delimiter = targetDelimiter
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
	observations, categories, answers, accuracy, err := scoreChoiceGroups(ctx, repository, plan, scorer, compiled.choice.delimited(), compiled.categories)
	if err != nil {
		return mmluProReport{}, err
	}
	report := mmluProReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.choice.dataset, Answers: answers,
		Observations: observations, Categories: categories, Accuracy: accuracy,
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return mmluProReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(ctx, repository, report.Plan, report.Dataset, report.ID, mmluProReportContract, report, answers); err != nil {
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
