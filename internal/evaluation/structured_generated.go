package evaluation

import (
	"context"
	"errors"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
)

const (
	structuredGeneratedKind          = "structured-generated"
	extractorDollarSpan              = "dollar-span-or-full"
	equivalenceLatexSurface          = "latex-surface"
	structuredGeneratedDatasetMedia  = "application/vnd.overgo.structured-generated-dataset+json"
	structuredGeneratedDatasetSchema = "overgo/structured-generated-dataset/v1"
	structuredGeneratedSplitMedia    = "application/vnd.overgo.structured-generated-split+json"
	structuredGeneratedSplitSchema   = "overgo/structured-generated-split/v1"
	structuredGeneratedReportMedia   = "application/vnd.overgo.structured-generated-report+json"
	structuredGeneratedReportSchema  = "overgo/structured-generated-report/v1"
)

var (
	structuredGeneratedDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: structuredGeneratedDatasetMedia, Schema: structuredGeneratedDatasetSchema,
	}
	structuredGeneratedSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: structuredGeneratedSplitMedia, Schema: structuredGeneratedSplitSchema,
	}
	structuredGeneratedReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: structuredGeneratedReportMedia, Schema: structuredGeneratedReportSchema,
	}
	latexSurfaceReplacer = strings.NewReplacer(
		"\n", "", `\!`, "", `\\`, `\`, "tfrac", "frac", "dfrac", "frac",
		`\left`, "", `\right`, "", `^{\circ}`, "", `^\circ`, "", `\$`, "", `\%`, "", "%", "",
	)
)

type structuredGeneratedCase struct {
	Name      string   `json:"name"`
	Group     string   `json:"group"`
	Prompt    string   `json:"prompt"`
	MaxTokens int      `json:"max_tokens"`
	Answers   []string `json:"answers"`
}

type structuredGeneratedSuite struct {
	Kind        string                    `json:"kind"`
	Schema      string                    `json:"schema"`
	Source      string                    `json:"source"`
	Extractor   string                    `json:"extractor"`
	Equivalence string                    `json:"equivalence"`
	Cases       []structuredGeneratedCase `json:"cases"`
}

type structuredGeneratedPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    structuredGeneratedSuite
}

type structuredGeneratedObservation struct {
	Name       string `json:"name"`
	Raw        string `json:"raw"`
	Extracted  string `json:"extracted"`
	Normalized string `json:"normalized"`
	Accepted   bool   `json:"accepted"`
	Prompt     int    `json:"prompt_tokens"`
	Generated  int    `json:"generated_tokens"`
}

type structuredGeneratedReport struct {
	ID           artifact.ID                      `json:"-"`
	Version      uint16                           `json:"version"`
	Plan         artifact.ID                      `json:"plan"`
	Dataset      artifact.ID                      `json:"dataset"`
	Observations []structuredGeneratedObservation `json:"observations"`
	Groups       []accuracyGroup                  `json:"groups"`
	Accuracy     float64                          `json:"accuracy"`
}

func compileStructuredGenerated(suite structuredGeneratedSuite) (structuredGeneratedPlan, error) {
	if suite.Kind != structuredGeneratedKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		suite.Extractor != extractorDollarSpan || suite.Equivalence != equivalenceLatexSurface || len(suite.Cases) == 0 {
		return structuredGeneratedPlan{}, errors.New("evaluation: invalid structured-generated suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	names := make(map[string]struct{}, len(suite.Cases))
	for index := range suite.Cases {
		testCase := &suite.Cases[index]
		testCase.Answers = slices.Clone(testCase.Answers)
		if strings.TrimSpace(testCase.Name) == "" || strings.TrimSpace(testCase.Group) != testCase.Group || testCase.Group == "" ||
			testCase.Prompt == "" || testCase.MaxTokens <= 0 || len(testCase.Answers) == 0 {
			return structuredGeneratedPlan{}, errors.New("evaluation: invalid structured-generated case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return structuredGeneratedPlan{}, errors.New("evaluation: duplicate structured-generated case")
		}
		names[testCase.Name] = struct{}{}
		for _, answer := range testCase.Answers {
			if normalizeLatexSurface(answer) == "" {
				return structuredGeneratedPlan{}, errors.New("evaluation: empty structured-generated answer")
			}
		}
	}
	identity, dataset, split, err := suiteIdentities(suite, suite.Cases)
	if err != nil {
		return structuredGeneratedPlan{}, err
	}
	return structuredGeneratedPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func bindStructuredGenerated(compiled structuredGeneratedPlan, authorities ExactAuthorities) (Plan, error) {
	scorer := struct {
		Version     uint16 `json:"version"`
		Kind        string `json:"kind"`
		Extractor   string `json:"extractor"`
		Equivalence string `json:"equivalence"`
	}{
		Version: artifact.InitialDocumentVersion, Kind: structuredGeneratedKind,
		Extractor: compiled.suite.Extractor, Equivalence: compiled.suite.Equivalence,
	}
	return bindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, scorer, authorities)
}

func evaluateStructuredGenerated(
	ctx context.Context,
	repository artifact.Repository,
	generator Generator,
	compiled structuredGeneratedPlan,
	plan Plan,
) (structuredGeneratedReport, error) {
	if ctx == nil || repository == nil || generator == nil || compiled.identity != plan.body.CaseProfile {
		return structuredGeneratedReport{}, errors.New("evaluation: structured-generated authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases,
		structuredGeneratedDatasetContract, structuredGeneratedSplitContract,
	)
	if err != nil {
		return structuredGeneratedReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return structuredGeneratedReport{}, err
	}
	report := structuredGeneratedReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]structuredGeneratedObservation, len(compiled.suite.Cases)),
	}
	groups, accepted := make([]string, len(compiled.suite.Cases)), make([]bool, len(compiled.suite.Cases))
	for index, testCase := range compiled.suite.Cases {
		result, err := recordInstruction(ctx, generator, testCase.Name, testCase.Prompt, testCase.MaxTokens)
		if err != nil {
			return structuredGeneratedReport{}, err
		}
		extracted := extractDollarSpan(result.Text)
		normalized := normalizeLatexSurface(extracted)
		for _, answer := range testCase.Answers {
			if normalized == normalizeLatexSurface(answer) {
				accepted[index] = true
				break
			}
		}
		groups[index] = testCase.Group
		report.Observations[index] = structuredGeneratedObservation{
			Name: testCase.Name, Raw: result.Text, Extracted: extracted, Normalized: normalized, Accepted: accepted[index],
			Prompt: result.PromptTokens, Generated: result.GeneratedTokens,
		}
	}
	report.Groups, err = aggregateAccuracy(groups, accepted)
	if err != nil {
		return structuredGeneratedReport{}, err
	}
	correct := 0
	for _, value := range accepted {
		if value {
			correct++
		}
	}
	report.Accuracy = float64(correct) / float64(len(accepted))
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return structuredGeneratedReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(ctx, repository, report.Plan, report.Dataset, report.ID, structuredGeneratedReportContract, report); err != nil {
		return structuredGeneratedReport{}, err
	}
	return report, nil
}

func extractDollarSpan(value string) string {
	first, last := strings.IndexByte(value, '$'), strings.LastIndexByte(value, '$')
	if first >= 0 && last > first {
		return value[first+1 : last]
	}
	return value
}

func normalizeLatexSurface(value string) string {
	value = latexSurfaceReplacer.Replace(value)
	if before, _, ok := strings.Cut(value, `\text{ `); ok {
		value = before
	}
	value = strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, value)
	value = normalizeUnaryLatex(value, `\sqrt`)
	value = normalizeFractionLatex(value)
	if strings.HasPrefix(value, ".") {
		value = "0" + value
	}
	return value
}

func normalizeUnaryLatex(value, operator string) string {
	for offset := 0; ; {
		index := strings.Index(value[offset:], operator)
		if index < 0 {
			return value
		}
		index += offset + len(operator)
		if index < len(value) && value[index] != '{' {
			value = value[:index] + "{" + value[index:index+1] + "}" + value[index+1:]
			index += 2
		}
		offset = index
	}
}

func normalizeFractionLatex(value string) string {
	for offset := 0; ; {
		index := strings.Index(value[offset:], `\frac`)
		if index < 0 {
			return value
		}
		index += offset + len(`\frac`)
		for operand := 0; operand < 2 && index < len(value); operand++ {
			if value[index] == '{' {
				end := latexGroupEnd(value, index)
				if end < 0 {
					return value
				}
				index = end
				continue
			}
			value = value[:index] + "{" + value[index:index+1] + "}" + value[index+1:]
			index += 3
		}
		offset = index
	}
}

func latexGroupEnd(value string, start int) int {
	depth := 0
	for index := start; index < len(value); index++ {
		switch value[index] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return index + 1
			}
		}
	}
	return -1
}
