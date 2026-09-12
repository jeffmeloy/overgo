package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
)

const (
	InstructionRulesKind          = "instruction-rules"
	RuleContainsAll               = "contains-all"
	RuleExcludesAll               = "excludes-all"
	RuleSubstringCount            = "substring-count"
	RuleRegex                     = "regex"
	RuleJSON                      = "json"
	RuleUppercase                 = "uppercase"
	RuleLowercase                 = "lowercase"
	RulePrefix                    = "prefix"
	RuleSuffix                    = "suffix"
	RuleNoComma                   = "no-comma"
	RelationEqual                 = "equal"
	RelationAtLeast               = "at-least"
	RelationAtMost                = "at-most"
	instructionRulesDatasetMedia  = "application/vnd.overgo.instruction-rules-dataset+json"
	instructionRulesDatasetSchema = "overgo/instruction-rules-dataset/v1"
	instructionRulesSplitMedia    = "application/vnd.overgo.instruction-rules-split+json"
	instructionRulesSplitSchema   = "overgo/instruction-rules-split/v1"
	instructionRulesReportMedia   = "application/vnd.overgo.instruction-rules-report+json"
	instructionRulesReportSchema  = "overgo/instruction-rules-report/v1"
)

var (
	instructionRulesDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: instructionRulesDatasetMedia, Schema: instructionRulesDatasetSchema,
	}
	instructionRulesSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: instructionRulesSplitMedia, Schema: instructionRulesSplitSchema,
	}
	instructionRulesReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: instructionRulesReportMedia, Schema: instructionRulesReportSchema,
	}
)

type CountRule struct {
	Relation string `json:"relation"`
	Value    int    `json:"value"`
}

type InstructionRule struct {
	Name   string     `json:"name"`
	Kind   string     `json:"kind"`
	Values []string   `json:"values,omitempty"`
	Count  *CountRule `json:"count,omitempty"`
	// Ordinal selects the one-based paragraph in indexed paragraph rules.
	Ordinal int `json:"ordinal,omitzero"`
	// Loose binds independently resolved native operands for the loose view.
	Loose *InstructionRule `json:"loose,omitempty"`
}

type InstructionRulesCase struct {
	Name      string            `json:"name"`
	Prompt    string            `json:"prompt"`
	MaxTokens int               `json:"max_tokens"`
	Rules     []InstructionRule `json:"rules"`
}

type InstructionRulesSuite struct {
	Kind   string                 `json:"kind"`
	Schema string                 `json:"schema"`
	Source string                 `json:"source"`
	Cases  []InstructionRulesCase `json:"cases"`
}

type compiledInstructionRule struct {
	tokenizer *ifevalTokenizer
	source    InstructionRule
	pattern   *regexp.Regexp
	patterns  []*regexp.Regexp
	loose     *compiledInstructionRule
}

type InstructionRulesPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    InstructionRulesSuite
	rules    [][]compiledInstructionRule
}

type InstructionRulesObservation struct {
	Name      string `json:"name"`
	Raw       string `json:"raw"`
	Strict    []bool `json:"strict"`
	Loose     []bool `json:"loose"`
	Prompt    int    `json:"prompt_tokens"`
	Generated int    `json:"generated_tokens"`
}

type InstructionRulesReport struct {
	ID                artifact.ID                   `json:"-"`
	Version           uint16                        `json:"version"`
	Plan              artifact.ID                   `json:"plan"`
	Dataset           artifact.ID                   `json:"dataset"`
	Observations      []InstructionRulesObservation `json:"observations"`
	PromptStrict      float64                       `json:"prompt_strict"`
	InstructionStrict float64                       `json:"instruction_strict"`
	PromptLoose       float64                       `json:"prompt_loose"`
	InstructionLoose  float64                       `json:"instruction_loose"`
}

func CompileInstructionRules(suite InstructionRulesSuite) (InstructionRulesPlan, error) {
	if suite.Kind != InstructionRulesKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return InstructionRulesPlan{}, errors.New("evaluation: invalid instruction-rules suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	compiled := make([][]compiledInstructionRule, len(suite.Cases))
	names := make(map[string]struct{}, len(suite.Cases))
	for index := range suite.Cases {
		testCase := &suite.Cases[index]
		testCase.Rules = slices.Clone(testCase.Rules)
		if strings.TrimSpace(testCase.Name) == "" || testCase.Prompt == "" || testCase.MaxTokens <= 0 || len(testCase.Rules) == 0 {
			return InstructionRulesPlan{}, errors.New("evaluation: invalid instruction-rules case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return InstructionRulesPlan{}, errors.New("evaluation: duplicate instruction-rules case")
		}
		names[testCase.Name] = struct{}{}
		compiled[index] = make([]compiledInstructionRule, len(testCase.Rules))
		ruleNames := make(map[string]struct{}, len(testCase.Rules))
		for ruleIndex := range testCase.Rules {
			rule, err := compileInstructionRule(testCase.Rules[ruleIndex])
			if err != nil {
				return InstructionRulesPlan{}, err
			}
			if _, duplicate := ruleNames[rule.source.Name]; duplicate {
				return InstructionRulesPlan{}, errors.New("evaluation: duplicate instruction rule")
			}
			ruleNames[rule.source.Name] = struct{}{}
			compiled[index][ruleIndex] = rule
			testCase.Rules[ruleIndex] = rule.source
		}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return InstructionRulesPlan{}, err
	}
	datasetID, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return InstructionRulesPlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: datasetID})
	if err != nil {
		return InstructionRulesPlan{}, err
	}
	return InstructionRulesPlan{identity: identity, dataset: datasetID, split: split, suite: suite, rules: compiled}, nil
}

func BindInstructionRules(compiled InstructionRulesPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, InstructionRulesKind, authorities)
}

func EvaluateInstructionRules(
	ctx context.Context,
	repository artifact.Repository,
	generator Generator,
	compiled InstructionRulesPlan,
	plan Plan,
) (InstructionRulesReport, error) {
	if ctx == nil || repository == nil || generator == nil || compiled.identity != plan.body.CaseProfile {
		return InstructionRulesReport{}, errors.New("evaluation: instruction-rules authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases,
		instructionRulesDatasetContract, instructionRulesSplitContract,
	)
	if err != nil {
		return InstructionRulesReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return InstructionRulesReport{}, err
	}
	report := InstructionRulesReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]InstructionRulesObservation, len(compiled.suite.Cases)),
	}
	strictPrompts, loosePrompts, strictRules, looseRules, rules := 0, 0, 0, 0, 0
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for index, testCase := range compiled.suite.Cases {
		result, err := recordInstruction(ctx, generator, testCase.Name, testCase.Prompt, testCase.MaxTokens)
		if err != nil {
			return InstructionRulesReport{}, err
		}
		strict, loose, err := evaluateInstructionViews(result.Text, compiled.rules[index])
		if err != nil {
			return InstructionRulesReport{}, fmt.Errorf("evaluation: case %s: %w", testCase.Name, err)
		}
		progress.hit(allTrue(strict))
		if allTrue(strict) {
			strictPrompts++
		}
		if allTrue(loose) {
			loosePrompts++
		}
		strictRules += countTrue(strict)
		looseRules += countTrue(loose)
		rules += len(strict)
		report.Observations[index] = InstructionRulesObservation{
			Name: testCase.Name, Raw: result.Text, Strict: strict, Loose: loose,
			Prompt: result.PromptTokens, Generated: result.GeneratedTokens,
		}
	}
	report.PromptStrict = float64(strictPrompts) / float64(len(report.Observations))
	report.PromptLoose = float64(loosePrompts) / float64(len(report.Observations))
	report.InstructionStrict = float64(strictRules) / float64(rules)
	report.InstructionLoose = float64(looseRules) / float64(rules)
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return InstructionRulesReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(ctx, repository, report.Plan, report.Dataset, report.ID, instructionRulesReportContract, report); err != nil {
		return InstructionRulesReport{}, err
	}
	return report, nil
}

func compileInstructionRule(source InstructionRule) (compiledInstructionRule, error) {
	if source.Ordinal != 0 && source.Kind != ifevalIndexedParagraph {
		return compiledInstructionRule{}, errors.New("evaluation: ordinal requires an indexed paragraph rule")
	}
	source.Values = slices.Clone(source.Values)
	if source.Count != nil {
		count := *source.Count
		source.Count = &count
	}
	if strings.TrimSpace(source.Name) == "" {
		return compiledInstructionRule{}, errors.New("evaluation: instruction rule name is absent")
	}
	result := compiledInstructionRule{source: source}
	switch source.Kind {
	case ifevalKeywordsRule, ifevalForbiddenRule:
		if len(source.Values) == 0 || source.Count != nil {
			return compiledInstructionRule{}, errors.New("evaluation: invalid IFEval keyword rule")
		}
		for _, value := range source.Values {
			pattern, err := compileIFEvalKeyword(value, source.Kind == ifevalForbiddenRule)
			if err != nil {
				return compiledInstructionRule{}, err
			}
			result.patterns = append(result.patterns, pattern)
		}
	case ifevalEndRule:
		if len(source.Values) != 1 || source.Count != nil {
			return compiledInstructionRule{}, errors.New("evaluation: invalid IFEval end phrase")
		}
		for _, r := range source.Values[0] {
			if r > unicode.MaxASCII {
				return compiledInstructionRule{}, errors.New("evaluation: unsupported non-ASCII IFEval end phrase")
			}
		}
	case RuleContainsAll, RuleExcludesAll:
		if len(source.Values) == 0 || source.Count != nil {
			return compiledInstructionRule{}, errors.New("evaluation: invalid instruction value rule")
		}
	case RuleSubstringCount:
		if len(source.Values) != 1 || !validCountRule(source.Count) {
			return compiledInstructionRule{}, errors.New("evaluation: invalid instruction count rule")
		}
	case RuleRegex:
		if len(source.Values) != 1 || source.Count != nil {
			return compiledInstructionRule{}, errors.New("evaluation: invalid instruction regex rule")
		}
		pattern, err := regexp.Compile(source.Values[0])
		if err != nil {
			return compiledInstructionRule{}, err
		}
		result.pattern = pattern
	case RulePrefix, RuleSuffix:
		if len(source.Values) != 1 || source.Values[0] == "" || source.Count != nil {
			return compiledInstructionRule{}, errors.New("evaluation: invalid instruction boundary rule")
		}
	case RuleJSON, ifevalJSONRule, RuleUppercase, RuleLowercase, RuleNoComma:
		if len(source.Values) != 0 || source.Count != nil {
			return compiledInstructionRule{}, errors.New("evaluation: invalid instruction unary rule")
		}
	default:
		var err error
		result, err = compileIFEvalStructure(source)
		if err != nil {
			return compiledInstructionRule{}, err
		}
	}
	for _, value := range source.Values {
		if value == "" {
			return compiledInstructionRule{}, errors.New("evaluation: empty instruction rule value")
		}
	}

	if source.Loose != nil {
		if source.Loose.Loose != nil || source.Loose.Name != source.Name || source.Loose.Kind != source.Kind {
			return compiledInstructionRule{}, errors.New("evaluation: incompatible loose instruction operands")
		}
		loose, err := compileInstructionRule(*source.Loose)
		if err != nil {
			return compiledInstructionRule{}, err
		}
		result.loose = &loose
		result.source.Loose = &loose.source
	}
	return result, nil
}

func validCountRule(rule *CountRule) bool {
	return rule != nil && rule.Value >= 0 && (rule.Relation == RelationEqual || rule.Relation == RelationAtLeast || rule.Relation == RelationAtMost || rule.Relation == ifevalLessThan)
}

func evaluateInstructionViews(response string, rules []compiledInstructionRule) ([]bool, []bool, error) {
	views := looseInstructionViews(response)
	strict, loose := make([]bool, len(rules)), make([]bool, len(rules))
	for index, rule := range rules {
		var err error
		if trimIFEvalSpace(response) != "" {
			strict[index], err = rule.matches(response)
			if err != nil {
				return nil, nil, fmt.Errorf("instruction %s strict: %w", rule.source.Name, err)
			}
		}
		if rule.loose != nil {
			rule = *rule.loose
		}
		for _, view := range views {
			if trimIFEvalSpace(view) == "" {
				continue
			}
			matched, err := rule.matches(view)
			if err != nil {
				return nil, nil, fmt.Errorf("instruction %s loose: %w", rule.source.Name, err)
			}
			if matched {
				loose[index] = true
				break
			}
		}
	}
	return strict, loose, nil
}

func looseInstructionViews(response string) []string {
	lines := strings.Split(response, "\n")
	removeFirst, removeLast, removeBoth := "", "", ""
	if len(lines) > 1 {
		removeFirst = trimIFEvalSpace(strings.Join(lines[1:], "\n"))
		removeLast = trimIFEvalSpace(strings.Join(lines[:len(lines)-1], "\n"))
	}
	if len(lines) > 2 {
		removeBoth = trimIFEvalSpace(strings.Join(lines[1:len(lines)-1], "\n"))
	}
	withoutMarks := func(value string) string { return strings.ReplaceAll(value, "*", "") }
	return []string{
		response, withoutMarks(response), removeFirst, removeLast, removeBoth,
		withoutMarks(removeFirst), withoutMarks(removeLast), withoutMarks(removeBoth),
	}
}

func (rule compiledInstructionRule) matches(response string) (bool, error) {
	if rule.tokenizer != nil {
		return rule.matchesIFEvalTokenizer(response)
	}
	switch rule.source.Kind {
	case ifevalKeywordsRule, ifevalForbiddenRule:
		for _, pattern := range rule.patterns {
			if pattern.MatchString(response) == (rule.source.Kind == ifevalForbiddenRule) {
				return false, nil
			}
		}
		return true, nil
	case ifevalEndRule:
		return matchesIFEvalEnd(response, rule.source.Values[0]), nil
	case RuleContainsAll:
		for _, value := range rule.source.Values {
			if !strings.Contains(response, value) {
				return false, nil
			}
		}
		return true, nil
	case RuleExcludesAll:
		for _, value := range rule.source.Values {
			if strings.Contains(response, value) {
				return false, nil
			}
		}
		return true, nil
	case RuleSubstringCount:
		return compareCount(strings.Count(response, rule.source.Values[0]), *rule.source.Count), nil
	case RuleRegex:
		return rule.pattern.MatchString(response), nil
	case RuleJSON:
		return json.Valid([]byte(response)), nil
	case ifevalJSONRule:
		return matchesIFEvalJSON(response), nil
	case RuleUppercase:
		return uniformLetterCase(response, unicode.IsUpper), nil
	case RuleLowercase:
		return uniformLetterCase(response, unicode.IsLower), nil
	case RulePrefix:
		return strings.HasPrefix(response, rule.source.Values[0]), nil
	case RuleSuffix:
		return strings.HasSuffix(response, rule.source.Values[0]), nil
	case RuleNoComma:
		return !strings.ContainsRune(response, ','), nil
	default:
		return rule.matchesIFEvalStructure(response), nil
	}
}

func compareCount(observed int, rule CountRule) bool {
	switch rule.Relation {
	case ifevalLessThan:
		return observed < rule.Value
	case RelationEqual:
		return observed == rule.Value
	case RelationAtLeast:
		return observed >= rule.Value
	case RelationAtMost:
		return observed <= rule.Value
	default:
		return false
	}
}

func uniformLetterCase(value string, accept func(rune) bool) bool {
	found := false
	for _, current := range value {
		if unicode.IsLetter(current) {
			found = true
			if !accept(current) {
				return false
			}
		}
	}
	return found
}

func allTrue(values []bool) bool { return countTrue(values) == len(values) }

func countTrue(values []bool) int {
	count := 0
	for _, value := range values {
		if value {
			count++
		}
	}
	return count
}
