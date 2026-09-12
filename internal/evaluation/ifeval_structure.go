package evaluation

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"
	"unicode"
)

const (
	ifevalLetters      = "lm-eval/ifeval/letter-frequency/v1"
	ifevalLessThan     = "less-than"
	ifevalWords        = "lm-eval/ifeval/words/v1"
	ifevalFrequency    = "lm-eval/ifeval/keyword-frequency/v1"
	ifevalPlaceholders = "lm-eval/ifeval/placeholders/v1"
	ifevalBullets      = "lm-eval/ifeval/bullets/v1"
	ifevalHighlights   = "lm-eval/ifeval/highlights/v1"
	ifevalSections     = "lm-eval/ifeval/sections/v1"
	ifevalParagraphs   = "lm-eval/ifeval/paragraphs/v1"
	ifevalPostscript   = "lm-eval/ifeval/postscript/v1"
	ifevalTitle        = "lm-eval/ifeval/title/v1"
	ifevalTwoResponses = "lm-eval/ifeval/two-responses/v1"
	ifevalRepeat       = "lm-eval/ifeval/repeat-prompt/v1"
)

func compileIFEvalStructure(source InstructionRule) (compiledInstructionRule, error) {
	if source.Kind == ifevalSentences || source.Kind == ifevalCapitalWords {
		return compileIFEvalTokenizerRule(source)
	}
	if source.Kind == ifevalIndexedParagraph {
		return compileIFEvalParagraph(source)
	}
	result := compiledInstructionRule{source: source}
	valueOperand, counted := false, false
	var patterns []string
	switch source.Kind {
	case ifevalWords:
		counted = true
		patterns = []string{`[\p{L}\p{N}_]+`}
	case ifevalFrequency, ifevalLetters:
		counted = true
		valueOperand = true
	case ifevalPlaceholders:
		counted = true
		patterns = []string{`\[.*?\]`}
	case ifevalBullets:
		counted = true
		patterns = []string{`(?m)^` + ifevalSpaceClass + `*\*[^*].*$`, `(?m)^` + ifevalSpaceClass + `*-.*$`}
	case ifevalHighlights:
		counted = true
		patterns = []string{`\*[^\n*]*\*`, `\*\*[^\n*]*\*\*`}
	case ifevalSections:
		counted = true
		valueOperand = true
	case ifevalParagraphs:
		counted = true
	case ifevalPostscript:
		valueOperand = true
	case ifevalTitle:
		patterns = []string{`<<[^\n]+>>`}
	case ifevalTwoResponses:
	case ifevalRepeat:
		valueOperand = true
	default:
		return result, errors.New("evaluation: unknown instruction rule")
	}
	validValues := len(source.Values) == 0
	if valueOperand {
		validValues = len(source.Values) == 1
	}
	if !validValues || counted && !validCountRule(source.Count) || !counted && source.Count != nil {
		return result, errors.New("evaluation: invalid IFEval structure operands")
	}
	if valueOperand {
		value := source.Values[0]
		if value == "" {
			return result, errors.New("evaluation: empty IFEval operand")
		}
		for _, r := range value {
			if r > unicode.MaxASCII {
				return result, errors.New("evaluation: unsupported non-ASCII IFEval operand")
			}
		}
		switch source.Kind {
		case ifevalLetters:
			if len(value) != 1 || !strings.ContainsAny(value, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") {
				return result, errors.New("evaluation: unresolved IFEval letter")
			}
		case ifevalFrequency:
			value = trimIFEvalSpace(value)
			if value == "" {
				return result, errors.New("evaluation: empty normalized IFEval keyword")
			}
			pattern, err := compileIFEvalKeyword(value, false)
			if err != nil {
				return result, err
			}
			result.patterns = []*regexp.Regexp{pattern}
		case ifevalSections:
			value = trimIFEvalSpace(value)
			if regexp.QuoteMeta(value) != value {
				return result, errors.New("evaluation: unsupported IFEval section regex")
			}
			patterns = []string{ifevalSpaceClass + `?` + value + ifevalSpaceClass + `?\p{Nd}+` + ifevalSpaceClass + `?`}
		case ifevalPostscript:
			value = trimIFEvalSpace(value)
			switch value {
			case "P.P.S":
				value = `p\.` + ifevalSpaceClass + `?p\.` + ifevalSpaceClass + `?s`
			case "P.S.":
				value = `p\.` + ifevalSpaceClass + `?s\.`
			default:
				if regexp.QuoteMeta(value) != value {
					return result, errors.New("evaluation: unsupported IFEval postscript regex")
				}
				value = strings.ToLower(value)
			}
			patterns = []string{`(?m)` + ifevalSpaceClass + `*` + value + `.*$`}
		}
	}
	for _, text := range patterns {
		pattern, err := regexp.Compile(text)
		if err != nil {
			return result, err
		}
		result.patterns = append(result.patterns, pattern)
	}
	return result, nil
}

func (rule compiledInstructionRule) matchesIFEvalStructure(response string) bool {
	if rule.source.Kind == ifevalIndexedParagraph {
		return rule.matchesIFEvalParagraph(response)
	}
	count := 0
	switch rule.source.Kind {
	case ifevalWords, ifevalFrequency, ifevalPlaceholders, ifevalBullets, ifevalSections:
		for _, pattern := range rule.patterns {
			count += len(pattern.FindAllStringIndex(response, -1))
		}
	case ifevalLetters:
		count = strings.Count(lowerIFEvalASCIIComparison(response), strings.ToLower(rule.source.Values[0]))
	case ifevalHighlights:
		for _, pattern := range rule.patterns {
			for _, match := range pattern.FindAllString(response, -1) {
				if trimIFEvalSpace(strings.Trim(match, "*")) != "" {
					count++
				}
			}
		}
	case ifevalParagraphs:
		parts, valid := splitIFEvalSections(response, "***")
		if !valid {
			return false
		}
		count = len(parts)
	case ifevalTwoResponses:
		parts, valid := splitIFEvalSections(response, "******")
		return valid && len(parts) == 2 && parts[0] != parts[1]
	case ifevalTitle:
		for _, match := range rule.patterns[0].FindAllString(response, -1) {
			if trimIFEvalSpace(strings.TrimRight(strings.TrimLeft(match, "<"), ">")) != "" {
				return true
			}
		}
		return false
	case ifevalPostscript:
		return rule.patterns[0].MatchString(lowerIFEvalASCIIComparison(response))
	case ifevalRepeat:
		return strings.HasPrefix(lowerIFEvalASCIIComparison(trimIFEvalSpace(response)), lowerIFEvalASCIIComparison(trimIFEvalSpace(rule.source.Values[0])))
	default:
		return false
	}
	return compareCount(count, *rule.source.Count)
}

func splitIFEvalSections(response, separator string) ([]string, bool) {
	parts := strings.Split(response, separator)
	var valid []string
	for index, part := range parts {
		part = trimIFEvalSpace(part)
		if part == "" {
			if index != 0 && index != len(parts)-1 {
				return nil, false
			}
		} else {
			valid = append(valid, part)
		}
	}
	return valid, true
}

func ifevalStructureFor(id string, kwargs map[string]json.RawMessage) ([]InstructionRule, bool) {
	if id == "length_constraints:nth_paragraph_first_word" {
		return ifevalParagraphFor(id, kwargs)
	}
	text := func(name string) (string, bool) {
		var value string
		err := json.Unmarshal(kwargs[name], &value)
		return value, err == nil && value != ""
	}
	count := func(name, relationName, relation string) *CountRule {
		var value *int
		if err := json.Unmarshal(kwargs[name], &value); err != nil || value == nil || *value <= 0 {
			return nil
		}
		if relationName != "" {
			choice, ok := text(relationName)
			if !ok {
				return nil
			}
			switch choice {
			case "less than":
				relation = ifevalLessThan
			case "at least":
				relation = RelationAtLeast
			default:
				return nil
			}
		}
		return &CountRule{Relation: relation, Value: *value}
	}
	rule := InstructionRule{Name: id}
	arg := ""
	switch id {
	case "length_constraints:number_sentences":
		rule.Kind = ifevalSentences
		rule.Count = count("num_sentences", "relation", "")
	case "change_case:capital_word_frequency":
		rule.Kind = ifevalCapitalWords
		rule.Count = count("capital_frequency", "capital_relation", "")
	case "length_constraints:number_words":
		rule.Kind = ifevalWords
		rule.Count = count("num_words", "relation", "")
	case "keywords:letter_frequency":
		rule.Kind = ifevalLetters
		rule.Count = count("let_frequency", "let_relation", "")
		arg = "letter"
	case "keywords:frequency":
		rule.Kind = ifevalFrequency
		rule.Count = count("frequency", "relation", "")
		arg = "keyword"
	case "detectable_content:number_placeholders":
		rule.Kind = ifevalPlaceholders
		rule.Count = count("num_placeholders", "", RelationAtLeast)
	case "detectable_format:number_bullet_lists":
		rule.Kind = ifevalBullets
		rule.Count = count("num_bullets", "", RelationEqual)
	case "detectable_format:number_highlighted_sections":
		rule.Kind = ifevalHighlights
		rule.Count = count("num_highlights", "", RelationAtLeast)
	case "detectable_format:multiple_sections":
		rule.Kind = ifevalSections
		rule.Count = count("num_sections", "", RelationAtLeast)
		arg = "section_spliter"
	case "length_constraints:number_paragraphs":
		rule.Kind = ifevalParagraphs
		rule.Count = count("num_paragraphs", "", RelationEqual)
	case "detectable_content:postscript":
		rule.Kind = ifevalPostscript
		arg = "postscript_marker"
	case "detectable_format:title":
		rule.Kind = ifevalTitle
	case "detectable_format:constrained_response":
		return []InstructionRule{{Name: id, Kind: RuleRegex, Values: []string{`My answer is (?:yes|no|maybe)\.`}}}, true
	case "combination:two_responses":
		rule.Kind = ifevalTwoResponses
	case "combination:repeat_prompt":
		rule.Kind = ifevalRepeat
		arg = "prompt_to_repeat"
	default:
		return nil, false
	}
	if arg != "" {
		value, ok := text(arg)
		if !ok {
			return nil, false
		}
		rule.Values = []string{value}
	}
	if _, err := compileIFEvalStructure(rule); err != nil {
		return nil, false
	}
	return []InstructionRule{rule}, true
}
