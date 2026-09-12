package evaluation

import (
	"encoding/json"
	"errors"
	"strings"
	"unicode"
)

const ifevalIndexedParagraph = "lm-eval/ifeval/indexed-paragraph/v1"

func compileIFEvalParagraph(source InstructionRule) (compiledInstructionRule, error) {
	result := compiledInstructionRule{source: source}
	if source.Count == nil || source.Count.Relation != RelationEqual || source.Count.Value <= 0 || source.Ordinal <= 0 || source.Ordinal > source.Count.Value || len(source.Values) != 1 || source.Values[0] == "" {
		return result, errors.New("evaluation: unresolved IFEval paragraph operands")
	}
	// The retained corpus declares ASCII target words. Other vocabularies need
	// a bound Unicode lowercase contract before admission.
	for _, r := range source.Values[0] {
		if r > unicode.MaxASCII {
			return result, errors.New("evaluation: unsupported non-ASCII IFEval paragraph operand")
		}
	}
	return result, nil
}

func (rule compiledInstructionRule) matchesIFEvalParagraph(response string) bool {
	paragraphs := strings.Split(response, "\n\n")
	count := 0
	for _, paragraph := range paragraphs {
		if trimIFEvalSpace(paragraph) != "" {
			count++
		}
	}
	if count != rule.source.Count.Value || rule.source.Ordinal > count {
		return false
	}
	// Native counts nonempty paragraphs but indexes the original split.
	paragraph := trimIFEvalSpace(paragraphs[rule.source.Ordinal-1])
	if paragraph == "" {
		return false
	}
	if end := strings.IndexFunc(paragraph, isIFEvalSpace); end >= 0 {
		paragraph = paragraph[:end]
	}
	word := strings.TrimLeft(strings.TrimLeft(paragraph, "'"), `"`)
	if end := strings.IndexAny(word, `.,?!'"`); end >= 0 {
		word = word[:end]
	}
	return lowerIFEvalASCIIComparison(word) == strings.ToLower(rule.source.Values[0])
}

func ifevalParagraphFor(id string, kwargs map[string]json.RawMessage) ([]InstructionRule, bool) {
	var count, ordinal int
	var first string
	if json.Unmarshal(kwargs["num_paragraphs"], &count) != nil || json.Unmarshal(kwargs["nth_paragraph"], &ordinal) != nil || json.Unmarshal(kwargs["first_word"], &first) != nil {
		return nil, false
	}
	rule := InstructionRule{Name: id, Kind: ifevalIndexedParagraph, Values: []string{first}, Count: &CountRule{Relation: RelationEqual, Value: count}, Ordinal: ordinal}
	if _, err := compileIFEvalParagraph(rule); err != nil {
		return nil, false
	}
	return []InstructionRule{rule}, true
}
