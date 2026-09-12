package evaluation

import "errors"

func compileIFEvalTokenizerRule(source InstructionRule) (compiledInstructionRule, error) {
	result := compiledInstructionRule{source: source}
	if source.Count == nil || source.Count.Value <= 0 || source.Count.Relation != RelationAtLeast && source.Count.Relation != ifevalLessThan || len(source.Values) != 0 || source.Ordinal != 0 {
		return result, errors.New("evaluation: unresolved native tokenizer operands")
	}
	tokenizer, err := compiledIFEvalTokenizer()
	if err != nil {
		return result, err
	}
	result.tokenizer = tokenizer
	return result, nil
}

func (rule compiledInstructionRule) matchesIFEvalTokenizer(response string) (bool, error) {
	var count int
	switch rule.source.Kind {
	case ifevalSentences:
		sentences, err := rule.tokenizer.sentences(response)
		if err != nil {
			return false, err
		}
		count = len(sentences)
	case ifevalCapitalWords:
		words, err := rule.tokenizer.words(response)
		if err != nil {
			return false, err
		}
		for _, word := range words {
			if ifevalUniformCase(word, nltkUpper) {
				count++
			}
		}
	default:
		return false, errors.New("evaluation: unknown native tokenizer rule")
	}
	return compareCount(count, *rule.source.Count), nil
}
