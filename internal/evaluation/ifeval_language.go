package evaluation

import (
	"errors"
	"math"
	"slices"
	"strings"
)

// Native langdetect normalization and classification. Profile attribution,
// license, ordering and source hashes are retained with its frozen inputs.
func (p *ifevalLanguageDetector) features(text string) (string, []string, error) {
	var err error
	if text, err = p.url.Replace(text, " ", 0, -1); err != nil {
		return "", nil, err
	}
	if text, err = p.mail.Replace(text, " ", 0, -1); err != nil {
		return "", nil, err
	}
	runes := []rune(text)
	var vietnamese strings.Builder
	for index := 0; index < len(runes); index++ {
		if index+1 < len(runes) {
			if mapped, found := p.input.Vietnamese[string(runes[index:index+2])]; found {
				vietnamese.WriteString(mapped)
				index++
				continue
			}
		}
		vietnamese.WriteRune(runes[index])
	}
	runes = []rune(vietnamese.String())
	var collapsed []rune
	var previous rune
	for _, r := range runes[:min(len(runes), p.input.Maximum)] {
		if r != ' ' || previous != ' ' {
			collapsed = append(collapsed, r)
		}
		previous = r
	}
	latin, nonLatin := 0, 0
	for _, r := range collapsed {
		if r >= 'A' && r <= 'z' {
			latin++
		} else if r >= 0x300 {
			nonLatin++
		}
	}
	// Native compares an integer block id to a string label, so it counts
	// every rune at or above U+0300 in this branch.
	if latin*2 < nonLatin {
		collapsed = slices.DeleteFunc(collapsed, func(r rune) bool { return r >= 'A' && r <= 'z' })
	}
	grams := []rune{' '}
	capital := false
	var features []string
	for _, raw := range collapsed {
		r := p.normalized(raw)
		last := grams[len(grams)-1]
		if last == ' ' {
			grams = []rune{' '}
			capital = false
			if r == ' ' {
				continue
			}
		} else if len(grams) >= p.input.NGram {
			grams = grams[1:]
		}
		grams = append(grams, r)
		if nltkUpper(r) {
			if nltkUpper(last) {
				capital = true
			}
		} else {
			capital = false
		}
		if capital {
			continue
		}
		for n := 1; n <= p.input.NGram; n++ {
			if len(grams) < n {
				break
			}
			word := string(grams[len(grams)-n:])
			if word != " " {
				if _, found := p.words[word]; found {
					features = append(features, word)
				}
			}
		}
	}
	return string(collapsed), features, nil
}

// CPython 3.13's compensated float sum preserves the native normalization order.
func sumIFEvalProbabilities(values []float64) float64 {
	sum, correction := 0.0, 0.0
	for _, value := range values {
		total := sum + value
		if math.Abs(sum) >= math.Abs(value) {
			correction += (sum - total) + value
		} else {
			correction += (value - total) + sum
		}
		sum = total
	}
	return sum + correction
}

func (p *ifevalLanguageDetector) probabilities(grams []string) ([]float64, error) {
	if len(grams) == 0 {
		return nil, nil
	}
	random := p.random()
	result := make([]float64, len(p.input.Languages))
	for range p.input.Trials {
		prob := make([]float64, len(result))
		for index := range prob {
			prob[index] = 1 / float64(len(prob))
		}
		alpha := p.input.Alpha + random.gauss()*p.input.AlphaWidth
		for iteration := 0; ; iteration++ {
			word := grams[random.choice(len(grams))]
			weight := alpha / p.input.Frequency
			for index := range prob {
				prob[index] *= weight + p.words[word][index]
			}
			// Native normalizes at iteration zero and after every five updates.
			if iteration%5 == 0 {
				sum := sumIFEvalProbabilities(prob)
				maximum := 0.0
				if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
					return nil, errors.New("evaluation: invalid language probability normalization")
				}
				for index := range prob {
					prob[index] /= sum
					maximum = max(maximum, prob[index])
				}
				if maximum > p.input.Convergence || iteration >= p.input.Iterations {
					break
				}
			}
		}
		for index := range result {
			result[index] += prob[index] / float64(p.input.Trials)
		}
	}
	return result, nil
}

func (p *ifevalLanguageDetector) detect(text string) (string, bool, error) {
	_, grams, err := p.features(text)
	if err != nil {
		return "", false, err
	}
	if len(grams) == 0 {
		return "", false, nil
	}
	prob, err := p.probabilities(grams)
	if err != nil {
		return "", false, err
	}
	best := -1
	for index, value := range prob {
		if value > p.input.Threshold && (best < 0 || value > prob[best]) {
			best = index
		}
	}
	if best < 0 {
		return "unknown", true, nil
	}
	return p.input.Languages[best].Name, true, nil
}

func compileIFEvalLanguageRule(source InstructionRule) (compiledInstructionRule, error) {
	result := compiledInstructionRule{source: source}
	if source.Count != nil || source.Ordinal != 0 {
		return result, errors.New("evaluation: invalid language operands")
	}
	p, err := compiledIFEvalLanguage()
	if err != nil {
		return result, err
	}
	if source.Kind == ifevalLanguageRule {
		if len(source.Values) != 1 || !slices.Contains(p.input.Declared, source.Values[0]) {
			return result, errors.New("evaluation: unresolved native language")
		}
	} else if len(source.Values) != 0 {
		return result, errors.New("evaluation: unexpected English-case operand")
	}
	result.language = p
	return result, nil
}

func (rule compiledInstructionRule) matchesIFEvalLanguage(response string) (bool, error) {
	expected := "en"
	switch rule.source.Kind {
	case ifevalLanguageRule:
		expected = rule.source.Values[0]
	case ifevalEnglishUpper:
		if !ifevalUniformCase(response, nltkUpper) {
			return false, nil
		}
	case ifevalEnglishLower:
		if !ifevalUniformCase(response, nltkLower) {
			return false, nil
		}
	default:
		return false, errors.New("evaluation: unknown language rule")
	}
	detected, features, err := rule.language.detect(response)
	if err != nil {
		return false, err
	}
	// Native catches CantDetectError as followed. Execution failures stay errors.
	return !features || detected == expected, nil
}
