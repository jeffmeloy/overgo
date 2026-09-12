package evaluation

import (
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2/v2"
)

type nltkToken struct {
	text, kind                            string
	stop, abbreviation, ellipsis, initial bool
}

var nltkNumeric = regexp.MustCompile(`^-?[.,]?\p{Nd}[\p{Nd},.\-]*\.?$`)

func makeNLTKToken(text string) nltkToken {
	kind := lowerIFEvalASCIIComparison(text)
	if nltkNumeric.MatchString(kind) {
		kind = "##number##"
	}
	runes := []rune(text)
	initial := len(runes) == 2 && runes[1] == '.' && (unicode.IsLetter(runes[0]) || unicode.IsNumber(runes[0]) || runes[0] == '_') && !unicode.Is(unicode.Nd, runes[0])
	return nltkToken{text: text, kind: kind, initial: initial}
}

func (token nltkToken) noPeriod() string {
	if len(token.kind) > 1 {
		return strings.TrimSuffix(token.kind, ".")
	}
	return token.kind
}
func (token nltkToken) noSentencePeriod() string {
	if token.stop {
		return token.noPeriod()
	}
	return token.kind
}

// The native orthographic heuristic returns true, false or unknown.
func (p *ifevalTokenizer) sentenceStarter(token nltkToken) (bool, bool) {
	if slices.Contains([]string{";", ":", ",", ".", "!", "?"}, token.text) {
		return false, true
	}
	context := p.ortho[token.noSentencePeriod()]
	first, _ := utf8.DecodeRuneInString(token.text)
	if nltkUpper(first) && context&nltkLowerContexts != 0 && context&nltkMiddleUpper == 0 {
		return true, true
	}
	if nltkLower(first) && (context&nltkUpperContexts != 0 || context&nltkBeginLower == 0) {
		return false, true
	}
	return false, false
}

func (p *ifevalTokenizer) containsSentenceBreak(text string) (bool, error) {
	var tokens []nltkToken
	for line := range strings.SplitSeq(text, "\n") {
		matches, err := nltkMatches(p.word, line)
		if err != nil {
			return false, err
		}
		for _, m := range matches {
			token := makeNLTKToken(m.String())
			s := token.text
			switch {
			case s == "." || s == "?" || s == "!":
				token.stop = true
			case len(s) >= 2 && strings.Trim(s, ".") == "":
				token.ellipsis = true
			case strings.HasSuffix(s, ".") && !strings.HasSuffix(s, ".."):
				key := lowerIFEvalASCIIComparison(strings.TrimSuffix(s, "."))
				parts := strings.Split(key, "-")
				token.abbreviation = p.abbreviations[key] || p.abbreviations[parts[len(parts)-1]]
				token.stop = !token.abbreviation
			}
			tokens = append(tokens, token)
		}
	}
	for index := 0; index+1 < len(tokens); index++ {
		a, b := &tokens[index], tokens[index+1]
		if strings.HasSuffix(a.text, ".") {
			typ, next := a.noPeriod(), b.noSentencePeriod()
			first, _ := utf8.DecodeRuneInString(b.text)
			if p.collocations[[2]string{typ, next}] {
				a.stop = false
				a.abbreviation = true
			} else {
				starter, known := p.sentenceStarter(b)
				if (a.abbreviation || a.ellipsis) && !a.initial && (starter || nltkUpper(first) && p.starters[next]) {
					a.stop = true
				}
				if (a.initial || typ == "##number##") && (known && !starter || !known && a.initial && nltkUpper(first) && p.ortho[next]&nltkLowerContexts == 0) {
					a.stop = false
					a.abbreviation = true
				}
			}
		}
		// The native contains-break predicate ignores the last token.
		if a.stop {
			return true, nil
		}
	}
	return false, nil
}

type nltkSpan struct{ start, end int }
type nltkPotential struct {
	match *regexp2.Match
	prior nltkSpan
}

func (p *ifevalTokenizer) sentences(text string) ([]string, error) {
	var previous *nltkPotential
	var contexts []nltkPotential
	matches, err := nltkMatches(p.period, text)
	if err != nil {
		return nil, err
	}
	for _, match := range matches {
		start, _ := match.ByteRange()
		old := nltkSpan{}
		if previous != nil {
			old = previous.prior
		}
		before := text[old.end:start]
		last := strings.LastIndexAny(before, " \t\n\r\v\f")
		begin := old.start
		// NLTK's overlap removal uses string.whitespace and treats index zero
		// as absent. Preserve that rule independently of Unicode token spacing.
		if last > 0 {
			begin = last + old.end + 1
		}
		current := nltkPotential{match, nltkSpan{begin, start}}
		if previous != nil && old.end <= begin {
			contexts = append(contexts, *previous)
		}
		previous = &current
	}
	if previous != nil {
		contexts = append(contexts, *previous)
	}
	last := 0
	var spans []nltkSpan
	for _, c := range contexts {
		m := c.match
		start, length := m.ByteRange()
		stop, err := p.containsSentenceBreak(text[c.prior.start:c.prior.end] + m.String() + m.GroupByName("after_tok").String())
		if err != nil {
			return nil, err
		}
		if stop {
			spans = append(spans, nltkSpan{last, start + length})
			next := m.GroupByName("next_tok")
			if next.String() != "" {
				last, _ = next.ByteRange()
			} else {
				last = start + length
			}
		}
	}
	spans = append(spans, nltkSpan{last, len(strings.TrimRightFunc(text, isIFEvalSpace))})
	var sentences []string
	realign := 0
	for index, s := range spans {
		s.start += realign
		if index+1 < len(spans) {
			next := spans[index+1]
			match, err := p.realignment.FindStringMatch(text[next.start:next.end])
			if err != nil {
				return nil, err
			}
			at, length := 0, 0
			if match != nil {
				at, length = match.ByteRange()
			}
			if match != nil && at == 0 {
				sentences = append(sentences, text[s.start:next.start+len(strings.TrimRightFunc(match.String(), isIFEvalSpace))])
				realign = length
				continue
			}
			realign = 0
		}
		if s.start < s.end {
			sentences = append(sentences, text[s.start:s.end])
		}
	}
	return sentences, nil
}

func (p *ifevalTokenizer) words(text string) ([]string, error) {
	sentences, err := p.sentences(text)
	if err != nil {
		return nil, err
	}
	var words []string
	for _, sentence := range sentences {
		for _, name := range []string{"STARTING_QUOTES", "PUNCTUATION", "PARENS_BRACKETS", "DOUBLE_DASHES", "ENDING_QUOTES", "CONTRACTIONS2", "CONTRACTIONS3"} {
			if name == "ENDING_QUOTES" {
				sentence = " " + sentence + " "
			}
			for _, rule := range p.substitutions[name] {
				sentence, err = rule.pattern.Replace(sentence, rule.replacement, 0, -1)
				if err != nil {
					return nil, err
				}
			}
		}
		words = append(words, strings.FieldsFunc(sentence, isIFEvalSpace)...)
	}
	return words, nil
}
