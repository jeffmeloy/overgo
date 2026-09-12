package evaluation

import (
	"errors"
	"regexp"
	"strings"
	"unicode"
)

const (
	ifevalKeywordsRule  = "lm-eval/ifeval/keywords/v1"
	ifevalForbiddenRule = "lm-eval/ifeval/forbidden-words/v1"
	ifevalEndRule       = "lm-eval/ifeval/end-phrase/v1"
)

// Admit the fixed-width pattern grammar used by the retained IFEval corpus.
// Unknown regex syntax must not silently become literal substring matching.
func compileIFEvalKeyword(value string, boundary bool) (*regexp.Regexp, error) {
	word := func(r rune) bool {
		return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_'
	}
	var pattern strings.Builder
	for _, r := range value {
		switch {
		case r == 'i' || r == 'I':
			// Python IGNORECASE also matches dotted and dotless I.
			pattern.WriteString("[iIİı]")
		case word(r) || r == ' ' || r == '-' || r == '.':
			pattern.WriteRune(r)
		default:
			return nil, errors.New("evaluation: unsupported IFEval keyword regex syntax")
		}
	}
	source := pattern.String()
	if boundary {
		if value == "" || !word(rune(value[0])) || !word(rune(value[len(value)-1])) {
			return nil, errors.New("evaluation: unsupported IFEval word-boundary edge")
		}
		source = `(?:^|[^\p{L}\p{N}_])(?:` + source + `)(?:$|[^\p{L}\p{N}_])`
	}
	return regexp.Compile("(?i)" + source)
}

// Python str.strip includes the ASCII information separators.
func trimIFEvalSpace(value string) string {
	return strings.TrimFunc(value, func(r rune) bool { return unicode.IsSpace(r) || r >= '\x1c' && r <= '\x1f' })
}

func matchesIFEvalEnd(response, phrase string) bool {
	// Python lower expands dotted I; the admitted phrase vocabulary is ASCII.
	lower := func(value string) string { return strings.ToLower(strings.ReplaceAll(value, "İ", "i\u0307")) }
	return strings.HasSuffix(lower(strings.Trim(trimIFEvalSpace(response), `"`)), lower(trimIFEvalSpace(phrase)))
}
