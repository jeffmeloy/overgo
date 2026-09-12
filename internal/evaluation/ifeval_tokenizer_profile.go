package evaluation

import (
	"crypto/sha256"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2/v2"
)

//go:embed testdata/ifeval_nltk_profile.json
var ifevalNLTKProfile []byte

const (
	ifevalSentences    = "lm-eval/ifeval/sentences/nltk-3.9.2/v1"
	ifevalCapitalWords = "lm-eval/ifeval/capital-words/nltk-3.9.2/v1"
	// Python whitespace in regexp2 syntax; POSIX classes belong to Go's RE2.
	nltkSpaceClass = `[\t\n\v\f\r \u001c-\u001f\u0085\p{Z}]`
	nltkWordClass  = `[\p{L}\p{N}_]`
	// Orthographic flags in NLTK's punkt_tab format.
	nltkBeginLower    = 1 << 4
	nltkMiddleUpper   = 1 << 2
	nltkUpperContexts = 1<<1 | nltkMiddleUpper | 1<<3
	nltkLowerContexts = nltkBeginLower | 1<<5 | 1<<6
)

type nltkSubstitution struct {
	pattern     *regexp2.Regexp
	replacement string
}
type ifevalTokenizer struct {
	word, period, realignment *regexp2.Regexp
	abbreviations, starters   map[string]bool
	collocations              map[[2]string]bool
	ortho                     map[string]int
	substitutions             map[string][]nltkSubstitution
}

var compiledIFEvalTokenizer = sync.OnceValues(func() (*ifevalTokenizer, error) { return loadIFEvalTokenizer(ifevalNLTKProfile) })

func loadIFEvalTokenizer(data []byte) (*ifevalTokenizer, error) {
	if fmt.Sprintf("%x", sha256.Sum256(data)) != "3d2055c31e46ee2542569ceab92d2e7beabe336b196e4b7698662e4fe1c23892" {
		return nil, errors.New("evaluation: frozen NLTK profile identity differs")
	}
	var input struct {
		Schema, Version, License string
		Files                    map[string]struct{ Content, SHA256 string }
		Regexes                  map[string]string
		Rules                    map[string][]struct {
			Pattern, Replacement string
			Flags                int
		}
	}
	if err := json.Unmarshal(data, &input); err != nil {
		return nil, err
	}
	if input.Schema != "overgo/ifeval-nltk-profile/v1" || input.Version != "3.9.2" || input.License == "" || len(input.Files) != 4 || len(input.Regexes) != 3 || len(input.Rules) != 7 {
		return nil, errors.New("evaluation: invalid frozen NLTK profile")
	}
	p := &ifevalTokenizer{abbreviations: map[string]bool{}, starters: map[string]bool{}, collocations: map[[2]string]bool{}, ortho: map[string]int{}, substitutions: map[string][]nltkSubstitution{}}
	for name, file := range input.Files {
		if !utf8.ValidString(file.Content) || fmt.Sprintf("%x", sha256.Sum256([]byte(file.Content))) != file.SHA256 {
			return nil, errors.New("evaluation: NLTK table identity differs")
		}
		for line := range strings.SplitSeq(strings.TrimSuffix(file.Content, "\n"), "\n") {
			switch name {
			case "abbrev_types.txt":
				if p.abbreviations[line] {
					return nil, errors.New("evaluation: duplicate NLTK abbreviation")
				}
				p.abbreviations[line] = true
			case "sent_starters.txt":
				if p.starters[line] {
					return nil, errors.New("evaluation: duplicate NLTK sentence starter")
				}
				p.starters[line] = true
			case "collocations.tab":
				a, b, ok := strings.Cut(line, "\t")
				key := [2]string{a, b}
				if !ok || p.collocations[key] {
					return nil, errors.New("evaluation: invalid NLTK collocation")
				}
				p.collocations[key] = true
			case "ortho_context.tab":
				a, b, ok := strings.Cut(line, "\t")
				var value int
				if !ok || json.Unmarshal([]byte(b), &value) != nil || value < 0 || value & ^(nltkUpperContexts|nltkLowerContexts) != 0 {
					return nil, errors.New("evaluation: invalid NLTK orthographic flags")
				}
				if _, found := p.ortho[a]; found {
					return nil, errors.New("evaluation: duplicate NLTK orthographic word")
				}
				p.ortho[a] = value
			default:
				return nil, errors.New("evaluation: unknown NLTK table")
			}
		}
	}
	var err error
	if p.word, err = compileNLTKRegex(input.Regexes["word"], regexp2.IgnorePatternWhitespace); err != nil {
		return nil, err
	}
	if p.period, err = compileNLTKRegex(input.Regexes["period"], regexp2.IgnorePatternWhitespace); err != nil {
		return nil, err
	}
	if p.realignment, err = compileNLTKRegex(input.Regexes["realignment"], regexp2.Multiline); err != nil {
		return nil, err
	}
	refs := regexp.MustCompile(`\\([0-9]+)`)
	for _, name := range []string{"STARTING_QUOTES", "PUNCTUATION", "PARENS_BRACKETS", "DOUBLE_DASHES", "ENDING_QUOTES", "CONTRACTIONS2", "CONTRACTIONS3"} {
		rules := input.Rules[name]
		if len(rules) == 0 {
			return nil, errors.New("evaluation: missing NLTK substitution stage")
		}
		for _, rule := range rules {
			flags := regexp2.None
			// These source rules declare Python UNICODE, with optional IGNORECASE.
			if rule.Flags != 32 && rule.Flags != 34 {
				return nil, errors.New("evaluation: unsupported NLTK regex flags")
			}
			if rule.Flags == 34 {
				flags |= regexp2.IgnoreCase
			}
			pattern, err := compileNLTKRegex(rule.Pattern, flags)
			if err != nil {
				return nil, err
			}
			replacement := strings.ReplaceAll(rule.Replacement, `\g<0>`, `${0}`)
			replacement = refs.ReplaceAllStringFunc(replacement, func(s string) string { return "${" + s[1:] + "}" })
			p.substitutions[name] = append(p.substitutions[name], nltkSubstitution{pattern, replacement})
		}
	}
	return p, nil
}

func compileNLTKRegex(pattern string, flags regexp2.RegexOptions) (*regexp2.Regexp, error) {
	if pattern == "" {
		return nil, errors.New("evaluation: empty NLTK regex")
	}
	if flags&regexp2.IgnoreCase != 0 {
		pattern = strings.ReplaceAll(strings.ReplaceAll(pattern, "(?i)", ""), "(?#X)", "")
		var expanded strings.Builder
		escaped := false
		for _, r := range pattern {
			if escaped {
				expanded.WriteRune(r)
				escaped = false
				continue
			}
			if r == '\\' {
				expanded.WriteRune(r)
				escaped = true
				continue
			}
			switch r {
			case '[':
				return nil, errors.New("evaluation: unsupported case-insensitive NLTK character class")
			case 'i', 'I':
				expanded.WriteString("[iIİı]")
			case 's', 'S':
				expanded.WriteString("[sSſ]")
			case 'k', 'K':
				expanded.WriteString("[kKK]")
			default:
				expanded.WriteRune(r)
			}
		}
		pattern = expanded.String()
	}
	pattern = strings.ReplaceAll(pattern, `\s`, nltkSpaceClass)
	pattern = strings.ReplaceAll(pattern, `\S`, `[^`+nltkSpaceClass[1:len(nltkSpaceClass)-1]+`]`)
	pattern = strings.ReplaceAll(pattern, `\w`, nltkWordClass)
	pattern = strings.ReplaceAll(pattern, `\b`, `(?:(?<=`+nltkWordClass+`)(?!`+nltkWordClass+`)|(?<!`+nltkWordClass+`)(?=`+nltkWordClass+`))`)
	pattern = strings.ReplaceAll(pattern, `(?P<`, `(?<`)
	return regexp2.Compile(pattern, flags)
}

func nltkMatches(pattern *regexp2.Regexp, text string) ([]*regexp2.Match, error) {
	var matches []*regexp2.Match
	for m, err := pattern.FindStringMatch(text); ; m, err = pattern.FindNextMatch(m) {
		if err != nil {
			return nil, err
		}
		if m == nil {
			return matches, nil
		}
		matches = append(matches, m)
	}
}

func nltkUpper(r rune) bool { return unicode.IsUpper(r) || unicode.Is(unicode.Other_Uppercase, r) }
func nltkLower(r rune) bool { return unicode.IsLower(r) || unicode.Is(unicode.Other_Lowercase, r) }
func ifevalUniformCase(text string, accept func(rune) bool) bool {
	found := false
	for _, r := range text {
		if nltkUpper(r) || nltkLower(r) || unicode.IsTitle(r) {
			if !accept(r) {
				return false
			}
			found = true
		}
	}
	return found
}
