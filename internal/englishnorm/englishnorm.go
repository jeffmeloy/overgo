// Package englishnorm writes an English text the way the evaluations of
// speech recognizers write a reference and a transcript before counting the
// words that differ between them: the English normalizer of Whisper's
// evaluation, which the Open ASR leaderboard and the recognizers' own model
// cards count word error rates with.
//
// A recognizer writes what it hears in its own conventions. One writes "10",
// "3rd" and "was not" where the speaker's text has "ten", "third" and
// "wasn't"; another writes capitals and commas. Counted word against word,
// each of those is an error that no one misheard. The normalizer writes both
// sides one way: lower case, no bracketed asides or hesitations, contractions
// and titles spelled out, symbols and diacritics removed, and numbers as
// digits.
//
// It is a port of that code, step for step and in its order, including what
// it does that a reader might not choose: "pierc'd" becomes "pierc would",
// and "and a half" at the start of a text is dropped. A word error rate is
// comparable with a published one only if it is counted the same way, and
// the port is held to the code's own output on the evaluations' texts.
//
// Two things that code has are not here. It is given a table of British and
// American spellings, which ships with a Whisper model's files and is not on
// the machine the port was made on; without the table that code's spelling
// step changes nothing, and that is the behaviour ported. And it unfolds
// compatibility characters by Unicode's decomposition, which Go's standard
// library does not carry: the port embeds what that code gives for each
// character that decomposes in the first two planes of Unicode, apart from
// Hangul syllables and CJK compatibility ideographs, and treats any other
// character by its category alone.
package englishnorm

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2/v2"
)

// tablesJSON is the normalizer's own tables as its code builds them, written
// by scripts/python_golden_references/whisper_english_normalizer.py.
//
//go:embed tables.json
var tablesJSON []byte

// The patterns the normalizer's code holds in its statements, as that code
// writes them. Its tables of patterns, the hesitations and the contractions,
// are in the embedded tables.
const (
	bracketedPattern   = `[<\[][^>\]]*[>\]]`
	parenthesesPattern = `\(([^)]+?)\)`
	apostrophePattern  = `\s+'`
	digitCommaPattern  = `(\d),(\d)`
	periodPattern      = `\.([^0-9]|$)`
	strandedPattern    = `[.$¢€£]([^0-9])`
	percentPattern     = `([^0-9])%`
	spacesPattern      = `\s+`
	halfPattern        = `\band\s+a\s+half\b`
	letterDigitPattern = `([a-z])([0-9])`
	digitLetterPattern = `([0-9])([a-z])`
	suffixPattern      = `([0-9])\s+(st|nd|rd|th|s)\b`
	centsPattern       = `([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b`
	loneCentsPattern   = `[€£$]0.([0-9]{1,2})\b`
	onePattern         = `\b1(s?)\b`
)

// suffixedNumber is a number word that carries an ending, as "thirds" carries
// three and "rds".
type suffixedNumber struct {
	Value  string `json:"value"`
	Suffix string `json:"suffix"`
}

// tables is the embedded file: the normalizer's tables under that code's own
// names for them.
type tables struct {
	Ignored             string                       `json:"ignored"`
	Replacers           [][]string                   `json:"replacers"`
	Kept                string                       `json:"kept"`
	Folded              map[string]string            `json:"folded"`
	Zeros               []string                     `json:"zeros"`
	Ones                map[string]string            `json:"ones"`
	OnesSuffixed        map[string]suffixedNumber    `json:"ones_suffixed"`
	Tens                map[string]string            `json:"tens"`
	TensSuffixed        map[string]suffixedNumber    `json:"tens_suffixed"`
	Multipliers         map[string]string            `json:"multipliers"`
	MultipliersSuffixed map[string]suffixedNumber    `json:"multipliers_suffixed"`
	PrecedingPrefixers  map[string]string            `json:"preceding_prefixers"`
	FollowingPrefixers  map[string]string            `json:"following_prefixers"`
	Suffixers           map[string]string            `json:"suffixers"`
	SplitSuffixers      map[string]map[string]string `json:"split_suffixers"`
	Specials            []string                     `json:"specials"`
	Repeats             map[string]int               `json:"repeats"`
}

// rewrite replaces every match of a pattern, as that code's re.sub does.
type rewrite struct {
	pattern *regexp2.Regexp
	with    string
}

// apply rewrites the text. The pattern engine fails only when a match runs
// past a time limit, and none is set.
func (r rewrite) apply(text string) string {
	result, err := r.pattern.Replace(text, r.with, 0, -1)
	if err != nil {
		panic(fmt.Sprintf("englishnorm: %s: %v", r.pattern, err))
	}
	return result
}

// literal is a replacement written out as it stands, with no reference to
// what the pattern matched.
func literal(text string) string { return strings.ReplaceAll(text, "$", "$$") }

// numberWord is a number word's value and the ending it carries, if any.
type numberWord struct {
	value  *big.Int
	suffix string
}

// normalizer is the tables made ready: patterns compiled and number words
// read.
type normalizer struct {
	before, numbered, after []rewrite
	half, cents, loneCents  *regexp2.Regexp
	one                     rewrite
	kept                    map[rune]bool
	folded                  map[rune]string
	words, zeros, decimals  map[string]bool
	prefixes                map[rune]bool
	ones, tens, multipliers map[string]*big.Int
	suffixed                map[string]numberWord
	onesEnded, tensEnded    map[string]bool
	preceding, following    map[string]string
	suffixers               map[string]string
	splitSuffixers          map[string]map[string]string
	specials                map[string]bool
	repeats                 map[string]int
}

var loaded = sync.OnceValue(func() *normalizer {
	var source tables
	if err := json.Unmarshal(tablesJSON, &source); err != nil {
		panic("englishnorm: embedded tables: " + err.Error())
	}
	n := &normalizer{
		half: regexp2.MustCompile(halfPattern), cents: regexp2.MustCompile(centsPattern), loneCents: regexp2.MustCompile(loneCentsPattern),
		one:  rewrite{regexp2.MustCompile(onePattern), "one$1"},
		kept: map[rune]bool{}, folded: map[rune]string{}, words: map[string]bool{}, zeros: map[string]bool{}, decimals: map[string]bool{},
		prefixes: map[rune]bool{}, ones: map[string]*big.Int{}, tens: map[string]*big.Int{}, multipliers: map[string]*big.Int{},
		suffixed: map[string]numberWord{}, onesEnded: map[string]bool{}, tensEnded: map[string]bool{},
		preceding: source.PrecedingPrefixers, following: source.FollowingPrefixers,
		suffixers: source.Suffixers, splitSuffixers: source.SplitSuffixers, specials: map[string]bool{}, repeats: source.Repeats,
	}
	n.before = []rewrite{
		{regexp2.MustCompile(bracketedPattern), ""}, {regexp2.MustCompile(parenthesesPattern), ""},
		{regexp2.MustCompile(source.Ignored), ""}, {regexp2.MustCompile(apostrophePattern), "'"},
	}
	for _, replacer := range source.Replacers {
		n.before = append(n.before, rewrite{regexp2.MustCompile(replacer[0]), literal(replacer[1])})
	}
	n.before = append(n.before, rewrite{regexp2.MustCompile(digitCommaPattern), "$1$2"}, rewrite{regexp2.MustCompile(periodPattern), " $1"})
	n.numbered = []rewrite{
		{regexp2.MustCompile(letterDigitPattern), "$1 $2"}, {regexp2.MustCompile(digitLetterPattern), "$1 $2"},
		{regexp2.MustCompile(suffixPattern), "$1$2"},
	}
	n.after = []rewrite{
		{regexp2.MustCompile(strandedPattern), " $1"}, {regexp2.MustCompile(percentPattern), "$1 "}, {regexp2.MustCompile(spacesPattern), " "},
	}
	for _, r := range source.Kept {
		n.kept[r] = true
	}
	for character, written := range source.Folded {
		r, _ := utf8.DecodeRuneInString(character)
		n.folded[r] = written
	}
	whole := func(digits string) *big.Int {
		value, ok := new(big.Int).SetString(digits, tenUnit)
		if !ok {
			panic("englishnorm: embedded tables: " + digits + " is no number")
		}
		return value
	}
	for _, word := range source.Zeros {
		n.zeros[word], n.decimals[word], n.words[word] = true, true, true
	}
	for word, digits := range source.Ones {
		n.ones[word], n.decimals[word], n.words[word] = whole(digits), true, true
	}
	for word, digits := range source.Tens {
		n.tens[word], n.decimals[word], n.words[word] = whole(digits), true, true
	}
	for word, digits := range source.Multipliers {
		n.multipliers[word], n.words[word] = whole(digits), true
	}
	for word, entry := range source.OnesSuffixed {
		n.suffixed[word], n.onesEnded[word], n.words[word] = numberWord{whole(entry.Value), entry.Suffix}, true, true
	}
	for word, entry := range source.TensSuffixed {
		n.suffixed[word], n.tensEnded[word], n.words[word] = numberWord{whole(entry.Value), entry.Suffix}, true, true
	}
	for word, entry := range source.MultipliersSuffixed {
		n.suffixed[word], n.words[word] = numberWord{whole(entry.Value), entry.Suffix}, true
	}
	for _, prefixers := range []map[string]string{source.PrecedingPrefixers, source.FollowingPrefixers} {
		for word, symbol := range prefixers {
			r, _ := utf8.DecodeRuneInString(symbol)
			n.prefixes[r], n.words[word] = true, true
		}
	}
	for word := range source.Suffixers {
		n.words[word] = true
	}
	for word := range source.SplitSuffixers {
		n.words[word] = true
	}
	for _, word := range source.Specials {
		n.specials[word], n.words[word] = true, true
	}
	return n
})

// Text writes an English text the way a word error rate is counted over: see
// the package. What it returns may open or close with a space, as that
// code's does; its words are what is compared.
func Text(text string) string {
	n := loaded()
	text = strings.ToLower(text)
	for _, step := range n.before {
		text = step.apply(text)
	}
	text = n.numbers(n.fold(text))
	for _, step := range n.after {
		text = step.apply(text)
	}
	return text
}

// fold removes symbols and diacritics: a mark that sits on a letter is
// dropped, any other mark, symbol or punctuation becomes a space, the few
// symbols a number is written with are kept, and a character that decomposes
// becomes what that code makes of its parts.
func (n *normalizer) fold(text string) string {
	var folded strings.Builder
	for _, r := range text {
		switch written, decomposes := n.folded[r]; {
		case decomposes:
			folded.WriteString(written)
		case n.kept[r]:
			folded.WriteRune(r)
		case unicode.Is(unicode.Mn, r):
		case unicode.Is(unicode.M, r), unicode.Is(unicode.S, r), unicode.Is(unicode.P, r):
			folded.WriteByte(' ')
		default:
			folded.WriteRune(r)
		}
	}
	return folded.String()
}
