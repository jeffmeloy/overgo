package sampling

import (
	"strings"
	"testing"
)

func TestGBNFLiteralsClassesGroupsAndRepetition(t *testing.T) {
	pieces := bytePieces("a", "b", "c", "1", "2", "3", "x", "", "a1", "12")
	grammar, err := NewGBNFGrammar(`
root ::= ("a" | [b-c]) [0-9]{2,3}
`, "root", pieces, []int{7})
	if err != nil {
		t.Fatal(err)
	}
	for _, tokens := range [][]int{
		{0, 3, 4, 7},
		{1, 9, 7},
		{2, 3, 4, 5, 7},
		{8, 4, 7},
	} {
		if !gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("grammar rejected tokens %v", tokens)
		}
	}
	for _, tokens := range [][]int{
		{6},
		{0, 3, 7},
		{0, 3, 4, 5, 5, 7},
		{0, 3, 4, 6},
	} {
		if gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("grammar accepted invalid tokens %v", tokens)
		}
	}
}

func TestGBNFRightRecursiveBalancedInput(t *testing.T) {
	grammar, err := NewGBNFGrammar(`
root ::= "(" root? ")"
`, "root", bytePieces("(", ")", "", "x", "()"), []int{2})
	if err != nil {
		t.Fatal(err)
	}
	for _, tokens := range [][]int{
		{0, 1, 2},
		{0, 0, 1, 1, 2},
		{4, 2},
	} {
		if !gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("balanced grammar rejected %v", tokens)
		}
	}
	for _, tokens := range [][]int{
		{0, 2},
		{0, 0, 1, 2},
		{3},
	} {
		if gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("balanced grammar accepted %v", tokens)
		}
	}
}

func TestGBNFUnicodeCanSpanByteTokens(t *testing.T) {
	grammar, err := NewGBNFGrammar(
		`root ::= "€"`,
		"root",
		[][]byte{{0xE2}, {0x82}, {0xAC}, nil, {'x'}},
		[]int{3},
	)
	if err != nil {
		t.Fatal(err)
	}
	state, err := grammar.initialState()
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range []int{0, 1, 2, 3} {
		var ok bool
		state, ok = grammar.advanceToken(state, token)
		if !ok {
			t.Fatalf("Unicode grammar rejected token %d", token)
		}
	}
	if !state.terminated {
		t.Fatal("Unicode grammar did not terminate on EOS")
	}
	state, err = grammar.initialState()
	if err != nil {
		t.Fatal(err)
	}
	state, ok := grammar.advanceToken(state, 0)
	if !ok {
		t.Fatal("Unicode grammar rejected a valid leading byte")
	}
	if _, ok := grammar.advanceToken(state, 4); ok {
		t.Fatal("Unicode grammar accepted an invalid continuation")
	}
}

func TestGBNFNegatedClassWildcardAndEscapes(t *testing.T) {
	grammar, err := NewGBNFGrammar(
		`root ::= [^a-c] . "\n" "\u20AC"`,
		"root",
		bytePieces("x", "a", "Z", "\n", "€", "", "xy"),
		[]int{5},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !gbnfAcceptsTokens(t, grammar, []int{0, 2, 3, 4, 5}) {
		t.Fatal("negated/wildcard grammar rejected valid input")
	}
	if gbnfAcceptsTokens(t, grammar, []int{1, 2, 3, 4, 5}) {
		t.Fatal("negated class accepted excluded input")
	}
}

func TestGBNFRejectsUndefinedLeftRecursiveAndMalformedRules(t *testing.T) {
	cases := []struct {
		source string
		want   string
	}{
		{`root ::= missing`, "undefined rule"},
		{`root ::= root "a" | "b"`, "left recursion"},
		{`root ::= "a"{3,2}`, "repetition"},
		{`root ::= [z-a]`, "reversed"},
		{`root ::= <[9]>`, "outside vocabulary"},
		{`other ::= "a"`, "root rule"},
	}
	for _, item := range cases {
		_, err := NewGBNFGrammar(item.source, "root", bytePieces("a", ""), []int{1})
		if err == nil || !strings.Contains(err.Error(), item.want) {
			t.Fatalf("grammar %q error = %v, want %q", item.source, err, item.want)
		}
	}
}

func TestGBNFTokenTerminalsMatchPinnedUpstreamSemantics(t *testing.T) {
	pieces := make([][]byte, 32)
	for index := range pieces {
		pieces[index] = []byte("x")
	}
	pieces[31] = nil
	grammar, err := NewGBNFGrammar(`
root ::= <[10]> content <[11]>
content ::= (!<[11]>)*
`, "root", pieces, []int{31})
	if err != nil {
		t.Fatal(err)
	}
	for _, tokens := range [][]int{
		{10, 11, 31},
		{10, 0, 11, 31},
		{10, 12, 13, 14, 11, 31},
	} {
		if !gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("token-terminal grammar rejected %v", tokens)
		}
	}
	for _, tokens := range [][]int{
		{10, 31},
		{11, 10, 31},
		{10, 11, 11, 31},
	} {
		if gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("token-terminal grammar accepted %v", tokens)
		}
	}
}

func TestGBNFNamedTokenTerminalResolution(t *testing.T) {
	grammar, err := NewGBNFGrammarWithTokens(
		`root ::= <start> !<end> <end>`,
		"root",
		bytePieces("<start>", "x", "<end>", ""),
		[]int{3},
		map[string]int{"<start>": 0, "<end>": 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !gbnfAcceptsTokens(t, grammar, []int{0, 1, 2, 3}) {
		t.Fatal("named token-terminal grammar rejected valid IDs")
	}
	if gbnfAcceptsTokens(t, grammar, []int{0, 2, 2, 3}) {
		t.Fatal("inverse named token terminal accepted its excluded ID")
	}
}

func TestLazyGBNFPatternReplaysOverlappingTokenPieces(t *testing.T) {
	grammar, err := NewGBNFGrammarWithOptions(
		`root ::= "JSON:" [0-9]+`,
		"root",
		bytePieces("prefixJS", "ON:42", "", "unconstrained"),
		[]int{2},
		nil,
		GBNFLazyOptions{
			Enabled:  true,
			Patterns: []string{`prefix(JSON:)`},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	state, err := grammar.initialState()
	if err != nil {
		t.Fatal(err)
	}
	if !state.awaitingTrigger {
		t.Fatal("lazy grammar did not begin in trigger-waiting state")
	}
	for _, token := range []int{0, 1} {
		var ok bool
		state, ok = grammar.advanceToken(state, token)
		if !ok {
			t.Fatalf("lazy grammar rejected buffered token %d", token)
		}
	}
	if state.awaitingTrigger || len(state.triggerBuffer) != 0 {
		t.Fatal("lazy grammar did not activate and clear its trigger buffer")
	}
	state, ok := grammar.advanceToken(state, 2)
	if !ok || !state.terminated {
		t.Fatal("activated lazy grammar did not accept EOS")
	}
}

func TestLazyGBNFECMAScriptTriggerFeatures(t *testing.T) {
	tests := []struct {
		name    string
		source  string
		pieces  []string
		pattern string
		tokens  []int
	}{
		{
			name:    "lookbehind",
			source:  `root ::= "JSON:" [0-9]+`,
			pieces:  []string{"prefix", "JSON:", "42", ""},
			pattern: `(?<=prefix)(JSON:)`,
			tokens:  []int{0, 1, 2, 3},
		},
		{
			name:    "lookahead",
			source:  `root ::= "JSON:" [0-9]+`,
			pieces:  []string{"prefixJSON:", "42", ""},
			pattern: `prefix((JSON:))(?=[0-9])`,
			tokens:  []int{0, 1, 2},
		},
		{
			name:    "backreference",
			source:  `root ::= "JSON:" "JSON:" [0-9]+`,
			pieces:  []string{"prefixJSON:", "JSON:42", ""},
			pattern: `prefix((JSON:))\2`,
			tokens:  []int{0, 1, 2},
		},
		{
			name:    "unicode-byte-offset",
			source:  `root ::= "JSON:" [0-9]+`,
			pieces:  []string{"πrefixJS", "ON:42", ""},
			pattern: `πrefix(JSON:)`,
			tokens:  []int{0, 1, 2},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			grammar, err := NewGBNFGrammarWithOptions(
				test.source,
				"root",
				bytePieces(test.pieces...),
				[]int{len(test.pieces) - 1},
				nil,
				GBNFLazyOptions{Enabled: true, Patterns: []string{test.pattern}},
			)
			if err != nil {
				t.Fatal(err)
			}
			if !gbnfAcceptsTokens(t, grammar, test.tokens) {
				t.Fatal("ECMAScript trigger rejected valid sequence")
			}
		})
	}
}

func TestLazyGBNFTriggerResourceLimits(t *testing.T) {
	pieces := bytePieces("a", "")
	options := GBNFLazyOptions{Enabled: true, Patterns: []string{strings.Repeat("a", maxGBNFTriggerPatternBytes+1)}}
	if _, err := NewGBNFGrammarWithOptions(
		`root ::= "a"`, "root", pieces, []int{1}, nil, options,
	); err == nil {
		t.Fatalf("lazy grammar accepted oversized trigger options %+v", options)
	}

	grammar, err := NewGBNFGrammarWithOptions(
		`root ::= "x"`,
		"root",
		bytePieces(strings.Repeat("x", maxGBNFSourceBytes), "y", ""),
		[]int{2},
		nil,
		GBNFLazyOptions{Enabled: true, Patterns: []string{"trigger"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	state, err := grammar.initialState()
	if err != nil {
		t.Fatal(err)
	}
	state, ok := grammar.advanceToken(state, 0)
	if !ok {
		t.Fatal("lazy grammar rejected trigger buffer at limit")
	}
	if _, ok = grammar.advanceToken(state, 1); ok {
		t.Fatal("lazy grammar accepted trigger buffer above limit")
	}

	bounded, err := compileGBNFTriggerRegex(`(?:^){40000}`)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = bounded.findStart(nil); err == nil {
		t.Fatal("bounded trigger accepted excessive backtracking stack")
	}
}

func TestLazyGBNFTokenTriggerIsIncludedInGrammar(t *testing.T) {
	grammar, err := NewGBNFGrammarWithOptions(
		`root ::= <[2]> "x"`,
		"root",
		bytePieces("free", "other", "<trigger>", "x", ""),
		[]int{4},
		nil,
		GBNFLazyOptions{Enabled: true, Tokens: []int{2}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !gbnfAcceptsTokens(t, grammar, []int{0, 1, 2, 3, 4}) {
		t.Fatal("token-triggered lazy grammar rejected valid sequence")
	}
	if gbnfAcceptsTokens(t, grammar, []int{0, 2, 1, 4}) {
		t.Fatal("token-triggered lazy grammar accepted invalid constrained suffix")
	}
}

func TestLazyGBNFRequiresValidTriggers(t *testing.T) {
	pieces := bytePieces("a", "")
	for _, options := range []GBNFLazyOptions{
		{Enabled: true},
		{Enabled: true, Patterns: []string{"("}},
		{Enabled: true, Tokens: []int{9}},
	} {
		if _, err := NewGBNFGrammarWithOptions(
			`root ::= "a"`, "root", pieces, []int{1}, nil, options,
		); err == nil {
			t.Fatalf("lazy grammar accepted invalid options %+v", options)
		}
	}
}

func TestGBNFCommentsMultilineGroupsAndNamedRoot(t *testing.T) {
	grammar, err := NewGBNFGrammar(`
# leading comment
value ::= (
  "yes" # first choice
  | "no"
)
`, "value", bytePieces("yes", "no", "", "maybe"), []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if !gbnfAcceptsTokens(t, grammar, []int{0, 2}) ||
		!gbnfAcceptsTokens(t, grammar, []int{1, 2}) ||
		gbnfAcceptsTokens(t, grammar, []int{3}) {
		t.Fatal("multiline named-root grammar behavior differs")
	}
}

func TestGBNFMatchesPinnedUpstreamIntegrationCorpus(t *testing.T) {
	assertGBNFCorpus(
		t,
		`
root ::= expr
expr ::= term ("+" term)*
term ::= number
number ::= [0-9]+
`,
		[]string{"42", "1+2+3+4+5", "123+456"},
		[]string{"+", "/ 3", "1+2+3+4+5+", "12a45"},
	)
	assertGBNFCorpus(
		t,
		`
root ::= expression
expression ::= term ws (("+"|"-") ws term)*
term ::= factor ws (("*"|"/") ws factor)*
factor ::= number | variable | "(" expression ")" | function-call
number ::= [0-9]+
variable ::= [a-zA-Z_][a-zA-Z0-9_]*
function-call ::= variable ws "(" (expression ("," ws expression)*)? ")"
ws ::= [ \t\n\r]?
`,
		[]string{
			"42", "x", "x+10", "(a+b)*(c-d)", "func()",
			"func(x,y+2)", "f(g(x), h(y, z))",
		},
		[]string{
			"+", "/ 3x", "x + + y", "func(,)", "func(x y)",
			"(a + b", "x + y)", "42 +",
		},
	)
	assertGBNFCorpus(
		t,
		`root ::= ("0x" [A-F0-9]{2} " "?){3,5}`,
		[]string{"0xFF 0x12 0xAB", "0xFF 0x12 0xAB 0x00 0x00"},
		[]string{"", "0xFF", "0xFF 0x12", "0xFF 0x12 0xAB 0x00 0x00 0x00"},
	)
	// nullable nested repetition is upstream regression case; has
	// epsilon cycle but does not grow parser stack
	assertGBNFCorpus(
		t,
		`root ::= ( [x]* )*`,
		[]string{"", "x", "xx"},
		[]string{"y", "yy"},
	)
}

func bytePieces(values ...string) [][]byte {
	result := make([][]byte, len(values))
	for index, value := range values {
		result[index] = []byte(value)
	}
	return result
}

func assertGBNFCorpus(
	t *testing.T,
	source string,
	passing, failing []string,
) {
	t.Helper()
	pieces := make([][]byte, 0, len(passing)+len(failing)+1)
	tokenFor := make(map[string]int)
	for _, value := range append(append([]string(nil), passing...), failing...) {
		if value == "" {
			continue
		}
		if _, exists := tokenFor[value]; exists {
			continue
		}
		tokenFor[value] = len(pieces)
		pieces = append(pieces, []byte(value))
	}
	eos := len(pieces)
	pieces = append(pieces, nil)
	grammar, err := NewGBNFGrammar(source, "root", pieces, []int{eos})
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range passing {
		tokens := []int{eos}
		if value != "" {
			tokens = []int{tokenFor[value], eos}
		}
		if !gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("pinned upstream corpus rejected valid string %q", value)
		}
	}
	for _, value := range failing {
		tokens := []int{eos}
		if value != "" {
			tokens = []int{tokenFor[value], eos}
		}
		if gbnfAcceptsTokens(t, grammar, tokens) {
			t.Fatalf("pinned upstream corpus accepted invalid string %q", value)
		}
	}
}

func gbnfAcceptsTokens(t *testing.T, grammar *GBNFGrammar, tokens []int) bool {
	t.Helper()
	state, err := grammar.initialState()
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range tokens {
		next, ok := grammar.advanceToken(state, token)
		if !ok {
			return false
		}
		state = next
	}
	return state.terminated
}
