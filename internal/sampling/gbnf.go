package sampling

import (
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"maps"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"overgo/internal/binaryschema"
)

const (
	maxGBNFSourceBytes   = 1 << 20
	maxGBNFWorkUnits     = 1 << 20
	gbnfSingleOccurrence = 1
)

type gbnfSymbolKind uint8

const (
	gbnfTerminalSymbol gbnfSymbolKind = iota
	gbnfRuleSymbol
	gbnfTokenSymbol
	gbnfTokenNotSymbol
)

type gbnfSymbol struct {
	kind  gbnfSymbolKind
	index int
}

type gbnfRange struct {
	low  rune
	high rune
}

type gbnfTerminal struct {
	negated bool
	ranges  []gbnfRange
}

type gbnfRule struct {
	alternatives [][]gbnfSymbol
}

// GBNFGrammar: immutable llama.cpp-style character/token grammar compiled
// for one vocabulary; supports literals, Unicode character classes,
// wildcard characters, token terminals, rule references, groups, alternation,
// right recursion, and *, +, ?, or {m,n} repetition
type GBNFGrammar struct {
	rules           []gbnfRule
	terminals       []gbnfTerminal
	root            int
	tokenPieces     [][]byte
	eos             []bool
	signature       uint64
	lazy            bool
	triggerTokens   map[int]struct{}
	triggerPatterns []gbnfTriggerPattern
}

// GBNFVocabulary owns immutable decoded pieces, terminal flags and token names.
// Multiple grammars may share it; parser and sampler state remain independent.
type GBNFVocabulary struct {
	pieces   [][]byte
	eos      []bool
	tokenIDs map[string]int
}

// NewGBNFVocabulary copies caller-owned vocabulary data once for reuse.
func NewGBNFVocabulary(tokenPieces [][]byte, eosTokens []int, tokenIDs map[string]int) (*GBNFVocabulary, error) {
	if len(tokenPieces) == 0 {
		return nil, errors.New("GBNF token vocabulary is empty")
	}
	vocabulary := &GBNFVocabulary{
		pieces:   make([][]byte, len(tokenPieces)),
		eos:      make([]bool, len(tokenPieces)),
		tokenIDs: maps.Clone(tokenIDs),
	}
	for index, piece := range tokenPieces {
		vocabulary.pieces[index] = slices.Clone(piece)
	}
	for _, token := range eosTokens {
		if token >= 0 && token < len(vocabulary.eos) {
			vocabulary.eos[token] = true
		}
	}
	return vocabulary, nil
}

type gbnfState struct {
	stacks           [][]gbnfSymbol
	partial          []byte
	terminated       bool
	awaitingTrigger  bool
	triggerBuffer    []byte
	triggerPositions []gbnfTriggerPosition
}

type gbnfTriggerPattern struct {
	source string
	regex  *gbnfTriggerRegex
}

type gbnfTriggerPosition struct {
	token int
	start int
	end   int
}

// GBNFLazyOptions: deferred grammar activation.
// Patterns: bounded ECMAScript regex; first non-empty capture sets replay start.
type GBNFLazyOptions struct {
	Enabled  bool
	Patterns []string
	Tokens   []int
}

type gbnfExpr struct {
	alternatives [][]gbnfTerm
}

type gbnfTermKind uint8

const (
	gbnfLiteralTerm gbnfTermKind = iota
	gbnfClassTerm
	gbnfReferenceTerm
	gbnfGroupTerm
	gbnfTokenTerm
)

type gbnfTerm struct {
	kind     gbnfTermKind
	literal  []rune
	class    gbnfTerminal
	name     string
	group    *gbnfExpr
	token    int
	tokenNot bool
	min      int
	max      int
}

type gbnfNamedRule struct {
	name string
	expr gbnfExpr
}

type gbnfParser struct {
	source     string
	offset     int
	tokenIDs   map[string]int
	vocabulary int
}

// NewGBNFGrammar: parses source and binds it to decoded token pieces; EOS token
// IDs: permitted only when grammar is in accepting state
func NewGBNFGrammar(
	source, root string,
	tokenPieces [][]byte,
	eosTokens []int,
) (*GBNFGrammar, error) {
	return NewGBNFGrammarWithTokens(source, root, tokenPieces, eosTokens, nil)
}

// NewGBNFGrammarWithTokens additionally resolves named token terminals such as
// <|im_start|>; Numeric <[id]> terminals do not require name map
func NewGBNFGrammarWithTokens(
	source, root string,
	tokenPieces [][]byte,
	eosTokens []int,
	tokenIDs map[string]int,
) (*GBNFGrammar, error) {
	return NewGBNFGrammarWithOptions(
		source, root, tokenPieces, eosTokens, tokenIDs, GBNFLazyOptions{},
	)
}

// NewGBNFGrammarWithOptions: compiles GBNF with optional lazy activation
func NewGBNFGrammarWithOptions(
	source, root string,
	tokenPieces [][]byte,
	eosTokens []int,
	tokenIDs map[string]int,
	lazy GBNFLazyOptions,
) (*GBNFGrammar, error) {
	vocabulary, err := NewGBNFVocabulary(tokenPieces, eosTokens, tokenIDs)
	if err != nil {
		return nil, err
	}
	return vocabulary.Compile(source, root, lazy)
}

// Compile binds a grammar to this vocabulary without copying vocabulary data.
func (v *GBNFVocabulary) Compile(source, root string, lazy GBNFLazyOptions) (*GBNFGrammar, error) {
	if source == "" {
		return nil, errors.New("GBNF source is empty")
	}
	if len(source) > maxGBNFSourceBytes {
		return nil, fmt.Errorf("GBNF source exceeds %d bytes", maxGBNFSourceBytes)
	}
	root = cmp.Or(root, "root")
	if v == nil || len(v.pieces) == 0 {
		return nil, errors.New("GBNF token vocabulary is empty")
	}
	parser := gbnfParser{
		source:     source,
		tokenIDs:   v.tokenIDs,
		vocabulary: len(v.pieces),
	}
	named, err := parser.parse()
	if err != nil {
		return nil, err
	}
	grammar, err := compileGBNF(named, root)
	if err != nil {
		return nil, err
	}
	grammar.tokenPieces = v.pieces
	grammar.eos = v.eos
	grammar.lazy = lazy.Enabled
	grammar.triggerTokens = make(map[int]struct{}, len(lazy.Tokens))
	for _, token := range lazy.Tokens {
		if token < 0 || token >= len(v.pieces) {
			return nil, fmt.Errorf("GBNF trigger token %d is outside vocabulary", token)
		}
		grammar.triggerTokens[token] = struct{}{}
	}
	grammar.triggerPatterns = make([]gbnfTriggerPattern, len(lazy.Patterns))
	for index, pattern := range lazy.Patterns {
		compiled, compileErr := compileGBNFTriggerRegex(pattern)
		if compileErr != nil {
			return nil, fmt.Errorf(
				"GBNF trigger pattern %d: %w",
				index,
				compileErr,
			)
		}
		grammar.triggerPatterns[index] = gbnfTriggerPattern{
			source: pattern,
			regex:  compiled,
		}
	}
	if grammar.lazy &&
		len(grammar.triggerTokens) == 0 &&
		len(grammar.triggerPatterns) == 0 {
		return nil, errors.New("lazy GBNF needs at least one trigger")
	}
	grammar.signature = grammarSignature(source, root, v.pieces, v.eos, grammar)
	if _, err := grammar.initialState(); err != nil {
		return nil, err
	}
	return grammar, nil
}

func validateGBNFGrammar(grammar *GBNFGrammar) error {
	if grammar == nil {
		return nil
	}
	if len(grammar.rules) == 0 ||
		grammar.root < 0 || grammar.root >= len(grammar.rules) ||
		len(grammar.tokenPieces) == 0 ||
		len(grammar.eos) != len(grammar.tokenPieces) {
		return errors.New("grammar structure or vocabulary is invalid")
	}
	if _, err := grammar.initialState(); err != nil {
		return err
	}
	return nil
}

func (p *gbnfParser) parse() ([]gbnfNamedRule, error) {
	var rules []gbnfNamedRule
	seen := make(map[string]struct{})
	p.skipSpace(true)
	for p.offset < len(p.source) {
		name, err := p.parseName()
		if err != nil {
			return nil, err
		}
		if _, ok := seen[name]; ok {
			return nil, p.errorf("duplicate rule %q", name)
		}
		seen[name] = struct{}{}
		p.skipSpace(false)
		if !p.consume("::=") {
			return nil, p.errorf("rule %q is missing ::=", name)
		}
		p.skipSpace(true)
		expr, err := p.parseExpr(false)
		if err != nil {
			return nil, fmt.Errorf("GBNF rule %q: %w", name, err)
		}
		rules = append(rules, gbnfNamedRule{name: name, expr: expr})
		if p.offset < len(p.source) {
			if p.source[p.offset] == '\r' {
				p.offset++
				if p.offset < len(p.source) && p.source[p.offset] == '\n' {
					p.offset++
				}
			} else if p.source[p.offset] == '\n' {
				p.offset++
			} else {
				return nil, p.errorf("expected newline after rule %q", name)
			}
		}
		p.skipSpace(true)
	}
	if len(rules) == 0 {
		return nil, errors.New("GBNF contains no rules")
	}
	return rules, nil
}

func (p *gbnfParser) parseExpr(nested bool) (gbnfExpr, error) {
	result := gbnfExpr{}
	for {
		sequence, err := p.parseSequence(nested)
		if err != nil {
			return gbnfExpr{}, err
		}
		result.alternatives = append(result.alternatives, sequence)
		if p.offset >= len(p.source) || p.source[p.offset] != '|' {
			break
		}
		p.offset++
		p.skipSpace(true)
	}
	return result, nil
}

func (p *gbnfParser) parseSequence(nested bool) ([]gbnfTerm, error) {
	var result []gbnfTerm
	for p.offset < len(p.source) {
		value := p.source[p.offset]
		if value == '|' || (nested && value == ')') ||
			(!nested && (value == '\r' || value == '\n')) {
			break
		}
		var term gbnfTerm
		var err error
		switch value {
		case '"':
			term, err = p.parseLiteral()
		case '[':
			term, err = p.parseClass()
		case '(':
			p.offset++
			p.skipSpace(true)
			var group gbnfExpr
			group, err = p.parseExpr(true)
			if err == nil {
				if p.offset >= len(p.source) || p.source[p.offset] != ')' {
					err = p.errorf("expected )")
				} else {
					p.offset++
					term = gbnfTerm{kind: gbnfGroupTerm, group: &group, min: gbnfSingleOccurrence, max: gbnfSingleOccurrence}
				}
			}
		case '.':
			p.offset++
			term = gbnfTerm{
				kind:  gbnfClassTerm,
				class: gbnfTerminal{ranges: []gbnfRange{{low: 0, high: utf8.MaxRune}}},
				min:   gbnfSingleOccurrence,
				max:   gbnfSingleOccurrence,
			}
		case '<', '!':
			term, err = p.parseTokenTerminal()
		default:
			if isGBNFNameByte(value) {
				var name string
				name, err = p.parseName()
				term = gbnfTerm{
					kind: gbnfReferenceTerm,
					name: name,
					min:  gbnfSingleOccurrence,
					max:  gbnfSingleOccurrence,
				}
			} else {
				return nil, p.errorf("unexpected character %q", value)
			}
		}
		if err != nil {
			return nil, err
		}
		p.skipSpace(nested)
		if err := p.parseRepetition(&term, nested); err != nil {
			return nil, err
		}
		result = append(result, term)
	}
	return result, nil
}

func (p *gbnfParser) parseTokenTerminal() (gbnfTerm, error) {
	inverse := false
	if p.source[p.offset] == '!' {
		inverse = true
		p.offset++
		if p.offset >= len(p.source) || p.source[p.offset] != '<' {
			return gbnfTerm{}, p.errorf("expected < after !")
		}
	}
	start := p.offset
	end := strings.IndexByte(p.source[start:], '>')
	if end < 0 {
		return gbnfTerm{}, p.errorf("unterminated token terminal")
	}
	end += start
	raw := p.source[start : end+1]
	p.offset = end + 1
	token := -1
	if strings.HasPrefix(raw, "<[") && strings.HasSuffix(raw, "]>") {
		digits := raw[2 : len(raw)-2]
		if digits == "" {
			return gbnfTerm{}, p.errorf("empty numeric token terminal")
		}
		value, err := strconv.ParseUint(digits, binaryschema.DecimalRadix, 31)
		if err != nil {
			return gbnfTerm{}, p.errorf("invalid numeric token terminal %q", raw)
		}
		token = int(value)
	} else {
		var ok bool
		token, ok = p.tokenIDs[raw]
		if !ok {
			return gbnfTerm{}, p.errorf("unknown named token terminal %q", raw)
		}
	}
	if token < 0 || token >= p.vocabulary {
		return gbnfTerm{}, p.errorf(
			"token terminal %q ID %d is outside vocabulary",
			raw,
			token,
		)
	}
	return gbnfTerm{
		kind:     gbnfTokenTerm,
		token:    token,
		tokenNot: inverse,
		min:      gbnfSingleOccurrence,
		max:      gbnfSingleOccurrence,
	}, nil
}

func (p *gbnfParser) parseLiteral() (gbnfTerm, error) {
	p.offset++
	var literal []rune
	for {
		if p.offset >= len(p.source) {
			return gbnfTerm{}, p.errorf("unterminated literal")
		}
		if p.source[p.offset] == '"' {
			p.offset++
			return gbnfTerm{
				kind:    gbnfLiteralTerm,
				literal: literal,
				min:     gbnfSingleOccurrence,
				max:     gbnfSingleOccurrence,
			}, nil
		}
		value, err := p.parseCharacter()
		if err != nil {
			return gbnfTerm{}, err
		}
		literal = append(literal, value)
	}
}

func (p *gbnfParser) parseClass() (gbnfTerm, error) {
	p.offset++
	terminal := gbnfTerminal{}
	if p.offset < len(p.source) && p.source[p.offset] == '^' {
		terminal.negated = true
		p.offset++
	}
	for {
		if p.offset >= len(p.source) {
			return gbnfTerm{}, p.errorf("unterminated character class")
		}
		if p.source[p.offset] == ']' {
			if len(terminal.ranges) == 0 {
				return gbnfTerm{}, p.errorf("empty character class")
			}
			p.offset++
			terminal.ranges = normalizeGBNFRanges(terminal.ranges)
			return gbnfTerm{
				kind:  gbnfClassTerm,
				class: terminal,
				min:   gbnfSingleOccurrence,
				max:   gbnfSingleOccurrence,
			}, nil
		}
		first, err := p.parseCharacter()
		if err != nil {
			return gbnfTerm{}, err
		}
		last := first
		if p.offset+1 < len(p.source) &&
			p.source[p.offset] == '-' &&
			p.source[p.offset+1] != ']' {
			p.offset++
			last, err = p.parseCharacter()
			if err != nil {
				return gbnfTerm{}, err
			}
			if last < first {
				return gbnfTerm{}, p.errorf(
					"character range U+%04X-U+%04X is reversed",
					first,
					last,
				)
			}
		}
		terminal.ranges = append(terminal.ranges, gbnfRange{low: first, high: last})
	}
}

func (p *gbnfParser) parseCharacter() (rune, error) {
	if p.offset >= len(p.source) {
		return 0, p.errorf("unexpected end of input")
	}
	if p.source[p.offset] != '\\' {
		value, size := utf8.DecodeRuneInString(p.source[p.offset:])
		if value == utf8.RuneError && size == 1 {
			return 0, p.errorf("invalid UTF-8")
		}
		p.offset += size
		return value, nil
	}
	p.offset++
	if p.offset >= len(p.source) {
		return 0, p.errorf("unterminated escape")
	}
	escape := p.source[p.offset]
	p.offset++
	switch escape {
	case 't':
		return '\t', nil
	case 'r':
		return '\r', nil
	case 'n':
		return '\n', nil
	case '\\', '"', '[', ']':
		return rune(escape), nil
	case 'x':
		return p.parseHexRune(2)
	case 'u':
		return p.parseHexRune(4)
	case 'U':
		return p.parseHexRune(8)
	default:
		return 0, p.errorf("unknown escape \\%c", escape)
	}
}

func (p *gbnfParser) parseHexRune(digits int) (rune, error) {
	if len(p.source)-p.offset < digits {
		return 0, p.errorf("hex escape needs %d digits", digits)
	}
	raw := p.source[p.offset : p.offset+digits]
	value, err := strconv.ParseUint(raw, binaryschema.HexRadix, binaryschema.Width32Bits)
	if err != nil {
		return 0, p.errorf("invalid hex escape %q", raw)
	}
	p.offset += digits
	if value > utf8.MaxRune || value >= 0xD800 && value <= 0xDFFF {
		return 0, p.errorf("hex escape U+%X is not a Unicode scalar", value)
	}
	return rune(value), nil
}

func (p *gbnfParser) parseRepetition(term *gbnfTerm, nested bool) error {
	if p.offset >= len(p.source) {
		return nil
	}
	switch p.source[p.offset] {
	case '*':
		term.min, term.max = 0, -1
		p.offset++
	case '+':
		term.min, term.max = 1, -1
		p.offset++
	case '?':
		term.min, term.max = 0, 1
		p.offset++
	case '{':
		p.offset++
		p.skipSpace(nested)
		minimum, err := p.parseInteger()
		if err != nil {
			return err
		}
		p.skipSpace(nested)
		maximum := minimum
		if p.offset < len(p.source) && p.source[p.offset] == ',' {
			p.offset++
			p.skipSpace(nested)
			maximum = -1
			if p.offset < len(p.source) &&
				p.source[p.offset] >= '0' && p.source[p.offset] <= '9' {
				maximum, err = p.parseInteger()
				if err != nil {
					return err
				}
				p.skipSpace(nested)
			}
		}
		if p.offset >= len(p.source) || p.source[p.offset] != '}' {
			return p.errorf("expected } after repetition")
		}
		p.offset++
		if maximum >= 0 && maximum < minimum {
			return p.errorf("invalid or excessive repetition {%d,%d}", minimum, maximum)
		}
		term.min, term.max = minimum, maximum
	default:
		return nil
	}
	p.skipSpace(nested)
	if p.offset < len(p.source) {
		switch p.source[p.offset] {
		case '*', '+', '?', '{':
			return p.errorf("multiple repetition operators on one term")
		}
	}
	return nil
}

func (p *gbnfParser) parseInteger() (int, error) {
	start := p.offset
	for p.offset < len(p.source) &&
		p.source[p.offset] >= '0' && p.source[p.offset] <= '9' {
		p.offset++
	}
	if start == p.offset {
		return 0, p.errorf("expected repetition integer")
	}
	value, err := strconv.ParseUint(p.source[start:p.offset], binaryschema.DecimalRadix, 31)
	if err != nil {
		return 0, p.errorf("invalid repetition integer")
	}
	return int(value), nil
}

func (p *gbnfParser) parseName() (string, error) {
	start := p.offset
	for p.offset < len(p.source) && isGBNFNameByte(p.source[p.offset]) {
		p.offset++
	}
	if start == p.offset {
		return "", p.errorf("expected rule name")
	}
	return p.source[start:p.offset], nil
}

func (p *gbnfParser) skipSpace(newline bool) {
	for p.offset < len(p.source) {
		switch p.source[p.offset] {
		case ' ', '\t':
			p.offset++
		case '#':
			for p.offset < len(p.source) &&
				p.source[p.offset] != '\r' &&
				p.source[p.offset] != '\n' {
				p.offset++
			}
		case '\r', '\n':
			if !newline {
				return
			}
			p.offset++
		default:
			return
		}
	}
}

func (p *gbnfParser) consume(value string) bool {
	if !strings.HasPrefix(p.source[p.offset:], value) {
		return false
	}
	p.offset += len(value)
	return true
}

func (p *gbnfParser) errorf(format string, arguments ...any) error {
	line, column := 1, 1
	for index, value := range p.source {
		if index >= p.offset {
			break
		}
		if value == '\n' {
			line, column = line+1, 1
		} else {
			column++
		}
	}
	return fmt.Errorf("GBNF line %d column %d: %s", line, column, fmt.Sprintf(format, arguments...))
}

func isGBNFNameByte(value byte) bool {
	return value >= 'a' && value <= 'z' ||
		value >= 'A' && value <= 'Z' ||
		value >= '0' && value <= '9' ||
		value == '-'
}

type gbnfCompiler struct {
	grammar     *GBNFGrammar
	ruleIDs     map[string]int
	terminalIDs map[string]int
	symbolCount int
}

func compileGBNF(named []gbnfNamedRule, root string) (*GBNFGrammar, error) {
	compiler := gbnfCompiler{
		grammar:     &GBNFGrammar{rules: make([]gbnfRule, len(named))},
		ruleIDs:     make(map[string]int, len(named)),
		terminalIDs: make(map[string]int),
	}
	for index, rule := range named {
		compiler.ruleIDs[rule.name] = index
	}
	rootID, ok := compiler.ruleIDs[root]
	if !ok {
		return nil, fmt.Errorf("GBNF does not contain root rule %q", root)
	}
	compiler.grammar.root = rootID
	for index, rule := range named {
		if err := compiler.compileExpr(index, rule.expr); err != nil {
			return nil, fmt.Errorf("GBNF rule %q: %w", rule.name, err)
		}
	}
	if compiler.symbolCount > maxGBNFWorkUnits-len(compiler.grammar.rules) {
		return nil, fmt.Errorf("GBNF expansion exceeds %d work units", maxGBNFWorkUnits)
	}
	if err := compiler.validateReferencesAndRecursion(); err != nil {
		return nil, err
	}
	return compiler.grammar, nil
}

func (c *gbnfCompiler) compileExpr(ruleID int, expr gbnfExpr) error {
	alternatives := make([][]gbnfSymbol, len(expr.alternatives))
	for index, terms := range expr.alternatives {
		var sequence []gbnfSymbol
		for _, term := range terms {
			base, err := c.compileTermBase(term)
			if err != nil {
				return err
			}
			repeated, err := c.compileRepetition(base, term.min, term.max)
			if err != nil {
				return err
			}
			sequence = append(sequence, repeated...)
			c.symbolCount += len(repeated)
			if c.symbolCount > maxGBNFWorkUnits-len(c.grammar.rules) {
				return fmt.Errorf("GBNF expansion exceeds %d work units", maxGBNFWorkUnits)
			}
		}
		alternatives[index] = sequence
	}
	c.grammar.rules[ruleID] = gbnfRule{alternatives: alternatives}
	return nil
}

func (c *gbnfCompiler) compileTermBase(term gbnfTerm) ([]gbnfSymbol, error) {
	switch term.kind {
	case gbnfLiteralTerm:
		result := make([]gbnfSymbol, 0, len(term.literal))
		for _, value := range term.literal {
			id := c.addTerminal(gbnfTerminal{
				ranges: []gbnfRange{{low: value, high: value}},
			})
			result = append(result, gbnfSymbol{kind: gbnfTerminalSymbol, index: id})
		}
		return result, nil
	case gbnfClassTerm:
		id := c.addTerminal(term.class)
		return []gbnfSymbol{{kind: gbnfTerminalSymbol, index: id}}, nil
	case gbnfReferenceTerm:
		id, ok := c.ruleIDs[term.name]
		if !ok {
			return nil, fmt.Errorf("undefined rule identifier %q", term.name)
		}
		return []gbnfSymbol{{kind: gbnfRuleSymbol, index: id}}, nil
	case gbnfGroupTerm:
		id, err := c.addGeneratedRule(*term.group)
		if err != nil {
			return nil, err
		}
		return []gbnfSymbol{{kind: gbnfRuleSymbol, index: id}}, nil
	case gbnfTokenTerm:
		kind := gbnfTokenSymbol
		if term.tokenNot {
			kind = gbnfTokenNotSymbol
		}
		return []gbnfSymbol{{kind: kind, index: term.token}}, nil
	default:
		return nil, errors.New("invalid GBNF term")
	}
}

func (c *gbnfCompiler) compileRepetition(
	base []gbnfSymbol,
	minimum, maximum int,
) ([]gbnfSymbol, error) {
	if minimum < 0 || maximum >= 0 && maximum < minimum ||
		len(base) > 0 && minimum > maxGBNFWorkUnits/len(base) {
		return nil, errors.New("invalid GBNF repetition")
	}
	var result []gbnfSymbol
	for range minimum {
		result = append(result, base...)
	}
	if maximum == minimum {
		return result, nil
	}
	if len(base) == 0 {
		return result, nil
	}
	if maximum < 0 {
		id := len(c.grammar.rules)
		if id >= maxGBNFWorkUnits-c.symbolCount {
			return nil, errors.New("GBNF expansion exceeds work budget")
		}
		c.grammar.rules = append(c.grammar.rules, gbnfRule{})
		recursive := append(slices.Clone(base),
			gbnfSymbol{kind: gbnfRuleSymbol, index: id})
		c.grammar.rules[id] = gbnfRule{
			alternatives: [][]gbnfSymbol{recursive, nil},
		}
		c.symbolCount += len(recursive)
		if c.symbolCount > maxGBNFWorkUnits-len(c.grammar.rules) {
			return nil, errors.New("GBNF expansion exceeds work budget")
		}
		return append(result, gbnfSymbol{kind: gbnfRuleSymbol, index: id}), nil
	}
	optional := maximum - minimum
	var next = -1
	for range optional {
		id := len(c.grammar.rules)
		if id >= maxGBNFWorkUnits-c.symbolCount {
			return nil, errors.New("GBNF expansion exceeds work budget")
		}
		sequence := slices.Clone(base)
		if next >= 0 {
			sequence = append(sequence, gbnfSymbol{kind: gbnfRuleSymbol, index: next})
		}
		c.grammar.rules = append(c.grammar.rules, gbnfRule{
			alternatives: [][]gbnfSymbol{sequence, nil},
		})
		c.symbolCount += len(sequence)
		if c.symbolCount > maxGBNFWorkUnits-len(c.grammar.rules) {
			return nil, errors.New("GBNF expansion exceeds work budget")
		}
		next = id
	}
	if next >= 0 {
		result = append(result, gbnfSymbol{kind: gbnfRuleSymbol, index: next})
	}
	return result, nil
}

func (c *gbnfCompiler) addGeneratedRule(expr gbnfExpr) (int, error) {
	id := len(c.grammar.rules)
	if id >= maxGBNFWorkUnits-c.symbolCount {
		return 0, errors.New("GBNF expansion exceeds work budget")
	}
	c.grammar.rules = append(c.grammar.rules, gbnfRule{})
	if err := c.compileExpr(id, expr); err != nil {
		return 0, err
	}
	return id, nil
}

func (c *gbnfCompiler) addTerminal(terminal gbnfTerminal) int {
	terminal.ranges = normalizeGBNFRanges(terminal.ranges)
	var key strings.Builder
	if terminal.negated {
		key.WriteByte('!')
	}
	var encoded [8]byte
	for _, item := range terminal.ranges {
		binary.LittleEndian.PutUint32(encoded[:4], uint32(item.low))
		binary.LittleEndian.PutUint32(encoded[4:], uint32(item.high))
		key.Write(encoded[:])
	}
	if id, ok := c.terminalIDs[key.String()]; ok {
		return id
	}
	id := len(c.grammar.terminals)
	c.terminalIDs[key.String()] = id
	c.grammar.terminals = append(c.grammar.terminals, terminal)
	return id
}

func (c *gbnfCompiler) validateReferencesAndRecursion() error {
	count := len(c.grammar.rules)
	nullable := make([]bool, count)
	changed := true
	for changed {
		changed = false
		for ruleID, rule := range c.grammar.rules {
			if nullable[ruleID] {
				continue
			}
			for _, alternative := range rule.alternatives {
				allNullable := true
				for _, symbol := range alternative {
					if symbol.kind != gbnfRuleSymbol || !nullable[symbol.index] {
						allNullable = false
						break
					}
				}
				if allNullable {
					nullable[ruleID] = true
					changed = true
					break
				}
			}
		}
	}
	// Prefix-reference edges model expansions that can occur before
	// terminal: consumed; cycle is unsafe only when some edge leaves
	// suffix on stack: that is growing form of left recursion
	// Zero-growth epsilon cycles are harmless because normalizeStacks dedupes
	// identical configurations (for example upstream's `( [x]* )*` case)
	edges := make([]map[int]bool, count)
	for ruleID, rule := range c.grammar.rules {
		edges[ruleID] = make(map[int]bool)
		for _, alternative := range rule.alternatives {
			for symbolIndex, symbol := range alternative {
				if symbol.kind != gbnfRuleSymbol {
					break
				}
				if symbol.index < 0 || symbol.index >= count {
					return fmt.Errorf("GBNF rule %d references invalid rule %d", ruleID, symbol.index)
				}
				growing := symbolIndex+1 < len(alternative)
				edges[ruleID][symbol.index] =
					edges[ruleID][symbol.index] || growing
				if !nullable[symbol.index] {
					break
				}
			}
		}
	}
	component := stronglyConnectedGBNFRules(edges)
	for rule, outgoing := range edges {
		for next, growing := range outgoing {
			if growing && component[rule] == component[next] {
				return fmt.Errorf(
					"GBNF contains unsupported growing left recursion at rule %d",
					rule,
				)
			}
		}
	}
	return nil
}

func stronglyConnectedGBNFRules(edges []map[int]bool) []int {
	index := 0
	indices := make([]int, len(edges))
	low := make([]int, len(edges))
	onStack := make([]bool, len(edges))
	for item := range indices {
		indices[item] = -1
	}
	var stack []int
	components := make([]int, len(edges))
	componentCount := 0
	var visit func(int)
	visit = func(rule int) {
		indices[rule] = index
		low[rule] = index
		index++
		stack = append(stack, rule)
		onStack[rule] = true
		for next := range edges[rule] {
			if indices[next] < 0 {
				visit(next)
				low[rule] = min(low[rule], low[next])
			} else if onStack[next] {
				low[rule] = min(low[rule], indices[next])
			}
		}
		if low[rule] != indices[rule] {
			return
		}
		for {
			last := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			onStack[last] = false
			components[last] = componentCount
			if last == rule {
				break
			}
		}
		componentCount++
	}
	for rule := range edges {
		if indices[rule] < 0 {
			visit(rule)
		}
	}
	return components
}

func normalizeGBNFRanges(input []gbnfRange) []gbnfRange {
	result := slices.Clone(input)
	sort.Slice(result, func(i, j int) bool {
		if result[i].low == result[j].low {
			return result[i].high < result[j].high
		}
		return result[i].low < result[j].low
	})
	merged := result[:0]
	for _, item := range result {
		if len(merged) == 0 || item.low > merged[len(merged)-1].high+1 {
			merged = append(merged, item)
			continue
		}
		if item.high > merged[len(merged)-1].high {
			merged[len(merged)-1].high = item.high
		}
	}
	return merged
}

func (g *GBNFGrammar) initialState() (gbnfState, error) {
	stacks, err := g.normalizeStacks([][]gbnfSymbol{{
		{kind: gbnfRuleSymbol, index: g.root},
	}})
	if err != nil {
		return gbnfState{}, err
	}
	return gbnfState{stacks: stacks, awaitingTrigger: g.lazy}, nil
}

func (g *GBNFGrammar) advanceToken(state gbnfState, token int) (gbnfState, bool) {
	if state.terminated || token < 0 || token >= len(g.tokenPieces) {
		return gbnfState{}, false
	}
	if state.awaitingTrigger {
		return g.advanceAwaitingTrigger(state, token)
	}
	return g.advanceConstrainedToken(state, token, g.tokenPieces[token])
}

func (g *GBNFGrammar) advanceAwaitingTrigger(
	state gbnfState,
	token int,
) (gbnfState, bool) {
	if _, triggered := g.triggerTokens[token]; triggered {
		state.awaitingTrigger = false
		state.triggerBuffer = nil
		state.triggerPositions = nil
		return g.advanceConstrainedToken(state, token, g.tokenPieces[token])
	}
	buffer := slices.Clone(state.triggerBuffer)
	positions := slices.Clone(state.triggerPositions)
	if len(buffer)+len(g.tokenPieces[token]) > maxGBNFSourceBytes {
		return gbnfState{}, false
	}
	start := len(buffer)
	buffer = append(buffer, g.tokenPieces[token]...)
	positions = append(positions, gbnfTriggerPosition{
		token: token,
		start: start,
		end:   len(buffer),
	})
	for _, trigger := range g.triggerPatterns {
		matchStart, matched, err := trigger.regex.findStart(buffer)
		if err != nil {
			return gbnfState{}, false
		}
		if !matched {
			continue
		}
		state.awaitingTrigger = false
		state.triggerBuffer = nil
		state.triggerPositions = nil
		for _, position := range positions {
			if position.end <= matchStart {
				continue
			}
			pieceStart := max(position.start, matchStart)
			var ok bool
			state, ok = g.advanceConstrainedToken(
				state,
				position.token,
				buffer[pieceStart:position.end],
			)
			if !ok {
				return gbnfState{}, false
			}
		}
		return state, true
	}
	state.triggerBuffer = buffer
	state.triggerPositions = positions
	return state, true
}

func (g *GBNFGrammar) advanceConstrainedToken(
	state gbnfState,
	token int,
	piece []byte,
) (gbnfState, bool) {
	if g.eos[token] {
		if len(state.partial) != 0 || !stacksAccepting(state.stacks) {
			return gbnfState{}, false
		}
		return gbnfState{terminated: true}, true
	}
	if len(piece) == 0 {
		return gbnfState{}, false
	}
	codePoints, partial, ok := decodeGBNFTokenPiece(state.partial, piece)
	if !ok {
		return gbnfState{}, false
	}
	var tokenPending, characterStacks [][]gbnfSymbol
	for _, stack := range state.stacks {
		if len(stack) == 0 {
			continue
		}
		symbol := stack[0]
		switch symbol.kind {
		case gbnfTokenSymbol:
			if token == symbol.index {
				tokenPending = append(tokenPending, slices.Clone(stack[1:]))
			}
		case gbnfTokenNotSymbol:
			if token != symbol.index {
				tokenPending = append(tokenPending, slices.Clone(stack[1:]))
			}
		case gbnfTerminalSymbol:
			characterStacks = append(characterStacks, stack)
		}
	}
	var survivors [][]gbnfSymbol
	if len(tokenPending) > 0 {
		var err error
		survivors, err = g.normalizeStacks(tokenPending)
		if err != nil {
			return gbnfState{}, false
		}
	}
	for _, value := range codePoints {
		var err error
		characterStacks, err = g.acceptRune(characterStacks, value)
		if err != nil || len(characterStacks) == 0 {
			characterStacks = nil
			break
		}
	}
	if len(partial) > 0 {
		characterStacks = g.filterPartialStacks(characterStacks, partial)
	}
	survivors = append(survivors, characterStacks...)
	if len(survivors) == 0 {
		return gbnfState{}, false
	}
	survivors, err := g.normalizeStacks(survivors)
	if err != nil {
		return gbnfState{}, false
	}
	return gbnfState{
		stacks:  survivors,
		partial: partial,
	}, true
}

func decodeGBNFTokenPiece(
	previous, piece []byte,
) ([]rune, []byte, bool) {
	input := make([]byte, 0, len(previous)+len(piece))
	input = append(input, previous...)
	input = append(input, piece...)
	var result []rune
	for len(input) > 0 {
		value, size := utf8.DecodeRune(input)
		if value == utf8.RuneError && size == 1 {
			if !utf8.FullRune(input) {
				if _, ok := partialUTF8Ranges(input); !ok {
					return nil, nil, false
				}
				return result, slices.Clone(input), true
			}
			return nil, nil, false
		}
		result = append(result, value)
		input = input[size:]
	}
	return result, nil, true
}

func (g *GBNFGrammar) acceptRune(
	stacks [][]gbnfSymbol,
	value rune,
) ([][]gbnfSymbol, error) {
	var pending [][]gbnfSymbol
	for _, stack := range stacks {
		if len(stack) == 0 {
			continue
		}
		symbol := stack[0]
		if symbol.kind != gbnfTerminalSymbol ||
			!g.terminals[symbol.index].matches(value) {
			continue
		}
		pending = append(pending, slices.Clone(stack[1:]))
	}
	if len(pending) == 0 {
		return nil, nil
	}
	return g.normalizeStacks(pending)
}

func (g *GBNFGrammar) normalizeStacks(
	input [][]gbnfSymbol,
) ([][]gbnfSymbol, error) {
	todo := append([][]gbnfSymbol(nil), input...)
	seen := make(map[string]struct{}, len(todo))
	var result [][]gbnfSymbol
	steps := 0
	for len(todo) > 0 {
		steps++
		if steps > maxGBNFWorkUnits {
			return nil, errors.New("GBNF stack expansion exceeds safety limit")
		}
		stack := todo[len(todo)-1]
		todo = todo[:len(todo)-1]
		key := gbnfStackKey(stack)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		if len(stack) == 0 || stack[0].kind != gbnfRuleSymbol {
			result = append(result, stack)
			continue
		}
		ruleID := stack[0].index
		if ruleID < 0 || ruleID >= len(g.rules) {
			return nil, errors.New("GBNF stack references an invalid rule")
		}
		rest := stack[1:]
		for _, alternative := range g.rules[ruleID].alternatives {
			next := make([]gbnfSymbol, 0, len(alternative)+len(rest))
			next = append(next, alternative...)
			next = append(next, rest...)
			todo = append(todo, next)
		}
	}
	sort.Slice(result, func(i, j int) bool {
		return gbnfStackKey(result[i]) < gbnfStackKey(result[j])
	})
	return result, nil
}

func gbnfStackKey(stack []gbnfSymbol) string {
	data := make([]byte, len(stack)*5)
	for index, symbol := range stack {
		data[index*5] = byte(symbol.kind)
		binary.LittleEndian.PutUint32(data[index*5+1:], uint32(symbol.index))
	}
	return string(data)
}

func (t gbnfTerminal) matches(value rune) bool {
	found := false
	for _, item := range t.ranges {
		if value >= item.low && value <= item.high {
			found = true
			break
		}
	}
	if t.negated {
		return !found
	}
	return found
}

func (g *GBNFGrammar) partialCanMatch(stacks [][]gbnfSymbol, partial []byte) bool {
	ranges, ok := partialUTF8Ranges(partial)
	if !ok {
		return false
	}
	for _, stack := range stacks {
		if len(stack) == 0 || stack[0].kind != gbnfTerminalSymbol {
			continue
		}
		terminal := g.terminals[stack[0].index]
		for _, possible := range ranges {
			if terminal.intersects(possible) {
				return true
			}
		}
	}
	return false
}

func (g *GBNFGrammar) filterPartialStacks(
	stacks [][]gbnfSymbol,
	partial []byte,
) [][]gbnfSymbol {
	result := make([][]gbnfSymbol, 0, len(stacks))
	for _, stack := range stacks {
		if g.partialCanMatch([][]gbnfSymbol{stack}, partial) {
			result = append(result, stack)
		}
	}
	return result
}

func (t gbnfTerminal) intersects(possible gbnfRange) bool {
	if !t.negated {
		for _, item := range t.ranges {
			if item.low <= possible.high && possible.low <= item.high {
				return true
			}
		}
		return false
	}
	cursor := possible.low
	for _, item := range t.ranges {
		if item.high < cursor {
			continue
		}
		if item.low > cursor {
			return true
		}
		if item.high >= cursor {
			cursor = item.high + 1
			if cursor > possible.high {
				return false
			}
		}
	}
	return cursor <= possible.high
}

func partialUTF8Ranges(partial []byte) ([]gbnfRange, bool) {
	if len(partial) == 0 {
		return nil, false
	}
	first := partial[0]
	length := 0
	var accumulated uint32
	switch {
	case first >= 0xC2 && first <= 0xDF:
		length, accumulated = 2, uint32(first&0x1F)
	case first >= 0xE0 && first <= 0xEF:
		length, accumulated = 3, uint32(first&0x0F)
	case first >= 0xF0 && first <= 0xF4:
		length, accumulated = 4, uint32(first&0x07)
	default:
		return nil, false
	}
	if len(partial) >= length {
		return nil, false
	}
	for _, value := range partial[1:] {
		if value < 0x80 || value > 0xBF {
			return nil, false
		}
		accumulated = accumulated<<6 | uint32(value&0x3F)
	}
	remaining := length - len(partial)
	low := rune(accumulated << (6 * remaining))
	high := rune(uint32(low) | (1<<(6*remaining) - 1))
	minimum := []rune{0, 0, 0x80, 0x800, 0x10000}[length]
	low = max(low, minimum)
	if high > utf8.MaxRune {
		high = utf8.MaxRune
	}
	if low > high {
		return nil, false
	}
	if high < 0xD800 || low > 0xDFFF {
		return []gbnfRange{{low: low, high: high}}, true
	}
	var result []gbnfRange
	if low < 0xD800 {
		result = append(result, gbnfRange{low: low, high: 0xD7FF})
	}
	if high > 0xDFFF {
		result = append(result, gbnfRange{low: 0xE000, high: high})
	}
	return result, len(result) > 0
}

func stacksAccepting(stacks [][]gbnfSymbol) bool {
	for _, stack := range stacks {
		if len(stack) == 0 {
			return true
		}
	}
	return false
}

func grammarSignature(
	source, root string,
	pieces [][]byte,
	eos []bool,
	grammar *GBNFGrammar,
) uint64 {
	hash := fnv.New64a()
	writeBytes := func(value []byte) {
		var length [8]byte
		binary.LittleEndian.PutUint64(length[:], uint64(len(value)))
		_, _ = hash.Write(length[:])
		_, _ = hash.Write(value)
	}
	writeBytes([]byte(source))
	writeBytes([]byte(root))
	for index, piece := range pieces {
		writeBytes(piece)
		if eos[index] {
			_, _ = hash.Write([]byte{1})
		} else {
			_, _ = hash.Write([]byte{0})
		}
	}
	var encoded [8]byte
	for _, rule := range grammar.rules {
		binary.LittleEndian.PutUint64(encoded[:], uint64(len(rule.alternatives)))
		_, _ = hash.Write(encoded[:])
		for _, alternative := range rule.alternatives {
			binary.LittleEndian.PutUint64(encoded[:], uint64(len(alternative)))
			_, _ = hash.Write(encoded[:])
			for _, symbol := range alternative {
				_, _ = hash.Write([]byte{byte(symbol.kind)})
				binary.LittleEndian.PutUint64(encoded[:], uint64(symbol.index))
				_, _ = hash.Write(encoded[:])
			}
		}
	}
	if grammar.lazy {
		_, _ = hash.Write([]byte{1})
	} else {
		_, _ = hash.Write([]byte{0})
	}
	triggerTokens := make([]int, 0, len(grammar.triggerTokens))
	for token := range grammar.triggerTokens {
		triggerTokens = append(triggerTokens, token)
	}
	slices.Sort(triggerTokens)
	binary.LittleEndian.PutUint64(encoded[:], uint64(len(triggerTokens)))
	_, _ = hash.Write(encoded[:])
	for _, token := range triggerTokens {
		binary.LittleEndian.PutUint64(encoded[:], uint64(token))
		_, _ = hash.Write(encoded[:])
	}
	binary.LittleEndian.PutUint64(encoded[:], uint64(len(grammar.triggerPatterns)))
	_, _ = hash.Write(encoded[:])
	for _, pattern := range grammar.triggerPatterns {
		writeBytes([]byte(pattern.source))
	}
	return hash.Sum64()
}
