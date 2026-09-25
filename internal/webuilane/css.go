package webuilane

import "strings"

// cssRule is one rule of a stylesheet: its prelude (a selector list or an
// at-rule, whitespace collapsed), its declarations, and the rules its block
// nests (an at-rule's rules, or nested style rules).
type cssRule struct {
	prelude      string
	declarations []cssDeclaration
	rules        []cssRule
}

// cssDeclaration is one property and its value, trimmed.
type cssDeclaration struct {
	property, value string
}

// parseCSS reads a stylesheet into its top-level rules. Comments are
// dropped; strings and parenthesised values are read whole, so a brace, a
// colon or a semicolon inside them splits nothing.
func parseCSS(source string) []cssRule {
	parser := cssParser{source: source}
	block := parser.block()
	return block.rules
}

type cssParser struct {
	source string
	index  int
}

// block reads declarations and nested rules up to the brace that closes
// the block, or the end of the source.
func (parser *cssParser) block() cssRule {
	var block cssRule
	var pending strings.Builder
	depth := 0
	flush := func() {
		text := strings.TrimSpace(pending.String())
		pending.Reset()
		if property, value, found := strings.Cut(text, ":"); found && text != "" {
			block.declarations = append(block.declarations, cssDeclaration{strings.TrimSpace(property), strings.TrimSpace(value)})
		}
	}
	for parser.index < len(parser.source) {
		character := parser.source[parser.index]
		switch {
		case strings.HasPrefix(parser.source[parser.index:], "/*"):
			end := strings.Index(parser.source[parser.index+len("/*"):], "*/")
			if end < 0 {
				parser.index = len(parser.source)
				continue
			}
			parser.index += len("/*") + end + len("*/")
			continue
		case character == '"' || character == '\'':
			end := skipQuoted(parser.source, parser.index)
			pending.WriteString(parser.source[parser.index:end])
			parser.index = end
			continue
		case character == '(':
			depth++
		case character == ')':
			depth--
		case character == ';' && depth == 0:
			flush()
			parser.index++
			continue
		case character == '{' && depth == 0:
			prelude := strings.Join(strings.Fields(pending.String()), " ")
			pending.Reset()
			parser.index++
			nested := parser.block()
			nested.prelude = prelude
			block.rules = append(block.rules, nested)
			continue
		case character == '}' && depth == 0:
			flush()
			parser.index++
			return block
		}
		pending.WriteByte(character)
		parser.index++
	}
	flush()
	return block
}

// rule answers the nested rule with this prelude.
func (rule cssRule) rule(prelude string) (cssRule, bool) {
	for _, nested := range rule.rules {
		if nested.prelude == prelude {
			return nested, true
		}
	}
	return cssRule{}, false
}

// walk visits the rule and every rule it nests.
func (rule cssRule) walk(visit func(cssRule)) {
	visit(rule)
	for _, nested := range rule.rules {
		nested.walk(visit)
	}
}
