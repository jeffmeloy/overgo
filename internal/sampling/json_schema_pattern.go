package sampling

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type schemaPatternPart struct {
	text    string
	literal bool
}

type schemaPatternParser struct {
	converter *schemaConverter
	pattern   string
	name      string
	offset    int
	subrules  map[string]string
}

func (converter *schemaConverter) visitPattern(pattern, name string) (string, error) {
	if len(pattern) < 2 || pattern[0] != '^' || pattern[len(pattern)-1] != '$' {
		return "", errors.New("JSON schema pattern must start with ^ and end with $")
	}
	parser := &schemaPatternParser{
		converter: converter,
		pattern:   pattern[1 : len(pattern)-1],
		name:      name,
		subrules:  make(map[string]string),
	}
	part, err := parser.parse(false)
	if err != nil {
		return "", err
	}
	if parser.offset != len(parser.pattern) {
		return "", errors.New("JSON schema pattern was not fully consumed")
	}
	return converter.addRule(
		name,
		`"\"" (`+formatSchemaPatternPart(part)+`) "\""`,
	), nil
}

func formatSchemaPatternPart(part schemaPatternPart) string {
	if part.literal {
		return `"` + part.text + `"`
	}
	return part.text
}

func (parser *schemaPatternParser) parse(group bool) (schemaPatternPart, error) {
	parts := make([]schemaPatternPart, 0)
	for parser.offset < len(parser.pattern) {
		character := parser.pattern[parser.offset]
		switch character {
		case ')':
			if !group {
				return schemaPatternPart{}, errors.New("unbalanced pattern parenthesis")
			}
			parser.offset++
			return joinSchemaPatternParts(parts), nil
		case '(':
			parser.offset++
			if parser.offset < len(parser.pattern) && parser.pattern[parser.offset] == '?' {
				if parser.offset+1 >= len(parser.pattern) || parser.pattern[parser.offset+1] != ':' {
					return schemaPatternPart{}, errors.New("pattern lookaround/modifiers are unsupported")
				}
				parser.offset += 2
			}
			part, err := parser.parse(true)
			if err != nil {
				return schemaPatternPart{}, err
			}
			parts = append(parts, schemaPatternPart{
				text: "(" + formatSchemaPatternPart(part) + ")",
			})
		case '[':
			start := parser.offset
			parser.offset++
			for parser.offset < len(parser.pattern) && parser.pattern[parser.offset] != ']' {
				if parser.pattern[parser.offset] == '\\' {
					parser.offset++
				}
				parser.offset++
			}
			if parser.offset >= len(parser.pattern) {
				return schemaPatternPart{}, errors.New("unbalanced pattern character class")
			}
			parser.offset++
			parts = append(parts, schemaPatternPart{text: parser.pattern[start:parser.offset]})
		case '.':
			dot := parser.converter.addRule("dot", `[^\x0A\x0D]`)
			parts = append(parts, schemaPatternPart{text: dot})
			parser.offset++
		case '|':
			parts = append(parts, schemaPatternPart{text: "|"})
			parser.offset++
		case '*', '+', '?':
			if len(parts) == 0 {
				return schemaPatternPart{}, errors.New("pattern quantifier has no operand")
			}
			last := parts[len(parts)-1]
			parts[len(parts)-1] = schemaPatternPart{
				text: formatSchemaPatternPart(last) + string(character),
			}
			parser.offset++
		case '{':
			if len(parts) == 0 {
				return schemaPatternPart{}, errors.New("pattern repetition has no operand")
			}
			end := strings.IndexByte(parser.pattern[parser.offset:], '}')
			if end < 0 {
				return schemaPatternPart{}, errors.New("unbalanced pattern repetition")
			}
			end += parser.offset
			fields := strings.Split(parser.pattern[parser.offset+1:end], ",")
			minimum := 0
			if text := strings.TrimSpace(fields[0]); text != "" {
				value, err := strconv.Atoi(text)
				if err != nil || value < 0 {
					return schemaPatternPart{}, errors.New("invalid pattern repetition")
				}
				minimum = value
			}
			var maximum *int
			if len(fields) == 1 {
				value := minimum
				maximum = &value
			} else if len(fields) == 2 && strings.TrimSpace(fields[1]) != "" {
				value, parseErr := strconv.Atoi(strings.TrimSpace(fields[1]))
				if parseErr != nil || value < 0 {
					return schemaPatternPart{}, errors.New("invalid pattern repetition")
				}
				maximum = &value
			} else if len(fields) > 2 {
				return schemaPatternPart{}, errors.New("invalid pattern repetition")
			}
			if maximum != nil && *maximum < minimum {
				return schemaPatternPart{}, errors.New("pattern repetition maximum is below minimum")
			}
			last := parts[len(parts)-1]
			operand := formatSchemaPatternPart(last)
			if !last.literal {
				if existing := parser.subrules[last.text]; existing != "" {
					operand = existing
				} else {
					rule := parser.converter.addRule(
						fmt.Sprintf("%s-%d", parser.name, len(parser.subrules)+1),
						last.text,
					)
					parser.subrules[last.text] = rule
					operand = rule
				}
			}
			parts[len(parts)-1] = schemaPatternPart{
				text: buildSchemaRepetition(operand, minimum, maximum, ""),
			}
			parser.offset = end + 1
		default:
			start := parser.offset
			var literal strings.Builder
			for parser.offset < len(parser.pattern) {
				current := parser.pattern[parser.offset]
				if current == '\\' && parser.offset+1 == len(parser.pattern) {
					return schemaPatternPart{}, errors.New("pattern has a dangling escape")
				}
				if current == '\\' && parser.offset+1 < len(parser.pattern) {
					next := parser.pattern[parser.offset+1]
					if strings.ContainsRune(`^$.[]()|{}*+?`, rune(next)) {
						literal.WriteByte(next)
					} else {
						literal.WriteByte(current)
						literal.WriteByte(next)
					}
					parser.offset += 2
					continue
				}
				nonLiteral := strings.ContainsRune(`|.()[]{}*+?`, rune(current))
				nextIsNonLiteral := parser.offset+1 < len(parser.pattern) &&
					strings.ContainsRune(`|.()[]{}*+?`, rune(parser.pattern[parser.offset+1]))
				if nonLiteral ||
					(parser.offset+1 < len(parser.pattern) &&
						literal.Len() != 0 &&
						parser.pattern[parser.offset+1] != '.' &&
						nextIsNonLiteral) {
					break
				}
				if current == '"' {
					literal.WriteString(`\"`)
				} else {
					literal.WriteByte(current)
				}
				parser.offset++
			}
			if parser.offset == start || literal.Len() == 0 {
				return schemaPatternPart{}, fmt.Errorf("unsupported pattern byte %q", character)
			}
			parts = append(parts, schemaPatternPart{
				text:    literal.String(),
				literal: true,
			})
		}
	}
	if group {
		return schemaPatternPart{}, errors.New("unclosed pattern group")
	}
	return joinSchemaPatternParts(parts), nil
}

func joinSchemaPatternParts(parts []schemaPatternPart) schemaPatternPart {
	if len(parts) == 0 {
		return schemaPatternPart{text: "", literal: true}
	}
	merged := make([]schemaPatternPart, 0, len(parts))
	for _, part := range parts {
		if part.literal && len(merged) > 0 && merged[len(merged)-1].literal {
			merged[len(merged)-1].text += part.text
			continue
		}
		merged = append(merged, part)
	}
	if len(merged) == 1 {
		return merged[0]
	}
	values := make([]string, len(merged))
	for index, part := range merged {
		values[index] = formatSchemaPatternPart(part)
	}
	return schemaPatternPart{text: strings.Join(values, " ")}
}
