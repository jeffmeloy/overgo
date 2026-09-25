package webuilane

import (
	"slices"
	"strings"
	"unicode"
)

// jsTokenKind classifies one JavaScript token: the census reads kinds, so
// a word inside a string or a comment is never taken for code.
type jsTokenKind int

const (
	jsIdentifier jsTokenKind = iota
	jsNumber
	jsString
	jsTemplate
	jsRegex
	jsComment
	jsPunctuator
)

// jsToken is one token with the line it starts on.
type jsToken struct {
	kind jsTokenKind
	text string
	line int
}

// value is a string token's contents between its quotes.
func (token jsToken) value() string {
	if token.kind != jsString || len(token.text) < len(`""`) {
		return ""
	}
	return token.text[1 : len(token.text)-1]
}

func (token jsToken) is(kind jsTokenKind, text string) bool {
	return token.kind == kind && token.text == text
}

// jsPunctuators are the multi-character punctuators the census tells
// apart: an arrow, and "?." and "??", which are not a ternary's "?". Every
// other punctuator is read one character at a time.
var jsPunctuators = []string{"=>", "?.", "??"}

// regexAfterWords are the words after which a slash opens a regular
// expression rather than dividing.
var regexAfterWords = []string{"return", "typeof", "case", "do", "else", "in", "of", "new", "delete", "void", "throw", "instanceof", "yield", "await"}

// tokenizeJS splits JavaScript source into tokens. A template literal is one
// token, its substitutions included; a slash opens a regular expression
// where no expression has just ended.
func tokenizeJS(source string) []jsToken {
	var tokens []jsToken
	var line int
	for index := 0; index < len(source); {
		character := source[index]
		start := index
		switch {
		case character == '\n':
			line++
			index++
			continue
		case character == ' ' || character == '\t' || character == '\r':
			index++
			continue
		case strings.HasPrefix(source[index:], "//"):
			index += strings.IndexByte(source[index:]+"\n", '\n')
			tokens = append(tokens, jsToken{jsComment, source[start:index], line})
			continue
		case strings.HasPrefix(source[index:], "/*"):
			end := strings.Index(source[index+len("/*"):], "*/")
			if end < 0 {
				index = len(source)
			} else {
				index += len("/*") + end + len("*/")
			}
		case character == '"' || character == '\'':
			index = skipQuoted(source, index)
		case character == '`':
			index = skipTemplate(source, index)
		case character == '/' && regexAllowed(tokens):
			index = skipRegex(source, index)
		case isIdentifierByte(character):
			for index < len(source) && isIdentifierByte(source[index]) {
				index++
			}
		default:
			index++
			for _, punctuator := range jsPunctuators {
				if strings.HasPrefix(source[start:], punctuator) {
					index = start + len(punctuator)
					break
				}
			}
		}
		text := source[start:index]
		tokens = append(tokens, jsToken{classifyJS(text), text, line})
		line += strings.Count(text, "\n")
	}
	return tokens
}

func classifyJS(text string) jsTokenKind {
	switch first := text[0]; {
	case strings.HasPrefix(text, "/*"):
		return jsComment
	case first == '"' || first == '\'':
		return jsString
	case first == '`':
		return jsTemplate
	case first == '/' && len(text) > len("/"):
		return jsRegex
	case first >= '0' && first <= '9':
		return jsNumber
	case isIdentifierByte(first):
		return jsIdentifier
	}
	return jsPunctuator
}

func isIdentifierByte(character byte) bool {
	return character == '_' || character == '$' || character >= 'a' && character <= 'z' ||
		character >= 'A' && character <= 'Z' || character >= '0' && character <= '9' || rune(character) > unicode.MaxASCII
}

// regexAllowed: a slash opens a regular expression unless the token before
// it ends an expression (a name, a literal, a closing bracket).
func regexAllowed(tokens []jsToken) bool {
	for index := len(tokens) - 1; index >= 0; index-- {
		previous := tokens[index]
		switch previous.kind {
		case jsComment:
			continue
		case jsIdentifier:
			return slices.Contains(regexAfterWords, previous.text)
		case jsNumber, jsString, jsTemplate, jsRegex:
			return false
		}
		return previous.text != ")" && previous.text != "]" && previous.text != "}"
	}
	return true
}

func skipQuoted(source string, index int) int {
	quote := source[index]
	for index++; index < len(source) && source[index] != quote && source[index] != '\n'; index++ {
		if source[index] == '\\' {
			index++
		}
	}
	return min(index+1, len(source))
}

// skipTemplate passes a template literal and its substitutions, whose
// braces, strings and nested templates are balanced.
func skipTemplate(source string, index int) int {
	for index++; index < len(source); index++ {
		switch {
		case source[index] == '\\':
			index++
		case source[index] == '`':
			return index + 1
		case strings.HasPrefix(source[index:], "${"):
			depth := 0
			for index += len("$"); index < len(source); index++ {
				switch source[index] {
				case '{':
					depth++
				case '}':
					depth--
				case '"', '\'':
					index = skipQuoted(source, index) - 1
				case '`':
					index = skipTemplate(source, index) - 1
				}
				if depth == 0 {
					break
				}
			}
		}
	}
	return len(source)
}

// skipRegex passes a regular expression literal, its classes and flags.
func skipRegex(source string, index int) int {
	class := false
	for index++; index < len(source) && source[index] != '\n'; index++ {
		switch source[index] {
		case '\\':
			index++
		case '[':
			class = true
		case ']':
			class = false
		case '/':
			if !class {
				for index++; index < len(source) && isIdentifierByte(source[index]); index++ {
				}
				return index
			}
		}
	}
	return index
}

// closing is the index of the token that closes the bracket opened at open.
func closing(tokens []jsToken, open int) int {
	depth := 0
	for index := open; index < len(tokens); index++ {
		if tokens[index].kind != jsPunctuator {
			continue
		}
		switch tokens[index].text {
		case "(", "[", "{":
			depth++
		case ")", "]", "}":
			if depth--; depth == 0 {
				return index
			}
		}
	}
	return len(tokens) - 1
}

// arguments splits a call's argument tokens, between its brackets, at its
// own commas.
func arguments(tokens []jsToken) [][]jsToken {
	var split [][]jsToken
	depth, start := 0, 0
	for index, token := range tokens {
		if token.kind != jsPunctuator {
			continue
		}
		switch token.text {
		case "(", "[", "{":
			depth++
		case ")", "]", "}":
			depth--
		case ",":
			if depth == 0 {
				split = append(split, tokens[start:index])
				start = index + 1
			}
		}
	}
	if start < len(tokens) {
		split = append(split, tokens[start:])
	}
	return split
}
