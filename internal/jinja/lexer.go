package jinja

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type chunkKind int

const (
	cText chunkKind = iota
	cOutput
	cBlock
	cComment
)

type chunk struct {
	kind         chunkKind
	text         string // cText
	name         string // cBlock: control-structure name
	toks         []etok // cOutput / cBlock expression tokens
	trimL, trimR bool
}

type etKind int

const (
	etName etKind = iota
	etStr
	etInt
	etFloat
	etSym
)

type etok struct {
	kind etKind
	val  string // for etStr: the decoded string; for numbers: raw; for sym: the symbol; for name: identifier
}

// normalizeInput mirrors gonja tokens.normalizeInput with default config
// (KeepTrailingNewline=false).
func normalizeInput(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	if strings.HasSuffix(s, "\n") {
		s = s[:len(s)-1]
	}
	return s
}

// lex splits the template into chunks. Expression content is pre-tokenized.
func lex(src string) ([]chunk, error) {
	src = normalizeInput(src)
	var chunks []chunk
	i := 0
	n := len(src)
	textStart := 0
	flushText := func(end int) {
		if end > textStart {
			chunks = append(chunks, chunk{kind: cText, text: src[textStart:end]})
		}
	}
	for i < n {
		if src[i] == '{' && i+1 < n {
			switch src[i+1] {
			case '{', '%', '#':
				flushText(i)
				var endDelim string
				var kind chunkKind
				switch src[i+1] {
				case '{':
					endDelim, kind = "}}", cOutput
				case '%':
					endDelim, kind = "%}", cBlock
				case '#':
					endDelim, kind = "#}", cComment
				}
				// left trim control
				trimL := false
				inner := i + 2
				if inner < n && (src[inner] == '-' || src[inner] == '+') {
					trimL = src[inner] == '-'
					inner++
				}
				// find end delimiter, honoring right-trim control (- or + before it)
				idx := strings.Index(src[inner:], endDelim)
				if idx < 0 {
					return nil, fmt.Errorf("jinja: unclosed %q", endDelim)
				}
				contentEnd := inner + idx
				trimR := false
				body := src[inner:contentEnd]
				if kind == cComment {
					if strings.HasSuffix(body, "-") {
						trimR = true
					} else if strings.HasSuffix(body, "+") {
						body = body[:len(body)-1]
					}
					chunks = append(chunks, chunk{kind: cComment, trimL: trimL, trimR: trimR})
				} else {
					// trailing whitespace-control char immediately before delimiter
					trimmed := strings.TrimRight(body, " \t")
					if strings.HasSuffix(trimmed, "-") {
						trimR = true
						body = strings.TrimSuffix(trimmed, "-")
					} else if strings.HasSuffix(trimmed, "+") {
						body = strings.TrimSuffix(trimmed, "+")
					}
					toks, name, err := lexExpr(body, kind)
					if err != nil {
						return nil, err
					}
					chunks = append(chunks, chunk{kind: kind, name: name, toks: toks, trimL: trimL, trimR: trimR})
				}
				i = contentEnd + len(endDelim)
				textStart = i
				continue
			}
		}
		i++
	}
	flushText(n)
	return chunks, nil
}

// lexExpr tokenizes the inner content of a {{ }} or {% %} tag. For blocks it
// also returns the leading control-structure name.
func lexExpr(body string, kind chunkKind) ([]etok, string, error) {
	var toks []etok
	name := ""
	i := 0
	n := len(body)
	skipSpace := func() {
		for i < n && (body[i] == ' ' || body[i] == '\t' || body[i] == '\n') {
			i++
		}
	}
	if kind == cBlock {
		skipSpace()
		start := i
		for i < n && isIdentRune(rune(body[i])) {
			i++
		}
		name = body[start:i]
	}
	for i < n {
		c := body[i]
		if c == ' ' || c == '\t' || c == '\n' {
			i++
			continue
		}
		switch {
		case c == '"' || c == '\'':
			s, ni, err := scanString(body, i)
			if err != nil {
				return nil, "", err
			}
			toks = append(toks, etok{kind: etStr, val: s})
			i = ni
		case c >= '0' && c <= '9':
			tok, ni := scanNumber(body, i)
			toks = append(toks, tok)
			i = ni
		case isIdentStart(rune(c)):
			start := i
			for i < n && isIdentRune(rune(body[i])) {
				i++
			}
			toks = append(toks, etok{kind: etName, val: body[start:i]})
		default:
			sym, ni := scanSymbol(body, i)
			if sym == "" {
				// unknown; skip to avoid infinite loop
				return nil, "", fmt.Errorf("jinja: unexpected character %q in expression", string(c))
			}
			toks = append(toks, etok{kind: etSym, val: sym})
			i = ni
		}
	}
	return toks, name, nil
}

func scanSymbol(body string, i int) (string, int) {
	two := ""
	if i+1 < len(body) {
		two = body[i : i+2]
	}
	switch two {
	case "**", "//", "==", "!=", "<=", ">=":
		return two, i + 2
	}
	switch body[i] {
	case '+', '-', '~', ':', '.', '%', '/', '<', '>', '*', '(', ')', '[', ']', '{', '}', ',', '|', '=':
		return string(body[i]), i + 1
	}
	return "", i
}

func scanNumber(body string, i int) (etok, int) {
	start := i
	n := len(body)
	isFloatTok := false
	// hex/bin/oct
	if body[i] == '0' && i+1 < n && (body[i+1] == 'x' || body[i+1] == 'X' || body[i+1] == 'b' || body[i+1] == 'B' || body[i+1] == 'o' || body[i+1] == 'O') {
		i += 2
		for i < n && (isHex(body[i]) || body[i] == '_') {
			i++
		}
		return etok{kind: etInt, val: body[start:i]}, i
	}
	for i < n && (body[i] >= '0' && body[i] <= '9' || body[i] == '_') {
		i++
	}
	if i < n && body[i] == '.' && i+1 < n && body[i+1] >= '0' && body[i+1] <= '9' {
		isFloatTok = true
		i++
		for i < n && (body[i] >= '0' && body[i] <= '9' || body[i] == '_') {
			i++
		}
	}
	if i < n && (body[i] == 'e' || body[i] == 'E') {
		j := i + 1
		if j < n && (body[j] == '+' || body[j] == '-') {
			j++
		}
		if j < n && body[j] >= '0' && body[j] <= '9' {
			isFloatTok = true
			i = j
			for i < n && body[i] >= '0' && body[i] <= '9' {
				i++
			}
		}
	}
	if isFloatTok {
		return etok{kind: etFloat, val: body[start:i]}, i
	}
	return etok{kind: etInt, val: body[start:i]}, i
}

func isHex(b byte) bool {
	return b >= '0' && b <= '9' || b >= 'a' && b <= 'f' || b >= 'A' && b <= 'F'
}

// scanString reads a quoted string and decodes escapes exactly as gonja does:
// the lexer strips quotes and unescapes \" and \', then parseString runs
// strconv.Quote -> replace(\\ -> \) -> strconv.Unquote.
func scanString(body string, i int) (string, int, error) {
	quote := body[i]
	n := len(body)
	j := i + 1
	var raw strings.Builder
	for j < n {
		c := body[j]
		if c == '\\' && j+1 < n {
			nxt := body[j+1]
			if nxt == quote {
				raw.WriteByte(nxt)
				j += 2
				continue
			}
			raw.WriteByte('\\')
			raw.WriteByte(nxt)
			j += 2
			continue
		}
		if c == quote {
			j++
			// gonja lexer unescape also replaces \" and \' regardless of quote.
			tokVal := raw.String()
			tokVal = strings.ReplaceAll(tokVal, `\"`, `"`)
			tokVal = strings.ReplaceAll(tokVal, `\'`, `'`)
			decoded := decodeStringLiteral(tokVal)
			return decoded, j, nil
		}
		raw.WriteByte(c)
		j++
	}
	return "", i, fmt.Errorf("jinja: unclosed string literal")
}

// decodeStringLiteral mirrors parser.parseString: Quote, collapse \\ -> \,
// then Unquote.
func decodeStringLiteral(v string) string {
	q := strconv.Quote(v)
	q = strings.ReplaceAll(q, `\\`, `\`)
	s, err := strconv.Unquote(q)
	if err != nil {
		return v
	}
	return s
}

func isIdentStart(r rune) bool {
	return r == '_' || unicode.IsLetter(r)
}

func isIdentRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}

var _ = utf8.RuneLen
