package server

import (
	"bytes"
	"compress/zlib"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"overgo/internal/binaryschema"
)

// PDF documents attach to a conversation as text (professional GUI
// campaign, gui-multimodal/import): the server extracts the text the page
// content streams draw and hands the model the same file part a plain-text
// document would carry. The extraction reads the string operands of the
// text-showing operators; text drawn through composite fonts as two-byte
// glyph identifiers has no character mapping here and is left out rather
// than rendered as noise.

var pdfStreamDictionary = regexp.MustCompile(`(?s)<<(.*)>>\s*stream\r?\n`)

// pdfText extracts the drawn text of every content stream in the document.
func pdfText(data []byte) (string, error) {
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		return "", errors.New("not a PDF document")
	}
	var out strings.Builder
	for object := range bytes.SplitSeq(data, []byte("endobj")) {
		match := pdfStreamDictionary.FindSubmatchIndex(object)
		if match == nil {
			continue
		}
		dictionary := string(object[match[2]:match[3]])
		body := object[match[1]:]
		end := bytes.LastIndex(body, []byte("endstream"))
		if end < 0 {
			continue
		}
		body = bytes.TrimRight(body[:end], "\r\n")
		if strings.Contains(dictionary, "/Image") || strings.Contains(dictionary, "/Length1") ||
			strings.Contains(dictionary, "/ObjStm") || strings.Contains(dictionary, "/XRef") {
			continue
		}
		if strings.Contains(dictionary, "/FlateDecode") {
			inflated, err := inflatePDFStream(body)
			if err != nil {
				continue
			}
			body = inflated
		} else if strings.Contains(dictionary, "/Filter") {
			continue
		}
		pdfContentText(&out, body)
	}
	text := strings.TrimSpace(out.String())
	if text == "" {
		return "", errors.New("no extractable text")
	}
	return text, nil
}

func inflatePDFStream(body []byte) ([]byte, error) {
	reader, err := zlib.NewReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	inflated, err := io.ReadAll(io.LimitReader(reader, maxMediaBytes))
	if err != nil && len(inflated) == 0 {
		return nil, err
	}
	return inflated, nil
}

// pdfContentText walks one content stream's tokens: string operands
// accumulate until a text-showing operator emits them, positioning
// operators and the end of a text object break the line, and an inline
// image's binary body is skipped.
func pdfContentText(out *strings.Builder, content []byte) {
	var pending []string
	lineOpen := false
	newline := func() {
		if lineOpen {
			out.WriteByte('\n')
			lineOpen = false
		}
		pending = nil
	}
	emit := func(leadingBreak bool) {
		if leadingBreak {
			newline()
		}
		text := strings.Join(pending, "")
		pending = nil
		if text != "" {
			out.WriteString(text)
			lineOpen = true
		}
	}
	for i := 0; i < len(content); {
		c := content[i]
		switch {
		case c == '(':
			text, next := pdfLiteralString(content, i)
			pending = append(pending, text)
			i = next
		case c == '<' && i+1 < len(content) && content[i+1] != '<':
			text, next := pdfHexString(content, i)
			pending = append(pending, text)
			i = next
		case c == '%':
			for i < len(content) && content[i] != '\n' && content[i] != '\r' {
				i++
			}
		case c == '/':
			i++
			for i < len(content) && !pdfDelimiter(content[i]) {
				i++
			}
		case c == '-' || c == '+' || c == '.' || (c >= '0' && c <= '9'):
			start := i
			for i < len(content) && !pdfDelimiter(content[i]) {
				i++
			}
			// A large negative adjustment inside a TJ array moves the next
			// glyph right by more than a glyph's width: a word space.
			if value, err := strconv.ParseFloat(string(content[start:i]), binaryschema.Width64Bits); err == nil && value < -pdfWordGapThousandths {
				pending = append(pending, " ")
			}
		case (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '\'' || c == '"' || c == '*':
			start := i
			for i < len(content) && !pdfDelimiter(content[i]) {
				i++
			}
			switch operator := string(content[start:i]); operator {
			case "Tj", "TJ":
				emit(false)
			case "'", "\"":
				emit(true)
			case "Td", "TD", "T*", "Tm", "ET":
				newline()
			case "ID":
				if end := bytes.Index(content[i:], []byte("EI")); end >= 0 {
					i += end + len("EI")
				} else {
					i = len(content)
				}
				pending = nil
			default:
				pending = nil
			}
		default:
			i++
		}
	}
	newline()
}

// pdfWordGapThousandths: a TJ adjustment beyond this many thousandths of
// the text space unit is read as a word gap rather than kerning.
const pdfWordGapThousandths = 200

func pdfDelimiter(c byte) bool {
	return c <= ' ' || strings.IndexByte("()<>[]{}/%", c) >= 0
}

// pdfLiteralString decodes a parenthesized string starting at content[start].
func pdfLiteralString(content []byte, start int) (string, int) {
	var text strings.Builder
	depth := 0
	i := start
	for ; i < len(content); i++ {
		c := content[i]
		switch c {
		case '\\':
			i++
			if i >= len(content) {
				return text.String(), i
			}
			switch escaped := content[i]; escaped {
			case 'n':
				text.WriteByte('\n')
			case 'r', 't', 'b', 'f':
				text.WriteByte(' ')
			case '\n':
			case '\r':
				if i+1 < len(content) && content[i+1] == '\n' {
					i++
				}
			default:
				if escaped >= '0' && escaped <= '7' {
					// An octal escape carries as many digits as still fit a byte.
					value := 0
					for ; i < len(content) && content[i] >= '0' && content[i] <= '7'; i++ {
						next := value*binaryschema.OctalRadix + int(content[i]-'0')
						if next > math.MaxUint8 {
							break
						}
						value = next
					}
					i--
					writePDFByte(&text, byte(value))
				} else {
					writePDFByte(&text, escaped)
				}
			}
		case '(':
			if depth > 0 {
				text.WriteByte('(')
			}
			depth++
		case ')':
			depth--
			if depth == 0 {
				return text.String(), i + 1
			}
			text.WriteByte(')')
		default:
			writePDFByte(&text, c)
		}
	}
	return text.String(), i
}

// pdfHexString decodes an angle-bracketed hex string; a string of two-byte
// glyph identifiers (every high byte zero) carries no characters and is
// left out.
func pdfHexString(content []byte, start int) (string, int) {
	end := bytes.IndexByte(content[start:], '>')
	if end < 0 {
		return "", len(content)
	}
	digits := strings.Join(strings.Fields(string(content[start+1:start+end])), "")
	raw, err := hex.DecodeString(digits)
	if err != nil && !errors.Is(err, hex.ErrLength) {
		return "", start + end + 1
	}
	glyphIDs := len(raw) > 0 && len(raw)%binaryschema.Uint16Bytes == 0
	for j := 0; glyphIDs && j < len(raw); j += binaryschema.Uint16Bytes {
		glyphIDs = raw[j] == 0
	}
	var text strings.Builder
	if !glyphIDs {
		for _, c := range raw {
			writePDFByte(&text, c)
		}
	}
	return text.String(), start + end + 1
}

// writePDFByte keeps the printable characters a simple font encoding
// shares with Latin-1; control bytes are dropped.
func writePDFByte(text *strings.Builder, c byte) {
	if r := rune(c); r == ' ' || unicode.IsPrint(r) {
		text.WriteRune(r)
	}
}
