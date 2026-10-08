package seq2seq

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// The model was trained on tools named in snake case. Its own code rewrites
// each tool's name that way before the tools are read, and writes the names
// the caller gave back into the calls the model answers with. This file does
// the two rewrites as that code does them, down to the text it leaves: a
// tools list goes into the encoder as the characters that code would have
// written, since those are what the model was shown, and an answer leaves as
// the characters that code would have returned.

// nameKey is the key a tool, and a call, is named under.
const nameKey = "name"

func upperLetter(char byte) bool { return char >= 'A' && char <= 'Z' }
func lowerLetter(char byte) bool { return char >= 'a' && char <= 'z' }
func digit(char byte) bool       { return char >= '0' && char <= '9' }

// nameCharacter is a character a snake-case name keeps.
func nameCharacter(char byte) bool {
	return upperLetter(char) || lowerLetter(char) || digit(char) || char == '_'
}

// snakeCase rewrites a camel-case, Pascal-case or dotted name in snake case,
// by the model's own rule and in its order: every run of characters that are
// not ASCII letters, digits or underscores becomes one underscore; an
// underscore parts a lower-case letter or digit from a capital after it, and
// the last capital of a run of them from the lower-case letter it begins a
// word with; runs of underscores become one; and the name is lowered and
// loses the underscores at its ends.
func snakeCase(name string) string {
	plain := make([]byte, 0, len(name))
	for index := range len(name) {
		char := name[index]
		switch {
		case nameCharacter(char):
			plain = append(plain, char)
		case index == 0 || nameCharacter(name[index-1]):
			plain = append(plain, '_')
		}
	}
	parted := make([]byte, 0, len(plain))
	for index, char := range plain {
		if index > 0 && upperLetter(char) && (lowerLetter(plain[index-1]) || digit(plain[index-1])) {
			parted = append(parted, '_')
		}
		parted = append(parted, char)
	}
	worded := make([]byte, 0, len(parted))
	for index, char := range parted {
		if index > 0 && index+1 < len(parted) && upperLetter(char) && upperLetter(parted[index-1]) && lowerLetter(parted[index+1]) {
			worded = append(worded, '_')
		}
		worded = append(worded, char)
	}
	snake := make([]byte, 0, len(worded))
	for _, char := range worded {
		if char == '_' && len(snake) != 0 && snake[len(snake)-1] == '_' {
			continue
		}
		snake = append(snake, char)
	}
	return strings.Trim(strings.ToLower(string(snake)), "_")
}

// normalizeTools rewrites the name of each tool in a JSON list of tools in
// snake case and returns the list written compactly, with each new name's
// original. Any other JSON value is written back compactly with no names, and
// text that is not JSON is returned as it is.
func normalizeTools(tools string) (string, map[string]string) {
	value, err := parseOrdered(tools)
	if err != nil {
		return tools, nil
	}
	var names map[string]string
	if value.kind == kindList {
		names = map[string]string{}
		for index := range value.items {
			if name := value.items[index].named(); name != nil {
				snake := snakeCase(name.text)
				names[snake], name.text = name.text, snake
			}
		}
	}
	return value.compact(), names
}

// restoreToolNames writes the names the caller gave back into the calls the
// model answered with: by name in an answer that is JSON, a list of calls or
// one call, written back compactly; and by replacing each snake-case name
// where it stands, the longest first, in one that is not.
func restoreToolNames(answer string, names map[string]string) string {
	if len(names) == 0 {
		return answer
	}
	value, err := parseOrdered(answer)
	if err != nil {
		snakes := make([]string, 0, len(names))
		for snake := range names {
			snakes = append(snakes, snake)
		}
		slices.SortFunc(snakes, func(left, right string) int {
			return cmp.Or(cmp.Compare(len(right), len(left)), strings.Compare(left, right))
		})
		for _, snake := range snakes {
			answer = strings.ReplaceAll(answer, snake, names[snake])
		}
		return answer
	}
	calls := []*ordered{&value}
	if value.kind == kindList {
		calls = calls[:0]
		for index := range value.items {
			calls = append(calls, &value.items[index])
		}
	}
	for _, call := range calls {
		if name := call.named(); name != nil {
			if original, known := names[name.text]; known {
				name.text = original
			}
		}
	}
	return value.compact()
}

// The kinds of JSON value.
const (
	kindNull = iota
	kindBool
	kindNumber
	kindString
	kindList
	kindObject
)

// ordered is one JSON value with an object's keys in the order they came, a
// key given twice keeping its first place and its last value, as the model's
// own code reads one.
type ordered struct {
	kind   int
	text   string
	truth  bool
	items  []ordered
	keys   []string
	values []ordered
}

// named is the string an object is named by, nil for any other value.
func (value *ordered) named() *ordered {
	if value.kind != kindObject {
		return nil
	}
	at := slices.Index(value.keys, nameKey)
	if at < 0 || value.values[at].kind != kindString {
		return nil
	}
	return &value.values[at]
}

// parseOrdered reads text as one JSON value.
func parseOrdered(text string) (ordered, error) {
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	value, err := readOrdered(decoder)
	if err != nil {
		return ordered{}, err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return ordered{}, errors.New("seq2seq: the text holds more than one JSON value")
	}
	return value, nil
}

func readOrdered(decoder *json.Decoder) (ordered, error) {
	token, err := decoder.Token()
	if err != nil {
		return ordered{}, err
	}
	switch token := token.(type) {
	case json.Delim:
		value := ordered{kind: kindList}
		if token == '{' {
			value.kind = kindObject
		}
		for decoder.More() {
			key := ""
			if value.kind == kindObject {
				name, err := decoder.Token()
				if err != nil {
					return ordered{}, err
				}
				key = name.(string)
			}
			item, err := readOrdered(decoder)
			if err != nil {
				return ordered{}, err
			}
			switch at := slices.Index(value.keys, key); {
			case value.kind == kindList:
				value.items = append(value.items, item)
			case at >= 0:
				value.values[at] = item
			default:
				value.keys, value.values = append(value.keys, key), append(value.values, item)
			}
		}
		_, err := decoder.Token()
		return value, err
	case string:
		return ordered{kind: kindString, text: token}, nil
	case json.Number:
		return ordered{kind: kindNumber, text: token.String()}, nil
	case bool:
		return ordered{kind: kindBool, truth: token}, nil
	default:
		return ordered{kind: kindNull}, nil
	}
}

// compact writes the value as Python's json.dumps writes it with compact
// separators, which is the text the model's own code hands the tokenizer and
// hands back to its caller.
func (value ordered) compact() string {
	var out bytes.Buffer
	value.write(&out)
	return out.String()
}

func (value ordered) write(out *bytes.Buffer) {
	switch value.kind {
	case kindObject:
		out.WriteByte('{')
		for index, key := range value.keys {
			if index != 0 {
				out.WriteByte(',')
			}
			writePythonString(out, key)
			out.WriteByte(':')
			value.values[index].write(out)
		}
		out.WriteByte('}')
	case kindList:
		out.WriteByte('[')
		for index, item := range value.items {
			if index != 0 {
				out.WriteByte(',')
			}
			item.write(out)
		}
		out.WriteByte(']')
	case kindString:
		writePythonString(out, value.text)
	case kindNumber:
		out.WriteString(pythonNumber(value.text))
	case kindBool:
		fmt.Fprint(out, value.truth)
	default:
		out.WriteString("null")
	}
}

// The exponents between which Python writes a float in full and not by its
// power of ten, and the digits it gives a power at least.
const (
	fullFrom, fullBelow = -4, 16
	powerDigits         = 2
	floatBits           = 64
)

// pythonNumber writes a number as Python reads and writes it back: a whole
// number as it is, less the sign of a zero, and any other as the shortest
// digits that read back the same, in full between a ten-thousandth and ten
// to the sixteenth and by its power of ten outside.
func pythonNumber(literal string) string {
	if !strings.ContainsAny(literal, ".eE") {
		if strings.Trim(literal, "-0") == "" {
			return "0"
		}
		return literal
	}
	number, err := strconv.ParseFloat(literal, floatBits)
	if err != nil {
		return literal
	}
	mantissa, power, _ := strings.Cut(strconv.FormatFloat(number, 'e', -1, floatBits), "e")
	exponent, _ := strconv.Atoi(power)
	if exponent >= fullFrom && exponent < fullBelow {
		full := strconv.FormatFloat(number, 'f', -1, floatBits)
		if !strings.Contains(full, ".") {
			full += ".0"
		}
		return full
	}
	sign := "+"
	if exponent < 0 {
		sign, exponent = "-", -exponent
	}
	return fmt.Sprintf("%se%s%0*d", mantissa, sign, powerDigits, exponent)
}

// maxBasicPlane is the last character one escape of four hex digits can
// write; one beyond it is written as two.
const maxBasicPlane = 0xffff

// writePythonString writes a string as Python's json.dumps does: a quote and
// a backslash escaped, the named control characters by their names, any other
// control character and every character outside ASCII as four hex digits.
func writePythonString(out *bytes.Buffer, text string) {
	out.WriteByte('"')
	for _, char := range text {
		switch {
		case char == '"':
			out.WriteString(`\"`)
		case char == '\\':
			out.WriteString(`\\`)
		case char == '\n':
			out.WriteString(`\n`)
		case char == '\r':
			out.WriteString(`\r`)
		case char == '\t':
			out.WriteString(`\t`)
		case char == '\b':
			out.WriteString(`\b`)
		case char == '\f':
			out.WriteString(`\f`)
		case char > maxBasicPlane:
			high, low := utf16.EncodeRune(char)
			fmt.Fprintf(out, `\u%04x\u%04x`, high, low)
		case char < ' ' || char >= utf8.RuneSelf:
			fmt.Fprintf(out, `\u%04x`, char)
		default:
			out.WriteRune(char)
		}
	}
	out.WriteByte('"')
}
