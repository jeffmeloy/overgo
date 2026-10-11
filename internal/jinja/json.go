package jinja

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// DecodeJSON reads one JSON value as a template is to see a request's JSON,
// which is as Python's json.loads gives it to a template under transformers:
// an object as a Dict in the order its keys arrive, an array as a list, a
// number as an integer where it is written as one and as a float otherwise,
// and null as None.
func DecodeJSON(data []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	value, err := decodeJSONValue(decoder)
	if err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("jinja: JSON holds more than one value")
	}
	return value, nil
}

func decodeJSONValue(decoder *json.Decoder) (any, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	switch typed := token.(type) {
	case json.Delim:
		if typed == '{' {
			object := NewDict()
			for decoder.More() {
				key, err := decoder.Token()
				if err != nil {
					return nil, err
				}
				name, isName := key.(string)
				if !isName {
					return nil, fmt.Errorf("jinja: JSON object key %v is not a string", key)
				}
				value, err := decodeJSONValue(decoder)
				if err != nil {
					return nil, err
				}
				object.Set(name, value)
			}
			_, err := decoder.Token()
			return object, err
		}
		list := []any{}
		for decoder.More() {
			value, err := decodeJSONValue(decoder)
			if err != nil {
				return nil, err
			}
			list = append(list, value)
		}
		_, err := decoder.Token()
		return list, err
	case json.Number:
		if text := typed.String(); !strings.ContainsAny(text, ".eE") {
			if whole, err := strconv.Atoi(text); err == nil {
				return whole, nil
			}
		}
		return typed.Float64()
	case nil:
		return None, nil
	}
	return token, nil
}

// jsonStyle is how tojson writes: Python's json.dumps arguments.
type jsonStyle struct {
	// ascii writes every character past ASCII as an escape.
	ascii bool
	// sortKeys writes a mapping's keys in sorted order and not its own.
	sortKeys bool
	// indent, when indented, opens a line for each member, set in by indent
	// width times for each level it is nested. An indent given as a count is
	// one space that many times: it is not built until a line is written, so
	// a value with no members costs nothing for it.
	indented bool
	indent   string
	width    int
	// item and key part two members, and a key from its value.
	item, key string
}

// jsonOut is where tojson writes. With text it writes; without, it counts
// the bytes the text would be, in bytes either way, which stops at the most
// an integer holds and never wraps. tojson is counted before it is written,
// because what it writes is not bounded by what it is given: an indent
// repeats for every line and level, and an escape is six bytes for one.
type jsonOut struct {
	text  *strings.Builder
	bytes int
}

// jsonPadBytes bounds the run of an indent that is built to be written: a
// long indent is written as that run over and over, so writing it allocates
// no more than the run beside the text.
const jsonPadBytes = 4096

// repeat writes unit count times.
func (out *jsonOut) repeat(unit string, count int) {
	if grown := scaled(count, len(unit)); grown > math.MaxInt-out.bytes {
		out.bytes = math.MaxInt
	} else {
		out.bytes += grown
	}
	if out.text == nil || unit == "" {
		return
	}
	units := min(count, max(jsonPadBytes/len(unit), 1))
	for run := strings.Repeat(unit, units); count > 0; count -= units {
		out.text.WriteString(run[:min(count, units)*len(unit)])
	}
}

func (out *jsonOut) write(text string) { out.repeat(text, 1) }

func (out *jsonOut) writeByte(char byte) { out.writeRune(rune(char)) }

// writeRune writes one character as itself.
func (out *jsonOut) writeRune(char rune) {
	if out.bytes < math.MaxInt-utf8.UTFMax {
		out.bytes += utf8.RuneLen(char)
	}
	if out.text != nil {
		out.text.WriteRune(char)
	}
}

// The arguments transformers' tojson takes after the value, in its order, and
// the separators Python writes with and without an indent.
const (
	jsonEnsureASCII   = "ensure_ascii"
	jsonIndent        = "indent"
	jsonSeparators    = "separators"
	jsonSortKeys      = "sort_keys"
	jsonItemSeparator = ", "
	jsonLineSeparator = ","
	jsonKeySeparator  = ": "
	// jsonSeparatorCount is how many separators a caller gives: one between
	// members and one between a key and its value.
	jsonSeparatorCount = 2
)

// filterToJSON is tojson as transformers defines it for a chat template,
// which is not Jinja2's own: json.dumps(value, ensure_ascii=False,
// indent=None, separators=None, sort_keys=False). Characters past ASCII are
// written as themselves, nothing is escaped for HTML, and a mapping's keys
// keep their order.
func filterToJSON(in any, args []any, kwargs map[string]any) (any, error) {
	_, write, err := planJSON(in, args, kwargs)
	if err != nil {
		return nil, err
	}
	return write(), nil
}

// planJSON is tojson in two parts: how many bytes it will write for a value,
// counted without writing any of them, and what writes them, into text of
// exactly that size. A render holds the bytes before it has them written. A
// count past what an integer holds is past any limit, and is refused here,
// since a render with no limit would otherwise go on to try to build it; and
// a value that cannot be written is refused by the count, so the writing
// that follows it has nothing left to refuse.
func planJSON(in any, args []any, kwargs map[string]any) (int, func() any, error) {
	style, err := jsonStyleOf(args, kwargs)
	if err != nil {
		return 0, nil, err
	}
	var counted jsonOut
	if err := writeJSON(&counted, in, style, 0); err != nil {
		return 0, nil, err
	}
	if counted.bytes == math.MaxInt {
		return 0, nil, fmt.Errorf("%w: tojson of more bytes than can be counted", errLimit)
	}
	return counted.bytes, func() any {
		var text strings.Builder
		text.Grow(counted.bytes)
		_ = writeJSON(&jsonOut{text: &text}, in, style, 0)
		return text.String()
	}, nil
}

// jsonStyleOf reads tojson's arguments after the value.
func jsonStyleOf(args []any, kwargs map[string]any) (jsonStyle, error) {
	given := map[string]any{}
	for index, name := range []string{jsonEnsureASCII, jsonIndent, jsonSeparators, jsonSortKeys} {
		if index < len(args) {
			given[name] = args[index]
		}
	}
	for name, value := range kwargs {
		if !slices.Contains([]string{jsonEnsureASCII, jsonIndent, jsonSeparators, jsonSortKeys}, name) {
			return jsonStyle{}, fmt.Errorf("jinja: tojson takes no argument %q", name)
		}
		given[name] = value
	}
	style := jsonStyle{ascii: truthy(given[jsonEnsureASCII]), sortKeys: truthy(given[jsonSortKeys]), item: jsonItemSeparator, key: jsonKeySeparator}
	switch indent := given[jsonIndent]; {
	case indent == nil || isNone(indent):
	case isString(indent):
		style.indented, style.indent, style.width, style.item = true, indent.(string), 1, jsonLineSeparator
	case isInteger(indent):
		style.indented, style.indent, style.width, style.item = true, " ", max(toInt(indent), 0), jsonLineSeparator
	default:
		return jsonStyle{}, errors.New("jinja: tojson indent is neither a count nor a string")
	}
	if separators := given[jsonSeparators]; separators != nil && !isNone(separators) {
		pair, isList := separators.([]any)
		if !isList || len(pair) != jsonSeparatorCount || !isString(pair[0]) || !isString(pair[1]) {
			return jsonStyle{}, errors.New("jinja: tojson separators are not a pair of strings")
		}
		style.item, style.key = pair[0].(string), pair[1].(string)
	}
	return style, nil
}

// writeJSON writes one value as json.dumps writes it, and refuses, as it
// does, a value that holds itself: one that nests past maxNesting.
func writeJSON(out *jsonOut, value any, style jsonStyle, depth int) error {
	if depth > maxNesting {
		return fmt.Errorf("%w: tojson of a value that nests deeper than %d, or holds itself", errLimit, maxNesting)
	}
	line := func(level int) {
		if style.indented {
			out.writeByte('\n')
			out.repeat(style.indent, scaled(level, style.width))
		}
	}
	switch typed := value.(type) {
	case nil:
		return errors.New("jinja: tojson of a value that is not defined")
	case noneType:
		out.write("null")
	case bool:
		out.write(strconv.FormatBool(typed))
	case string:
		writeJSONString(out, typed, style.ascii)
	case float32, float64:
		number := toFloat(typed)
		switch {
		case math.IsNaN(number):
			out.write("NaN")
		case math.IsInf(number, 1):
			out.write("Infinity")
		case math.IsInf(number, -1):
			out.write("-Infinity")
		default:
			out.write(formatFloat(number))
		}
	case []any:
		if len(typed) == 0 {
			out.write("[]")
			return nil
		}
		out.writeByte('[')
		for index, item := range typed {
			if index != 0 {
				out.write(style.item)
			}
			line(depth + 1)
			if err := writeJSON(out, item, style, depth+1); err != nil {
				return err
			}
		}
		line(depth)
		out.writeByte(']')
	case []string:
		items := make([]any, len(typed))
		for index, item := range typed {
			items[index] = item
		}
		return writeJSON(out, items, style, depth)
	default:
		mapping, isMapping := asDictLike(value)
		if !isMapping {
			if isInteger(value) {
				out.write(strconv.Itoa(toInt(value)))
				return nil
			}
			return fmt.Errorf("jinja: tojson cannot write a %T", value)
		}
		keys := mapping.keys()
		if _, unordered := value.(map[string]any); unordered || style.sortKeys {
			// A Go map has no order of its own to keep.
			slices.Sort(keys)
		}
		if len(keys) == 0 {
			out.write("{}")
			return nil
		}
		out.writeByte('{')
		for index, key := range keys {
			if index != 0 {
				out.write(style.item)
			}
			line(depth + 1)
			writeJSONString(out, key, style.ascii)
			out.write(style.key)
			member, _ := mapping.lookup(key)
			if err := writeJSON(out, member, style, depth+1); err != nil {
				return err
			}
		}
		line(depth)
		out.writeByte('}')
	}
	return nil
}

// What a JSON string escapes: the quote, the backslash and the control
// characters, the five with a letter of their own by it and the rest by
// number.
const (
	jsonControlBelow = 0x20
	jsonHex          = "0123456789abcdef"
	hexDigitBits     = 4
	hexDigitMask     = 0xf
	hexDigits        = 4
)

// writeJSONString writes a string as json.dumps does: quoted, with the quote,
// the backslash and the control characters escaped, and every character past
// ASCII as itself, or as an escape of its UTF-16 units when ascii is set.
func writeJSONString(out *jsonOut, text string, ascii bool) {
	escape := func(unit rune) {
		out.write(`\u`)
		for digit := hexDigits - 1; digit >= 0; digit-- {
			out.writeByte(jsonHex[(unit>>(digit*hexDigitBits))&hexDigitMask])
		}
	}
	out.writeByte('"')
	for _, char := range text {
		switch char {
		case '"':
			out.write(`\"`)
		case '\\':
			out.write(`\\`)
		case '\n':
			out.write(`\n`)
		case '\r':
			out.write(`\r`)
		case '\t':
			out.write(`\t`)
		case '\b':
			out.write(`\b`)
		case '\f':
			out.write(`\f`)
		default:
			switch {
			case char < jsonControlBelow:
				escape(char)
			case ascii && char >= utf8.RuneSelf:
				if high, low := utf16.EncodeRune(char); high != utf8.RuneError || low != utf8.RuneError {
					escape(high)
					escape(low)
				} else {
					escape(char)
				}
			default:
				out.writeRune(char)
			}
		}
	}
	out.writeByte('"')
}
