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
	// once for each level it is nested.
	indented bool
	indent   string
	// item and key part two members, and a key from its value.
	item, key string
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
	given := map[string]any{}
	for index, name := range []string{jsonEnsureASCII, jsonIndent, jsonSeparators, jsonSortKeys} {
		if index < len(args) {
			given[name] = args[index]
		}
	}
	for name, value := range kwargs {
		if !slices.Contains([]string{jsonEnsureASCII, jsonIndent, jsonSeparators, jsonSortKeys}, name) {
			return nil, fmt.Errorf("jinja: tojson takes no argument %q", name)
		}
		given[name] = value
	}
	style := jsonStyle{ascii: truthy(given[jsonEnsureASCII]), sortKeys: truthy(given[jsonSortKeys]), item: jsonItemSeparator, key: jsonKeySeparator}
	switch indent := given[jsonIndent]; {
	case indent == nil || isNone(indent):
	case isString(indent):
		style.indented, style.indent, style.item = true, indent.(string), jsonLineSeparator
	case isInteger(indent):
		style.indented, style.indent, style.item = true, strings.Repeat(" ", max(toInt(indent), 0)), jsonLineSeparator
	default:
		return nil, errors.New("jinja: tojson indent is neither a count nor a string")
	}
	if separators := given[jsonSeparators]; separators != nil && !isNone(separators) {
		pair, isList := separators.([]any)
		if !isList || len(pair) != jsonSeparatorCount || !isString(pair[0]) || !isString(pair[1]) {
			return nil, errors.New("jinja: tojson separators are not a pair of strings")
		}
		style.item, style.key = pair[0].(string), pair[1].(string)
	}
	var out strings.Builder
	if err := writeJSON(&out, in, style, 0); err != nil {
		return nil, err
	}
	return out.String(), nil
}

// writeJSON writes one value as json.dumps writes it.
func writeJSON(out *strings.Builder, value any, style jsonStyle, depth int) error {
	line := func(level int) {
		if style.indented {
			out.WriteByte('\n')
			out.WriteString(strings.Repeat(style.indent, level))
		}
	}
	switch typed := value.(type) {
	case nil:
		return errors.New("jinja: tojson of a value that is not defined")
	case noneType:
		out.WriteString("null")
	case bool:
		out.WriteString(strconv.FormatBool(typed))
	case string:
		writeJSONString(out, typed, style.ascii)
	case float32, float64:
		number := toFloat(typed)
		switch {
		case math.IsNaN(number):
			out.WriteString("NaN")
		case math.IsInf(number, 1):
			out.WriteString("Infinity")
		case math.IsInf(number, -1):
			out.WriteString("-Infinity")
		default:
			out.WriteString(formatFloat(number))
		}
	case []any:
		if len(typed) == 0 {
			out.WriteString("[]")
			return nil
		}
		out.WriteByte('[')
		for index, item := range typed {
			if index != 0 {
				out.WriteString(style.item)
			}
			line(depth + 1)
			if err := writeJSON(out, item, style, depth+1); err != nil {
				return err
			}
		}
		line(depth)
		out.WriteByte(']')
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
				out.WriteString(strconv.Itoa(toInt(value)))
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
			out.WriteString("{}")
			return nil
		}
		out.WriteByte('{')
		for index, key := range keys {
			if index != 0 {
				out.WriteString(style.item)
			}
			line(depth + 1)
			writeJSONString(out, key, style.ascii)
			out.WriteString(style.key)
			member, _ := mapping.lookup(key)
			if err := writeJSON(out, member, style, depth+1); err != nil {
				return err
			}
		}
		line(depth)
		out.WriteByte('}')
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
func writeJSONString(out *strings.Builder, text string, ascii bool) {
	escape := func(unit rune) {
		out.WriteString(`\u`)
		for digit := hexDigits - 1; digit >= 0; digit-- {
			out.WriteByte(jsonHex[(unit>>(digit*hexDigitBits))&hexDigitMask])
		}
	}
	out.WriteByte('"')
	for _, char := range text {
		switch char {
		case '"':
			out.WriteString(`\"`)
		case '\\':
			out.WriteString(`\\`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
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
				out.WriteRune(char)
			}
		}
	}
	out.WriteByte('"')
}
