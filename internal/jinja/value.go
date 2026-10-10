// Package jinja is a bounded, stdlib-only interpreter for the part of Jinja2
// that served models' chat templates use. What it is held to is Python's
// Jinja2 in the environment transformers compiles a chat template in, since
// that is the language the templates are written for: its tests compare it,
// construct by construct and template by template, with what Python renders.
//
// It was first written to reproduce github.com/nikolalohinski/gonja/v2, a Go
// template library it replaced, and a comment that says a behaviour mirrors
// gonja records where that behaviour came from. Where gonja and Python part,
// in a string method's optional arguments, the order a mapping is walked in,
// what None prints as, how tojson writes and how block tags eat their lines,
// Python is followed.
package jinja

import (
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
)

// Getter is implemented by caller-supplied objects (chat tool adapters) to
// expose attribute/item access to templates. It mirrors gonja's
// AttributeGetter/ItemGetter contract, which resolve identically.
type Getter interface {
	JinjaGet(key string) (any, bool)
}

// Dict is a mapping that keeps its keys in the order they were first set, as
// a Python dict does. A template's dict literals are Dicts, and so are the
// objects of a request's JSON, which a template must read in the order the
// request holds them: a model's template writes a tool's schema as it finds
// it. A Go map handed to a template has no order of its own and is read in
// case-insensitively sorted key order.
type Dict struct {
	order []string
	vals  map[string]any
}

// NewDict returns an empty ordered mapping.
func NewDict() *Dict { return &Dict{vals: map[string]any{}} }

// Set binds key to value, at the end of the order when the key is new and in
// its first place when it is not, as assigning to a Python dict does.
func (d *Dict) Set(key string, value any) {
	if _, ok := d.vals[key]; !ok {
		d.order = append(d.order, key)
	}
	d.vals[key] = value
}

func (d *Dict) get(k string) (any, bool) {
	v, ok := d.vals[k]
	return v, ok
}

// noneType is the type of None.
type noneType struct{}

// None is Python's None as a template meets it: the literal none, a JSON
// null, a dict's get of a key it does not hold, and a variable a caller sets
// to nothing, as transformers sets tools when a request offers none. It is
// defined, it is none, it is false, and it prints as "None". Go's nil is
// something else to a template: a name or attribute that is not defined,
// which is not none and prints as nothing.
var None any = noneType{}

// isNone reports whether a value is None.
func isNone(v any) bool { _, ok := v.(noneType); return ok }

// callable is a template-invocable function (globals like range/namespace,
// registered runtime functions, and macros).
type callable struct {
	name string
	call func(args []any, kwargs map[string]any) (any, error)
}

// mapKeysSorted returns a Go map's keys sorted case-insensitively, matching
// gonja's Value.Keys()/iteration order for maps.
func mapKeysSorted(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.SliceStable(keys, func(i, j int) bool {
		return strings.ToLower(keys[i]) < strings.ToLower(keys[j])
	})
	return keys
}

// ---- type predicates (mirror gonja Value.IsX) ----

func isString(v any) bool { _, ok := v.(string); return ok }
func isBool(v any) bool   { _, ok := v.(bool); return ok }

func isInteger(v any) bool {
	switch v.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	}
	return false
}

func isFloat(v any) bool {
	switch v.(type) {
	case float32, float64:
		return true
	}
	return false
}

func isNumber(v any) bool { return isInteger(v) || isFloat(v) }

func isList(v any) bool { _, ok := v.([]any); return ok }

func isDict(v any) bool {
	switch v.(type) {
	case map[string]any, *Dict:
		return true
	}
	return false
}

// isIterable mirrors gonja: string, list, or dict.
func isIterable(v any) bool { return isString(v) || isList(v) || isDict(v) }

// ---- conversions ----

func toInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int8:
		return int(n)
	case int16:
		return int(n)
	case int32:
		return int(n)
	case int64:
		return int(n)
	case uint:
		return int(n)
	case uint8:
		return int(n)
	case uint16:
		return int(n)
	case uint32:
		return int(n)
	case uint64:
		return int(n)
	case float32:
		return int(n)
	case float64:
		return int(n)
	case string:
		f, err := strconv.ParseFloat(n, binaryschema.Width64Bits)
		if err != nil {
			return 0
		}
		return int(f)
	case bool:
		return 0
	}
	return 0
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case int:
		return float64(n)
	case int8:
		return float64(n)
	case int16:
		return float64(n)
	case int32:
		return float64(n)
	case int64:
		return float64(n)
	case uint:
		return float64(n)
	case uint32:
		return float64(n)
	case uint64:
		return float64(n)
	case float32:
		return float64(n)
	case float64:
		return n
	case string:
		f, err := strconv.ParseFloat(n, binaryschema.Width64Bits)
		if err != nil {
			return 0
		}
		return f
	}
	return 0
}

// asBool returns the underlying bool only when v is actually a bool, matching
// gonja Value.Bool() (used by the `true`/`false` tests).
func asBool(v any) bool {
	if b, ok := v.(bool); ok {
		return b
	}
	return false
}

// truthy mirrors gonja Value.IsTrue().
func truthy(v any) bool {
	switch n := v.(type) {
	case nil, noneType:
		return false
	case bool:
		return n
	case string:
		return len(n) > 0
	case []any:
		return len(n) > 0
	case map[string]any:
		return len(n) > 0
	case *Dict:
		return len(n.order) > 0
	}
	if isInteger(v) {
		return toInt(v) != 0
	}
	if isFloat(v) {
		return toFloat(v) != 0
	}
	// Structs (adapters) are always truthy in gonja.
	return true
}

// length mirrors gonja Value.Len().
func length(v any) int {
	switch n := v.(type) {
	case string:
		return len([]rune(n))
	case []any:
		return len(n)
	case map[string]any:
		return len(n)
	case *Dict:
		return len(n.order)
	}
	return 0
}

// stringify mirrors gonja Value.String().
func stringify(v any) string { return stringifyAt(v, 0) }

// stringifyAt is stringify at a depth inside the value it began with. A value
// that nests past maxNesting, as one that holds itself does without end, is
// written there as Python writes a container that holds itself.
func stringifyAt(v any, depth int) string {
	if depth > maxNesting {
		return "..."
	}
	switch n := v.(type) {
	case nil:
		return ""
	case noneType:
		return "None"
	case string:
		return n
	case bool:
		if n {
			return "True"
		}
		return "False"
	case float32:
		return formatFloat(float64(n))
	case float64:
		return formatFloat(n)
	case []any:
		var b strings.Builder
		b.WriteByte('[')
		for i, item := range n {
			if i > 0 {
				b.WriteString(", ")
			}
			if isString(item) {
				b.WriteByte('\'')
				b.WriteString(stringifyAt(item, depth+1))
				b.WriteByte('\'')
			} else {
				b.WriteString(stringifyAt(item, depth+1))
			}
		}
		b.WriteByte(']')
		return b.String()
	case map[string]any:
		pairs := make([]string, 0, len(n))
		for k, val := range n {
			pairs = append(pairs, mapPairRepr(k, val, depth+1))
		}
		slices.Sort(pairs)
		return "{" + strings.Join(pairs, ", ") + "}"
	case *Dict:
		pairs := make([]string, 0, len(n.order))
		for _, k := range n.order {
			pairs = append(pairs, mapPairRepr(k, n.vals[k], depth+1))
		}
		return "{" + strings.Join(pairs, ", ") + "}"
	}
	if isInteger(v) {
		return strconv.Itoa(toInt(v))
	}
	return ""
}

func mapPairRepr(key string, val any, depth int) string {
	k := "'" + key + "'"
	var vs string
	if isString(val) {
		vs = "'" + stringifyAt(val, depth) + "'"
	} else {
		vs = stringifyAt(val, depth)
	}
	return k + ": " + vs
}

// formatFloat writes a float as Python's repr does, which is how a template
// prints one and how tojson writes one: the shortest digits that read back as
// the same value, in fixed notation with a fraction always shown, and in
// exponent notation when the decimal point would stand more than sixteen
// digits right of the first digit or more than three zeros left of it.
func formatFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	scientific := strconv.FormatFloat(f, 'e', -1, binaryschema.Width64Bits)
	exponent, err := strconv.Atoi(scientific[strings.IndexByte(scientific, 'e')+1:])
	if err != nil || exponent < pythonFixedFrom || exponent >= pythonFixedBelow {
		return scientific
	}
	fixed := strconv.FormatFloat(f, 'f', -1, binaryschema.Width64Bits)
	if strings.Contains(fixed, ".") {
		return fixed
	}
	return fixed + ".0"
}

// Python's repr writes a float in fixed notation when its decimal exponent is
// from -4 up to, and not including, 16.
const (
	pythonFixedFrom  = -4
	pythonFixedBelow = 16
)

// equal mirrors gonja Value.EqualValueTo().
func equal(a, b any) bool { return equalAt(a, b, 0) }

// equalAt is equal at a depth inside the values it began with: two values
// that nest past maxNesting are followed no further and are not equal.
func equalAt(a, b any, depth int) bool {
	if depth > maxNesting {
		return false
	}
	if isNumber(a) && isNumber(b) {
		if isInteger(a) && isInteger(b) {
			return toInt(a) == toInt(b)
		}
		return toFloat(a) == toFloat(b)
	}
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if isNone(a) || isNone(b) {
		return isNone(a) && isNone(b)
	}
	if la, ok := a.([]any); ok {
		if lb, ok := b.([]any); ok {
			if len(la) != len(lb) {
				return false
			}
			for i := range la {
				if !equalAt(la[i], lb[i], depth+1) {
					return false
				}
			}
			return true
		}
		return false
	}
	if da, ok := asDictLike(a); ok {
		if db, ok := asDictLike(b); ok {
			ka := da.keys()
			kb := db.keys()
			if len(ka) != len(kb) {
				return false
			}
			for _, k := range ka {
				va, oka := da.lookup(k)
				vb, okb := db.lookup(k)
				if !oka || !okb || !equalAt(va, vb, depth+1) {
					return false
				}
			}
			return true
		}
		return false
	}
	if isString(a) && isString(b) {
		return a.(string) == b.(string)
	}
	if isBool(a) && isBool(b) {
		return a.(bool) == b.(bool)
	}
	return false
}

type dictLike interface {
	keys() []string
	lookup(k string) (any, bool)
}

type goMapDict map[string]any

func (m goMapDict) keys() []string { return mapKeysSorted(m) }
func (m goMapDict) lookup(k string) (any, bool) {
	v, ok := m[k]
	return v, ok
}

func (d *Dict) keys() []string              { return append([]string(nil), d.order...) }
func (d *Dict) lookup(k string) (any, bool) { return d.get(k) }

func asDictLike(v any) (dictLike, bool) {
	switch n := v.(type) {
	case map[string]any:
		return goMapDict(n), true
	case *Dict:
		return n, true
	}
	return nil, false
}
