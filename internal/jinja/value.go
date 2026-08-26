// Package jinja is a bounded, stdlib-only interpreter for the Jinja2 subset
// used by served-model chat templates. It is engineered to render those
// templates byte-identically to github.com/nikolalohinski/gonja/v2, which it
// replaces. Semantics (value stringification, truthiness, comparison, map key
// ordering, whitespace control, filter/test/method behavior) intentionally
// mirror gonja's observable output; see internal/jinja for the parity gate.
package jinja

import (
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

// odict is an insertion-ordered dictionary, used for dict literals and the
// dict() global. Go maps (map[string]any) model JSON objects and iterate in
// case-insensitively sorted key order, matching gonja.
type odict struct {
	order []string
	vals  map[string]any
}

func newODict() *odict { return &odict{vals: map[string]any{}} }

func (d *odict) set(k string, v any) {
	if _, ok := d.vals[k]; !ok {
		d.order = append(d.order, k)
	}
	d.vals[k] = v
}

func (d *odict) get(k string) (any, bool) {
	v, ok := d.vals[k]
	return v, ok
}

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
	case map[string]any, *odict:
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
	case nil:
		return false
	case bool:
		return n
	case string:
		return len(n) > 0
	case []any:
		return len(n) > 0
	case map[string]any:
		return len(n) > 0
	case *odict:
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
	case *odict:
		return len(n.order)
	}
	return 0
}

// stringify mirrors gonja Value.String().
func stringify(v any) string {
	switch n := v.(type) {
	case nil:
		return ""
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
				b.WriteString(stringify(item))
				b.WriteByte('\'')
			} else {
				b.WriteString(stringify(item))
			}
		}
		b.WriteByte(']')
		return b.String()
	case map[string]any:
		pairs := make([]string, 0, len(n))
		for k, val := range n {
			pairs = append(pairs, mapPairRepr(k, val))
		}
		sort.Strings(pairs)
		return "{" + strings.Join(pairs, ", ") + "}"
	case *odict:
		pairs := make([]string, 0, len(n.order))
		for _, k := range n.order {
			pairs = append(pairs, mapPairRepr(k, n.vals[k]))
		}
		return "{" + strings.Join(pairs, ", ") + "}"
	}
	if isInteger(v) {
		return strconv.Itoa(toInt(v))
	}
	return ""
}

func mapPairRepr(key string, val any) string {
	k := "'" + key + "'"
	var vs string
	if isString(val) {
		vs = "'" + stringify(val) + "'"
	} else {
		vs = stringify(val)
	}
	return k + ": " + vs
}

// formatFloat mirrors gonja's formatFloatString (compact 'g', with a trailing
// .0 for integral values). Chat templates rarely emit floats.
func formatFloat(f float64) string {
	s := strconv.FormatFloat(f, 'g', -1, binaryschema.Width64Bits)
	if s == "NaN" || strings.HasSuffix(s, "Inf") {
		return s
	}
	if strings.ContainsAny(s, "eE") {
		return s
	}
	if strings.ContainsAny(s, ".") {
		return s
	}
	return s + ".0"
}

// equal mirrors gonja Value.EqualValueTo().
func equal(a, b any) bool {
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
	if la, ok := a.([]any); ok {
		if lb, ok := b.([]any); ok {
			if len(la) != len(lb) {
				return false
			}
			for i := range la {
				if !equal(la[i], lb[i]) {
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
				if !oka || !okb || !equal(va, vb) {
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

func (d *odict) keys() []string              { return append([]string(nil), d.order...) }
func (d *odict) lookup(k string) (any, bool) { return d.get(k) }

func asDictLike(v any) (dictLike, bool) {
	switch n := v.(type) {
	case map[string]any:
		return goMapDict(n), true
	case *odict:
		return n, true
	}
	return nil, false
}
