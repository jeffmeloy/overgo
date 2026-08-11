package jinja

import (
	"fmt"
	"sort"
	"strings"
)

func registerBuiltinFilters(e *Env) {
	e.filters["default"] = filterDefault
	e.filters["d"] = filterDefault
	e.filters["length"] = filterLength
	e.filters["count"] = filterLength
	e.filters["list"] = filterList
	e.filters["trim"] = filterTrim
	e.filters["upper"] = filterUpper
	e.filters["lower"] = filterLower
	e.filters["safe"] = filterSafe
	e.filters["string"] = filterString
	e.filters["dictsort"] = filterDictsort
	e.filters["items"] = filterItems
	e.filters["min"] = filterMin
	e.filters["max"] = filterMax
	e.filters["map"] = e.filterMap
}

func filterDefault(in any, args []any, kwargs map[string]any) (any, error) {
	var def any
	if len(args) >= 1 {
		def = args[0]
	}
	boolean := false
	if len(args) >= 2 {
		boolean = truthy(args[1])
	} else if b, ok := kwargs["boolean"]; ok {
		boolean = truthy(b)
	}
	if in == nil {
		return def, nil
	}
	if boolean && !truthy(in) {
		return def, nil
	}
	return in, nil
}

func filterLength(in any, _ []any, _ map[string]any) (any, error) {
	return length(in), nil
}

func filterList(in any, _ []any, _ map[string]any) (any, error) {
	if s, ok := in.(string); ok {
		out := make([]any, 0, len(s))
		for _, r := range s {
			out = append(out, string(r))
		}
		return out, nil
	}
	entries := iterate(in)
	out := make([]any, len(entries))
	for i, e := range entries {
		out[i] = e.key
	}
	return out, nil
}

func filterTrim(in any, args []any, kwargs map[string]any) (any, error) {
	s, ok := in.(string)
	if !ok {
		return nil, fmt.Errorf("jinja: trim: %v is not a string", in)
	}
	var chars any
	if len(args) >= 1 {
		chars = args[0]
	} else if c, ok := kwargs["chars"]; ok {
		chars = c
	}
	if chars == nil {
		return strings.TrimSpace(s), nil
	}
	return strings.Trim(s, stringify(chars)), nil
}

func filterUpper(in any, _ []any, _ map[string]any) (any, error) {
	return strings.ToUpper(stringify(in)), nil
}

func filterLower(in any, _ []any, _ map[string]any) (any, error) {
	return strings.ToLower(stringify(in)), nil
}

func filterSafe(in any, _ []any, _ map[string]any) (any, error) {
	return in, nil
}

func filterString(in any, _ []any, _ map[string]any) (any, error) {
	return stringify(in), nil
}

func filterDictsort(in any, _ []any, _ map[string]any) (any, error) {
	dl, ok := asDictLike(in)
	if !ok {
		return nil, fmt.Errorf("jinja: dictsort requires a mapping")
	}
	type kv struct {
		k string
		v any
	}
	var pairs []kv
	for _, k := range dictAllKeys(in) {
		v, _ := dl.lookup(k)
		pairs = append(pairs, kv{k, v})
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		return strings.ToLower(pairs[i].k) < strings.ToLower(pairs[j].k)
	})
	out := make([]any, len(pairs))
	for i, p := range pairs {
		out[i] = []any{p.k, p.v}
	}
	return out, nil
}

// dictAllKeys returns keys in a stable, unsorted-source order (Go-map keys are
// deduplicated); dictsort re-sorts them anyway.
func dictAllKeys(in any) []string {
	switch d := in.(type) {
	case map[string]any:
		keys := make([]string, 0, len(d))
		for k := range d {
			keys = append(keys, k)
		}
		return keys
	case *odict:
		return append([]string(nil), d.order...)
	}
	return nil
}

func filterItems(in any, _ []any, _ map[string]any) (any, error) {
	if in == nil {
		return []any{}, nil
	}
	if isList(in) {
		return in, nil
	}
	dl, ok := asDictLike(in)
	if !ok {
		return nil, fmt.Errorf("jinja: items requires a mapping")
	}
	keys := dictAllKeys(in)
	sort.Strings(keys)
	out := make([]any, 0, len(keys))
	for _, k := range keys {
		v, _ := dl.lookup(k)
		out = append(out, []any{k, v})
	}
	return out, nil
}

func filterMin(in any, _ []any, _ map[string]any) (any, error) {
	return minMax(in, true), nil
}

func filterMax(in any, _ []any, _ map[string]any) (any, error) {
	return minMax(in, false), nil
}

func minMax(in any, wantMin bool) any {
	entries := iterate(in)
	if len(entries) == 0 {
		return ""
	}
	best := entries[0].key
	for _, e := range entries[1:] {
		v := e.key
		var less bool
		if isNumber(best) && isNumber(v) {
			less = toFloat(v) < toFloat(best)
		} else {
			less = strings.ToLower(stringify(v)) < strings.ToLower(stringify(best))
		}
		if wantMin == less {
			best = v
		}
	}
	return best
}

func (e *Env) filterMap(in any, args []any, kwargs map[string]any) (any, error) {
	if in == nil {
		return []any{}, nil
	}
	entries := iterate(in)
	if len(args) > 0 {
		name, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("jinja: map filter name must be a string")
		}
		f, ok := e.filters[name]
		if !ok {
			return nil, fmt.Errorf("jinja: unknown filter %q in map", name)
		}
		fargs := args[1:]
		out := make([]any, 0, len(entries))
		for _, en := range entries {
			v, err := f(en.key, fargs, kwargs)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
		}
		return out, nil
	}
	// attribute mode
	attr, _ := kwargs["attribute"]
	out := make([]any, 0, len(entries))
	for _, en := range entries {
		if attr != nil {
			v, _ := getAttr(en.key, stringify(attr))
			out = append(out, v)
		} else {
			out = append(out, en.key)
		}
	}
	return out, nil
}
