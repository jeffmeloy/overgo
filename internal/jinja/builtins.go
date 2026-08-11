package jinja

import (
	"fmt"
	"sort"
	"strings"
)

// ---- global functions ----

func rangeGlobal(args []any, _ map[string]any) (any, error) {
	start, stop, step := 0, 0, 1
	switch len(args) {
	case 1:
		stop = toInt(args[0])
	case 2:
		start, stop = toInt(args[0]), toInt(args[1])
	case 3:
		start, stop, step = toInt(args[0]), toInt(args[1]), toInt(args[2])
	default:
		return nil, fmt.Errorf("jinja: range expects 1-3 integer arguments")
	}
	if step == 0 {
		return nil, fmt.Errorf("jinja: range step cannot be 0")
	}
	var out []any
	if step > 0 {
		for i := start; i < stop; i += step {
			out = append(out, i)
		}
	} else {
		for i := start; i > stop; i += step {
			out = append(out, i)
		}
	}
	if out == nil {
		out = []any{}
	}
	return out, nil
}

func namespaceGlobal(_ []any, kwargs map[string]any) (any, error) {
	ns := map[string]any{}
	for k, v := range kwargs {
		ns[k] = v
	}
	return ns, nil
}

func dictGlobal(_ []any, kwargs map[string]any) (any, error) {
	d := newODict()
	keys := make([]string, 0, len(kwargs))
	for k := range kwargs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		d.set(k, kwargs[k])
	}
	return d, nil
}

// ---- method dispatch ----

func (r *renderer) dispatchMethod(obj any, name string, n callNode, sc *scope) (any, error) {
	args, kwargs, err := r.evalArgs(n, sc)
	if err != nil {
		return nil, err
	}
	switch o := obj.(type) {
	case string:
		return strMethod(o, name, args)
	case map[string]any:
		return dictMethod(o, name, args, kwargs)
	case *odict:
		return odictMethod(o, name, args)
	case Getter:
		// gonja dispatches methods by native type; adapters expose no methods.
		return nil, fmt.Errorf("jinja: %q is not callable on adapter", name)
	}
	return nil, fmt.Errorf("jinja: %q is not callable on %T", name, obj)
}

func strMethod(self, name string, args []any) (any, error) {
	switch name {
	case "split":
		if len(args) < 1 {
			return nil, fmt.Errorf("jinja: split requires a separator")
		}
		sep, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("jinja: split separator must be a string")
		}
		maxsplit := -1
		if len(args) >= 2 {
			maxsplit = toInt(args[1])
		}
		return pySplit(self, sep, maxsplit), nil
	case "strip":
		cut := ""
		if len(args) >= 1 {
			s, ok := args[0].(string)
			if !ok {
				return nil, fmt.Errorf("jinja: strip cutset must be a string")
			}
			cut = s
		}
		return pyStrip(self, cut, true, true), nil
	case "lstrip":
		if len(args) < 1 {
			return nil, fmt.Errorf("jinja: failed to validate argument 'cutset': not a string")
		}
		s, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("jinja: lstrip cutset must be a string")
		}
		return pyStrip(self, s, true, false), nil
	case "rstrip":
		if len(args) < 1 {
			return nil, fmt.Errorf("jinja: failed to validate argument 'cutset': not a string")
		}
		s, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("jinja: rstrip cutset must be a string")
		}
		return pyStrip(self, s, false, true), nil
	case "replace":
		if len(args) < 3 {
			return nil, fmt.Errorf("jinja: replace missing required 3rd positional argument 'count'")
		}
		old, _ := args[0].(string)
		nw, _ := args[1].(string)
		count := toInt(args[2])
		return strings.Replace(self, old, nw, count), nil
	case "startswith":
		if len(args) < 1 {
			return nil, fmt.Errorf("jinja: startswith requires a prefix")
		}
		if prefixes, ok := args[0].([]any); ok {
			for _, p := range prefixes {
				if ps, ok := p.(string); ok && strings.HasPrefix(self, ps) {
					return true, nil
				}
			}
			return false, nil
		}
		p, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("jinja: startswith prefix must be a string")
		}
		return strings.HasPrefix(self, p), nil
	case "endswith":
		if len(args) < 1 {
			return nil, fmt.Errorf("jinja: endswith requires a suffix")
		}
		p, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("jinja: endswith suffix must be a string")
		}
		return strings.HasSuffix(self, p), nil
	case "upper":
		return strings.ToUpper(self), nil
	case "lower":
		return strings.ToLower(self), nil
	}
	return nil, fmt.Errorf("jinja: unknown string method %q", name)
}

// pySplit implements Python str.split with an explicit separator.
func pySplit(s, sep string, maxsplit int) []any {
	if sep == "" {
		return []any{s}
	}
	var parts []string
	if maxsplit < 0 {
		parts = strings.Split(s, sep)
	} else {
		parts = strings.SplitN(s, sep, maxsplit+1)
	}
	out := make([]any, len(parts))
	for i, p := range parts {
		out[i] = p
	}
	return out
}

// pyStrip implements Python str.strip/lstrip/rstrip. An empty cutset strips
// Python whitespace.
func pyStrip(s, cut string, left, right bool) string {
	isCut := func(r rune) bool {
		if cut == "" {
			return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\v' || r == '\f'
		}
		return strings.ContainsRune(cut, r)
	}
	runes := []rune(s)
	start := 0
	end := len(runes)
	if left {
		for start < end && isCut(runes[start]) {
			start++
		}
	}
	if right {
		for end > start && isCut(runes[end-1]) {
			end--
		}
	}
	return string(runes[start:end])
}

func dictMethod(self map[string]any, name string, args []any, _ map[string]any) (any, error) {
	switch name {
	case "get":
		if len(args) < 1 || len(args) > 2 {
			return nil, fmt.Errorf("jinja: get() takes 1 or 2 positional arguments")
		}
		key := stringify(args[0])
		if v, ok := self[key]; ok {
			return v, nil
		}
		if len(args) == 2 {
			return args[1], nil
		}
		return nil, nil
	case "keys":
		keys := make([]string, 0, len(self))
		for k := range self {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make([]any, len(keys))
		for i, k := range keys {
			out[i] = k
		}
		return out, nil
	case "values":
		keys := make([]string, 0, len(self))
		for k := range self {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make([]any, len(keys))
		for i, k := range keys {
			out[i] = self[k]
		}
		return out, nil
	case "items":
		keys := make([]string, 0, len(self))
		for k := range self {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make([]any, len(keys))
		for i, k := range keys {
			out[i] = []any{k, self[k]}
		}
		return out, nil
	}
	return nil, fmt.Errorf("jinja: unknown dict method %q", name)
}

func odictMethod(self *odict, name string, args []any) (any, error) {
	switch name {
	case "get":
		if len(args) < 1 || len(args) > 2 {
			return nil, fmt.Errorf("jinja: get() takes 1 or 2 positional arguments")
		}
		key := stringify(args[0])
		if v, ok := self.get(key); ok {
			return v, nil
		}
		if len(args) == 2 {
			return args[1], nil
		}
		return nil, nil
	case "keys":
		out := make([]any, len(self.order))
		for i, k := range self.order {
			out[i] = k
		}
		return out, nil
	case "items":
		out := make([]any, len(self.order))
		for i, k := range self.order {
			out[i] = []any{k, self.vals[k]}
		}
		return out, nil
	case "values":
		out := make([]any, len(self.order))
		for i, k := range self.order {
			out[i] = self.vals[k]
		}
		return out, nil
	}
	return nil, fmt.Errorf("jinja: unknown dict method %q", name)
}
