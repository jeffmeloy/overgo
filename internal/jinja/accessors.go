package jinja

// getAttr resolves obj.name for attribute access (mirrors gonja
// GetAttribute with GetItem fallback for maps/adapters).
func getAttr(obj any, name string) (any, bool) {
	switch o := obj.(type) {
	case Getter:
		return o.JinjaGet(name)
	case map[string]any:
		v, ok := o[name]
		return v, ok
	case *odict:
		return o.get(name)
	}
	return nil, false
}

// getItem resolves obj[intIndex] for integer indexing.
func getItem(obj any, i int) (any, bool) {
	return indexInto(obj, i)
}

func indexInto(obj any, i int) (any, bool) {
	switch o := obj.(type) {
	case []any:
		n := len(o)
		if i < 0 {
			if -i > n {
				return nil, false
			}
			return o[n+i], true
		}
		if i >= n {
			return nil, false
		}
		return o[i], true
	case string:
		runes := []rune(o)
		n := len(runes)
		if i < 0 {
			if -i > n {
				return "", true
			}
			return string(runes[n+i]), true
		}
		if i < n {
			return string(runes[i]), true
		}
		return "", true
	}
	return nil, false
}

// getIndexOrKey resolves obj[key] where key is an int index or string key.
// Mirrors gonja evalGetItem: string keys try item then attribute.
func getIndexOrKey(obj any, key any) (any, bool) {
	if ki, ok := key.(int); ok {
		return indexInto(obj, ki)
	}
	ks, ok := key.(string)
	if !ok {
		return nil, false
	}
	switch o := obj.(type) {
	case map[string]any:
		if v, ok := o[ks]; ok {
			return v, true
		}
	case *odict:
		if v, ok := o.get(ks); ok {
			return v, true
		}
	case Getter:
		if v, ok := o.JinjaGet(ks); ok {
			return v, true
		}
	}
	return getAttr(obj, ks)
}

// contains mirrors gonja Value.Contains (the `in` operator / test).
func contains(container any, item any) bool {
	switch c := container.(type) {
	case string:
		return stringContains(c, stringify(item))
	case []any:
		for _, e := range c {
			if equal(e, item) {
				return true
			}
		}
		return false
	case map[string]any:
		if ks, ok := item.(string); ok {
			_, ok := c[ks]
			return ok
		}
		return false
	case *odict:
		if ks, ok := item.(string); ok {
			_, ok := c.get(ks)
			return ok
		}
		return false
	}
	// Structs/adapters: gonja uses reflect FieldByName on unexported fields,
	// which never matches — so `x in adapter` is always false.
	return false
}

func stringContains(s, sub string) bool {
	if sub == "" {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// iterate materializes an iterable into loop entries, matching gonja's
// iteration order (case-insensitively sorted keys for Go maps).
func iterate(v any) []loopEntry {
	switch o := v.(type) {
	case []any:
		out := make([]loopEntry, len(o))
		for i, e := range o {
			out[i] = loopEntry{key: e}
		}
		return out
	case map[string]any:
		keys := mapKeysSorted(o)
		out := make([]loopEntry, len(keys))
		for i, k := range keys {
			out[i] = loopEntry{key: k, val: o[k]}
		}
		return out
	case *odict:
		out := make([]loopEntry, len(o.order))
		for i, k := range o.order {
			out[i] = loopEntry{key: k, val: o.vals[k]}
		}
		return out
	case string:
		runes := []rune(o)
		out := make([]loopEntry, len(runes))
		for i, rn := range runes {
			out[i] = loopEntry{key: string(rn)}
		}
		return out
	}
	return nil
}

// loopInfo exposes Jinja's `loop` variable inside {% for %} bodies.
type loopInfo struct {
	index, index0, length, revindex int
	first, last                     bool
	previtem, nextitem              any
}

func (l *loopInfo) JinjaGet(name string) (any, bool) {
	switch name {
	case "index":
		return l.index, true
	case "index0":
		return l.index0, true
	case "length":
		return l.length, true
	case "revindex":
		return l.revindex, true
	case "revindex0":
		return l.length - l.index, true
	case "first":
		return l.first, true
	case "last":
		return l.last, true
	case "previtem":
		return l.previtem, true
	case "nextitem":
		return l.nextitem, true
	}
	return nil, false
}
