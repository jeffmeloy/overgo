package jinja

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
)

// Filter is a template filter: it transforms an input value using positional
// and keyword arguments.
type Filter func(in any, args []any, kwargs map[string]any) (any, error)

// GlobalFunc is a template-callable global function.
type GlobalFunc func(args []any, kwargs map[string]any) (any, error)

// Env holds the interpreter's globals and filters. Construct with New, then
// register runtime extensions (tojson, strftime_now, raise_exception) before
// rendering.
type Env struct {
	globals map[string]any
	filters map[string]Filter
	limits  Limits
}

// SetLimits declares what one render of this environment may take.
func (e *Env) SetLimits(limits Limits) { e.limits = limits }

// New returns an environment preloaded with the built-in globals (range,
// namespace, dict) and filters.
func New() *Env {
	e := &Env{
		globals: map[string]any{},
		filters: map[string]Filter{},
	}
	e.globals["namespace"] = callable{name: "namespace", call: namespaceGlobal}
	e.globals["dict"] = callable{name: "dict", call: dictGlobal}
	registerBuiltinFilters(e)
	return e
}

// SetGlobal registers a global function callable from templates.
func (e *Env) SetGlobal(name string, fn GlobalFunc) {
	e.globals[name] = callable{name: name, call: fn}
}

// Render parses and executes src with the given context variables, inside the
// environment's limits.
func (e *Env) Render(src string, ctx map[string]any) (string, error) {
	text, _, err := e.render(src, ctx)
	return text, err
}

// render is Render, with what the render left of its limits.
func (e *Env) render(src string, ctx map[string]any) (string, budget, error) {
	body, err := parseTemplate(src)
	if err != nil {
		return "", budget{}, err
	}
	root := newScope(nil)
	r := &renderer{env: e, root: root, budget: newBudget(e.limits)}
	maps.Copy(root.vars, e.globals)
	// A render's range charges the render for the items it makes.
	root.vars["range"] = callable{name: "range", call: r.rangeGlobal}
	maps.Copy(root.vars, ctx)
	var out strings.Builder
	if err := r.execAll(body, root, &out); err != nil {
		return "", r.budget, err
	}
	return out.String(), r.budget, nil
}

// ---- scope ----

type scope struct {
	vars   map[string]any
	parent *scope
}

func newScope(parent *scope) *scope {
	return &scope{vars: map[string]any{}, parent: parent}
}

func (s *scope) get(name string) (any, bool) {
	for c := s; c != nil; c = c.parent {
		if v, ok := c.vars[name]; ok {
			return v, true
		}
	}
	return nil, false
}

func (s *scope) set(name string, v any) { s.vars[name] = v }

// ---- renderer ----

type renderer struct {
	env  *Env
	root *scope
	budget
}

// filtersInOneStep are the filters that look at their input without reading
// through it: its length, whether it is there. A template asks a whole
// conversation for its length once a message, and is not charged the
// conversation each time.
var filtersInOneStep = []string{"length", "count", "default", "d", "safe"}

// write puts a value into the output, charged for all it holds before it is
// made into text.
func (r *renderer) write(out *strings.Builder, value any) error {
	if err := r.read(value); err != nil {
		return err
	}
	out.WriteString(stringify(value))
	return nil
}

func (r *renderer) execAll(body []stmt, sc *scope, out *strings.Builder) error {
	for _, s := range body {
		if err := r.exec(s, sc, out); err != nil {
			return err
		}
	}
	return nil
}

func (r *renderer) exec(s stmt, sc *scope, out *strings.Builder) error {
	if err := r.work(1); err != nil {
		return err
	}
	switch n := s.(type) {
	case textStmt:
		if err := r.hold(len(n.text)); err != nil {
			return err
		}
		out.WriteString(n.text)
		return nil
	case outputStmt:
		return r.execOutput(n, sc, out)
	case ifStmt:
		return r.execIf(n, sc, out)
	case forStmt:
		return r.execFor(n, sc, out)
	case setStmt:
		return r.execSet(n, sc, out)
	case macroStmt:
		return r.execMacro(n, sc)
	}
	return fmt.Errorf("jinja: unknown statement %T", s)
}

func (r *renderer) execOutput(n outputStmt, sc *scope, out *strings.Builder) error {
	if n.cond != nil {
		c, err := r.eval(n.cond, sc)
		if err != nil {
			return err
		}
		if truthy(c) {
			v, err := r.eval(n.expr, sc)
			if err != nil {
				return err
			}
			return r.write(out, v)
		}
		if n.alt != nil {
			v, err := r.eval(n.alt, sc)
			if err != nil {
				return err
			}
			return r.write(out, v)
		}
		return nil
	}
	v, err := r.eval(n.expr, sc)
	if err != nil {
		return err
	}
	return r.write(out, v)
}

func (r *renderer) execIf(n ifStmt, sc *scope, out *strings.Builder) error {
	for _, br := range n.branches {
		c, err := r.eval(br.cond, sc)
		if err != nil {
			return err
		}
		if truthy(c) {
			return r.execAll(br.body, sc, out)
		}
	}
	if n.elseBody != nil {
		return r.execAll(n.elseBody, sc, out)
	}
	return nil
}

type loopEntry struct {
	key any
	val any // nil for single-var / list iteration
}

func (r *renderer) execFor(n forStmt, sc *scope, out *strings.Builder) error {
	iterVal, err := r.eval(n.iter, sc)
	if err != nil {
		return err
	}
	entries := iterate(iterVal)
	// A loop is charged each item it walks, and the place it keeps it in until
	// it ends. A list its own expression built for it, a range or a slice, is
	// held by nothing but the loop, and is given back with it.
	walked := scaled(len(entries), wordBytes)
	if err := errors.Join(r.work(len(entries)), r.hold(walked)); err != nil {
		return err
	}
	switch n.iter.(type) {
	case listNode, tupleNode, sliceNode, binNode, filterNode, callNode:
		walked += size(iterVal)
	}
	defer r.free(walked)
	// Materialize applying for-if and destructuring.
	type binding struct {
		keyVal any
		valVal any // for two-var
		two    bool
	}
	var items []binding
	for _, e := range entries {
		var b binding
		if n.keyVar != "" {
			// two-var: destructure a 2-list key, else key/value pair
			if lst, ok := e.key.([]any); ok && !isString(e.key) && len(lst) == 2 && e.val == nil {
				b.keyVal, b.valVal, b.two = lst[0], lst[1], true
			} else {
				b.keyVal, b.valVal, b.two = e.key, e.val, true
			}
		} else {
			b.keyVal = e.key
		}
		if n.ifCond != nil {
			child := newScope(sc)
			bindLoopVars(child, n, b.keyVal, b.valVal)
			c, err := r.eval(n.ifCond, child)
			if err != nil {
				return err
			}
			if !truthy(c) {
				continue
			}
		}
		items = append(items, b)
	}
	length := len(items)
	if length == 0 {
		if n.elseBody != nil {
			child := newScope(sc)
			return r.execAll(n.elseBody, child, out)
		}
		return nil
	}
	itemKeyOrPair := func(i int) any {
		if items[i].two {
			return []any{items[i].keyVal, items[i].valVal}
		}
		return items[i].keyVal
	}
	for i := range items {
		child := newScope(sc)
		bindLoopVars(child, n, items[i].keyVal, items[i].valVal)
		li := &loopInfo{
			index0:   i,
			index:    i + 1,
			length:   length,
			first:    i == 0,
			last:     i == length-1,
			revindex: length - i,
		}
		if i > 0 {
			li.previtem = itemKeyOrPair(i - 1)
		}
		if i < length-1 {
			li.nextitem = itemKeyOrPair(i + 1)
		}
		child.set("loop", li)
		if err := r.execAll(n.body, child, out); err != nil {
			return err
		}
	}
	return nil
}

func bindLoopVars(sc *scope, n forStmt, keyVal, valVal any) {
	if n.keyVar != "" {
		sc.set(n.keyVar, keyVal)
		sc.set(n.valVar, valVal)
	} else {
		sc.set(n.valVar, keyVal)
	}
}

func (r *renderer) execSet(n setStmt, sc *scope, out *strings.Builder) error {
	var val any
	if n.body != nil {
		var buf strings.Builder
		child := newScope(sc)
		if err := r.execAll(n.body, child, &buf); err != nil {
			return err
		}
		val = buf.String()
	} else if n.cond != nil {
		c, err := r.eval(n.cond, sc)
		if err != nil {
			return err
		}
		if truthy(c) {
			val, err = r.eval(n.value, sc)
		} else if n.alt != nil {
			val, err = r.eval(n.alt, sc)
		} else {
			val = nil
		}
		if err != nil {
			return err
		}
	} else {
		v, err := r.eval(n.value, sc)
		if err != nil {
			return err
		}
		val = v
	}
	return r.assign(n.target, val, sc)
}

func (r *renderer) assign(target expr, val any, sc *scope) error {
	switch t := target.(type) {
	case nameNode:
		sc.set(t.name, val)
		return nil
	case getAttrNode:
		obj, err := r.eval(t.node, sc)
		if err != nil {
			return err
		}
		return setMember(obj, t.attr, val)
	case getItemNode:
		obj, err := r.eval(t.node, sc)
		if err != nil {
			return err
		}
		key, err := r.eval(t.arg, sc)
		if err == nil {
			err = r.read(key)
		}
		if err != nil {
			return err
		}
		return setMember(obj, stringify(key), val)
	}
	return fmt.Errorf("jinja: invalid set target")
}

func setMember(obj any, key string, val any) error {
	switch m := obj.(type) {
	case map[string]any:
		m[key] = val
		return nil
	case *Dict:
		m.Set(key, val)
		return nil
	}
	return fmt.Errorf("jinja: cannot set %q on %T", key, obj)
}

func (r *renderer) execMacro(n macroStmt, sc *scope) error {
	defScope := sc
	params := n.params
	body := n.body
	name := n.name
	var fn callable
	fn = callable{name: name, call: func(args []any, kwargs map[string]any) (any, error) {
		child := newScope(defScope)
		// bind positional
		bound := make([]bool, len(params))
		for i := 0; i < len(args) && i < len(params); i++ {
			child.set(params[i].name, args[i])
			bound[i] = true
		}
		// bind keyword
		for k, v := range kwargs {
			for i := range params {
				if params[i].name == k {
					child.set(params[i].name, v)
					bound[i] = true
				}
			}
		}
		// defaults / none
		for i := range params {
			if bound[i] {
				continue
			}
			if params[i].def != nil {
				dv, err := r.eval(params[i].def, defScope)
				if err != nil {
					return nil, err
				}
				child.set(params[i].name, dv)
			} else {
				child.set(params[i].name, nil)
			}
		}
		var buf strings.Builder
		if err := r.execAll(body, child, &buf); err != nil {
			return nil, err
		}
		return buf.String(), nil
	}}
	sc.set(name, fn)
	return nil
}

// ---- expression evaluation ----

// eval evaluates one expression: a step of the render's work, one level
// deeper than the expression it stands in, and charged for the value it builds
// where it builds one.
func (r *renderer) eval(e expr, sc *scope) (any, error) {
	if err := r.enter(); err != nil {
		return nil, err
	}
	defer r.leave()
	value, err := r.evalNode(e, sc)
	if err != nil {
		return nil, err
	}
	switch e.(type) {
	case listNode, tupleNode, dictNode, sliceNode, binNode, filterNode, callNode:
		err = r.made(value)
	}
	return value, err
}

func (r *renderer) evalNode(e expr, sc *scope) (any, error) {
	switch n := e.(type) {
	case litNode:
		return n.val, nil
	case nameNode:
		v, _ := sc.get(n.name)
		return v, nil
	case listNode:
		out := make([]any, len(n.elems))
		for i, el := range n.elems {
			v, err := r.eval(el, sc)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case tupleNode:
		out := make([]any, len(n.elems))
		for i, el := range n.elems {
			v, err := r.eval(el, sc)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case dictNode:
		d := NewDict()
		for _, p := range n.pairs {
			k, err := r.eval(p.key, sc)
			if err == nil {
				err = r.read(k)
			}
			if err != nil {
				return nil, err
			}
			v, err := r.eval(p.val, sc)
			if err != nil {
				return nil, err
			}
			d.Set(stringify(k), v)
		}
		return d, nil
	case getAttrNode:
		obj, err := r.eval(n.node, sc)
		if err != nil {
			return nil, err
		}
		if n.hasIndex {
			v, _ := indexInto(obj, n.index)
			return v, nil
		}
		v, _ := getAttr(obj, n.attr)
		return v, nil
	case getItemNode:
		obj, err := r.eval(n.node, sc)
		if err != nil {
			return nil, err
		}
		if n.arg == nil {
			return nil, nil
		}
		key, err := r.eval(n.arg, sc)
		if err != nil {
			return nil, err
		}
		v, _ := getIndexOrKey(obj, key)
		return v, nil
	case sliceNode:
		return r.evalSlice(n, sc)
	case unaryNode:
		v, err := r.eval(n.term, sc)
		if err != nil {
			return nil, err
		}
		if n.negative {
			if isInteger(v) {
				return -toInt(v), nil
			}
			if isFloat(v) {
				return -toFloat(v), nil
			}
		}
		return v, nil
	case notNode:
		v, err := r.eval(n.term, sc)
		if err != nil {
			return nil, err
		}
		return !truthy(v), nil
	case binNode:
		return r.evalBin(n, sc)
	case condNode:
		c, err := r.eval(n.cond, sc)
		if err != nil {
			return nil, err
		}
		if truthy(c) {
			return r.eval(n.value, sc)
		}
		if n.alt == nil {
			return nil, nil
		}
		return r.eval(n.alt, sc)
	case testNode:
		return r.evalTest(n, sc)
	case filterNode:
		return r.evalFilter(n, sc)
	case callNode:
		return r.evalCall(n, sc)
	}
	return nil, fmt.Errorf("jinja: cannot evaluate %T", e)
}

func (r *renderer) evalBin(n binNode, sc *scope) (any, error) {
	if n.op == "and" || n.op == "or" {
		l, err := r.eval(n.left, sc)
		if err != nil {
			return nil, err
		}
		if n.op == "and" {
			if !truthy(l) {
				return l, nil
			}
			return r.eval(n.right, sc)
		}
		// or
		if truthy(l) {
			return l, nil
		}
		return r.eval(n.right, sc)
	}
	l, err := r.eval(n.left, sc)
	if err != nil {
		return nil, err
	}
	rt, err := r.eval(n.right, sc)
	if err != nil {
		return nil, err
	}
	// Arithmetic on two numbers reads nothing; every other operator reads
	// through its operands, to join them, compare them or search them.
	if !isNumber(l) || !isNumber(rt) {
		if err := r.read(l, rt); err != nil {
			return nil, err
		}
	}
	switch n.op {
	case "+":
		if isList(l) {
			if isList(rt) {
				return append(slices.Clone(l.([]any)), rt.([]any)...), nil
			}
			return nil, fmt.Errorf("jinja: cannot concatenate list to non-list")
		}
		if isFloat(l) || isFloat(rt) {
			return toFloat(l) + toFloat(rt), nil
		}
		if isString(l) || isString(rt) {
			return stringify(l) + stringify(rt), nil
		}
		return toInt(l) + toInt(rt), nil
	case "-":
		if isFloat(l) || isFloat(rt) {
			return toFloat(l) - toFloat(rt), nil
		}
		return toInt(l) - toInt(rt), nil
	case "*":
		if isFloat(l) || isFloat(rt) {
			return toFloat(l) * toFloat(rt), nil
		}
		if isString(l) {
			// Python repeats a string no times for a count below one. The
			// whole of the repeated string is charged before it is built.
			text, count := stringify(l), max(toInt(rt), 0)
			if err := r.hold(scaled(len(text), count)); err != nil {
				return nil, err
			}
			return strings.Repeat(text, count), nil
		}
		return toInt(l) * toInt(rt), nil
	case "/":
		return toFloat(l) / toFloat(rt), nil
	case "//":
		return int(toFloat(l) / toFloat(rt)), nil
	case "%":
		if toInt(rt) == 0 {
			return nil, fmt.Errorf("jinja: modulo by zero")
		}
		return toInt(l) % toInt(rt), nil
	case "**":
		if isInteger(l) && isInteger(rt) && toInt(rt) >= 0 {
			// A whole number to a whole power is a whole number, as in Python,
			// reached one multiplication at a time, each a step.
			if err := r.work(toInt(rt)); err != nil {
				return nil, err
			}
			power := 1
			for range toInt(rt) {
				power *= toInt(l)
			}
			return power, nil
		}
		return math.Pow(toFloat(l), toFloat(rt)), nil
	case "~":
		return stringify(l) + stringify(rt), nil
	case "==":
		return equal(l, rt), nil
	case "!=":
		return !equal(l, rt), nil
	case "<":
		return cmpLess(l, rt), nil
	case ">":
		return cmpGreater(l, rt), nil
	case "<=":
		return cmpLessEq(l, rt), nil
	case ">=":
		return cmpGreaterEq(l, rt), nil
	}
	return nil, fmt.Errorf("jinja: unknown operator %q", n.op)
}

func cmpLess(l, rt any) bool {
	if isFloat(l) || isFloat(rt) {
		return toFloat(l) < toFloat(rt)
	}
	if isString(l) || isString(rt) {
		return stringify(l) < stringify(rt)
	}
	return toInt(l) < toInt(rt)
}
func cmpGreater(l, rt any) bool {
	if isFloat(l) || isFloat(rt) {
		return toFloat(l) > toFloat(rt)
	}
	if isString(l) || isString(rt) {
		return stringify(l) > stringify(rt)
	}
	return toInt(l) > toInt(rt)
}
func cmpLessEq(l, rt any) bool {
	if isFloat(l) || isFloat(rt) {
		return toFloat(l) <= toFloat(rt)
	}
	if isString(l) || isString(rt) {
		return stringify(l) <= stringify(rt)
	}
	return toInt(l) <= toInt(rt)
}
func cmpGreaterEq(l, rt any) bool {
	if isFloat(l) || isFloat(rt) {
		return toFloat(l) >= toFloat(rt)
	}
	if isString(l) || isString(rt) {
		return stringify(l) >= stringify(rt)
	}
	return toInt(l) >= toInt(rt)
}

func (r *renderer) evalTest(n testNode, sc *scope) (any, error) {
	v, err := r.eval(n.node, sc)
	if err != nil {
		return nil, err
	}
	var res bool
	switch n.name {
	case "defined":
		res = v != nil
	case "undefined":
		res = v == nil
	case "none":
		// A name that is not defined is not none; only None is.
		res = isNone(v)
	case "mapping":
		res = isDict(v)
	case "sequence":
		// Python's test asks for a length and for indexing, which a string and
		// a mapping have as a list does.
		res = isIterable(v)
	case "string":
		res = isString(v)
	case "iterable":
		res = isIterable(v)
	case "boolean":
		res = isBool(v)
	case "number":
		// A Python bool is a number, though not an integer to the integer test.
		res = isNumber(v) || isBool(v)
	case "integer":
		res = isInteger(v)
	case "float":
		res = isFloat(v)
	case "callable":
		_, res = v.(callable)
	case "true":
		res = asBool(v)
	case "false":
		// Only False is false to this test: None and an empty string are not.
		res = isBool(v) && !asBool(v)
	case "in":
		if n.arg == nil {
			res = false
			break
		}
		arg, err := r.eval(n.arg, sc)
		if err == nil {
			err = r.read(arg, v)
		}
		if err != nil {
			return nil, err
		}
		res = contains(arg, v)
	case "eq", "equalto", "==":
		arg, err := r.eval(n.arg, sc)
		if err == nil {
			err = r.read(arg, v)
		}
		if err != nil {
			return nil, err
		}
		res = equal(v, arg)
	default:
		return nil, fmt.Errorf("jinja: unknown test %q", n.name)
	}
	if n.neg {
		res = !res
	}
	return res, nil
}

func (r *renderer) evalSlice(n sliceNode, sc *scope) (any, error) {
	obj, err := r.eval(n.node, sc)
	if err != nil {
		return nil, err
	}
	var srcLen int
	var getStr []rune
	isStr := false
	switch o := obj.(type) {
	case string:
		getStr = []rune(o)
		srcLen = len(getStr)
		isStr = true
	case []any:
		srcLen = len(o)
	default:
		return nil, fmt.Errorf("jinja: cannot slice %T", obj)
	}
	step := 1
	if n.step != nil {
		sv, err := r.eval(n.step, sc)
		if err != nil {
			return nil, err
		}
		step = toInt(sv)
		if step == 0 {
			return nil, fmt.Errorf("jinja: slice step cannot be zero")
		}
	}
	var start, end int
	startProvided := n.start != nil
	endProvided := n.end != nil
	if startProvided {
		sv, err := r.eval(n.start, sc)
		if err != nil {
			return nil, err
		}
		start = toInt(sv)
	}
	if endProvided {
		ev, err := r.eval(n.end, sc)
		if err != nil {
			return nil, err
		}
		end = toInt(ev)
	}
	if step == 1 {
		s := 0
		if startProvided {
			s = start
			if s < 0 {
				s = srcLen + s
			}
		}
		ee := srcLen
		if endProvided {
			ee = end
			if ee < 0 {
				ee = srcLen + ee
			}
		}
		if s < 0 {
			s = 0
		}
		s = min(s, srcLen)
		ee = max(ee, s)
		ee = min(ee, srcLen)
		if isStr {
			return string(getStr[s:ee]), nil
		}
		return slices.Clone(obj.([]any)[s:ee]), nil
	}
	start, end = pythonSliceBounds(srcLen, start, end, step, startProvided, endProvided)
	if isStr {
		var outR []rune
		if step > 0 {
			for i := start; i < end && i < srcLen; i += step {
				if i < 0 {
					continue
				}
				outR = append(outR, getStr[i])
			}
		} else {
			for i := start; i > end && i >= 0; i += step {
				if i >= srcLen {
					continue
				}
				outR = append(outR, getStr[i])
			}
		}
		return string(outR), nil
	}
	src := obj.([]any)
	var out []any
	if step > 0 {
		for i := start; i < end && i < srcLen; i += step {
			if i < 0 {
				continue
			}
			out = append(out, src[i])
		}
	} else {
		for i := start; i > end && i >= 0; i += step {
			if i >= srcLen {
				continue
			}
			out = append(out, src[i])
		}
	}
	if out == nil {
		out = []any{}
	}
	return out, nil
}

func pythonSliceBounds(length, start, stop, step int, startProvided, stopProvided bool) (int, int) {
	if step > 0 {
		if !startProvided {
			start = 0
		} else {
			if start < 0 {
				start += length
			}
			if start < 0 {
				start = 0
			}
			start = min(start, length)
		}
		if !stopProvided {
			stop = length
		} else {
			if stop < 0 {
				stop += length
			}
			if stop < 0 {
				stop = 0
			}
			stop = min(stop, length)
		}
		return start, stop
	}
	if !startProvided {
		start = length - 1
	} else {
		if start < 0 {
			start += length
		}
		if start < 0 {
			start = -1
		}
		if start >= length {
			start = length - 1
		}
	}
	if !stopProvided {
		stop = -1
	} else {
		if stop < 0 {
			stop += length
		}
		if stop < -1 {
			stop = -1
		}
		if stop >= length {
			stop = length - 1
		}
	}
	return start, stop
}

func (r *renderer) evalFilter(n filterNode, sc *scope) (any, error) {
	in, err := r.eval(n.node, sc)
	if err != nil {
		return nil, err
	}
	args := make([]any, len(n.args))
	for i, a := range n.args {
		v, err := r.eval(a, sc)
		if err != nil {
			return nil, err
		}
		args[i] = v
	}
	kwargs := map[string]any{}
	for _, kw := range n.kwargs {
		v, err := r.eval(kw.val, sc)
		if err == nil {
			err = r.read(v)
		}
		if err != nil {
			return nil, err
		}
		kwargs[kw.name] = v
	}
	f, ok := r.env.filters[n.name]
	if !ok {
		return nil, fmt.Errorf("jinja: unknown filter %q", n.name)
	}
	// A filter reads through what it is given, to turn it into text or to
	// walk it, unless it is one that only looks at it.
	if !slices.Contains(filtersInOneStep, n.name) {
		if err := r.read(append(args, in)...); err != nil {
			return nil, err
		}
	}
	if n.name == mapFilter && len(args) > 0 && in != nil {
		return r.mapNamed(in, args, kwargs)
	}
	value, held, err := r.build(n.name, f, in, args, kwargs)
	// What the filter built is charged as the value of its expression (eval).
	r.free(held)
	return value, err
}

// mapFilter names the filter that runs another on each item.
const mapFilter = "map"

// planned are the filters that say how many bytes they will build before
// they build them, because it can be far more than they are given: each
// returns the count and what builds the value.
var planned = map[string]func(in any, args []any, kwargs map[string]any) (int, func() any, error){"tojson": planJSON}

// build runs a filter, and for one that plans what it builds (planned) holds
// the bytes first: a result the render cannot pay for is refused before any
// of it is allocated. It returns the bytes it held.
func (r *renderer) build(name string, f Filter, in any, args []any, kwargs map[string]any) (any, int, error) {
	plan, counted := planned[name]
	if !counted {
		value, err := f(in, args, kwargs)
		return value, 0, err
	}
	held, build, err := plan(in, args, kwargs)
	if err == nil {
		err = r.hold(held)
	}
	if err != nil {
		return nil, 0, err
	}
	return build(), held, nil
}

// mapNamed is map given a filter's name: that filter run on each item, with
// the arguments after the name. Each item is built as the filter alone builds
// it, and what was held for it stays held, by the list that keeps it.
func (r *renderer) mapNamed(in any, args []any, kwargs map[string]any) (any, error) {
	name, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("jinja: map filter name must be a string")
	}
	f, ok := r.env.filters[name]
	if !ok {
		return nil, fmt.Errorf("jinja: unknown filter %q in map", name)
	}
	entries := iterate(in)
	out := make([]any, 0, len(entries))
	for _, en := range entries {
		v, _, err := r.build(name, f, en.key, args[1:], kwargs)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

func (r *renderer) evalCall(n callNode, sc *scope) (any, error) {
	// Method dispatch: obj.method(args) where method is not a resolvable callable.
	if ga, ok := n.fn.(getAttrNode); ok && !ga.hasIndex {
		fnv, _ := r.evalGetAttrRaw(ga, sc)
		if c, ok := fnv.(callable); ok {
			return r.invoke(c, n, sc)
		}
		obj, err := r.eval(ga.node, sc)
		if err != nil {
			return nil, err
		}
		return r.dispatchMethod(obj, ga.attr, n, sc)
	}
	fnv, err := r.eval(n.fn, sc)
	if err != nil {
		return nil, err
	}
	if c, ok := fnv.(callable); ok {
		return r.invoke(c, n, sc)
	}
	return nil, fmt.Errorf("jinja: %v is not callable", n.fn)
}

// evalGetAttrRaw evaluates a getAttrNode returning the attribute value (used to
// probe whether obj.attr resolves to a callable before falling back to method
// dispatch).
func (r *renderer) evalGetAttrRaw(ga getAttrNode, sc *scope) (any, bool) {
	obj, err := r.eval(ga.node, sc)
	if err != nil {
		return nil, false
	}
	return getAttr(obj, ga.attr)
}

func (r *renderer) invoke(c callable, n callNode, sc *scope) (any, error) {
	args := make([]any, len(n.args))
	for i, a := range n.args {
		v, err := r.eval(a, sc)
		if err != nil {
			return nil, err
		}
		args[i] = v
	}
	kwargs := map[string]any{}
	for _, kw := range n.kwargs {
		v, err := r.eval(kw.val, sc)
		if err != nil {
			return nil, err
		}
		kwargs[kw.name] = v
	}
	return c.call(args, kwargs)
}

func (r *renderer) evalArgs(n callNode, sc *scope) ([]any, map[string]any, error) {
	args := make([]any, len(n.args))
	for i, a := range n.args {
		v, err := r.eval(a, sc)
		if err != nil {
			return nil, nil, err
		}
		args[i] = v
	}
	kwargs := map[string]any{}
	for _, kw := range n.kwargs {
		v, err := r.eval(kw.val, sc)
		if err != nil {
			return nil, nil, err
		}
		kwargs[kw.name] = v
	}
	return args, kwargs, nil
}
