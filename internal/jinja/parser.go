package jinja

import (
	"fmt"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
)

// parseTemplate lexes and parses a template source into a statement list.
func parseTemplate(src string) ([]stmt, error) {
	chunks, err := lex(src)
	if err != nil {
		return nil, err
	}
	applyTrim(chunks)
	dp := &docParser{chunks: chunks}
	body, term, err := dp.parseUntil()
	if err != nil {
		return nil, err
	}
	if term != "" {
		return nil, fmt.Errorf("jinja: unexpected {%% %s %%}", term)
	}
	return body, nil
}

// applyTrim implements gonja whitespace control: a tag with left-control (-)
// strips trailing whitespace of the preceding text; right-control (-) strips
// leading whitespace of the following text.
func applyTrim(chunks []chunk) {
	const ws = " \t\r\n"
	for i := range chunks {
		if chunks[i].kind != cText {
			continue
		}
		if i > 0 && chunks[i-1].kind != cText && chunks[i-1].trimR {
			chunks[i].text = strings.TrimLeft(chunks[i].text, ws)
		}
		if i+1 < len(chunks) && chunks[i+1].kind != cText && chunks[i+1].trimL {
			chunks[i].text = strings.TrimRight(chunks[i].text, ws)
		}
	}
}

type docParser struct {
	chunks []chunk
	pos    int
}

// parseUntil parses statements until one of the terminator block names is
// reached (or end of input). It returns the parsed statements and the
// terminator name encountered ("" for EOF).
func (dp *docParser) parseUntil(terms ...string) ([]stmt, string, error) {
	var out []stmt
	for dp.pos < len(dp.chunks) {
		c := dp.chunks[dp.pos]
		switch c.kind {
		case cText:
			dp.pos++
			if c.text != "" {
				out = append(out, textStmt{text: c.text})
			}
		case cComment:
			dp.pos++
		case cOutput:
			dp.pos++
			s, err := parseOutputChunk(c)
			if err != nil {
				return nil, "", err
			}
			out = append(out, s)
		case cBlock:
			if containsStr(terms, c.name) {
				return out, c.name, nil
			}
			dp.pos++
			s, err := dp.parseBlock(c)
			if err != nil {
				return nil, "", err
			}
			if s != nil {
				out = append(out, s)
			}
		}
	}
	return out, "", nil
}

func containsStr(xs []string, v string) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

func parseOutputChunk(c chunk) (stmt, error) {
	ep := &eparser{toks: c.toks}
	e, err := ep.parseExpr()
	if err != nil {
		return nil, err
	}
	o := outputStmt{expr: e}
	if ep.matchName("if") {
		cond, err := ep.parseExpr()
		if err != nil {
			return nil, err
		}
		o.cond = cond
		if ep.matchName("else") {
			alt, err := ep.parseExpr()
			if err != nil {
				return nil, err
			}
			o.alt = alt
		}
	}
	if !ep.done() {
		return nil, fmt.Errorf("jinja: trailing tokens in output expression")
	}
	return o, nil
}

func (dp *docParser) parseBlock(c chunk) (stmt, error) {
	switch c.name {
	case "if":
		return dp.parseIf(c)
	case "for":
		return dp.parseFor(c)
	case "set":
		return dp.parseSet(c)
	case "macro":
		return dp.parseMacro(c)
	default:
		return nil, fmt.Errorf("jinja: unsupported control structure {%% %s %%}", c.name)
	}
}

func (dp *docParser) parseIf(c chunk) (stmt, error) {
	var st ifStmt
	cond, err := parseSingleExpr(c.toks)
	if err != nil {
		return nil, err
	}
	body, term, err := dp.parseUntil("elif", "else", "endif")
	if err != nil {
		return nil, err
	}
	st.branches = append(st.branches, ifBranch{cond: cond, body: body})
	for term == "elif" {
		ec := dp.chunks[dp.pos]
		dp.pos++
		econd, err := parseSingleExpr(ec.toks)
		if err != nil {
			return nil, err
		}
		var ebody []stmt
		ebody, term, err = dp.parseUntil("elif", "else", "endif")
		if err != nil {
			return nil, err
		}
		st.branches = append(st.branches, ifBranch{cond: econd, body: ebody})
	}
	if term == "else" {
		dp.pos++
		var ebody []stmt
		ebody, term, err = dp.parseUntil("endif")
		if err != nil {
			return nil, err
		}
		st.elseBody = ebody
	}
	if term != "endif" {
		return nil, fmt.Errorf("jinja: unclosed {%% if %%}")
	}
	dp.pos++
	return st, nil
}

func (dp *docParser) parseFor(c chunk) (stmt, error) {
	ep := &eparser{toks: c.toks}
	var vars []string
	for {
		n := ep.cur()
		if n == nil || (n.kind == etName && n.val == "in") {
			break
		}
		if n.kind == etName {
			vars = append(vars, n.val)
			ep.pos++
		} else if n.kind == etSym && n.val == "," {
			ep.pos++
		} else {
			return nil, fmt.Errorf("jinja: malformed for targets")
		}
	}
	if !ep.matchName("in") {
		return nil, fmt.Errorf("jinja: for loop missing 'in'")
	}
	iter, err := ep.parseExpr()
	if err != nil {
		return nil, err
	}
	var st forStmt
	if len(vars) == 1 {
		st.valVar = vars[0]
	} else if len(vars) == 2 {
		st.keyVar, st.valVar = vars[0], vars[1]
	} else {
		return nil, fmt.Errorf("jinja: unsupported for target arity %d", len(vars))
	}
	st.iter = iter
	if ep.matchName("if") {
		cond, err := ep.parseExpr()
		if err != nil {
			return nil, err
		}
		st.ifCond = cond
	}
	body, term, err := dp.parseUntil("else", "endfor")
	if err != nil {
		return nil, err
	}
	st.body = body
	if term == "else" {
		dp.pos++
		var ebody []stmt
		ebody, term, err = dp.parseUntil("endfor")
		if err != nil {
			return nil, err
		}
		st.elseBody = ebody
	}
	if term != "endfor" {
		return nil, fmt.Errorf("jinja: unclosed {%% for %%}")
	}
	dp.pos++
	return st, nil
}

func (dp *docParser) parseSet(c chunk) (stmt, error) {
	ep := &eparser{toks: c.toks}
	target, err := ep.parsePrimary()
	if err != nil {
		return nil, err
	}
	if !ep.matchSym("=") {
		// block form
		if !ep.done() {
			return nil, fmt.Errorf("jinja: malformed block set")
		}
		body, term, err := dp.parseUntil("endset")
		if err != nil {
			return nil, err
		}
		if term != "endset" {
			return nil, fmt.Errorf("jinja: unclosed {%% set %%}")
		}
		dp.pos++
		return setStmt{target: target, body: body}, nil
	}
	val, err := ep.parseExpr()
	if err != nil {
		return nil, err
	}
	st := setStmt{target: target, value: val}
	if ep.matchName("if") {
		cond, err := ep.parseExpr()
		if err != nil {
			return nil, err
		}
		st.cond = cond
		if ep.matchName("else") {
			alt, err := ep.parseExpr()
			if err != nil {
				return nil, err
			}
			st.alt = alt
		}
	}
	if !ep.done() {
		return nil, fmt.Errorf("jinja: trailing tokens in set")
	}
	return st, nil
}

func (dp *docParser) parseMacro(c chunk) (stmt, error) {
	ep := &eparser{toks: c.toks}
	n := ep.cur()
	if n == nil || n.kind != etName {
		return nil, fmt.Errorf("jinja: macro requires a name")
	}
	name := n.val
	ep.pos++
	if !ep.matchSym("(") {
		return nil, fmt.Errorf("jinja: macro requires parameter list")
	}
	var params []macroParam
	for !ep.matchSym(")") {
		ep.matchSym(",")
		if ep.matchSym(")") {
			break
		}
		pn := ep.cur()
		if pn == nil || pn.kind != etName {
			return nil, fmt.Errorf("jinja: malformed macro parameter")
		}
		p := macroParam{name: pn.val}
		ep.pos++
		if ep.matchSym("=") {
			def, err := ep.parseExpr()
			if err != nil {
				return nil, err
			}
			p.def = def
		}
		params = append(params, p)
	}
	body, term, err := dp.parseUntil("endmacro")
	if err != nil {
		return nil, err
	}
	if term != "endmacro" {
		return nil, fmt.Errorf("jinja: unclosed {%% macro %%}")
	}
	dp.pos++
	return macroStmt{name: name, params: params, body: body}, nil
}

func parseSingleExpr(toks []etok) (expr, error) {
	ep := &eparser{toks: toks}
	e, err := ep.parseExpr()
	if err != nil {
		return nil, err
	}
	if !ep.done() {
		return nil, fmt.Errorf("jinja: trailing tokens in expression")
	}
	return e, nil
}

// ---- expression parser ----

type eparser struct {
	toks []etok
	pos  int
}

func (p *eparser) cur() *etok {
	if p.pos < len(p.toks) {
		return &p.toks[p.pos]
	}
	return nil
}

func (p *eparser) done() bool { return p.pos >= len(p.toks) }

func (p *eparser) peekSym(s string) bool {
	t := p.cur()
	return t != nil && t.kind == etSym && t.val == s
}

func (p *eparser) matchSym(s string) bool {
	if p.peekSym(s) {
		p.pos++
		return true
	}
	return false
}

func (p *eparser) peekName(s string) bool {
	t := p.cur()
	return t != nil && t.kind == etName && t.val == s
}

func (p *eparser) matchName(s string) bool {
	if p.peekName(s) {
		p.pos++
		return true
	}
	return false
}

var compareSyms = map[string]bool{"==": true, "!=": true, ">": true, ">=": true, "<": true, "<=": true}

func (p *eparser) parseExpr() (expr, error) {
	e, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	return p.parseFilters(e)
}

func (p *eparser) parseOr() (expr, error) {
	l, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.matchName("or") {
		r, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		l = binNode{op: "or", left: l, right: r}
	}
	return l, nil
}

func (p *eparser) parseAnd() (expr, error) {
	l, err := p.parseNot()
	if err != nil {
		return nil, err
	}
	for p.matchName("and") {
		r, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		l = binNode{op: "and", left: l, right: r}
	}
	return l, nil
}

func (p *eparser) parseNot() (expr, error) {
	if p.matchName("not") {
		t, err := p.parseCompare()
		if err != nil {
			return nil, err
		}
		return notNode{term: t}, nil
	}
	return p.parseCompare()
}

func (p *eparser) parseCompare() (expr, error) {
	l, err := p.parseMath()
	if err != nil {
		return nil, err
	}
	for {
		t := p.cur()
		if t == nil || t.kind != etSym || !compareSyms[t.val] {
			break
		}
		op := t.val
		p.pos++
		r, err := p.parseMath()
		if err != nil {
			return nil, err
		}
		l = binNode{op: op, left: l, right: r}
	}
	return p.parseTest(l)
}

func (p *eparser) parseTest(l expr) (expr, error) {
	l, err := p.parseFilters(l)
	if err != nil {
		return nil, err
	}
	if p.peekName("is") || p.peekName("in") || p.peekName("not") {
		p.matchName("is")
		neg := p.matchName("not")
		nm := p.cur()
		if nm == nil || nm.kind != etName {
			return nil, fmt.Errorf("jinja: expected test name")
		}
		name := nm.val
		p.pos++
		t := testNode{node: l, name: name, neg: neg}
		if !p.peekName("else") {
			// optional single argument (primary)
			if p.canStartArg() {
				save := p.pos
				arg, err := p.parsePrimary()
				if err != nil {
					p.pos = save
				} else {
					arg2, err := p.parseFilters(arg)
					if err == nil {
						arg = arg2
					}
					t.arg = arg
				}
			}
		}
		return t, nil
	}
	return l, nil
}

// canStartArg reports whether the current token can begin a test argument.
func (p *eparser) canStartArg() bool {
	t := p.cur()
	if t == nil {
		return false
	}
	switch t.kind {
	case etStr, etInt, etFloat:
		return true
	case etName:
		switch t.val {
		case "and", "or", "not", "in", "is", "if", "else":
			return false
		}
		return true
	case etSym:
		return t.val == "(" || t.val == "[" || t.val == "{"
	}
	return false
}

func (p *eparser) parseMath() (expr, error) {
	l, err := p.parseConcat()
	if err != nil {
		return nil, err
	}
	for p.peekSym("+") || p.peekSym("-") {
		op := p.cur().val
		p.pos++
		r, err := p.parseConcat()
		if err != nil {
			return nil, err
		}
		l = binNode{op: op, left: l, right: r}
	}
	return l, nil
}

func (p *eparser) parseConcat() (expr, error) {
	l, err := p.parseMathPrio()
	if err != nil {
		return nil, err
	}
	for p.peekSym("~") {
		p.pos++
		r, err := p.parseMathPrio()
		if err != nil {
			return nil, err
		}
		l = binNode{op: "~", left: l, right: r}
	}
	return l, nil
}

func (p *eparser) parseMathPrio() (expr, error) {
	l, err := p.parsePower()
	if err != nil {
		return nil, err
	}
	for p.peekSym("*") || p.peekSym("/") || p.peekSym("//") || p.peekSym("%") {
		op := p.cur().val
		p.pos++
		r, err := p.parsePower()
		if err != nil {
			return nil, err
		}
		l = binNode{op: op, left: l, right: r}
	}
	return l, nil
}

func (p *eparser) parsePower() (expr, error) {
	l, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for p.peekSym("**") {
		p.pos++
		r, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		l = binNode{op: "**", left: l, right: r}
	}
	return l, nil
}

func (p *eparser) parseUnary() (expr, error) {
	negative := false
	signed := false
	if p.peekSym("-") {
		negative = true
		signed = true
		p.pos++
	} else if p.peekSym("+") {
		signed = true
		p.pos++
	}
	e, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	if signed {
		e = unaryNode{negative: negative, term: e}
	}
	return p.parseFilters(e)
}

func (p *eparser) parseFilters(e expr) (expr, error) {
	for p.peekSym("|") {
		p.pos++
		nm := p.cur()
		if nm == nil || nm.kind != etName {
			return nil, fmt.Errorf("jinja: filter name must be an identifier")
		}
		f := filterNode{node: e, name: nm.val}
		p.pos++
		if p.matchSym("(") {
			args, kwargs, err := p.parseCallArgs()
			if err != nil {
				return nil, err
			}
			f.args = args
			f.kwargs = kwargs
		}
		e = f
	}
	return e, nil
}

// parseCallArgs parses the interior of a "(...)" after the opening paren has
// been consumed, mirroring gonja's arg loop (adjacent expressions become
// separate positional args; NAME=expr becomes a kwarg).
func (p *eparser) parseCallArgs() ([]expr, []kwarg, error) {
	var args []expr
	var kwargs []kwarg
	for {
		if p.matchSym(")") {
			break
		}
		if p.matchSym(",") {
			if p.matchSym(")") {
				break
			}
			continue
		}
		v, err := p.parseExpr()
		if err != nil {
			return nil, nil, err
		}
		if p.matchSym("=") {
			name := ""
			if nn, ok := v.(nameNode); ok {
				name = nn.name
			}
			val, err := p.parseExpr()
			if err != nil {
				return nil, nil, err
			}
			kwargs = append(kwargs, kwarg{name: name, val: val})
		} else {
			args = append(args, v)
		}
	}
	return args, kwargs, nil
}

func (p *eparser) parsePrimary() (expr, error) {
	t := p.cur()
	if t == nil {
		return nil, fmt.Errorf("jinja: unexpected end of expression")
	}
	var node expr
	switch t.kind {
	case etInt:
		iv, err := parseIntLit(t.val)
		if err != nil {
			return nil, err
		}
		node = litNode{val: iv}
		p.pos++
	case etFloat:
		fv, err := strconv.ParseFloat(strings.ReplaceAll(t.val, "_", ""), binaryschema.Width64Bits)
		if err != nil {
			return nil, err
		}
		node = litNode{val: fv}
		p.pos++
	case etStr:
		node = litNode{val: t.val}
		p.pos++
	case etName:
		switch t.val {
		case "true", "True":
			node = litNode{val: true}
		case "false", "False":
			node = litNode{val: false}
		case "none", "None", "nil":
			node = litNode{val: nil}
		default:
			node = nameNode{name: t.val}
		}
		p.pos++
	case etSym:
		switch t.val {
		case "(":
			p.pos++
			e, err := p.parseParenOrTuple()
			if err != nil {
				return nil, err
			}
			node = e
		case "[":
			p.pos++
			e, err := p.parseList()
			if err != nil {
				return nil, err
			}
			node = e
		case "{":
			p.pos++
			e, err := p.parseDict()
			if err != nil {
				return nil, err
			}
			node = e
		default:
			return nil, fmt.Errorf("jinja: unexpected symbol %q", t.val)
		}
	default:
		return nil, fmt.Errorf("jinja: unexpected token")
	}
	return p.parsePostfix(node)
}

func (p *eparser) parsePostfix(node expr) (expr, error) {
	var parent expr
	for {
		if p.matchSym(".") {
			nm := p.cur()
			if nm == nil {
				return nil, fmt.Errorf("jinja: expected name after '.'")
			}
			if nm.kind == etName {
				parent = node
				node = getAttrNode{node: node, attr: nm.val}
				p.pos++
			} else if nm.kind == etInt {
				iv, err := parseIntLit(nm.val)
				if err != nil {
					return nil, err
				}
				parent = node
				node = getAttrNode{node: node, hasIndex: true, index: iv}
				p.pos++
			} else {
				return nil, fmt.Errorf("jinja: expected name or integer after '.'")
			}
			continue
		}
		if p.matchSym("[") {
			g, err := p.parseGetterBracket(node)
			if err != nil {
				return nil, err
			}
			parent = node
			node = g
			continue
		}
		if p.matchSym("(") {
			args, kwargs, err := p.parseCallArgs()
			if err != nil {
				return nil, err
			}
			node = callNode{fn: node, args: args, kwargs: kwargs, parent: parent}
			continue
		}
		break
	}
	return node, nil
}

func (p *eparser) parseGetterBracket(node expr) (expr, error) {
	// item or slice: [expr] | [start:end[:step]]
	var start, end, step expr
	hasColon := false
	if !p.peekSym(":") && !p.peekSym("]") {
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		start = e
	}
	if p.matchSym("]") {
		if !hasColon {
			return getItemNode{node: node, arg: start}, nil
		}
	}
	if p.matchSym(":") {
		hasColon = true
		if !p.peekSym(":") && !p.peekSym("]") {
			e, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			end = e
		}
		if p.matchSym(":") {
			if !p.peekSym("]") {
				e, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				step = e
			}
		}
		if !p.matchSym("]") {
			return nil, fmt.Errorf("jinja: unbalanced slice bracket")
		}
		return sliceNode{node: node, start: start, end: end, step: step}, nil
	}
	return nil, fmt.Errorf("jinja: unbalanced bracket")
}

func (p *eparser) parseParenOrTuple() (expr, error) {
	if p.matchSym(")") {
		return tupleNode{}, nil
	}
	first, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if p.matchSym(")") {
		return first, nil
	}
	elems := []expr{first}
	for p.matchSym(",") {
		if p.peekSym(")") {
			break
		}
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		elems = append(elems, e)
	}
	if !p.matchSym(")") {
		return nil, fmt.Errorf("jinja: unbalanced parenthesis")
	}
	return tupleNode{elems: elems}, nil
}

func (p *eparser) parseList() (expr, error) {
	var elems []expr
	if p.matchSym("]") {
		return listNode{}, nil
	}
	e, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	elems = append(elems, e)
	for p.matchSym(",") {
		if p.peekSym("]") {
			break
		}
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		elems = append(elems, e)
	}
	if !p.matchSym("]") {
		return nil, fmt.Errorf("jinja: unbalanced bracket")
	}
	return listNode{elems: elems}, nil
}

func (p *eparser) parseDict() (expr, error) {
	var pairs []pairNode
	if p.matchSym("}") {
		return dictNode{}, nil
	}
	kv, err := p.parseDictPair()
	if err != nil {
		return nil, err
	}
	pairs = append(pairs, kv)
	for p.matchSym(",") {
		if p.peekSym("}") {
			break
		}
		kv, err := p.parseDictPair()
		if err != nil {
			return nil, err
		}
		pairs = append(pairs, kv)
	}
	if !p.matchSym("}") {
		return nil, fmt.Errorf("jinja: unbalanced brace")
	}
	return dictNode{pairs: pairs}, nil
}

func (p *eparser) parseDictPair() (pairNode, error) {
	k, err := p.parseExpr()
	if err != nil {
		return pairNode{}, err
	}
	if !p.matchSym(":") {
		return pairNode{}, fmt.Errorf("jinja: expected ':' in dict")
	}
	v, err := p.parseExpr()
	if err != nil {
		return pairNode{}, err
	}
	return pairNode{key: k, val: v}, nil
}

func parseIntLit(s string) (int, error) {
	s = strings.ReplaceAll(s, "_", "")
	if len(s) > 2 && s[0] == '0' {
		switch s[1] {
		case 'x', 'X':
			v, err := strconv.ParseInt(s[2:], binaryschema.HexRadix, binaryschema.Width64Bits)
			return int(v), err
		case 'b', 'B':
			v, err := strconv.ParseInt(s[2:], binaryschema.BinaryRadix, binaryschema.Width64Bits)
			return int(v), err
		case 'o', 'O':
			v, err := strconv.ParseInt(s[2:], binaryschema.OctalRadix, binaryschema.Width64Bits)
			return int(v), err
		}
	}
	v, err := strconv.ParseInt(s, binaryschema.DecimalRadix, binaryschema.Width64Bits)
	return int(v), err
}
