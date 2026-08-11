package jinja

// ---- expression AST ----

type expr interface{ isExpr() }

type litNode struct{ val any }
type nameNode struct{ name string }
type listNode struct{ elems []expr }
type tupleNode struct{ elems []expr }

type pairNode struct {
	key expr
	val expr
}
type dictNode struct{ pairs []pairNode }

type getAttrNode struct {
	node     expr
	attr     string
	hasIndex bool
	index    int
}
type getItemNode struct {
	node expr
	arg  expr // may be nil
}
type sliceNode struct {
	node             expr
	start, end, step expr // any may be nil
}

type kwarg struct {
	name string
	val  expr
}
type callNode struct {
	fn     expr
	args   []expr
	kwargs []kwarg
	parent expr // node before the last getter (method dispatch)
}
type filterNode struct {
	node   expr
	name   string
	args   []expr
	kwargs []kwarg
}
type testNode struct {
	node expr
	name string
	arg  expr // may be nil
	neg  bool
}
type binNode struct {
	op          string
	left, right expr
}
type unaryNode struct {
	negative bool
	term     expr
}
type notNode struct{ term expr }

func (litNode) isExpr()     {}
func (nameNode) isExpr()    {}
func (listNode) isExpr()    {}
func (tupleNode) isExpr()   {}
func (dictNode) isExpr()    {}
func (getAttrNode) isExpr() {}
func (getItemNode) isExpr() {}
func (sliceNode) isExpr()   {}
func (callNode) isExpr()    {}
func (filterNode) isExpr()  {}
func (testNode) isExpr()    {}
func (binNode) isExpr()     {}
func (unaryNode) isExpr()   {}
func (notNode) isExpr()     {}

// ---- statement AST ----

type stmt interface{ isStmt() }

type textStmt struct{ text string }
type outputStmt struct {
	expr expr
	cond expr // ternary condition (nullable)
	alt  expr // ternary alternative (nullable)
}
type ifBranch struct {
	cond expr
	body []stmt
}
type ifStmt struct {
	branches []ifBranch
	elseBody []stmt
}
type forStmt struct {
	keyVar   string // "" if single-variable loop
	valVar   string
	iter     expr
	ifCond   expr // for-if filter (nullable)
	body     []stmt
	elseBody []stmt
}
type setStmt struct {
	target    expr
	value     expr   // expression form
	cond, alt expr   // ternary (nullable)
	body      []stmt // block form (nil if expression form)
}
type macroParam struct {
	name string
	def  expr // default (nullable)
}
type macroStmt struct {
	name   string
	params []macroParam
	body   []stmt
}

func (textStmt) isStmt()   {}
func (outputStmt) isStmt() {}
func (ifStmt) isStmt()     {}
func (forStmt) isStmt()    {}
func (setStmt) isStmt()    {}
func (macroStmt) isStmt()  {}
