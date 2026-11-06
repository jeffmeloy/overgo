package inference

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// layerLists are the names a model's layers are walked by: its weights' layers, an encoder's, or a local list of them.
var layerLists = map[string]bool{"Layers": true, "EncoderLayers": true, "layers": true}

// TestOneLayerWalkServesEveryForward holds each forward of the package's production code to one walk over a model's
// layers: no function ranges over a layer list twice, each time handing the activation through a per-layer step, as
// the non-causal forward did for its recurrent and its dense layers, whose steps take one signature. A forward that
// serves several kinds of layer chooses the step and walks once.
func TestOneLayerWalkServesEveryForward(t *testing.T) {
	t.Parallel()
	for _, name := range testcheck.Must(filepath.Glob("*.go"))(t) {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		parsed := testcheck.Must(parser.ParseFile(token.NewFileSet(), name, testcheck.Must(os.ReadFile(name))(t), parser.SkipObjectResolution))(t)
		for _, declaration := range parsed.Decls {
			function, isFunction := declaration.(*ast.FuncDecl)
			if !isFunction || function.Body == nil {
				continue
			}
			walks := 0
			ast.Inspect(function.Body, func(node ast.Node) bool {
				if loop, isRange := node.(*ast.RangeStmt); isRange && layerList(loop.X) && handsOnActivation(loop.Body) {
					walks++
				}
				return true
			})
			if walks > 1 {
				t.Errorf("%s: %s walks a model's layers %d times; choose the layer's step and walk once", name, function.Name.Name, walks)
			}
		}
	}
}

// layerList reports whether a range expression is a list of a model's layers.
func layerList(expression ast.Expr) bool {
	switch list := expression.(type) {
	case *ast.SelectorExpr:
		return layerLists[list.Sel.Name]
	case *ast.Ident:
		return layerLists[list.Name]
	}
	return false
}

// handsOnActivation reports whether a loop body assigns the activation a call returns, as a walk's per-layer step does.
func handsOnActivation(body *ast.BlockStmt) bool {
	for _, statement := range body.List {
		assign, isAssign := statement.(*ast.AssignStmt)
		if !isAssign || len(assign.Rhs) != 1 {
			continue
		}
		if _, isCall := assign.Rhs[0].(*ast.CallExpr); isCall && slicesContainsIdent(assign.Lhs, "activation") {
			return true
		}
	}
	return false
}

// slicesContainsIdent reports whether one of expressions is the identifier name.
func slicesContainsIdent(expressions []ast.Expr, name string) bool {
	for _, expression := range expressions {
		if isNamedIdent(expression, name) {
			return true
		}
	}
	return false
}
