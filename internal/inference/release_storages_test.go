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

// TestReleaseStoragesHasOneBody holds the package's production code to one loop that releases a failed step's cache
// storages, releaseStorages, so each failure path that must give them back calls it rather than writing the loop
// again.
func TestReleaseStoragesHasOneBody(t *testing.T) {
	t.Parallel()
	var owners []string
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
			ast.Inspect(function.Body, func(node ast.Node) bool {
				loop, isRange := node.(*ast.RangeStmt)
				if !isRange || !isNamedIdent(loop.X, "storages") {
					return true
				}
				ast.Inspect(loop.Body, func(inner ast.Node) bool {
					if call, isCall := inner.(*ast.CallExpr); isCall {
						if selector, isSelector := call.Fun.(*ast.SelectorExpr); isSelector && selector.Sel.Name == "release" {
							owners = append(owners, name+": "+function.Name.Name)
							return false
						}
					}
					return true
				})
				return true
			})
		}
	}
	if len(owners) != 1 || !strings.HasSuffix(owners[0], ": releaseStorages") {
		t.Fatalf("the loops that release cache storages stand in %q, want releaseStorages alone", owners)
	}
}

func isNamedIdent(expression ast.Expr, name string) bool {
	ident, isIdent := expression.(*ast.Ident)
	return isIdent && ident.Name == name
}
