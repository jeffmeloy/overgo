package gate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestServerRouteTableIsTheOnlyDispatcher holds the server's route table to
// being the one place a path or a method chooses a handler: no server handler
// switches on the request path, and a method check survives only where the
// idle shell dispatches for itself (it serves without a route table). The
// server's TestRouteTableIsTheOnlyDispatcher holds each workspace
// precondition's answer on every route it guards.
func TestServerRouteTableIsTheOnlyDispatcher(t *testing.T) {
	t.Parallel()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "internal", "server")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	idleDispatch := map[string]bool{"idle_shell.go": true, "library_workspace.go": true, "http_helpers.go": true}
	files := token.NewFileSet()
	switches := 0
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(files, filepath.Join(dir, name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		methodChecks := 0
		ast.Inspect(parsed, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.SwitchStmt:
				if selector, ok := node.Tag.(*ast.SelectorExpr); ok && selector.Sel.Name == "Path" {
					switches++
					if name != "idle_shell.go" {
						t.Errorf("%s re-dispatches on the request path at %s", name, files.Position(node.Pos()))
					}
				}
			case *ast.CallExpr:
				if ident, ok := node.Fun.(*ast.Ident); ok && ident.Name == "requireMethod" {
					methodChecks++
				}
			}
			return true
		})
		if methodChecks > 0 && !idleDispatch[name] {
			t.Errorf("%s checks a request method %d times; the route table owns methods", name, methodChecks)
		}
	}
	// The idle shell's own dispatch is the one path switch, so the census reads real source.
	if switches != 1 {
		t.Errorf("found %d path switches, want the idle shell's one", switches)
	}
}
