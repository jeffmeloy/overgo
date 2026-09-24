package server

import (
	"go/ast"
	"go/parser"
	"go/token"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
)

// TestRouteTableIsTheOnlyDispatcher holds the route table to being the one
// place a path or a method chooses a handler: no server handler switches on
// the request path, a method check survives only where the idle shell
// dispatches for itself, and each workspace precondition answers the same
// way on every route it guards.
func TestRouteTableIsTheOnlyDispatcher(t *testing.T) {
	t.Parallel()
	sources, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	// The idle shell serves without a route table: it dispatches its own paths and methods.
	idleDispatch := map[string]bool{"idle_shell.go": true, "library_workspace.go": true, "http_helpers.go": true}
	files := token.NewFileSet()
	for _, source := range sources {
		if strings.HasSuffix(source, "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(files, source, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		methodChecks := 0
		ast.Inspect(parsed, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.SwitchStmt:
				if selector, ok := node.Tag.(*ast.SelectorExpr); ok && selector.Sel.Name == "Path" && source != "idle_shell.go" {
					t.Errorf("%s re-dispatches on the request path at %s", source, files.Position(node.Pos()))
				}
			case *ast.CallExpr:
				if name, ok := node.Fun.(*ast.Ident); ok && name.Name == "requireMethod" {
					methodChecks++
				}
			}
			return true
		})
		if methodChecks > 0 && !idleDispatch[source] {
			t.Errorf("%s checks a request method %d times; the route table owns methods", source, methodChecks)
		}
	}

	// Without the agent runtime every agent route answers alike; without a
	// workspace every peer or automation route answers what is missing.
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	for _, route := range routeCatalog {
		segment, _, _ := strings.Cut(strings.TrimPrefix(route.Path, "/"), "/")
		want := map[string]int{"agents": http.StatusServiceUnavailable, "peers": http.StatusNotImplemented, "automations": http.StatusNotImplemented}[segment]
		if want == 0 || route.Path == "/automations/webhook" {
			continue
		}
		answer := serveTestRequest(handler, route.Methods[0], route.Path, "{}")
		if answer.Code != want {
			t.Errorf("%s %s without its workspace answered %d, want %d: %s", route.Methods[0], route.Path, answer.Code, want, answer.Body)
		}
	}
}
