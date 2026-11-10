package server

import (
	"go/ast"
	"go/parser"
	"go/token"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// TestEveryJSONRouteRefusesABodyPastItsLimit holds every handler that reads a JSON body to the bounded decoder
// (decodeJSONBounded, its 1 MiB limit and its media-type refusal): five handlers read theirs whole through
// strictjson.Decode (serving-bounds/body-limit, 2026-11-09). No function of the package's source reads a request's body
// through strictjson.Decode or io.ReadAll but the bounded decoder itself, so a handler added later is held by the
// same reading; and a body one byte past the limit, sent to the composition and hub routes that read theirs whole, is
// refused as too large.
func TestEveryJSONRouteRefusesABodyPastItsLimit(t *testing.T) {
	t.Parallel()
	sources := testcheck.Must(filepath.Glob("*.go"))(t)
	files := token.NewFileSet()
	for _, source := range sources {
		if strings.HasSuffix(source, "_test.go") {
			continue
		}
		parsed := testcheck.Must(parser.ParseFile(files, source, nil, 0))(t)
		for _, declaration := range parsed.Decls {
			function, isFunction := declaration.(*ast.FuncDecl)
			if !isFunction || function.Body == nil || function.Name.Name == "decodeJSONBounded" {
				continue
			}
			ast.Inspect(function.Body, func(node ast.Node) bool {
				call, isCall := node.(*ast.CallExpr)
				if !isCall || len(call.Args) == 0 {
					return true
				}
				reader, readsWhole := call.Fun.(*ast.SelectorExpr)
				body, ofBody := call.Args[0].(*ast.SelectorExpr)
				if !readsWhole || !ofBody || body.Sel.Name != "Body" {
					return true
				}
				if owner, named := reader.X.(*ast.Ident); named && (owner.Name == "strictjson" && reader.Sel.Name == "Decode" || owner.Name == "io" && reader.Sel.Name == "ReadAll") {
					t.Errorf("%s: %s reads a request's body whole through %s.%s, past the bounded decoder", files.Position(call.Pos()), function.Name.Name, owner.Name, reader.Sel.Name)
				}
				return true
			})
		}
	}

	repository := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	t.Cleanup(func() { testcheck.NoError(t, repository.Close()) })
	handler := newTestHandlerForRepository(t, repository, &fakeGenerator{}, func(config *Config) { config.HubDownloadRoot = t.TempDir() })
	defer handler.Close()
	past := `{"x":"` + strings.Repeat("a", maxRequestBytes) + `"}`
	for _, path := range []string{"/compositions/generate", "/compositions/activate", "/hub/downloads"} {
		answer := serveTestRequest(handler, http.MethodPost, path, past)
		if answer.Code != http.StatusBadRequest || !strings.Contains(answer.Body.String(), "too large") {
			t.Errorf("POST %s with a body past %d bytes answered %d: %.200s; want it refused as too large", path, maxRequestBytes, answer.Code, answer.Body)
		}
	}
}
