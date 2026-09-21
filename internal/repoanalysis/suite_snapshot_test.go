package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"testing"
)

// TestRepositorySnapshotIsSharedAcrossTheSuite holds this suite to computing
// the census of the live repository through its two shared censuses and
// nowhere else. Each costs seconds of type checking over the whole tree; the
// suite once computed four where it needs two, and a fifth arrives unnoticed
// with any test that builds its own. A test function that both reaches the
// live repository, by the parent path every test here reaches it by, and
// calls a census builder is refused; a test of a fixture tree it wrote itself
// names no parent path and is free to.
func TestRepositorySnapshotIsSharedAcrossTheSuite(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	builders := map[string]bool{"BuildModernGoCensus": true, "ModernGoCensusSnapshot": true}
	checked := 0
	for _, name := range files {
		syntax, err := parser.ParseFile(token.NewFileSet(), name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range syntax.Decls {
			function, isFunction := declaration.(*ast.FuncDecl)
			if !isFunction || function.Body == nil {
				continue
			}
			checked++
			builds, live := false, false
			ast.Inspect(function.Body, func(node ast.Node) bool {
				switch typed := node.(type) {
				case *ast.CallExpr:
					if callee, isIdent := typed.Fun.(*ast.Ident); isIdent && builders[callee.Name] {
						builds = true
					}
				case *ast.BasicLit:
					if value, err := strconv.Unquote(typed.Value); err == nil && value == ".." {
						live = true
					}
				}
				return true
			})
			if builds && live {
				t.Errorf("%s: %s computes a census of the live repository itself; read modernGoRepositoryCensus or modernGoSnapshotCensus", name, function.Name.Name)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no test function was inspected")
	}
}
