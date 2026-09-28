package testutil

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"testing"
)

// TestSuitesStayOutOfTheRepositoryDirectory holds the suites seen re-running
// on landings that changed none of their inputs to changing directory only
// into a temporary directory. t.Chdir restores the old directory through a
// file handle the test log does not record, so go test's cache attributes
// every later relative read of the package to the directory it entered;
// entered at the repository, internal/server's own listing of its test files
// was hashed as the repository's, whose .git moves on every commit, and its
// suite re-ran on every landing. Measured 2026-09-27: 20-29 s a landing.
func TestSuitesStayOutOfTheRepositoryDirectory(t *testing.T) {
	t.Parallel()
	for _, directory := range []string{"../server", "../../cmd/validate"} {
		files, err := filepath.Glob(filepath.Join(directory, "*_test.go"))
		if err != nil || len(files) == 0 {
			t.Fatalf("no test files in %s: %v", directory, err)
		}
		for _, file := range files {
			fileSet := token.NewFileSet()
			syntax, err := parser.ParseFile(fileSet, file, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, position := range chdirsOutsideTemporary(syntax) {
				t.Errorf("%s: a directory change leaves the test's temporary directories; pass the path to the code under test instead", fileSet.Position(position))
			}
		}
	}
}

// chdirsOutsideTemporary are the directory changes in a file whose target
// is neither t.TempDir() nor a name its function assigns from t.TempDir().
func chdirsOutsideTemporary(syntax *ast.File) []token.Pos {
	var outside []token.Pos
	for _, declaration := range syntax.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Body == nil {
			continue
		}
		temporary := map[string]bool{}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			if assignment, ok := node.(*ast.AssignStmt); ok && len(assignment.Lhs) == len(assignment.Rhs) {
				for index, value := range assignment.Rhs {
					if name, ok := assignment.Lhs[index].(*ast.Ident); ok && types.ExprString(value) == "t.TempDir()" {
						temporary[name.Name] = true
					}
				}
			}
			call, ok := node.(*ast.CallExpr)
			if !ok || len(call.Args) != 1 {
				return true
			}
			if selector, ok := call.Fun.(*ast.SelectorExpr); !ok || selector.Sel.Name != "Chdir" {
				return true
			}
			target := types.ExprString(call.Args[0])
			if target != "t.TempDir()" && !temporary[target] {
				outside = append(outside, call.Pos())
			}
			return true
		})
	}
	return outside
}
