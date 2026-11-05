package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// storeFlagNames are the flags that name an OvergoDB store.
var storeFlagNames = []string{"repo", "store"}

// storeFlagFindings names, for the command tree under root, each store flag with
// a default of its own and each package that defines a store flag and never
// resolves one. A relative default reads a store only from the directory that
// holds one, and a writable open of it creates an empty store anywhere else.
func storeFlagFindings(t *testing.T, root string) []string {
	t.Helper()
	var findings []string
	defined, resolved := map[string]bool{}, map[string]bool{}
	files := token.NewFileSet()
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() || filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
			return err
		}
		syntax, err := parser.ParseFile(files, path, nil, 0)
		if err != nil {
			return err
		}
		directory := filepath.Dir(path)
		ast.Inspect(syntax, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if owner, ok := selector.X.(*ast.Ident); ok && owner.Name == "dataroot" {
				resolved[directory] = true
			}
			// String takes name, default; StringVar takes the variable first.
			nameAt := -1
			switch selector.Sel.Name {
			case "String":
				nameAt = 0
			case "StringVar":
				nameAt = 1
			}
			if nameAt < 0 || len(call.Args) <= nameAt+1 {
				return true
			}
			name, ok := call.Args[nameAt].(*ast.BasicLit)
			if !ok || name.Kind != token.STRING {
				return true
			}
			text, err := strconv.Unquote(name.Value)
			if err != nil || !slices.Contains(storeFlagNames, text) {
				return true
			}
			defined[directory] = true
			if fallback, ok := call.Args[nameAt+1].(*ast.BasicLit); !ok || fallback.Value != `""` {
				findings = append(findings, filepath.ToSlash(path)+": -"+text+" has a default of its own")
			}
			return true
		})
		return nil
	})
	testcheck.NoError(t, err)
	for directory := range defined {
		if !resolved[directory] {
			findings = append(findings, filepath.ToSlash(directory)+" defines a store flag and never asks the data root")
		}
	}
	return findings
}

// TestEveryStoreFlagResolvesThroughTheDataRoot holds every command's store flag
// to an empty default that the data root resolves.
func TestEveryStoreFlagResolvesThroughTheDataRoot(t *testing.T) {
	t.Parallel()
	for _, finding := range storeFlagFindings(t, filepath.Join("..", "..", "cmd")) {
		t.Error(finding)
	}
}
