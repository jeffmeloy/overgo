// Package testvocab guards canonical vocabularies in tests by parsing Go
// source; it lives apart from testutil so the packages that only compare
// numbers inherit no Go-source reach from their test helpers.
package testvocab

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// ForbidVocabularyRedefinition scans one package directory for
// string-typed constant declarations outside the owning package that
// restate any of the given vocabulary values, so a canonical
// vocabulary cannot quietly fork per subsystem. The caller passes its
// own package directory; allowed names exempt exact constant names
// (mappings that intentionally share wire values).
func ForbidVocabularyRedefinition[T ~string](t testing.TB, packageDir string, vocabulary []T, allowed ...string) {
	t.Helper()
	values := map[string]bool{}
	for _, value := range vocabulary {
		values[string(value)] = true
	}
	exempt := map[string]bool{}
	for _, name := range allowed {
		exempt[name] = true
	}
	entries, err := os.ReadDir(packageDir)
	if err != nil {
		t.Fatal(err)
	}
	fileSet := token.NewFileSet()
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		syntax, err := parser.ParseFile(fileSet, filepath.Join(packageDir, name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			spec, ok := node.(*ast.ValueSpec)
			if !ok {
				return true
			}
			for index, value := range spec.Values {
				literal, ok := value.(*ast.BasicLit)
				if !ok || literal.Kind != token.STRING {
					continue
				}
				unquoted, err := strconv.Unquote(literal.Value)
				if err != nil || !values[unquoted] || index >= len(spec.Names) || exempt[spec.Names[index].Name] {
					continue
				}
				t.Errorf("%s redefines canonical vocabulary value %q as %s; use the owning type", name, unquoted, spec.Names[index].Name)
			}
			return true
		})
	}
}
