package composition

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNoDirectCompositionConstructors(t *testing.T) {
	directory := filepath.Join("..", "inference")
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	protected := map[string]map[string]struct{}{
		"ProductionComposition": {
			"OpenProductionComposition": {},
		},
		"ExternalCrossAttentionProgram": {
			"OpenExternalCrossAttention": {},
		},
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		path := filepath.Join(directory, name)
		file, parseErr := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		for _, declaration := range file.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || !function.Name.IsExported() || function.Type.Results == nil {
				continue
			}
			for _, result := range function.Type.Results.List {
				resultName := compositionResultType(result.Type)
				allowed, guarded := protected[resultName]
				if !guarded {
					continue
				}
				if _, ok := allowed[function.Name.Name]; !ok {
					t.Errorf("%s exports direct %s constructor %s", path, resultName, function.Name.Name)
				}
			}
		}
	}
}

func compositionResultType(expression ast.Expr) string {
	switch value := expression.(type) {
	case *ast.Ident:
		return value.Name
	case *ast.StarExpr:
		return compositionResultType(value.X)
	case *ast.SelectorExpr:
		return value.Sel.Name
	default:
		return ""
	}
}
