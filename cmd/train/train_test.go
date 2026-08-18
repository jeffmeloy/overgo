package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"testing"
)

func TestTrainCommandUsesNativeWorkflow(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "main.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	imports := map[string]bool{}
	for _, spec := range file.Imports {
		path, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			t.Fatal(err)
		}
		imports[path] = true
	}
	if !imports["overgo/internal/trainingworkflow"] || imports["overgo/internal/densecausal"] || imports["overgo/internal/trainingprogram"] {
		t.Fatalf("command imports=%v", imports)
	}
	declarations := 0
	for _, declaration := range file.Decls {
		if _, ok := declaration.(*ast.FuncDecl); ok {
			declarations++
		}
	}
	if declarations != 3 {
		t.Fatalf("command function count=%d", declarations)
	}
}
