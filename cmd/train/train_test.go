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
	executesWorkflow := false
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		owner, ownerOK := selector.X.(*ast.Ident)
		if ownerOK && owner.Name == "trainingworkflow" && selector.Sel.Name == "Execute" {
			executesWorkflow = true
		}
		return true
	})
	if !executesWorkflow {
		t.Fatal("command does not execute the native training workflow")
	}
}
