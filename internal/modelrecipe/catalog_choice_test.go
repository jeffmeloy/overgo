package modelrecipe

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/workflowrecipe"
)

// TestOneCatalogForEachTask holds the package to one choice of the catalog a recipe's modules come from,
// definitionCatalog: a projection or a training recipe takes the workflow's catalog, a representation bridge and every
// other task this package's own, inference among them; and no other function of the package's production code names
// the workflow's catalog, so no second choice by task can stand beside it.
func TestOneCatalogForEachTask(t *testing.T) {
	t.Parallel()
	bridge := []recipe.Node{{ID: "capture", Module: ModuleCaptureRepresentation}}
	for _, choice := range []struct {
		task     recipe.Task
		nodes    []recipe.Node
		workflow bool
	}{
		{recipe.TaskInference, nil, false},
		{recipe.TaskGeneration, nil, false},
		{recipe.TaskImageGen, nil, false},
		{recipe.TaskTranscription, nil, false},
		{recipe.TaskProjection, nil, true},
		{recipe.TaskTraining, nil, true},
		{recipe.TaskProjection, bridge, false},
	} {
		want := catalog
		if choice.workflow {
			want = workflowrecipe.Catalog()
		}
		if got := definitionCatalog(recipe.Definition{Task: choice.task, Nodes: choice.nodes}); got != want {
			t.Errorf("a %s recipe (bridge %t) takes the other catalog", choice.task, choice.nodes != nil)
		}
	}
	for _, name := range testcheck.Must(filepath.Glob("*.go"))(t) {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		parsed := testcheck.Must(parser.ParseFile(token.NewFileSet(), name, testcheck.Must(os.ReadFile(name))(t), parser.SkipObjectResolution))(t)
		for _, declaration := range parsed.Decls {
			function, isFunction := declaration.(*ast.FuncDecl)
			if !isFunction || function.Name.Name == "definitionCatalog" {
				continue
			}
			ast.Inspect(function, func(node ast.Node) bool {
				if selector, isSelector := node.(*ast.SelectorExpr); isSelector && selector.Sel.Name == "Catalog" {
					if owner, isIdent := selector.X.(*ast.Ident); isIdent && owner.Name == "workflowrecipe" {
						t.Errorf("%s: %s names the workflow's catalog; choose it through definitionCatalog", name, function.Name.Name)
					}
				}
				return true
			})
		}
	}
}
