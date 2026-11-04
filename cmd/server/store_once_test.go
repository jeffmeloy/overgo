package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// TestTheServerOpensItsStoreOnce holds a server start to one read-only open of the store: the server's own files call
// overgodb.OpenReadOnly once, for the remote, transcription, projector and draft resolutions alike, and open the runner
// over that store (ModelFlags.OpenRunnerFromStore) rather than through an open of its own. Each resolution opening its
// own cost a start four opens, 0.8 s each on this machine's store.
func TestTheServerOpensItsStoreOnce(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	calls := map[string]int{}
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if call, ok := node.(*ast.CallExpr); ok {
				if selector, ok := call.Fun.(*ast.SelectorExpr); ok {
					calls[selector.Sel.Name]++
				}
			}
			return true
		})
	}
	if calls["OpenReadOnly"] != 1 || calls["OpenRunnerFromStore"] != 1 || calls["OpenRunnerWithOptions"] != 0 || calls["OpenRunner"] != 0 {
		t.Fatalf("the server opens the store read-only %d times and its runner %d times over it, %d and %d times through opens of their own; want one open, shared",
			calls["OpenReadOnly"], calls["OpenRunnerFromStore"], calls["OpenRunnerWithOptions"], calls["OpenRunner"])
	}
}
