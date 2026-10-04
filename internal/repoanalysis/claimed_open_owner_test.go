package repoanalysis

import (
	"go/ast"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gosyntax"
)

// hashingOpeners are the production files that still resolve a model through
// ResolveActiveGGUF, which hashes the whole weights file before anything is
// uploaded, with the reason each cannot take modelcli.OpenRunner yet. A
// model that the discovery memo remembers opens through OpenRunner without
// that first pass, and the open proves the remembered identity in the pass
// that uploads; a row that moves one of these there removes it from the list,
// and nothing adds to it.
var hashingOpeners = map[string]string{
	"internal/evaluation/transcription_projected.go": "checks the language model's identity against the plan before it loads the projector and only then opens",
	"internal/servingtest/active_gguf.go":            "the fixture helper that serves a test with an exact session and residency policy",
}

// TestModelOpensShareTheClaimedLoad holds the opening of a model to its owner:
// no production file outside internal/modelrecipe resolves a GGUF with
// ResolveActiveGGUF except the ones listed, so every evaluation, measurement
// and server open takes the load that proves a remembered identity while it
// uploads instead of hashing the file first.
func TestModelOpensShareTheClaimedLoad(t *testing.T) {
	t.Parallel()
	snapshot, err := gosyntax.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	var found []string
	for _, file := range snapshot.Files {
		if file.Test || filepath.ToSlash(filepath.Dir(file.Path)) == "internal/modelrecipe" {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			if selector, ok := node.(*ast.SelectorExpr); ok && selector.Sel.Name == "ResolveActiveGGUF" && !slices.Contains(found, file.Path) {
				found = append(found, file.Path)
			}
			return true
		})
	}
	slices.Sort(found)
	var listed []string
	for name := range hashingOpeners {
		listed = append(listed, name)
	}
	slices.Sort(listed)
	if !slices.Equal(found, listed) {
		t.Fatalf("production files resolving a model with ResolveActiveGGUF = %v, the list holds %v: open through modelcli.OpenRunner, which does not hash the file first", found, listed)
	}
}
