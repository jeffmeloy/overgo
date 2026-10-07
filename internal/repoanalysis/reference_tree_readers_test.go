package repoanalysis

import (
	"go/ast"
	"go/token"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// referenceTree is the directory name of the tree overgo was ported from,
// kept apart from any separator so that this file names no path into it.
const referenceTree = "adaptive_new"

// namesReferenceTree reports whether a string names a path into the reference
// tree: the tree's directory followed by a path separator. The tree's name in
// prose, a repository label or a file name is not a path into it.
func namesReferenceTree(value string) bool {
	return strings.Contains(value, referenceTree+"/") || strings.Contains(value, referenceTree+`\`)
}

// TestNoTestReadsAdaptiveNew holds overgo's tests and commands to their own
// data: no Go file under internal or cmd names a path into the reference tree
// in a string. A model is found through the data roots, where this machine's
// configuration says which home holds it; a golden a test compares against is
// a fixture in git, or, when git is too small for it, a file among the
// references beside the store. A test that hard-codes a path into the
// reference tree runs only on the machine that has the tree, and a command
// that defaults to one reads another project's home without being told to.
func TestNoTestReadsAdaptiveNew(t *testing.T) {
	t.Parallel()
	for value, want := range map[string]bool{
		`C:\Users\someone\` + referenceTree + `\models\A-Model`: true,
		"C:/Users/someone/" + referenceTree + "/fixtures":       true,
		"local/" + referenceTree:                                false,
		"docs/" + referenceTree + "_parity_report.md":           false,
		referenceTree + " is the reference tree":                false,
	} {
		if namesReferenceTree(value) != want {
			t.Fatalf("namesReferenceTree(%q) = %v, want %v", value, !want, want)
		}
	}
	snapshot, err := gosyntax.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	var found []string
	for _, file := range snapshot.Files {
		syntax, err := file.Syntax()
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			literal, ok := node.(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING || slices.Contains(found, file.Path) {
				return true
			}
			if value, err := strconv.Unquote(literal.Value); err == nil && namesReferenceTree(value) {
				found = append(found, file.Path)
			}
			return true
		})
	}
	if len(found) > 0 {
		slices.Sort(found)
		t.Fatalf("Go files naming a path into the reference tree: %v: find a model through the data roots (testutil.ModelArtifactDir, modeltest.Directory), keep a golden in fixtures, and one too large for git among the references beside the store", found)
	}
}
