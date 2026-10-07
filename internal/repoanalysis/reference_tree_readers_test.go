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

// referenceTreeReaders are the Go files that still name a path into the
// reference tree, each with what it reads there and what takes it off the
// list. A model overgo's own models directory holds is found through the data
// roots, and a golden a test compares against is a fixture in git; a row that
// brings one of these home removes its file from the list, and nothing adds
// to it.
var referenceTreeReaders = map[string]string{
	"cmd/validate/parity_sensenova.go":                             "flag defaults for the SenseNova model and the reference fixtures its parity is checked against; the fixtures live outside git",
	"cmd/validate/parity_vqa.go":                                   "flag defaults for the RxBrain model, its reference fixtures and demo image; the fixtures live outside git",
	"cmd/validate/training_proof.go":                               "the fixtures directory the mixture-of-transformers training route reads its reference batches from, which live outside git",
	"internal/latentimage/runtime_leadership_cuda_windows_test.go": "the picture the reference tree's image lane rendered, compared against as a golden that lives outside git",
	"internal/latentimage/spec_test.go":                            "the Krea-2-Turbo directory and a census of the models root, named outright where the data roots would find them",
	"internal/routedlm/flowterminal_test.go":                       "the SenseNova generation oracle, a reference fixture that lives outside git",
	"internal/routedlm/rope_test.go":                               "the SenseNova model directory, named outright where the data roots would find it",
	"internal/sensenovarecipe/seededlatent_cuda_windows_test.go":   "the SenseNova model directory, named outright where the data roots would find it",
}

// namesReferenceTree reports whether a string names a path into the reference
// tree: the tree's directory followed by a path separator. The tree's name in
// prose, a repository label or a file name is not a path into it.
func namesReferenceTree(value string) bool {
	return strings.Contains(value, referenceTree+"/") || strings.Contains(value, referenceTree+`\`)
}

// TestReferenceTreeReadersAreListed holds overgo's tests and commands to their
// own data: no Go file under internal or cmd names a path into the reference
// tree in a string, except the ones listed with what still keeps them there.
// A test that hard-codes such a path runs only on the machine that has the
// tree, and reads a model or golden that overgo's own roots and fixtures are
// meant to hold.
func TestReferenceTreeReadersAreListed(t *testing.T) {
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
	slices.Sort(found)
	var listed []string
	for name := range referenceTreeReaders {
		listed = append(listed, name)
	}
	slices.Sort(listed)
	if !slices.Equal(found, listed) {
		t.Fatalf("Go files naming a path into the reference tree = %v, the list holds %v: find a model through the data roots (testutil.ModelArtifactDir, modeltest.Directory) and keep a golden in fixtures", found, listed)
	}
}
