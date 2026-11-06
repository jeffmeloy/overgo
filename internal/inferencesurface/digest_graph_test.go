package inferencesurface

import (
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// graphReaders are the code identities and what each reads the package graph through: the inference surface's digest
// and execution identities, the cycle's trigger and the lanes' reach.
var graphReaders = map[string]string{
	"internal/inferencesurface": "ListGraph",
	"cmd/loop":                  "ListGraph",
	"internal/landing":          "HostBuildClosure",
}

// TestEveryCodeIdentityReadsOneDigestGraph holds the code identities to one package graph, the one gosource.ListGraph
// lists: the production Go of the module asks the go tool for a dependency closure, -deps, only in internal/gosource,
// and the inference surface, the cycle's trigger and the lanes' reach each read the graph through it, so no identity
// walks a closure of its own with its own roots, pruning and file rules.
func TestEveryCodeIdentityReadsOneDigestGraph(t *testing.T) {
	t.Parallel()
	root := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	reads := map[string]bool{}
	for _, tree := range []string{"cmd", "internal"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(root, tree), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return err
			}
			directory := filepath.ToSlash(testcheck.Must(filepath.Rel(root, filepath.Dir(path)))(t))
			source := testcheck.Must(os.ReadFile(path))(t)
			var words scanner.Scanner
			words.Init(token.NewFileSet().AddFile(path, -1, len(source)), source, nil, 0)
			for _, kind, literal := words.Scan(); kind != token.EOF; _, kind, literal = words.Scan() {
				if kind == token.IDENT && literal == graphReaders[directory] {
					reads[directory] = true
				}
				if text, _ := strconv.Unquote(literal); kind == token.STRING && text == "-deps" && directory != "internal/gosource" {
					t.Errorf("%s asks the go tool for a closure of its own; read gosource.ListGraph", filepath.ToSlash(testcheck.Must(filepath.Rel(root, path))(t)))
				}
			}
			return nil
		}))
	}
	for directory, reader := range graphReaders {
		if !reads[directory] {
			t.Errorf("%s does not read the package graph through gosource.%s", directory, reader)
		}
	}
}
