package inferencesurface

import (
	"go/scanner"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// formatPackages are the serving packages keyed by a format the model file
// declares beside its architecture: the tokenizer by tokenizer.ggml.model and
// its pre-tokenizer, the projector by clip.projector_type. Their names can
// coincide with an architecture's ("llama" names SentencePiece to the
// tokenizer), but they select a vocabulary or a vision tower, not the
// architecture.
var formatPackages = []string{"/internal/tokenizer", "/internal/projector"}

// ServingNames maps each string literal in the surface packages' non-test Go
// files, outside the format packages, to the files that hold it, named by
// import path. An architecture whose name is among them has serving code
// selected by that name rather than by what its profile declares.
func ServingNames(packages []Package) (map[string][]string, error) {
	named := map[string][]string{}
	for _, pkg := range packages {
		if slices.ContainsFunc(formatPackages, func(suffix string) bool { return strings.HasSuffix(pkg.ImportPath, suffix) }) {
			continue
		}
		files, err := filepath.Glob(filepath.Join(pkg.Dir, "*.go"))
		if err != nil {
			return nil, err
		}
		for _, file := range files {
			if strings.HasSuffix(file, "_test.go") {
				continue
			}
			source, err := os.ReadFile(file)
			if err != nil {
				return nil, err
			}
			var scan scanner.Scanner
			scan.Init(token.NewFileSet().AddFile(file, -1, len(source)), source, nil, 0)
			name := path.Join(pkg.ImportPath, filepath.Base(file))
			for _, kind, literal := scan.Scan(); kind != token.EOF; _, kind, literal = scan.Scan() {
				if value, err := strconv.Unquote(literal); kind == token.STRING && err == nil && !slices.Contains(named[value], name) {
					named[value] = append(named[value], name)
				}
			}
		}
	}
	return named, nil
}
