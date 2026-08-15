// Package testscope derives test ownership from Go package metadata.
package testscope

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// Package is the subset of `go list -json` metadata needed to map changed
// source and embedded asset files to their compiler-recognized owners.
type Package struct {
	ImportPath      string
	Dir             string
	EmbedFiles      []string
	TestEmbedFiles  []string
	XTestEmbedFiles []string
}

// DecodePackages decodes the concatenated JSON objects emitted by go list.
func DecodePackages(r io.Reader) ([]Package, error) {
	decoder := json.NewDecoder(r)
	var packages []Package
	for {
		var pkg Package
		if err := decoder.Decode(&pkg); err != nil {
			if err == io.EOF {
				return packages, nil
			}
			return nil, fmt.Errorf("decode go list package: %w", err)
		}
		packages = append(packages, pkg)
	}
}

// DirectPackages returns packages directly owning changed Go files or changed
// files selected by a go:embed directive. EmbedFiles is resolved by go list,
// so this policy follows compiler metadata instead of a parallel path table.
func DirectPackages(repo string, changed []string, packages []Package) []string {
	owners := map[string]bool{}
	for _, name := range changed {
		absolute := filepath.Join(repo, filepath.FromSlash(name))
		for _, pkg := range packages {
			if strings.EqualFold(filepath.Ext(name), ".go") && samePath(filepath.Dir(absolute), pkg.Dir) {
				owners[pkg.ImportPath] = true
				continue
			}
			for _, embedded := range packageEmbedFiles(pkg) {
				if samePath(absolute, filepath.Join(pkg.Dir, filepath.FromSlash(embedded))) {
					owners[pkg.ImportPath] = true
					break
				}
			}
		}
	}
	out := make([]string, 0, len(owners))
	for owner := range owners {
		if owner != "" {
			out = append(out, owner)
		}
	}
	sort.Strings(out)
	return out
}

func packageEmbedFiles(pkg Package) []string {
	files := make([]string, 0, len(pkg.EmbedFiles)+len(pkg.TestEmbedFiles)+len(pkg.XTestEmbedFiles))
	files = append(files, pkg.EmbedFiles...)
	files = append(files, pkg.TestEmbedFiles...)
	return append(files, pkg.XTestEmbedFiles...)
}

func samePath(a, b string) bool {
	a, b = filepath.Clean(a), filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}
