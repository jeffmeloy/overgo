// Package testscope derives test ownership from Go package metadata.
package testscope

import (
	"path/filepath"
	"runtime"
	"slices"
	"strings"
)

// Package binds compiler metadata and undeclared repository resources to a Go owner.
type Package struct {
	ImportPath      string
	Dir             string
	ProductionFiles []string
	TestEmbedFiles  []string
	XTestEmbedFiles []string
	// ResourceFiles are repository inputs without a compiler declaration.
	// Their owning package remains production-affected until independence is known.
	ResourceFiles []string
}

// DirectPackages returns packages directly owning changed source or resource
// files, and the subset whose production inputs changed. Metadata must include
// native compiler inputs and resolved test embeds from go list -test.
// A test source embedded in production remains a production input.
func DirectPackages(repo string, changed []string, packages []Package) (direct, production []string) {
	owners := map[string]bool{}
	for _, name := range changed {
		absolute := filepath.Join(repo, filepath.FromSlash(name))
		for _, pkg := range packages {
			if strings.EqualFold(filepath.Ext(name), ".go") && samePath(filepath.Dir(absolute), pkg.Dir) {
				owners[pkg.ImportPath] = owners[pkg.ImportPath] || !strings.HasSuffix(name, "_test.go")
			}
			for _, group := range []struct {
				files      []string
				production bool
			}{{pkg.ProductionFiles, true}, {pkg.TestEmbedFiles, false}, {pkg.XTestEmbedFiles, false}, {pkg.ResourceFiles, true}} {
				for _, embedded := range group.files {
					if samePath(absolute, filepath.Join(pkg.Dir, filepath.FromSlash(embedded))) {
						owners[pkg.ImportPath] = owners[pkg.ImportPath] || group.production
					}
				}
			}
		}
	}
	for owner, changedProduction := range owners {
		if owner != "" {
			direct = append(direct, owner)
			if changedProduction {
				production = append(production, owner)
			}
		}
	}
	slices.Sort(direct)
	slices.Sort(production)
	return direct, production
}

func samePath(a, b string) bool {
	a, b = filepath.Clean(a), filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}
