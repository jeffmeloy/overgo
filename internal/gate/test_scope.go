package gate

import (
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/testscope"
)

type packageTestScope struct {
	direct, dependent, productionPaths []string
	unresolved                         []string
	excluded                           int
}

func (g *gateContext) deriveTestScope() (packageTestScope, error) {
	graph, err := g.inputGraph()
	if err != nil {
		return packageTestScope{}, err
	}
	// Deleted inputs are absent from the candidate index. Retain their owner
	// conservatively so removing a resource cannot remove its acceptance too.
	graph.bindResourceFiles(g.paths)
	var roots []goPackageInput
	var packages []testscope.Package
	for _, node := range graph.nodes {
		// Match identifies real packages selected by ./...; test binaries have
		// no match, and ForTest identifies compiler-generated test variants.
		if len(node.Match) == 0 || node.ForTest != "" {
			continue
		}
		roots = append(roots, node)
		pkg := testscope.Package{
			ImportPath: node.ImportPath, Dir: node.Dir, ProductionFiles: node.productionFiles(),
			TestEmbedFiles: node.TestEmbedFiles, XTestEmbedFiles: node.XTestEmbedFiles,
		}
		for _, path := range graph.resourceFiles[node.Dir] {
			relative, err := filepath.Rel(node.Dir, path)
			if err != nil {
				return packageTestScope{}, err
			}
			pkg.ResourceFiles = append(pkg.ResourceFiles, relative)
		}
		packages = append(packages, pkg)
	}
	direct, production := testscope.DirectPackages(g.repo, g.paths, packages)
	scope := packageTestScope{direct: direct}
	for _, path := range g.paths {
		if path == "go.mod" || path == "go.sum" {
			scope.unresolved = append(scope.unresolved, path)
			continue
		}
		if strings.EqualFold(filepath.Ext(path), ".go") {
			owners, _ := testscope.DirectPackages(g.repo, []string{path}, packages)
			if len(owners) == 0 {
				scope.unresolved = append(scope.unresolved, path)
			}
		}
	}
	productionDirs := map[string]bool{}
	for _, node := range roots {
		if len(scope.unresolved) == 0 && !slices.Contains(production, node.ImportPath) {
			continue
		}
		productionDirs[node.Dir] = true
		// The boundary owner consumes package directories through source paths.
		// Include actual compiled sources so nested embedded assets reach the
		// same assembly checks as their owning production package.
		for _, name := range append(slices.Clone(node.GoFiles), node.CgoFiles...) {
			relative, err := filepath.Rel(g.repo, filepath.Join(node.Dir, name))
			if err != nil {
				return packageTestScope{}, err
			}
			scope.productionPaths = append(scope.productionPaths, filepath.ToSlash(relative))
		}
	}
	// A dependency recompiled for another package's tests has that other
	// package in ForTest. Seed by physical owner, then follow the compiler's
	// actual imports; do not turn test imports into production dependencies.
	affected := map[string]bool{}
	for _, node := range graph.nodes {
		if productionDirs[node.Dir] {
			affected[node.ImportPath] = true
		}
	}
	for changed := true; changed; {
		changed = false
		for _, node := range graph.nodes {
			if affected[node.ImportPath] {
				continue
			}
			for _, imported := range node.Imports {
				if affected[imported] {
					affected[node.ImportPath] = true
					changed = true
					break
				}
			}
		}
	}
	testTargets := map[string]bool{}
	for _, node := range graph.nodes {
		if affected[node.ImportPath] && node.ForTest != "" {
			testTargets[node.ForTest] = true
		}
	}
	for _, node := range roots {
		if slices.Contains(direct, node.ImportPath) {
			continue
		}
		if affected[node.ImportPath] || testTargets[node.ImportPath] {
			scope.dependent = append(scope.dependent, node.ImportPath)
		} else {
			scope.excluded++
		}
	}
	slices.Sort(scope.dependent)
	return scope, nil
}
