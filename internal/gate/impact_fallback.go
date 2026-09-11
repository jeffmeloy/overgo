package gate

import (
	"maps"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
)

// consumerBoundaryReason prefixes the external uncertainties the consumer
// census raises for method dispatch and command consumers.
const consumerBoundaryReason = "consumer census boundary: "

// reachabilityOnlyUncertainty reports whether every uncertainty concerns who
// can reach a changed symbol: dynamic dispatch, reflection, cgo and the
// consumer census boundaries, all bounded by the dependency closure with
// execution edges. A non-Go, build-selection, outside-snapshot, generated or
// analysis uncertainty keeps the complete plan.
func reachabilityOnlyUncertainty(impact codemanifest.Impact) bool {
	for _, item := range impact.Uncertainty {
		switch item.Kind {
		case codemanifest.UncertaintyInterface, codemanifest.UncertaintyReflection, codemanifest.UncertaintyCgo:
			continue
		case codemanifest.UncertaintyExternal:
			if strings.HasPrefix(item.Reason, consumerBoundaryReason) {
				continue
			}
			return false
		default:
			return false
		}
	}
	return len(impact.Uncertainty) != 0
}

// changedPackages lists the packages a manifest impact seeds from and the
// owners of its changed external inputs.
func changedPackages(impact codemanifest.Impact) []string {
	packages := map[string]bool{}
	for _, seed := range impact.Seeds {
		packages[seed.Package] = true
	}
	for _, change := range impact.ExternalInputs {
		if change.Base != nil {
			packages[change.Base.Owner] = true
		}
		if change.Candidate != nil {
			packages[change.Candidate.Owner] = true
		}
	}
	delete(packages, "")
	return slices.Sorted(maps.Keys(packages))
}

// dependencyResolver decides whether an owned package is, or transitively
// imports, a changed package over the live package graph, test imports
// included since the lanes run tests. A package whose import closure reads
// files or runs commands at run time may consume any candidate source, so
// it reaches every changed package until a declared runtime contract narrows
// that; a changed package the graph no longer holds resolves to run. Packages are
// named repo-relative.
func (g *gateContext) dependencyResolver() (automationcheck.DependencyResolver, error) {
	graph, err := g.inputGraph()
	if err != nil {
		return nil, err
	}
	imports := make(map[string][]string, len(graph.nodes))
	relative := make(map[string]string, len(graph.nodes))
	known := map[string]bool{}
	for _, node := range graph.nodes {
		// Recompiled test variants and the generated test mains (".test")
		// carry the harness's own os import, not the package's reach.
		if node.ForTest != "" || strings.HasSuffix(node.ImportPath, ".test") {
			continue
		}
		imports[node.ImportPath] = slices.Concat(node.Imports, node.TestImports, node.XTestImports)
		// Only packages under the repository root are repository packages;
		// the standard library and the module cache resolve outside it.
		directory, err := filepath.Rel(graph.root, node.Dir)
		if err != nil || directory == ".." || filepath.IsAbs(directory) || strings.HasPrefix(directory, ".."+string(filepath.Separator)) {
			continue
		}
		relative[node.ImportPath] = filepath.ToSlash(directory)
		known[filepath.ToSlash(directory)] = true
	}
	opaque := map[string]bool{}
	return func(ownership automationcheck.Ownership, changed string) bool {
		// A changed package the graph no longer holds (deleted or renamed)
		// cannot be resolved; the owned check runs.
		if !known[changed] {
			return true
		}
		for importPath, directory := range relative {
			if !ownership.OwnedPackage(directory) {
				continue
			}
			if reaches(importPath, changed, imports, relative, map[string]bool{}) {
				return true
			}
			// A test process that reads files or runs commands anywhere in
			// its import closure may consume any candidate source; the
			// package keeps the complete plan until a declared runtime
			// contract narrows what it reaches.
			if reachesOpaqueRuntime(importPath, imports, relative, opaque, map[string]bool{}) {
				return true
			}
		}
		return false
	}, nil
}

// opaqueRuntimeImports are the standard packages whose importers may read
// repository files or run repository commands at run time.
var opaqueRuntimeImports = []string{"os", "os/exec", "io/ioutil"}

// reachesOpaqueRuntime reports whether importPath or a repository package in
// its import closure imports an opaque runtime package itself; the standard
// library is not followed, since testing reaching os says nothing about the
// repository code under test. memo caches settled packages.
func reachesOpaqueRuntime(importPath string, imports map[string][]string, relative map[string]string, memo map[string]bool, visiting map[string]bool) bool {
	if settled, ok := memo[importPath]; ok {
		return settled
	}
	if visiting[importPath] {
		return false
	}
	visiting[importPath] = true
	result := false
	for _, dependency := range imports[importPath] {
		if slices.Contains(opaqueRuntimeImports, dependency) {
			result = true
			break
		}
		if _, repository := relative[dependency]; repository && reachesOpaqueRuntime(dependency, imports, relative, memo, visiting) {
			result = true
			break
		}
	}
	memo[importPath] = result
	return result
}

// reaches reports whether importPath is the changed package or transitively
// imports it.
func reaches(importPath, changed string, imports map[string][]string, relative map[string]string, seen map[string]bool) bool {
	if relative[importPath] == changed {
		return true
	}
	if seen[importPath] {
		return false
	}
	seen[importPath] = true
	return slices.ContainsFunc(imports[importPath], func(dependency string) bool {
		return reaches(dependency, changed, imports, relative, seen)
	})
}
