package gate

import (
	"maps"
	"path/filepath"
	"slices"

	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
)

// reachabilityOnlyUncertainty: every uncertainty concerns symbol reachability
// (dynamic dispatch, reflection, cgo), which the dependency closure bounds;
// a non-Go, build-selection or outside-snapshot uncertainty keeps the
// complete plan.
func reachabilityOnlyUncertainty(impact codemanifest.Impact) bool {
	for _, item := range impact.Uncertainty {
		switch item.Kind {
		case codemanifest.UncertaintyInterface, codemanifest.UncertaintyReflection, codemanifest.UncertaintyCgo:
			continue
		default:
			return false
		}
	}
	return len(impact.Uncertainty) != 0
}

// changedPackages: the packages a manifest impact seeds from, and the
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

// dependencyResolver: owned packages are, or transitively import, a changed
// package; edges come from the live package input graph, test imports
// included since the lanes run tests. Packages are named repo-relative.
func (g *gateContext) dependencyResolver() (automationcheck.DependencyResolver, error) {
	graph, err := g.inputGraph()
	if err != nil {
		return nil, err
	}
	imports := make(map[string][]string, len(graph.nodes))
	relative := make(map[string]string, len(graph.nodes))
	for _, node := range graph.nodes {
		if node.ForTest != "" {
			continue
		}
		imports[node.ImportPath] = slices.Concat(node.Imports, node.TestImports, node.XTestImports)
		if directory, err := filepath.Rel(graph.root, node.Dir); err == nil {
			relative[node.ImportPath] = filepath.ToSlash(directory)
		}
	}
	return func(ownership automationcheck.Ownership, changed string) bool {
		for importPath, directory := range relative {
			if ownership.OwnedPackage(directory) && reaches(importPath, changed, imports, relative, map[string]bool{}) {
				return true
			}
		}
		return false
	}, nil
}

// reaches: importPath is the changed package or transitively imports it.
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
