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
// included since the lanes run tests. A package that imports os/exec may run
// any repository command, so it also reaches every package some command
// reaches until a declared execution contract narrows that. Packages are
// named repo-relative.
func (g *gateContext) dependencyResolver() (automationcheck.DependencyResolver, error) {
	graph, err := g.inputGraph()
	if err != nil {
		return nil, err
	}
	imports := make(map[string][]string, len(graph.nodes))
	relative := make(map[string]string, len(graph.nodes))
	executes := map[string]bool{}
	var commands []string
	for _, node := range graph.nodes {
		if node.ForTest != "" {
			continue
		}
		edges := slices.Concat(node.Imports, node.TestImports, node.XTestImports)
		imports[node.ImportPath] = edges
		if directory, err := filepath.Rel(graph.root, node.Dir); err == nil {
			relative[node.ImportPath] = filepath.ToSlash(directory)
		}
		if slices.Contains(edges, "os/exec") {
			executes[node.ImportPath] = true
		}
		if strings.HasPrefix(relative[node.ImportPath], "cmd/") {
			commands = append(commands, node.ImportPath)
		}
	}
	commandReaches := func(changed string) bool {
		return slices.ContainsFunc(commands, func(command string) bool {
			return reaches(command, changed, imports, relative, map[string]bool{})
		})
	}
	return func(ownership automationcheck.Ownership, changed string) bool {
		for importPath, directory := range relative {
			if !ownership.OwnedPackage(directory) {
				continue
			}
			if reaches(importPath, changed, imports, relative, map[string]bool{}) {
				return true
			}
			if executes[importPath] && commandReaches(changed) {
				return true
			}
		}
		return false
	}, nil
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
