package gate

import (
	"maps"
	"os"
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

// dependencyResolver uses the receipt input graph for lane reachability.
// Unknown owners and removed inputs retain the complete selection.
func (g *gateContext) dependencyResolver() (automationcheck.DependencyResolver, error) {
	graph, err := g.inputGraph()
	if err != nil {
		return nil, err
	}
	// The candidate alone cannot establish a removed input's former readers.
	for _, name := range g.paths {
		if _, err := os.Stat(filepath.Join(graph.root, filepath.FromSlash(name))); err != nil {
			return func(automationcheck.Ownership, string) bool { return true }, nil
		}
	}
	relative := map[string]string{}
	known := map[string]bool{}
	for _, node := range graph.nodes {
		if node.ForTest != "" || strings.HasSuffix(node.ImportPath, ".test") {
			continue
		}
		directory, err := filepath.Rel(graph.root, node.Dir)
		if err != nil || !filepath.IsLocal(directory) {
			continue
		}
		relative[node.ImportPath] = filepath.ToSlash(directory)
		known[filepath.ToSlash(directory)] = true
	}
	inputs := map[string][]string{}
	return func(ownership automationcheck.Ownership, changed string) bool {
		if !known[changed] {
			reach, attributed := graph.documentReach(ownership, changed, g.attributedDocuments, relative)
			if attributed {
				return reach
			}
			return true
		}
		matched := false
		for importPath, directory := range relative {
			if !ownership.OwnedPackage(directory) {
				continue
			}
			matched = true
			if directory == changed {
				return true
			}
			paths, found := inputs[importPath]
			if !found {
				files, err := graph.laneInputFiles(importPath)
				if err != nil {
					return true
				}
				for name := range files {
					if path, err := filepath.Rel(graph.root, name); err == nil && filepath.IsLocal(path) {
						paths = append(paths, filepath.ToSlash(path))
					}
				}
				inputs[importPath] = paths
			}
			for _, path := range paths {
				if strings.HasPrefix(path, changed+"/") {
					return true
				}
			}
		}
		return !matched
	}, nil
}
