package gate

import (
	"errors"
	"maps"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/repoanalysis"
)

// harnessBoundaryCoverage reports, for the changed shared harness owners, the tested
// assembly packages that exercise them and any owner left without a boundary.
type harnessBoundaryCoverage struct {
	Changed    []string `json:"changed"`
	Boundaries []string `json:"boundaries"`
	Missing    []string `json:"missing,omitempty"`
}

var sharedHarnessOwners = map[string]bool{
	"internal/recipe":    true,
	"internal/agenttool": true,
	"internal/runrecord": true,
	"internal/plan":      true,
}

// agentHarnessBoundaryCoverage maps each changed shared harness owner to the
// tested packages that reach it and assemble more than one harness owner.
func agentHarnessBoundaryCoverage(snapshot repoanalysis.SourceSnapshot, changedPaths []string) (harnessBoundaryCoverage, error) {
	graph := map[string]map[string]bool{}
	tested := map[string]bool{}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return harnessBoundaryCoverage{}, err
		}
		if generated {
			continue
		}
		pkg := filepath.ToSlash(filepath.Dir(source.Path))
		if source.Test {
			tested[pkg] = true
			continue
		}
		parsed, err := source.Syntax()
		if err != nil {
			return harnessBoundaryCoverage{}, err
		}
		if graph[pkg] == nil {
			graph[pkg] = map[string]bool{}
		}
		for _, spec := range parsed.Imports {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return harnessBoundaryCoverage{}, err
			}
			if local := localPackagePath(path); local != "" {
				graph[pkg][local] = true
			}
		}
	}
	changedSet := map[string]bool{}
	for _, path := range changedPaths {
		pkg := filepath.ToSlash(filepath.Dir(path))
		if sharedHarnessOwners[pkg] {
			changedSet[pkg] = true
		}
	}
	coverage := harnessBoundaryCoverage{Changed: slices.Sorted(maps.Keys(changedSet))}
	boundaries, covered := map[string]bool{}, map[string]bool{}
	for candidate := range graph {
		if !tested[candidate] || len(coverage.Changed) == 0 {
			continue
		}
		owners := reachableHarnessOwners(graph, candidate)
		for _, changed := range coverage.Changed {
			if candidate == changed || !owners[changed] {
				continue
			}
			for other := range owners {
				if other != changed {
					boundaries[candidate], covered[changed] = true, true
					break
				}
			}
		}
	}
	for _, changed := range coverage.Changed {
		if !covered[changed] {
			coverage.Missing = append(coverage.Missing, changed)
		}
	}
	coverage.Boundaries = slices.Sorted(maps.Keys(boundaries))
	return coverage, nil
}

// requireAgentHarnessBoundaries fails when a changed owner has no tested
// assembly boundary or a boundary package is missing from the derived test scope.
func requireAgentHarnessBoundaries(coverage harnessBoundaryCoverage, selectedImports []string) error {
	if len(coverage.Missing) != 0 {
		return errors.New("automation ownership: changed agent contract has no tested assembly boundary: " + strings.Join(coverage.Missing, ","))
	}
	selected := map[string]bool{}
	for _, importPath := range selectedImports {
		if local := localPackagePath(importPath); local != "" {
			selected[local] = true
		}
	}
	for _, boundary := range coverage.Boundaries {
		if !selected[boundary] {
			return errors.New("automation ownership: assembled boundary omitted from derived test scope: " + boundary)
		}
	}
	return nil
}

// Collect shared owners once; cycles and repeated edges visit each node once.
func reachableHarnessOwners(graph map[string]map[string]bool, start string) map[string]bool {
	owners, seen := map[string]bool{}, map[string]bool{}
	pending := []string{start}
	for len(pending) != 0 {
		current := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if seen[current] {
			continue
		}
		seen[current] = true
		if sharedHarnessOwners[current] {
			owners[current] = true
		}
		pending = slices.AppendSeq(pending, maps.Keys(graph[current]))
	}
	return owners
}

func localPackagePath(importPath string) string {
	path := filepath.ToSlash(strings.TrimSpace(importPath))
	for _, root := range []string{"internal/", "cmd/"} {
		if strings.HasPrefix(path, root) {
			return path
		}
		if _, after, ok := strings.Cut(path, "/"+root); ok {
			return root + after
		}
	}
	return ""
}
