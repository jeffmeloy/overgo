package automationcheck

import (
	"errors"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/repoanalysis"
)

// BoundaryCoverage reports, for the changed shared harness owners, the tested
// assembly packages that exercise them and any owner left without a boundary.
type BoundaryCoverage struct {
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

// AgentHarnessBoundaryCoverage maps each changed shared harness owner to the
// tested packages that reach it and assemble more than one harness owner.
func AgentHarnessBoundaryCoverage(snapshot repoanalysis.SourceSnapshot, changedPaths []string) (BoundaryCoverage, error) {
	graph := map[string]map[string]bool{}
	tested := map[string]bool{}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return BoundaryCoverage{}, err
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
			return BoundaryCoverage{}, err
		}
		if graph[pkg] == nil {
			graph[pkg] = map[string]bool{}
		}
		for _, spec := range parsed.Imports {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return BoundaryCoverage{}, err
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
	coverage := BoundaryCoverage{Changed: sortedKeys(changedSet)}
	boundaries := map[string]bool{}
	for _, changed := range coverage.Changed {
		found := false
		for candidate := range graph {
			if !tested[candidate] || candidate == changed || !reaches(graph, candidate, changed) || !assemblesHarnessOwners(graph, candidate) {
				continue
			}
			boundaries[candidate] = true
			found = true
		}
		if !found {
			coverage.Missing = append(coverage.Missing, changed)
		}
	}
	coverage.Boundaries = sortedKeys(boundaries)
	return coverage, nil
}

// RequireAgentHarnessBoundaries fails when a changed owner has no tested
// assembly boundary or a boundary package is missing from the derived test scope.
func RequireAgentHarnessBoundaries(coverage BoundaryCoverage, selectedImports []string) error {
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

func reaches(graph map[string]map[string]bool, start, target string) bool {
	seen := map[string]bool{}
	var visit func(string) bool
	visit = func(current string) bool {
		if current == target {
			return true
		}
		if seen[current] {
			return false
		}
		seen[current] = true
		for next := range graph[current] {
			if visit(next) {
				return true
			}
		}
		return false
	}
	return visit(start)
}

func assemblesHarnessOwners(graph map[string]map[string]bool, start string) bool {
	seen := map[string]bool{}
	found := ""
	var visit func(string) bool
	visit = func(current string) bool {
		if seen[current] {
			return false
		}
		seen[current] = true
		if sharedHarnessOwners[current] {
			if found != "" && found != current {
				return true
			}
			found = current
		}
		for next := range graph[current] {
			if visit(next) {
				return true
			}
		}
		return false
	}
	return visit(start)
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

func sortedKeys(values map[string]bool) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}
