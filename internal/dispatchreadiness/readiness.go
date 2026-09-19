// Package dispatchreadiness resolves the cheap, read-only admission facts that
// precede any store replay, completion-authority resolution, candidate
// preparation, analysis, or model load. The gate, preflight, and plan prompt
// project one typed snapshot instead of each recomputing branch, HEAD, changed
// and staged scope, data-root resolution, and the dirty verification inputs a
// declared scope does not cover. It carries facts, never refusals: each consumer
// decides what blocks, what is advisory, and what is unresolved.
package dispatchreadiness

import (
	"context"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
)

// Readiness is the cheap admission projection resolved before expensive work.
type Readiness struct {
	Branch      string
	Head        string
	DataRoot    string
	RootSource  string
	Changed     []string
	Staged      []string
	Untracked   []string
	Provisional bool
	OutOfScope  []string
}

// Resolve gathers the cheap facts for root against a declared repo-relative
// write scope. It runs only git plain-text queries and the data-root
// resolution: it opens no store and loads no model. An empty declaredScope
// marks the scope provisional rather than inventing a refusal.
func Resolve(root string, declaredScope []string) (Readiness, error) {
	readiness := Readiness{Provisional: len(declaredScope) == 0}
	branch, err := gitLine(root, "rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return Readiness{}, err
	}
	head, err := gitLine(root, "rev-parse", "--verify", "HEAD")
	if err != nil {
		return Readiness{}, err
	}
	readiness.Branch, readiness.Head = branch, head
	status, err := gitOutput(root, "status", "--porcelain=v1", "--untracked-files=all")
	if err != nil {
		return Readiness{}, err
	}
	readiness.Changed, readiness.Staged, readiness.Untracked = parseStatus(status)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		return Readiness{}, err
	}
	readiness.DataRoot, readiness.RootSource = roots.Store, roots.Source
	readiness.OutOfScope = outsideScope(declaredScope, readiness.Changed, readiness.Staged, readiness.Untracked)
	return readiness, nil
}

// String renders one line naming the branch and HEAD prefix, whether the scope
// is provisional or declared, the dirty verification inputs outside that scope,
// and the resolved data root, so a driver reads readiness at a glance.
func (r Readiness) String() string {
	scope := "declared"
	if r.Provisional {
		scope = "provisional"
	}
	head := r.Head
	if len(head) > headPrefix {
		head = head[:headPrefix]
	}
	fields := []string{
		"branch=" + r.Branch,
		"head=" + head,
		"scope=" + scope,
		fmt.Sprintf("changed=%d staged=%d untracked=%d", len(r.Changed), len(r.Staged), len(r.Untracked)),
		"data-root=" + r.DataRoot,
	}
	if len(r.OutOfScope) != 0 {
		fields = append(fields, "dirty-outside-scope="+strings.Join(r.OutOfScope, ","))
	}
	return strings.Join(fields, " ")
}

// headPrefix bounds the rendered HEAD to a readable short commit.
const headPrefix = 12

func gitLine(root string, args ...string) (string, error) {
	out, err := gitOutput(root, args...)
	return strings.TrimSpace(out), err
}

func gitOutput(root string, args ...string) (string, error) {
	out, err := gitauthority.Query(context.Background(), root, args...)
	if err != nil {
		return "", fmt.Errorf("dispatch readiness: %w", err)
	}
	return string(out), nil
}

// parseStatus splits a porcelain v1 listing into worktree-changed, index-staged,
// and untracked repo-relative paths, each sorted and free of duplicates.
func parseStatus(status string) (changed, staged, untracked []string) {
	changedSet, stagedSet, untrackedSet := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for line := range strings.SplitSeq(status, "\n") {
		if len(line) < statusPrefix {
			continue
		}
		index, worktree, path := line[0], line[1], strings.TrimSpace(line[statusPrefix:])
		if _, renamed, found := strings.Cut(path, " -> "); found {
			path = renamed
		}
		path = strings.Trim(path, `"`)
		switch {
		case index == '?' && worktree == '?':
			untrackedSet[path] = true
		default:
			if index != ' ' {
				stagedSet[path] = true
			}
			if worktree != ' ' {
				changedSet[path] = true
			}
		}
	}
	return sortedKeys(changedSet), sortedKeys(stagedSet), sortedKeys(untrackedSet)
}

// statusPrefix is the two status columns plus the separating space before a path.
const statusPrefix = 3

func outsideScope(declaredScope []string, groups ...[]string) []string {
	if len(declaredScope) == 0 {
		return nil
	}
	outside := map[string]bool{}
	for _, group := range groups {
		for _, path := range group {
			if !underScope(declaredScope, path) {
				outside[path] = true
			}
		}
	}
	return sortedKeys(outside)
}

func underScope(declaredScope []string, path string) bool {
	path = filepath.ToSlash(filepath.Clean(path))
	for _, prefix := range declaredScope {
		prefix = filepath.ToSlash(filepath.Clean(prefix))
		if path == prefix || strings.HasPrefix(path, prefix+"/") {
			return true
		}
	}
	return false
}

func sortedKeys(set map[string]bool) []string {
	return slices.Sorted(maps.Keys(set))
}
