package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// sourceOvergoDB is the OvergoDB store of the registered worktree that holds
// the merge source at snapshot, or empty when no worktree holds it.
func sourceOvergoDB(root, source, snapshot, selector string) (string, error) {
	raw, err := gitOutput(root, "worktree", "list", "--porcelain", "-z")
	if err != nil {
		return "", err
	}
	branchRaw, err := gitOutput(root, "rev-parse", "--symbolic-full-name", source)
	if err != nil {
		return "", err
	}
	return sourceOvergoDBFromPorcelain(raw, snapshot, strings.TrimSpace(string(branchRaw)), selector, func(candidate string) bool {
		info, err := os.Stat(candidate)
		return err == nil && info.IsDir()
	})
}

// sourceOvergoDBFromPorcelain resolves the source worktree's OvergoDB store from
// a `git worktree list` porcelain listing. Registered checkouts at the exact
// snapshot are candidates; a caller-supplied selector names one of them
// directly, so a prepared merge is portable across worktrees. Multiple
// checkouts at one commit are normal but can hold genuinely different store
// authorities, so without a unique branch match or selector the ambiguity is
// reported rather than silently resolved.
func sourceOvergoDBFromPorcelain(
	raw []byte,
	snapshot string,
	preferredBranch string,
	selector string,
	usable func(string) bool,
) (string, error) {
	type storeCandidate struct {
		store  string
		branch string
	}
	var candidates []storeCandidate
	var worktree, head, branch string
	flush := func() error {
		if worktree == "" || head != snapshot {
			return nil
		}
		candidate := filepath.Join(filepath.FromSlash(worktree), "overgodb-store")
		if !usable(candidate) {
			return nil
		}
		candidates = append(candidates, storeCandidate{store: candidate, branch: branch})
		return nil
	}
	for encoded := range bytes.SplitSeq(raw, []byte("\x00")) {
		field := string(encoded)
		if field == "" {
			if err := flush(); err != nil {
				return "", err
			}
			worktree, head = "", ""
			continue
		}
		key, value, found := strings.Cut(field, " ")
		if !found {
			continue
		}
		switch key {
		case "worktree":
			if worktree != "" {
				if err := flush(); err != nil {
					return "", err
				}
				head, branch = "", ""
			}
			worktree = value
		case "HEAD":
			head = value
		case "branch":
			branch = value
		}
	}
	if err := flush(); err != nil {
		return "", err
	}
	if selector != "" {
		for _, candidate := range candidates {
			if sameStorePath(candidate.store, selector) {
				return candidate.store, nil
			}
		}
		return "", fmt.Errorf("prepare-merge: -merge-source-store %s is not a registered OvergoDB worktree at %s", selector, snapshot)
	}
	if preferredBranch != "" {
		var preferred []storeCandidate
		for _, candidate := range candidates {
			if candidate.branch == preferredBranch {
				preferred = append(preferred, candidate)
			}
		}
		if len(preferred) == 1 {
			return preferred[0].store, nil
		}
		if len(preferred) > 1 {
			return "", fmt.Errorf("prepare-merge: multiple OvergoDB worktrees match branch %s; select one with -merge-source-store <path>", preferredBranch)
		}
	}
	if len(candidates) > 1 {
		return "", fmt.Errorf("prepare-merge: multiple OvergoDB worktrees match %s; select one with -merge-source-store <path>", snapshot)
	}
	if len(candidates) == 1 {
		return candidates[0].store, nil
	}
	return "", nil
}

// sameStorePath compares two OvergoDB store paths as the filesystem resolves
// them, case-insensitively on Windows, so a selector matches a candidate
// regardless of separators or casing.
func sameStorePath(left, right string) bool {
	leftAbs, leftErr := filepath.Abs(filepath.Clean(left))
	rightAbs, rightErr := filepath.Abs(filepath.Clean(right))
	if leftErr != nil || rightErr != nil {
		return false
	}
	if runtime.GOOS == "windows" {
		return strings.EqualFold(leftAbs, rightAbs)
	}
	return leftAbs == rightAbs
}
