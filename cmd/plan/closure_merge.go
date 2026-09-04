package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

type closureEvidenceSnapshot struct {
	store    string
	source   string
	head     artifact.CommitID
	sequence uint64
	cleanup  func()
}

func captureClosureEvidence(root, gitSource, gitSnapshot string) (closureEvidenceSnapshot, error) {
	source, err := sourceOvergoDB(root, gitSource, gitSnapshot)
	if err != nil || source == "" {
		return closureEvidenceSnapshot{}, err
	}
	if _, err := os.Stat(source); errors.Is(err, os.ErrNotExist) {
		return closureEvidenceSnapshot{}, nil
	} else if err != nil {
		return closureEvidenceSnapshot{}, err
	}
	store, err := overgodb.OpenReadOnly(source)
	if err != nil {
		return closureEvidenceSnapshot{}, err
	}
	parent, err := os.MkdirTemp("", "overgo-merge-evidence-")
	if err != nil {
		_ = store.Close()
		return closureEvidenceSnapshot{}, err
	}
	cleanup := func() { _ = os.RemoveAll(parent) }
	destination := filepath.Join(parent, "overgodb-store")
	head, sequence, backupErr := store.Backup(destination)
	closeErr := store.Close()
	if backupErr != nil || closeErr != nil {
		cleanup()
		return closureEvidenceSnapshot{}, fmt.Errorf("prepare-merge: snapshot closure evidence: %w", errors.Join(backupErr, closeErr))
	}
	return closureEvidenceSnapshot{
		store: destination, source: source, head: head, sequence: sequence, cleanup: cleanup,
	}, nil
}

func sourceOvergoDB(root, source, snapshot string) (string, error) {
	raw, err := gitOutput(root, "worktree", "list", "--porcelain", "-z")
	if err != nil {
		return "", err
	}
	branchRaw, err := gitOutput(root, "rev-parse", "--symbolic-full-name", source)
	if err != nil {
		return "", err
	}
	return sourceOvergoDBFromPorcelain(raw, snapshot, strings.TrimSpace(string(branchRaw)), func(candidate string) bool {
		info, err := os.Stat(candidate)
		return err == nil && info.IsDir()
	})
}

func sourceOvergoDBFromPorcelain(
	raw []byte,
	snapshot string,
	preferredBranch string,
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
			return "", fmt.Errorf("prepare-merge: multiple OvergoDB worktrees match branch %s", preferredBranch)
		}
	}
	if len(candidates) > 1 {
		return "", fmt.Errorf("prepare-merge: multiple OvergoDB worktrees match %s", snapshot)
	}
	if len(candidates) == 1 {
		return candidates[0].store, nil
	}
	return "", nil
}
