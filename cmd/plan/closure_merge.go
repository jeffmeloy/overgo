package main

import (
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
	head     artifact.CommitID
	sequence uint64
	cleanup  func()
}

func captureClosureEvidence(root, gitSnapshot string) (closureEvidenceSnapshot, error) {
	source, err := sourceOvergoDB(root, gitSnapshot)
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
	return closureEvidenceSnapshot{store: destination, head: head, sequence: sequence, cleanup: cleanup}, nil
}

func sourceOvergoDB(root, snapshot string) (string, error) {
	raw, err := gitOutput(root, "worktree", "list", "--porcelain")
	if err != nil {
		return "", err
	}
	fields := strings.Fields(string(raw))
	var match string
	for index := 0; index+3 < len(fields); index++ {
		if fields[index] != "worktree" || fields[index+2] != "HEAD" || fields[index+3] != snapshot {
			continue
		}
		candidate := filepath.Join(filepath.FromSlash(fields[index+1]), "overgodb-store")
		if match != "" {
			return "", fmt.Errorf("prepare-merge: multiple OvergoDB worktrees match %s", snapshot)
		}
		match = candidate
	}
	return match, nil
}
