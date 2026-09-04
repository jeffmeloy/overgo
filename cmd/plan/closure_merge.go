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
	return closureEvidenceSnapshot{
		store: destination, source: source, head: head, sequence: sequence, cleanup: cleanup,
	}, nil
}

func sourceOvergoDB(root, snapshot string) (string, error) {
	raw, err := gitOutput(root, "worktree", "list", "--porcelain", "-z")
	if err != nil {
		return "", err
	}
	return sourceOvergoDBFromPorcelain(raw, snapshot, func(candidate string) bool {
		info, err := os.Stat(candidate)
		return err == nil && info.IsDir()
	})
}

func sourceOvergoDBFromPorcelain(
	raw []byte,
	snapshot string,
	usable func(string) bool,
) (string, error) {
	var match string
	var worktree, head string
	flush := func() error {
		if worktree == "" || head != snapshot {
			return nil
		}
		candidate := filepath.Join(filepath.FromSlash(worktree), "overgodb-store")
		if !usable(candidate) {
			return nil
		}
		if match != "" {
			return fmt.Errorf("prepare-merge: multiple OvergoDB worktrees match %s", snapshot)
		}
		match = candidate
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
				head = ""
			}
			worktree = value
		case "HEAD":
			head = value
		}
	}
	if err := flush(); err != nil {
		return "", err
	}
	return match, nil
}
