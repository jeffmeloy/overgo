package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/fsatomic"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
)

type closureEvidenceSnapshot struct {
	store    string
	source   string
	head     artifact.CommitID
	sequence uint64
	cleanup  func()
	backup   *overgodb.BackupReport
}

func captureClosureEvidence(ctx context.Context, root, gitSource, gitSnapshot, selector string) (closureEvidenceSnapshot, error) {
	source, err := sourceOvergoDB(root, gitSource, gitSnapshot, selector)
	if err != nil || source == "" {
		return closureEvidenceSnapshot{}, err
	}
	if _, err := os.Stat(source); errors.Is(err, os.ErrNotExist) {
		return closureEvidenceSnapshot{}, nil
	} else if err != nil {
		return closureEvidenceSnapshot{}, err
	}
	// One source workspace survives retries; the lease pins it through import.
	identity := sha256.Sum256([]byte(source))
	parent := filepath.Join(root, "tmp", "merge-evidence", fmt.Sprintf("%x", identity))
	err = os.MkdirAll(parent, clioptions.OutputDirectoryMode)
	if err != nil {
		return closureEvidenceSnapshot{}, err
	}
	lease, err := processlock.AcquireContext(ctx, filepath.Join(parent, "lease.lock"), clioptions.PrivateFileMode)
	if err != nil {
		return closureEvidenceSnapshot{}, err
	}
	cleanup := func() { _ = lease.Close() }
	destination := filepath.Join(parent, "overgodb-store")
	if _, err := os.Lstat(destination); err == nil {
		if _, err := os.Lstat(destination + ".partial"); !os.IsNotExist(err) {
			cleanup()
			return closureEvidenceSnapshot{}, errors.New("prepare-merge: published and partial evidence snapshots overlap")
		}
		if err := fsatomic.Replace(destination, destination+".partial"); err != nil {
			cleanup()
			return closureEvidenceSnapshot{}, err
		}
	} else if !os.IsNotExist(err) {
		cleanup()
		return closureEvidenceSnapshot{}, err
	}
	store, err := overgodb.OpenReadOnly(source)
	if err != nil {
		cleanup()
		return closureEvidenceSnapshot{}, err
	}
	report, backupErr := store.Backup(ctx, destination)
	closeErr := store.Close()
	if backupErr != nil || closeErr != nil {
		cleanup()
		return closureEvidenceSnapshot{}, fmt.Errorf("prepare-merge: snapshot closure evidence IO=%+v: %w", report, errors.Join(backupErr, closeErr))
	}
	return closureEvidenceSnapshot{
		store: destination, source: source, head: report.Head, sequence: report.Sequence, cleanup: cleanup, backup: report,
	}, nil
}

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
