package gitauthority

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// CanonicalOvergoDBDirectory is the only store location a registered worktree may own.
const CanonicalOvergoDBDirectory = "overgodb-store"

type registeredWorktree struct {
	root string
	head string
}

// RequireRegisteredWorktreeStore proves that storePath is the canonical
// OvergoDB directory of one registered worktree whose immutable HEAD contains
// revision. It rejects arbitrary directories and ambient Git overrides.
func RequireRegisteredWorktreeStore(
	ctx context.Context,
	repository, storePath, revision string,
) (string, error) {
	if ctx == nil || strings.TrimSpace(storePath) == "" || !ValidObjectID(revision) {
		return "", errors.New("git authority: worktree store requires context, path, and exact revision")
	}
	repository, err := RepositoryRoot(ctx, repository)
	if err != nil {
		return "", err
	}
	storePath, err = filepath.Abs(filepath.Clean(storePath))
	if err != nil {
		return "", fmt.Errorf("git authority: resolve worktree store: %w", err)
	}
	if filepath.Base(storePath) != CanonicalOvergoDBDirectory {
		return "", errors.New("git authority: source store is not the canonical OvergoDB directory")
	}
	info, err := os.Stat(storePath)
	if err != nil {
		return "", fmt.Errorf("git authority: inspect worktree store: %w", err)
	}
	if !info.IsDir() {
		return "", errors.New("git authority: worktree store is not a directory")
	}
	worktreeRoot, err := RepositoryRoot(ctx, filepath.Dir(storePath))
	if err != nil {
		return "", fmt.Errorf("git authority: source store owner: %w", err)
	}
	listing, err := output(ctx, repository, "worktree", "list", "--porcelain", "-z")
	if err != nil {
		return "", err
	}
	worktrees, err := parseRegisteredWorktrees(listing)
	if err != nil {
		return "", err
	}
	var registered *registeredWorktree
	for index := range worktrees {
		if sameWorktreePath(worktrees[index].root, worktreeRoot) {
			if registered != nil {
				return "", errors.New("git authority: source store worktree is registered more than once")
			}
			registered = &worktrees[index]
		}
	}
	if registered == nil || !ValidObjectID(registered.head) {
		return "", errors.New("git authority: source store owner is not a registered worktree with an exact HEAD")
	}
	resolved, err := output(ctx, repository, "rev-parse", "--verify", revision+"^{commit}")
	if err != nil || strings.TrimSpace(string(resolved)) != revision {
		return "", errors.New("git authority: source revision is not one exact commit")
	}
	if _, err := output(ctx, repository, "merge-base", "--is-ancestor", revision, registered.head); err != nil {
		return "", fmt.Errorf("git authority: source worktree HEAD does not contain revision: %w", err)
	}
	return filepath.Join(worktreeRoot, CanonicalOvergoDBDirectory), nil
}

func parseRegisteredWorktrees(raw []byte) ([]registeredWorktree, error) {
	var result []registeredWorktree
	var current registeredWorktree
	flush := func() error {
		if current.root == "" && current.head == "" {
			return nil
		}
		if current.root == "" || current.head == "" || !filepath.IsAbs(current.root) || !ValidObjectID(current.head) {
			return errors.New("git authority: malformed registered worktree listing")
		}
		result = append(result, current)
		current = registeredWorktree{}
		return nil
	}
	for field := range bytes.SplitSeq(raw, []byte{0}) {
		if len(field) == 0 {
			if err := flush(); err != nil {
				return nil, err
			}
			continue
		}
		key, value, found := strings.Cut(string(field), " ")
		if !found {
			continue
		}
		switch key {
		case "worktree":
			if current.root != "" {
				if err := flush(); err != nil {
					return nil, err
				}
			}
			current.root = filepath.Clean(value)
		case "HEAD":
			if current.head != "" {
				return nil, errors.New("git authority: registered worktree repeats HEAD")
			}
			current.head = value
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return result, nil
}

func sameWorktreePath(left, right string) bool {
	left, leftErr := filepath.Abs(filepath.Clean(left))
	right, rightErr := filepath.Abs(filepath.Clean(right))
	if leftErr != nil || rightErr != nil {
		return false
	}
	if runtime.GOOS == "windows" {
		return strings.EqualFold(left, right)
	}
	return left == right
}
