package gitauthority

import (
	"context"
	"errors"
	"fmt"
	"path"
	"strings"
)

// AncestorFiles reads repository-relative files from one exact ancestor commit.
// Complete, unreplaced history and the caller's explicit repository bind the
// read; a branch name, tag object or unrelated commit cannot supply authority.
func AncestorFiles(ctx context.Context, repository, revision string, paths ...string) ([][]byte, error) {
	if !ValidObjectID(revision) || len(paths) == 0 {
		return nil, errors.New("git authority: ancestor files require an exact commit and paths")
	}
	for _, name := range paths {
		if name == "" || name == "." || path.IsAbs(name) || path.Clean(name) != name ||
			strings.HasPrefix(name, "../") || strings.ContainsAny(name, "\\:\x00\r\n") {
			return nil, fmt.Errorf("git authority: noncanonical ancestor path %q", name)
		}
	}
	root, err := RepositoryRoot(ctx, repository)
	if err != nil {
		return nil, err
	}
	if err := RequireCompleteHistory(ctx, root); err != nil {
		return nil, err
	}
	resolved, err := output(ctx, root, "rev-parse", "--verify", revision+"^{commit}")
	if err != nil {
		return nil, fmt.Errorf("git authority: resolve ancestor commit: %w", err)
	}
	if strings.TrimSpace(string(resolved)) != revision {
		return nil, errors.New("git authority: ancestor revision is not one exact commit")
	}
	if _, err := output(ctx, root, "merge-base", "--is-ancestor", revision, "HEAD"); err != nil {
		return nil, fmt.Errorf("git authority: revision is not an ancestor: %w", err)
	}
	files := make([][]byte, len(paths))
	for index, name := range paths {
		files[index], err = output(ctx, root, "cat-file", "blob", revision+":"+name)
		if err != nil {
			return nil, err
		}
	}
	return files, nil
}
