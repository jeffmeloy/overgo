package longform

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Affected reports whether changed paths can alter the current platform's
// inference dependency closure. Unknown scope requires measurement. Package
// resources are included even when they are not Go source; ordinary tests are
// excluded, but an embedded test-named file remains a runtime input.
// Kernel changes conservatively affect every model until finer equivalence is
// proven. The caller still owns corpus and model-input identity validation.
func Affected(ctx context.Context, root string, paths []string) (bool, string, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return false, "", err
	}
	packages, err := surfacePackages(ctx, root)
	if err != nil {
		return false, "", err
	}
	var directories, embedded []string
	for _, pkg := range packages {
		directories = append(directories, pkg.Dir)
		for _, file := range pkg.EmbedFiles {
			embedded = append(embedded, filepath.Join(pkg.Dir, file))
		}
	}
	return affectedPaths(root, paths, directories, embedded...)
}

func affectedPaths(root string, paths, directories []string, embedded ...string) (bool, string, error) {
	if len(paths) == 0 {
		return true, "unknown changed-path scope; full selected catalog required", nil
	}
	var owners []string
	embedInputs := map[string]bool{}
	for _, file := range embedded {
		embedInputs[impactPath(file)] = true
	}
	for _, directory := range directories {
		relative, err := filepath.Rel(root, directory)
		if err != nil {
			return false, "", err
		}
		owners = append(owners, impactPath(relative))
	}
	for _, raw := range paths {
		path := impactPath(raw)
		if path == "." || filepath.IsAbs(path) || path == ".." || strings.HasPrefix(path, "../") || strings.Contains(path, ":") {
			return false, "", errors.New("longform: changed paths must name repository-relative files")
		}
		if path == "go.mod" || path == "go.sum" || strings.HasPrefix(path, "kernels/") || strings.HasPrefix(path, "internal/cuda/") {
			if !strings.HasSuffix(path, "_test.go") {
				return true, "runtime or kernel authority changed; full selected catalog required", nil
			}
		}
		if strings.HasSuffix(path, "_test.go") && !embedInputs[impactPath(filepath.Join(root, filepath.FromSlash(path)))] {
			continue
		}
		for _, owner := range owners {
			if owner == "." || strings.HasPrefix(path, owner+"/") {
				return true, "inference dependency or package resource changed; full selected catalog required", nil
			}
		}
		// A removed package can disappear from the current compiler graph.
		// Without the previous graph its ownership is unknown, never disjoint.
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(path))); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return true, "deleted or unknown path; full selected catalog required", nil
			}
			return false, "", err
		}
	}
	return false, "changed files are disjoint from the current inference dependency closure and kernel authority; no model measurements required", nil
}

func impactPath(path string) string {
	path = filepath.ToSlash(filepath.Clean(strings.ReplaceAll(path, "\\", "/")))
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return path
}
