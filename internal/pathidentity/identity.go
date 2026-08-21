// Package pathidentity answers local path authority questions using filesystem
// identity instead of string spelling. It keeps filesystem policy independent
// of model, dataset, artifact, and scheduler data shapes.
package pathidentity

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Canonical returns the absolute, link-resolved spelling of an existing path.
func Canonical(path string) (string, error) {
	if path == "" {
		return "", errors.New("path identity: path is empty")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("path identity: absolute path: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(filepath.Clean(absolute))
	if err != nil {
		return "", fmt.Errorf("path identity: resolve path: %w", err)
	}
	if _, err := os.Stat(resolved); err != nil {
		return "", fmt.Errorf("path identity: inspect path: %w", err)
	}
	return filepath.Clean(resolved), nil
}

// Same reports whether two existing spellings resolve to one filesystem object.
func Same(left, right string) (bool, error) {
	leftInfo, err := os.Stat(left)
	if err != nil {
		return false, fmt.Errorf("path identity: inspect left path: %w", err)
	}
	rightInfo, err := os.Stat(right)
	if err != nil {
		return false, fmt.Errorf("path identity: inspect right path: %w", err)
	}
	return os.SameFile(leftInfo, rightInfo), nil
}

// Contains reports whether path is root itself or descends from root. The root
// must exist and be a directory. Path may name a future descendant: the walk
// continues through nonexistent components until it reaches an existing
// ancestor, while links and case aliases are compared by os.SameFile.
func Contains(root, path string) (bool, error) {
	rootInfo, err := os.Stat(root)
	if err != nil {
		return false, fmt.Errorf("path identity: inspect root: %w", err)
	}
	if !rootInfo.IsDir() {
		return false, errors.New("path identity: root is not a directory")
	}
	current, err := filepath.Abs(path)
	if err != nil {
		return false, fmt.Errorf("path identity: absolute candidate: %w", err)
	}
	for {
		_, statErr := os.Stat(current)
		if statErr == nil {
			resolved, resolveErr := filepath.EvalSymlinks(current)
			if resolveErr != nil {
				return false, fmt.Errorf("path identity: resolve candidate ancestor: %w", resolveErr)
			}
			for ancestor := filepath.Clean(resolved); ; ancestor = filepath.Dir(ancestor) {
				info, inspectErr := os.Stat(ancestor)
				if inspectErr != nil {
					return false, fmt.Errorf("path identity: inspect resolved ancestor: %w", inspectErr)
				}
				if os.SameFile(rootInfo, info) {
					return true, nil
				}
				parent := filepath.Dir(ancestor)
				if parent == ancestor {
					return false, nil
				}
			}
		} else if !errors.Is(statErr, os.ErrNotExist) {
			return false, fmt.Errorf("path identity: inspect candidate: %w", statErr)
		}
		parent := filepath.Dir(current)
		if parent == current {
			return false, nil
		}
		current = parent
	}
}
