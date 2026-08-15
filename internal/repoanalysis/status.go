// Package repoanalysis owns reusable facts derived from repository source and
// Git-visible worktree state. It contains no admission policy.
package repoanalysis

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

type DirtyPath struct {
	Path           string `json:"path"`
	IndexStatus    string `json:"index_status"`
	WorktreeStatus string `json:"worktree_status"`
	OriginalPath   string `json:"original_path,omitempty"`
}

// NormalizeDirty canonicalizes and orders repository-relative paths.
func NormalizeDirty(paths []DirtyPath) []DirtyPath {
	out := make([]DirtyPath, 0, len(paths))
	for _, path := range paths {
		path.Path = filepath.ToSlash(strings.TrimSpace(path.Path))
		path.OriginalPath = filepath.ToSlash(strings.TrimSpace(path.OriginalPath))
		if path.Path != "" {
			out = append(out, path)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].OriginalPath < out[j].OriginalPath
	})
	return out
}

// ParseDirtyStatus parses `git status --porcelain=v1 -z`. Rename and copy
// entries carry the original path in the following NUL-delimited field.
func ParseDirtyStatus(raw []byte) ([]DirtyPath, error) {
	fields := strings.Split(string(raw), "\x00")
	var paths []DirtyPath
	for i := 0; i < len(fields); i++ {
		entry := fields[i]
		if entry == "" {
			continue
		}
		if len(entry) < 4 || entry[2] != ' ' {
			return nil, fmt.Errorf("malformed porcelain status entry %q", entry)
		}
		path := DirtyPath{
			IndexStatus: string(entry[0]), WorktreeStatus: string(entry[1]), Path: entry[3:],
		}
		if strings.ContainsAny(entry[:2], "RC") {
			i++
			if i >= len(fields) || fields[i] == "" {
				return nil, fmt.Errorf("rename/copy status for %q lacks original path", path.Path)
			}
			path.OriginalPath = fields[i]
		}
		paths = append(paths, path)
	}
	return NormalizeDirty(paths), nil
}
