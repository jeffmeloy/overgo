package plan

import (
	"io/fs"
	"os"
	"path/filepath"
)

// CompetingPlans walks root and returns every live plan.json outside the
// canonical Path, as root-relative slash paths. One rule keeps nested
// checkouts out of plan authority: a directory other than root holding a
// .git marker -- file or directory, covering worktrees and submodules
// alike -- is another repository, so nothing under it can compete with
// this repository's plan. Volatile non-source trees (tmp, OvergoDB
// stores, imported external trees, vendored modules) are skipped by
// their own names because they are not checkouts yet still hold foreign
// files.
func CompetingPlans(root string) ([]string, error) {
	canonical := filepath.ToSlash(Path)
	var competing []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path == root {
				return nil
			}
			switch entry.Name() {
			case ".git", "tmp", "overgodb-store", "external", "vendor":
				return fs.SkipDir
			}
			if _, err := os.Lstat(filepath.Join(path, ".git")); err == nil {
				return fs.SkipDir
			}
			return nil
		}
		if entry.Name() != "plan.json" {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if slashed := filepath.ToSlash(relative); slashed != canonical {
			competing = append(competing, slashed)
		}
		return nil
	})
	return competing, err
}
