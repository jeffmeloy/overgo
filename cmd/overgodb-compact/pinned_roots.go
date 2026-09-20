package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// identityPattern matches a content-addressed identity as committed documents spell it.
var identityPattern = regexp.MustCompile(`[a-z][a-z0-9-]*:sha256:[0-9a-f]{64}`)

// allMatches is regexp's negative match count, which returns every match.
const allMatches = -1

// pinnedRoots collects every store identity the checkout's committed JSON
// documents name -- docs/**/*.json and the root compatibility.json -- once
// each, sorted. These are claims the repository depends on without an alias,
// so retention keeps them alongside the alias-rooted closure.
func pinnedRoots(checkout string) ([]artifact.ID, error) {
	var files []string
	if err := filepath.WalkDir(filepath.Join(checkout, "docs"), func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".json") {
			files = append(files, path)
		}
		return nil
	}); err != nil {
		return nil, err
	}
	files = append(files, filepath.Join(checkout, "compatibility.json"))
	seen := map[artifact.ID]bool{}
	var roots []artifact.ID
	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		for _, match := range identityPattern.FindAll(data, allMatches) {
			id, err := artifact.ParseID(string(match))
			if err != nil || seen[id] {
				continue
			}
			seen[id] = true
			roots = append(roots, id)
		}
	}
	slices.SortFunc(roots, artifact.CompareID)
	return roots, nil
}
