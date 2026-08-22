package main

import (
	"fmt"
	"path/filepath"
	"strings"
)

func sourceRepoDB(root, snapshot string) (string, error) {
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
		candidate := filepath.Join(filepath.FromSlash(fields[index+1]), "repodb-store")
		if match != "" {
			return "", fmt.Errorf("prepare-merge: multiple RepoDB worktrees match %s", snapshot)
		}
		match = candidate
	}
	return match, nil
}
