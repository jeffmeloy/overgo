package repoanalysis

import (
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"strings"
)

// ValidateDocsInventory rejects files outside the declared document classes and media areas.
func ValidateDocsInventory(root string) error {
	docs := filepath.Join(root, "docs")
	var findings []error
	err := filepath.WalkDir(docs, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}
		relative, err := filepath.Rel(docs, path)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		if relative == ".loop_state" || relative == ".dispatch" {
			return nil
		}
		switch strings.ToLower(filepath.Ext(relative)) {
		case ".md", ".json", ".jsonl", ".txt", ".jinja":
			return nil
		case ".png", ".gif", ".wav", ".svg":
			area, _, _ := strings.Cut(relative, "/")
			switch area {
			case "assets", "media_samples", "verification":
				return nil
			}
		}
		findings = append(findings, fmt.Errorf("docs holds unplanned file %q: reference material enters through a plan row", relative))
		return nil
	})
	return errors.Join(append(findings, err)...)
}
