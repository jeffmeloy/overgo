package repoanalysis

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	markdownLinkPattern = regexp.MustCompile(`\[[^\]]+\]\(([^)]+)\)`)
	commandPathPattern  = regexp.MustCompile("`cmd/([a-z0-9-]+)`")
)

// ValidateDocsInventory checks document placement and README references.
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
	return errors.Join(append(findings, err, validateReadmeReferences(root))...)
}

func validateReadmeReferences(root string) error {
	raw, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		return err
	}
	text := string(raw)
	var findings []error
	// Each pattern consumes input; the byte count bounds its match count.
	for _, match := range markdownLinkPattern.FindAllStringSubmatch(text, len(text)) {
		target, _, _ := strings.Cut(match[1], "#")
		if target == "" || strings.Contains(target, "://") {
			continue
		}
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(target))); err != nil {
			findings = append(findings, fmt.Errorf("README link %q: %w", match[1], err))
		}
	}
	for _, match := range commandPathPattern.FindAllStringSubmatch(text, len(text)) {
		if _, err := os.Stat(filepath.Join(root, "cmd", match[1])); err != nil {
			findings = append(findings, fmt.Errorf("README command %q: %w", match[1], err))
		}
	}
	return errors.Join(findings...)
}
