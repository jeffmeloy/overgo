package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

var (
	markdownLinkPattern = regexp.MustCompile(`\[[^\]]+\]\(([^)]+)\)`)
	commandPathPattern  = regexp.MustCompile("`cmd/([a-z0-9-]+)`")
)

func TestReadmeReferencesLiveRepositorySurfaces(t *testing.T) {
	root := filepath.Join("..", "..")
	raw, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, match := range markdownLinkPattern.FindAllStringSubmatch(text, -1) {
		target := strings.SplitN(match[1], "#", 2)[0]
		if target == "" || strings.HasPrefix(target, "#") || strings.Contains(target, "://") {
			continue
		}
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(target))); err != nil {
			t.Errorf("README link %q: %v", match[1], err)
		}
	}
	for _, match := range commandPathPattern.FindAllStringSubmatch(text, -1) {
		if _, err := os.Stat(filepath.Join(root, "cmd", match[1])); err != nil {
			t.Errorf("README command %q: %v", match[1], err)
		}
	}
}
