package main

import (
	"os"
	"path/filepath"
	"strings"
)

// memoryTexts are the project's memory files, found where Claude keeps them
// for this checkout: the project path with its separators turned to dashes.
func memoryTexts(root string) map[string]string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	slug := strings.NewReplacer(":", "-", `\`, "-", "/", "-").Replace(root)
	matches, _ := filepath.Glob(filepath.Join(home, ".claude", "projects", slug, "memory", "*.md"))
	texts := map[string]string{}
	for _, match := range matches {
		if data, err := os.ReadFile(match); err == nil {
			texts["memory/"+filepath.Base(match)] = string(data)
		}
	}
	return texts
}
