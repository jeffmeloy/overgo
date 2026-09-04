package longform

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// corpusPrefix is the prose the corpus starts with: the README, then
// the tracked documents in name order.
var corpusPrefix = []string{"README.md", "docs"}

// corpusSources is the tree the corpus continues into once the prose
// is spent: the Go sources under internal in path order, tests
// excluded. A long rung of the ladder reads a tree of the repository's
// own code, deterministic at every commit and long past any model's
// declared context.
const corpusSources = "internal"

// Corpus reads at least limit bytes of the repository's text at root
// in a fixed order: README.md, the documents under docs, then the Go
// sources under internal. The order is a function of the tree alone,
// so every model at one commit reads the same text. A repository with
// less text than the limit returns what it holds.
func Corpus(root string, limit int) (string, error) {
	var builder strings.Builder
	appendFile := func(path string) error {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		builder.Write(data)
		builder.WriteString("\n\n")
		return nil
	}
	if err := appendFile(filepath.Join(root, corpusPrefix[0])); err != nil {
		return "", err
	}
	docs, err := sortedFiles(filepath.Join(root, corpusPrefix[1]), ".md", false)
	if err != nil {
		return "", err
	}
	for _, path := range docs {
		if builder.Len() >= limit {
			return builder.String(), nil
		}
		if err := appendFile(path); err != nil {
			return "", err
		}
	}
	if builder.Len() >= limit {
		return builder.String(), nil
	}
	err = filepath.WalkDir(filepath.Join(root, corpusSources), func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		if builder.Len() >= limit {
			return filepath.SkipAll
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil
		}
		return appendFile(path)
	})
	if err != nil {
		return "", err
	}
	if builder.Len() == 0 {
		return "", errors.New("longform: the corpus is empty")
	}
	return builder.String(), nil
}

// sortedFiles lists the files with the suffix directly under directory
// in name order.
func sortedFiles(directory, suffix string, recursive bool) ([]string, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), suffix) {
			continue
		}
		files = append(files, filepath.Join(directory, entry.Name()))
	}
	slices.Sort(files)
	return files, nil
}
