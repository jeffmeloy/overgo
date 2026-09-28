package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unicode"
)

// referenceRoots are the top-level directories a repository path in prose
// starts with.
var referenceRoots = []string{"cmd/", "internal/", "docs/", "fixtures/", "scripts/"}

// testDeclaration opens a test function in a test file.
const testDeclaration = "func Test"

// shortestName is the shortest last path segment prose means as a name; a
// one-letter segment is a placeholder.
const shortestName = 2

// deadReferences are the references in the texts that name what the
// repository no longer holds: a path that does not exist, or a test no test
// file declares. Prose drifts from code; this names where, one per line as
// "<text>: <reference>".
func deadReferences(root string, texts map[string]string) []string {
	declared := map[string]bool{}
	for _, top := range []string{"cmd", "internal"} {
		_ = filepath.WalkDir(filepath.Join(root, top), func(name string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(name, "_test.go") {
				return nil
			}
			data, err := os.ReadFile(name)
			if err != nil {
				return nil
			}
			for rest := string(data); ; {
				at := strings.Index(rest, testDeclaration)
				if at < 0 {
					break
				}
				rest = rest[at+len("func "):]
				declared[rest[:strings.IndexFunc(rest, func(r rune) bool { return !isNameRune(r) })]] = true
			}
			return nil
		})
	}
	var dead []string
	for source, text := range texts {
		seen := map[string]bool{}
		for word := range strings.FieldsSeq(text) {
			reference, named := reference(word)
			if !named || seen[reference] {
				continue
			}
			seen[reference] = true
			if strings.HasPrefix(reference, "Test") {
				if !declared[reference] {
					dead = append(dead, source+": "+reference)
				}
			} else if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(reference))); err != nil {
				dead = append(dead, source+": "+reference)
			}
		}
	}
	slices.Sort(dead)
	return dead
}

// reference reads one word of prose as a repository path or a test name.
// A prefix or a template (the name followed by * or <), a number (a path
// segment that starts with a digit) and a one-letter placeholder name
// nothing.
func reference(word string) (string, bool) {
	start := strings.IndexFunc(word, func(r rune) bool { return unicode.IsLetter(r) })
	if start < 0 {
		return "", false
	}
	word = word[start:]
	end := strings.IndexFunc(word, func(r rune) bool { return !isNameRune(r) && r != '/' && r != '.' && r != '-' })
	name := word
	if end >= 0 {
		name, word = word[:end], word[end:]
		if strings.ContainsAny(word[:1], "*<") {
			return "", false
		}
	}
	name = strings.TrimRight(name, ".-/")
	if strings.HasPrefix(name, "Test") && len(name) > len("Test") && unicode.IsUpper(rune(name[len("Test")])) && !strings.ContainsAny(name, "/.-") {
		return name, true
	}
	if !slices.ContainsFunc(referenceRoots, func(root string) bool { return strings.HasPrefix(name, root) }) {
		return "", false
	}
	_, rest, _ := strings.Cut(name, "/")
	last := name[strings.LastIndex(name, "/")+1:]
	if rest == "" || unicode.IsDigit(rune(rest[0])) || len(last) < shortestName {
		return "", false
	}
	return name, true
}

func isNameRune(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) }

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
