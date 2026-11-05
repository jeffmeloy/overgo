package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"

	"overgo/internal/testcheck"
)

// TestEveryDoctrineSentenceCitesItsCheck holds the session-start text to skill.md's form: past its title, each line is
// a section's heading or a numbered rule that cites, as "(check: <package directory> <TestName>, ...)", tests that exist
// in the packages named. The text was a second doctrine in prose that nothing read, which is how "holds its worktree
// from its first message" stayed in it after the hold moved to a session's first landing.
func TestEveryDoctrineSentenceCitesItsCheck(t *testing.T) {
	t.Parallel()
	sources := map[string]string{}
	exists := func(directory, name string) bool {
		source, read := sources[directory]
		if !read {
			files, _ := filepath.Glob(filepath.Join("..", "..", filepath.FromSlash(directory), "*_test.go"))
			var all strings.Builder
			for _, file := range files {
				data, err := os.ReadFile(file)
				testcheck.NoError(t, err)
				all.Write(data)
			}
			source = all.String()
			sources[directory] = source
		}
		return strings.Contains(source, "func "+name+"(")
	}
	lines := strings.Split(doctrineText, "\n")
	for _, line := range lines[1:] {
		text := strings.TrimSpace(line)
		number, rule, numbered := strings.Cut(text, ". ")
		switch {
		case text == "" || strings.ToUpper(text) == text && strings.HasSuffix(text, "."):
			continue
		case !numbered || strings.TrimFunc(number, unicode.IsDigit) != "":
			t.Errorf("the session text holds prose that is no rule: %s", text)
			continue
		}
		_, cited, citing := strings.Cut(rule, "(check: ")
		if !citing {
			t.Errorf("rule %s cites no check: %s", number, rule)
			continue
		}
		cited, _, _ = strings.Cut(cited, ")")
		for entry := range strings.SplitSeq(cited, ",") {
			directory, name, named := strings.Cut(strings.TrimSpace(entry), " ")
			if !named || !strings.HasPrefix(name, "Test") || !exists(directory, name) {
				t.Errorf("rule %s cites %q, which is no test of a package: %s", number, strings.TrimSpace(entry), rule)
			}
		}
	}
}
