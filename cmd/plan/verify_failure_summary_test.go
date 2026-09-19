package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/planverify"
)

// TestVerifyFailureSummary holds the parser-to-command handoff plan -verify uses:
// a failing go test verifier reports the failed package and test name and the
// final assertion line, not a wall of raw output, while the run's exit code
// stays the authority. It runs one bounded failing package, never a suite.
func TestVerifyFailureSummary(t *testing.T) {
	t.Parallel()
	module := t.TempDir()
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(module, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module verifyfailurefixture\n\ngo 1.25\n")
	write("fixture_test.go", "package fixture\n\nimport \"testing\"\n\n"+
		"func TestPasses(t *testing.T) {}\n\n"+
		"func TestFails(t *testing.T) { t.Fatal(\"seeded assertion sentinel\") }\n")

	_, err := planverify.Execute(t.Context(), module, "go test . -run '^(TestPasses|TestFails)$' -count=1", nil)
	if err == nil {
		t.Fatal("a failing verifier reported success")
	}
	summary := err.Error()
	for _, want := range []string{"TestFails", "seeded assertion sentinel"} {
		if !strings.Contains(summary, want) {
			t.Fatalf("failure summary omits %q: %s", want, summary)
		}
	}
	if strings.Contains(summary, "TestPasses") {
		t.Fatalf("failure summary named a passing test: %s", summary)
	}
}
