package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/automationcheck"
)

func TestChangeSelectionUnownedRuntimeInputs(t *testing.T) {
	t.Parallel()
	for _, inputPath := range []string{"protocol.txt", "internal/other/protocol.txt", "internal/other/catalog.txt", "internal/other/other.go", "internal/other/reader_test.go"} {
		t.Run(inputPath, func(t *testing.T) {
			g := scopeCompilerFixture(t)
			write := func(name, content string) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(g.repo, filepath.FromSlash(name)), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			beforeText, afterText := "accepted", "regression"
			if filepath.Ext(inputPath) == ".go" {
				beforeText, afterText = "package other\n", "package other\n// changed source\n"
			}
			write(inputPath, beforeText)
			if inputPath == "internal/other/catalog.txt" {
				write("internal/other/other.go", "package other\nimport _ \"embed\"\n//go:embed catalog.txt\nvar Catalog string\n")
			}
			write("internal/client/runtime_test.go", "package client\nimport (\"os\"; \"testing\")\nfunc TestRuntimeInput(t *testing.T) { b,err := os.ReadFile(\"../../"+inputPath+"\"); if err != nil || string(b) != "+strconv.Quote(beforeText)+" { t.Fatalf(\"input=%s err=%v\", b,err) } }\n")
			runGitFixture(t, g.repo, "add", ".")
			const consumer = "overgo/internal/client"
			if report, err := g.runGoTests(t.Context(), []string{consumer}, false, nil); err != nil {
				t.Fatalf("baseline: %+v, %v", report, err)
			}
			graph, err := g.inputGraph()
			if err != nil {
				t.Fatal(err)
			}
			before, err := graph.identity(consumer)
			if err != nil {
				t.Fatal(err)
			}
			cache := automationcheck.NewEvidenceCache(lifecycleTestEnvironment(t).ID)
			if err := cache.RecordPackagePass(consumer, "complete", before); err != nil {
				t.Fatal(err)
			}
			write(inputPath, afterText)
			g.paths = []string{inputPath}
			g.packageGraph = nil
			full, fullErr := g.runGoTests(t.Context(), fixtureRootPackages(t, g), false, nil)
			if fullErr == nil || !slices.Contains(full.Failed, consumer) {
				t.Fatalf("missing seeded failure: %+v %v", full, fullErr)
			}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			selected := scope.selected()
			if missed := selectionCounterexamples(full.Failed, selected); len(missed) != 0 {
				t.Errorf("runtime consumer omitted: %v; selected=%v", missed, selected)
			}
			graph, err = g.inputGraph()
			if err != nil {
				t.Fatal(err)
			}
			after, err := graph.identity(consumer)
			if err != nil {
				t.Fatal(err)
			}
			if reused, err := cache.PackageReusable(consumer, "complete", after); err != nil || reused {
				t.Errorf("stale runtime receipt reusable=%t error=%v", reused, err)
			}
			if t.Failed() {
				return
			}
			report, err := g.runGoTests(t.Context(), selected, false, nil)
			if err == nil || !slices.Equal(report.Failed, full.Failed) {
				t.Fatalf("selected failures=%v full=%v error=%v", report.Failed, full.Failed, err)
			}
		})
	}
}
