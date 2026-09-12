package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/automationcheck"
)

// matrixCase is one frozen representative change: the paths the gate is
// told about, the edits that seed a real failure, and the packages the full
// gate must then fail, or the consumers a seedless change must still select.
type matrixCase struct {
	name     string
	paths    []string
	edits    map[string]string
	removed  []string
	required []string
}

// requiredSelectionMatrix freezes the change kinds the selector must cover
// on the compiler fixture: a cross-package constant, a fixture-only edit an
// embed consumes, a leaf dependency, a renamed source file, a test-only
// library and an intermediate consumer. Every case seeds a failure the full
// gate detects.
func requiredSelectionMatrix() []matrixCase {
	return []matrixCase{
		{
			name: "cross-package constant", paths: []string{"internal/plan/plan.go"},
			edits: map[string]string{"internal/plan/plan.go": "package plan\nimport _ \"embed\"\nconst Value = 2\n//go:embed catalog.txt\nvar Catalog string\n//go:embed payload_test.go\nvar Payload string\n"},
		},
		{
			name: "fixture-only", paths: []string{"internal/plan/testdata/fixture.txt"},
			edits: map[string]string{"internal/plan/testdata/fixture.txt": "changed fixture"}, required: []string{"overgo/internal/plan"},
		},
		{
			name: "leaf dependency", paths: []string{"internal/recipe/recipe.go"},
			edits: map[string]string{"internal/recipe/recipe.go": "package recipe\nconst Value = 5\n"},
		},
		{
			name: "renamed source", paths: []string{"internal/recipe/recipe.go", "internal/recipe/value.go"},
			edits: map[string]string{"internal/recipe/value.go": "package recipe\nconst Value = 7\n"}, removed: []string{"internal/recipe/recipe.go"},
		},
		{
			name: "intermediate consumer", paths: []string{"internal/consumer/consumer.go"},
			edits: map[string]string{"internal/consumer/consumer.go": "package consumer\nimport (\"overgo/internal/plan\"; \"overgo/internal/recipe\")\nfunc Value() int { return plan.Value + recipe.Value + 1 }\n"},
		},
	}
}

// TestChangeSelectionRequiredMatrix shadow-compares the selector against
// the full gate on the frozen matrix: every package the full gate fails is
// selected, the matrix denominator and per-case timings are reported, and
// at least one case excludes packages so the comparison proves something.
func TestChangeSelectionRequiredMatrix(t *testing.T) {
	matrix := requiredSelectionMatrix()
	excludedCases := 0
	for _, entry := range matrix {
		t.Run(entry.name, func(t *testing.T) {
			g := scopeCompilerFixture(t)
			for _, path := range entry.removed {
				if err := os.Remove(filepath.Join(g.repo, filepath.FromSlash(path))); err != nil {
					t.Fatal(err)
				}
			}
			for path, content := range entry.edits {
				if err := os.WriteFile(filepath.Join(g.repo, filepath.FromSlash(path)), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			g.paths = entry.paths
			roots := fixtureRootPackages(t, g)
			started := time.Now()
			full, err := g.runGoTests(t.Context(), roots, false, nil)
			fullWall := time.Since(started)
			if len(entry.required) == 0 && (err == nil || len(full.Failed) == 0) {
				t.Fatalf("matrix case seeded no full-gate failure: %v", err)
			}
			started = time.Now()
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			selectionWall := time.Since(started)
			selected := append(slices.Clone(scope.direct), scope.dependent...)
			if missed := selectionCounterexamples(full.Failed, selected); len(missed) != 0 {
				t.Fatalf("selection omitted failing packages %v; full failed=%v selected=%v", missed, full.Failed, selected)
			}
			for _, pkg := range entry.required {
				if !slices.Contains(selected, pkg) {
					t.Fatalf("selection omitted the required consumer %s: selected=%v", pkg, selected)
				}
			}
			if scope.excluded != 0 {
				excludedCases++
			}
			t.Logf("matrix=%d case=%s denominator=%d full_failed=%d selected=%d excluded=%d full_wall=%s selection_wall=%s",
				len(matrix), entry.name, len(roots), len(full.Failed), len(selected), scope.excluded, fullWall.Round(time.Millisecond), selectionWall.Round(time.Millisecond))
		})
	}
	if excludedCases == 0 {
		t.Fatal("no matrix case excluded a package; the comparison proves nothing")
	}
	names := make([]string, 0, len(matrix))
	for _, entry := range matrix {
		names = append(names, entry.name)
	}
	t.Logf("required matrix: %s", strings.Join(names, ", "))
	t.Run("lane exclusions", laneExclusionMatrix)
}

// laneExclusionMatrix drives the lane exclusion path itself, the dependency
// resolver under OwnershipByDependency, over the compiler fixture extended
// with a shared launcher, a runtime file reader and a deleted command: a lane
// whose owned package reaches a command only through a helper that runs it
// stays in scope, a lane owning a file reader stays in scope for any change,
// a changed package the graph no longer holds keeps every lane, and a lane
// owning a pure package leaves the scope of an unrelated change.
func laneExclusionMatrix(t *testing.T) {
	g := scopeCompilerFixture(t)
	write := func(name, content string) {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("cmd/tool/main.go", "package main\nimport \"fmt\"\nfunc main() { fmt.Print(1) }\n")
	write("internal/launcher/launcher.go", "package launcher\nimport (\"context\"; \"os/exec\")\nfunc Run(ctx context.Context) ([]byte, error) { return exec.CommandContext(ctx, \"go\", \"run\", \"../../cmd/tool\").CombinedOutput() }\n")
	write("internal/lane/lane.go", "package lane\nconst Value = 1\n")
	write("internal/lane/lane_test.go", "package lane\nimport (\"testing\"; \"overgo/internal/launcher\")\nfunc TestRun(t *testing.T) { if _, err := launcher.Run(t.Context()); err != nil { t.Skip(err) } }\n")
	write("internal/reader/reader.go", "package reader\nimport \"os\"\nfunc Read(name string) ([]byte, error) { return os.ReadFile(name) }\n")
	runGitFixture(t, g.repo, "add", ".")
	g.packageGraph = nil
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	owning := func(name string, fact automationcheck.Fact, packages ...string) automationcheck.Check {
		return automationcheck.Check{Descriptor: automationcheck.Descriptor{Name: name, Ownership: automationcheck.Ownership{Fact: fact, Packages: packages}}}
	}
	checks := []automationcheck.Check{
		owning("launcher-lane", "launcher fact", "internal/lane"),
		owning("reader-lane", "reader fact", "internal/reader"),
		owning("pure-lane", "pure fact", "internal/client"),
	}
	excluded := func(impact automationcheck.Impact) []string {
		var names []string
		for _, exclusion := range impact.Exclusions {
			names = append(names, exclusion.Check)
		}
		return names
	}
	cases := []struct {
		name     string
		changed  []string
		excluded []string
	}{
		{name: "shared launcher reaches the command", changed: []string{"cmd/tool"}, excluded: []string{"pure-lane"}},
		{name: "runtime file reader stays in scope", changed: []string{"internal/other"}, excluded: []string{"launcher-lane", "pure-lane"}},
		{name: "deleted command keeps every lane", changed: []string{"cmd/gone"}, excluded: nil},
		{name: "pure lane follows its imports", changed: []string{"internal/recipe"}, excluded: []string{"launcher-lane"}},
	}
	for _, entry := range cases {
		impact := automationcheck.OwnershipByDependency(checks, entry.changed, resolver)
		if got := excluded(impact); !slices.Equal(got, entry.excluded) {
			t.Fatalf("%s: excluded=%v facts=%v, want excluded %v", entry.name, got, impact.Facts, entry.excluded)
		}
		t.Logf("lane matrix case=%s changed=%v excluded=%v facts=%v", entry.name, entry.changed, excluded(impact), impact.Facts)
	}
}
