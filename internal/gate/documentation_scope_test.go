package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/testutil"
)

func TestDocumentationBoundaryAcceptance(t *testing.T) {
	t.Parallel()
	t.Run("repository README selects no runtime suites", func(t *testing.T) {
		root, err := filepath.Abs(filepath.Join("..", ".."))
		if err != nil {
			t.Fatal(err)
		}
		g := &gateContext{repo: root, paths: []string{"README.md"}}
		tree, err := g.plannedTree()
		if err != nil {
			t.Fatal(err)
		}
		err = g.withCandidateWorktree(tree, func(string) error {
			started := time.Now()
			planned, err := g.planPipeline()
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"device", automationcheck.WebUICheckName} {
				if _, excluded := planned.impact.ExclusionReason(name); !excluded {
					t.Fatalf("README selected %s: %+v", name, planned.surface)
				}
			}
			scope, err := g.deriveTestScope()
			if err != nil || len(scope.direct)+len(scope.dependent) != 0 || scope.excluded == 0 {
				t.Fatalf("README scope=%+v error=%v", scope, err)
			}
			if skipped, err := g.stepTestPlan(t.Context()); err != nil || !skipped || g.testPlan == nil || g.testPlan.pending != 0 {
				t.Fatalf("README did not establish empty package scope: skipped=%v error=%v", skipped, err)
			}
			if skipped, err := g.stepTestOwners(t.Context()); err != nil || !skipped || len(g.testExecutions) != 0 {
				t.Fatalf("README started package acquisition: skipped=%v error=%v", skipped, err)
			}
			for _, name := range []string{"docs", "acceptance", "architecture", "protection", "commit"} {
				if !slices.ContainsFunc(planned.invocations, func(check automationcheck.Invocation) bool { return check.Check.Name == name }) {
					t.Fatalf("README omitted required %s acceptance", name)
				}
			}
			t.Logf("repository packages=%d executed=0; device=excluded browser=excluded; planning=%s", scope.excluded, time.Since(started))
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	})
	t.Run("opaque fixture reader retains evidence across prose edits", func(t *testing.T) {
		g := scopeCompilerFixture(t)
		testutil.WriteTextFile(t, g.repo, "README.md", "before")
		testutil.WriteTextFile(t, g.repo, "internal/other/other.go", "package other\nimport \"os\"\nfunc Read(path string) ([]byte,error) { return os.ReadFile(path) }\n")
		runGitFixture(t, g.repo, "add", ".")
		graph, err := g.inputGraph()
		if err != nil {
			t.Fatal(err)
		}
		before, err := graph.identity("overgo/internal/other")
		if err != nil {
			t.Fatal(err)
		}
		testutil.WriteTextFile(t, g.repo, "README.md", "after")
		g.paths = []string{"README.md"}
		g.packageGraph = nil
		scope, err := g.deriveTestScope()
		if err != nil || len(scope.direct)+len(scope.dependent) != 0 {
			t.Fatalf("prose reached opaque fixture: %+v, %v", scope, err)
		}
		graph, err = g.inputGraph()
		if err != nil {
			t.Fatal(err)
		}
		after, err := graph.identity("overgo/internal/other")
		if err != nil || before != after {
			t.Fatalf("prose invalidated software receipt: %v", err)
		}
		for _, name := range []string{"internal/other/other.go", "internal/other/testdata/input.md", "docs/verification/input.md", "go.mod", "kernels/manifest.json"} {
			if documentationChanges([]string{"README.md", name}, graph) {
				t.Fatalf("mixed software input %s classified as prose", name)
			}
		}
		graph.bindResourceFiles([]string{"root_linux.go"})
		if documentationChanges([]string{"README.md"}, graph) {
			t.Fatal("prose bypassed a source owner omitted by the host compiler")
		}
	})
	for _, kind := range []string{"runtime Markdown", "embedded Markdown", "deleted runtime Markdown"} {
		t.Run(kind, func(t *testing.T) {
			g := scopeCompilerFixture(t)
			name := "docs/contract.md"
			consumer := "overgo/internal/other"
			testutil.WriteTextFile(t, g.repo, name, "good")
			testutil.WriteTextFile(t, g.repo, "internal/other/other_test.go", "package other\nimport (\"os\"; \"testing\")\nfunc TestContract(t *testing.T) { b,e:=os.ReadFile(\"../../docs/contract.md\"); if e!=nil || string(b)!=\"good\" { t.Fatalf(\"contract=%s error=%v\",b,e) } }\n")
			if kind == "embedded Markdown" {
				name, consumer = "README.md", "overgo"
				testutil.WriteTextFile(t, g.repo, name, "good")
				testutil.WriteTextFile(t, g.repo, "root.go", "package overgo\nimport _ \"embed\"\n//go:embed README.md\nvar Contract string\n")
				testutil.WriteTextFile(t, g.repo, "root_test.go", "package overgo\nimport \"testing\"\nfunc TestContract(t *testing.T) { if Contract!=\"good\" { t.Fatal(Contract) } }\n")
			}
			runGitFixture(t, g.repo, "add", ".")
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
			testutil.WriteTextFile(t, g.repo, name, "bad")
			if kind == "deleted runtime Markdown" {
				if err := os.Remove(filepath.Join(g.repo, filepath.FromSlash(name))); err != nil {
					t.Fatal(err)
				}
			}
			g.paths, g.packageGraph = []string{name}, nil
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			selected := scope.selected()
			if !slices.Contains(selected, consumer) {
				t.Fatalf("lost seeded %s failure: %v", kind, selected)
			}
			graph, err = g.inputGraph()
			if err != nil {
				t.Fatal(err)
			}
			after, err := graph.identity(consumer)
			if err != nil || before == after {
				t.Fatalf("changed fixture retained receipt: %v", err)
			}
			if report, err := g.runGoTests(t.Context(), selected, false, nil); err == nil || !slices.Contains(report.Failed, consumer) {
				t.Fatalf("selected tests missed seeded %s failure: %+v, %v", kind, report, err)
			}
		})
	}
}
