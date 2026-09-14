package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// shadowCandidate: one recorded candidate change over a fixture, the
// compiler fixture unless the candidate brings its own; edits replace whole
// files, paths is what the gate would be handed.
type shadowCandidate struct {
	name    string
	paths   []string
	edits   map[string]string
	fixture func(*testing.T) *gateContext
}

// selectionCounterexamples: packages the full gate failed that the selection
// omitted; the report names failures as packages and as "package: Test"
// entries, both reduced to the package; empty proves the selection missed no
// required check.
func selectionCounterexamples(fullFailed, selected []string) []string {
	var missed []string
	for _, failed := range fullFailed {
		failedPackage, _, _ := strings.Cut(failed, ": ")
		if !slices.Contains(selected, failedPackage) {
			missed = append(missed, failedPackage)
		}
	}
	slices.Sort(missed)
	return slices.Compact(missed)
}

// candidateFixture builds the candidate's own fixture, else the compiler
// fixture every recorded change shares.
func candidateFixture(t *testing.T, candidate shadowCandidate) *gateContext {
	t.Helper()
	if candidate.fixture != nil {
		return candidate.fixture(t)
	}
	return scopeCompilerFixture(t)
}

// fixtureRootPackages: every package ./... resolves in the fixture graph.
func fixtureRootPackages(t *testing.T, g *gateContext) []string {
	t.Helper()
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	var roots []string
	for _, node := range graph.nodes {
		if len(node.Match) != 0 && node.ForTest == "" {
			roots = append(roots, node.ImportPath)
		}
	}
	slices.Sort(roots)
	return roots
}

// TestChangeSelectiveVerificationShadow pins, per recorded candidate change:
// the full check set over every fixture package is executed after the change,
// every package it fails lies inside the gate's selected scope (no
// counterexample), the scope excluded at least one package (selection is
// selective), and a narrower selection that drops the dependents is caught as
// a counterexample by the same shadow.
func TestChangeSelectiveVerificationShadow(t *testing.T) {
	candidates := []shadowCandidate{
		{
			name: "production constant", paths: []string{"internal/plan/plan.go"},
			edits: map[string]string{"internal/plan/plan.go": "package plan\nimport _ \"embed\"\nconst Value = 2\n//go:embed catalog.txt\nvar Catalog string\n//go:embed payload_test.go\nvar Payload string\n"},
		},
		{
			name: "intermediate consumer", paths: []string{"internal/consumer/consumer.go"},
			edits: map[string]string{"internal/consumer/consumer.go": "package consumer\nimport (\"overgo/internal/plan\"; \"overgo/internal/recipe\")\nfunc Value() int { return plan.Value + recipe.Value + 1 }\n"},
		},
		{
			name: "leaf dependency", paths: []string{"internal/recipe/recipe.go"},
			edits: map[string]string{"internal/recipe/recipe.go": "package recipe\nconst Value = 5\n"},
		},
		{
			// Owner review 2026-09-10: a file a library reads at run time
			// changes; the library has no test, its caller's test fails.
			name: "library runtime input", paths: []string{"docs/config.txt"},
			edits:   map[string]string{"docs/config.txt": "2\n"},
			fixture: runtimeReaderFixture,
		},
	}
	for _, candidate := range candidates {
		t.Run(candidate.name, func(t *testing.T) {
			g := candidateFixture(t, candidate)
			for path, content := range candidate.edits {
				if err := os.WriteFile(filepath.Join(g.repo, filepath.FromSlash(path)), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			g.paths = candidate.paths
			roots := fixtureRootPackages(t, g)
			full, err := g.runGoTests(t.Context(), roots, false, nil)
			if err == nil || len(full.Failed) == 0 {
				t.Fatalf("recorded change produced no full-gate failure: %v", err)
			}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			selected := scope.selected()
			if missed := selectionCounterexamples(full.Failed, selected); len(missed) != 0 {
				t.Fatalf("selection omitted failing packages %v; full failed=%v selected=%v", missed, full.Failed, selected)
			}
			if scope.excluded == 0 {
				t.Fatalf("selection excluded nothing; the shadow proves nothing: %+v", scope)
			}
			t.Logf("candidate=%s full_failed=%d selected=%d excluded=%d", candidate.name, len(full.Failed), len(selected), scope.excluded)

			// Direct-only selection drops the importers, which fail on every
			// recorded change; the shadow must name each as a counterexample
			// from the uncertain or the dependent group.
			missed := selectionCounterexamples(full.Failed, scope.direct)
			if len(missed) == 0 {
				t.Fatal("shadow accepted a selection that omitted failing dependents")
			}
			dropped := slices.Concat(scope.uncertain, scope.dependent)
			for _, dependent := range missed {
				if !slices.Contains(dropped, dependent) {
					t.Fatalf("counterexample %s is not a dropped importer of %v", dependent, dropped)
				}
			}
			t.Logf("counterexample selection=direct-only missed=%s", strings.Join(missed, ","))
		})
	}
}
