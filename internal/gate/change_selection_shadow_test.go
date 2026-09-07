package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// shadowCandidate: one recorded candidate change over the compiler fixture;
// edits replace whole files, paths is what the gate would be handed.
type shadowCandidate struct {
	name  string
	paths []string
	edits map[string]string
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
	}
	for _, candidate := range candidates {
		t.Run(candidate.name, func(t *testing.T) {
			g := scopeCompilerFixture(t)
			for path, content := range candidate.edits {
				if err := os.WriteFile(filepath.Join(g.repo, filepath.FromSlash(path)), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			g.paths = candidate.paths
			roots := fixtureRootPackages(t, g)
			full, err := runGoTests(t.Context(), g.repo, roots, false, nil)
			if err == nil || len(full.Failed) == 0 {
				t.Fatalf("recorded change produced no full-gate failure: %v", err)
			}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			selected := append(slices.Clone(scope.direct), scope.dependent...)
			if missed := selectionCounterexamples(full.Failed, selected); len(missed) != 0 {
				t.Fatalf("selection omitted failing packages %v; full failed=%v selected=%v", missed, full.Failed, selected)
			}
			if scope.excluded == 0 {
				t.Fatalf("selection excluded nothing; the shadow proves nothing: %+v", scope)
			}
			t.Logf("candidate=%s full_failed=%d selected=%d excluded=%d", candidate.name, len(full.Failed), len(selected), scope.excluded)

			// Direct-only selection drops the dependents, which fail on every
			// recorded change; the shadow must name each as a counterexample.
			missed := selectionCounterexamples(full.Failed, scope.direct)
			if len(missed) == 0 {
				t.Fatal("shadow accepted a selection that omitted failing dependents")
			}
			for _, dependent := range missed {
				if !slices.Contains(scope.dependent, dependent) {
					t.Fatalf("counterexample %s is not a dropped dependent of %v", dependent, scope.dependent)
				}
			}
			t.Logf("counterexample selection=direct-only missed=%s", strings.Join(missed, ","))
		})
	}
}
