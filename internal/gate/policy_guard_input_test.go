package gate

import (
	"path/filepath"
	"slices"
	"testing"
)

// Static policy fixtures belong to their analyzer, not runtime consumer suites.
func TestPolicyGuardInputBoundary(t *testing.T) {
	liveRepositoryFixture(t).use(t, func(g *gateContext) {
		graph, err := g.inputGraph()
		if err != nil {
			t.Fatal(err)
		}
		for target, own := range map[string]string{
			"overgo/internal/evaluation":  "internal/evaluation/plan.go",
			"overgo/internal/modelrecipe": "internal/modelrecipe/text_answer.go",
		} {
			inputs, err := graph.inputFiles(target)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"internal/gate/deferred_lanes.go", "docs/api_manifest.json", "docs/modern_go_baseline.json", "docs/modern_go_census.json", "docs/plan.json", own} {
				want := name == own
				if got := inputs[filepath.Join(g.sourceRoot(), filepath.FromSlash(name))]; got != want {
					t.Errorf("%s input %s=%v, want %v", target, name, got, want)
				}
				g.paths = []string{name}
				scope, err := g.deriveTestScope()
				if err != nil {
					t.Fatal(err)
				}
				if got := slices.Contains(scope.selected(), target); got != want {
					t.Errorf("%s selected for %s=%v, want %v", target, name, got, want)
				}
			}
		}
		for _, name := range []string{"internal/closurescan/entry_authority.go", "internal/closurescan/entry_authority_test.go"} {
			g.paths = []string{name}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Contains(scope.selected(), "overgo/internal/closurescan") {
				t.Errorf("policy change %s omitted policy tests", name)
			}
		}
	})
}
