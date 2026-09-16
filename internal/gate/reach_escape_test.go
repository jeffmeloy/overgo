package gate

import (
	"slices"
	"strings"
	"testing"
)

// TestReachEscapeAcceptance pins the reach boundary on the live graph: an
// unnamed read observes Go source only when the package parses Go or runs a
// program it does not name, so a gate-only change no longer selects every
// package with a file read, while the analysis owners and every package that
// compiles the change stay selected. The counts are the measured witness;
// the fixture-level detection of a seeded data regression lives in
// TestRuntimeOpaqueCallerSourceInput and the lane cases in the matrix.
func TestReachEscapeAcceptance(t *testing.T) {
	t.Parallel()
	live := liveRepositoryFixture(t)
	g := live.context("internal/gate/preflight.go")
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	classes := map[string]bool{}
	for _, node := range graph.nodes {
		if node.ForTest == "" && len(node.Match) != 0 {
			classes[strings.TrimPrefix(node.ImportPath, "overgo/")] = node.sourceReader
		}
	}
	for _, reader := range []string{"internal/repoanalysis", "internal/codemanifest", "internal/closurescan", "internal/gate"} {
		if !classes[reader] {
			t.Errorf("%s parses Go but is not a source reader", reader)
		}
	}
	for _, data := range []string{"internal/artifact", "internal/jsonfile", "internal/overgodb", "internal/gguf"} {
		if classes[data] {
			t.Errorf("%s reads data yet is classed as a source reader", data)
		}
	}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	total := len(scope.selected()) + scope.excluded
	if scope.excluded < total/5 {
		t.Fatalf("gate-only change excluded %d of %d packages; the data-reach boundary is not in effect", scope.excluded, total)
	}
	for _, required := range []string{"overgo/internal/gate", "overgo/cmd/gate"} {
		if !slices.Contains(scope.selected(), required) {
			t.Fatalf("gate-only change dropped %s", required)
		}
	}
	// A data reader whose tests reach no program runner leaves the scope;
	// gguf's tests import testevidence for a skip constant and inherit the
	// processcontrol launcher's unnamed program reach until that constant
	// moves to a leaf owner.
	for _, excluded := range []string{"overgo/internal/jsonfile"} {
		if slices.Contains(scope.selected(), excluded) {
			t.Fatalf("data reader %s selected by a gate-only change", excluded)
		}
	}
	t.Logf("gate-only change: direct=%d uncertain=%d dependent=%d excluded=%d of %d", len(scope.direct), len(scope.uncertain), len(scope.dependent), scope.excluded, total)
}
