package gate

import (
	"slices"
	"strings"
	"testing"
)

// TestCompatibilityInputBindingRatchet pins how widely the compatibility
// command is bound. The command reads paths its own source does not name, so
// the graph classes it an opaque source reader and binds it to every root:
// its receipt then expires on any change anywhere, and the ten media and
// specialized models whose currency that receipt judges cannot be called
// current again until a full run re-earns it. The binding is measured here
// rather than argued, and cannot grow while the row that narrows it is open.
func TestCompatibilityInputBindingRatchet(t *testing.T) {
	t.Parallel()
	live := liveRepositoryFixture(t)
	g := live.context()
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	const target = "overgo/cmd/compatibility"
	found := false
	for _, node := range graph.nodes {
		if node.ImportPath != target || node.ForTest != "" {
			continue
		}
		found = true
		// The legacy opaque binding reached 269 packages after the image
		// embedding owner arrived. Colibri's shared benchmarkrecord and the
		// composition authority fixture the compositions leg shares are
		// reviewed additions outside the capped harness. They must stay bound
		// while the remaining opaque reach keeps its original ceiling.
		const ceiling = 269
		reviewedAdditions := []string{"overgo/internal/benchmarkrecord", "overgo/internal/composition/compositiontest"}
		for _, addition := range reviewedAdditions {
			if !slices.Contains(node.inputDependencies, addition) {
				t.Errorf("%s lost its reviewed source binding %s", target, addition)
			}
		}
		legacyBound := len(node.inputDependencies) - len(reviewedAdditions)
		if legacyBound > ceiling {
			t.Errorf("%s binds %d legacy packages, ceiling %d", target, legacyBound, ceiling)
		}
		if !node.opaqueReader {
			t.Errorf("%s is no longer an opaque reader; lower the ceiling and say so", target)
		}
		for reason := range strings.SplitSeq(node.runtimeReason, "; ") {
			t.Log(reason)
		}
	}
	if !found {
		t.Fatalf("%s is absent from the live input graph", target)
	}
}
