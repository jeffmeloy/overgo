package gate

import (
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
		// Measured 2026-09-22: bound to 268 packages by ten files whose
		// reads the source does not name. A narrowing lands with a lower
		// ceiling; nothing may widen it.
		const ceiling = 268
		if len(node.inputDependencies) > ceiling {
			t.Errorf("%s binds %d packages, ceiling %d", target, len(node.inputDependencies), ceiling)
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
