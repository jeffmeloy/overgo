package closurescan

import (
	"testing"

	"overgo/internal/repoanalysis"
	"overgo/internal/testutil"
)

// preSessionHarnessNodes is the agent harness's production size before the
// control-plane batch of 2026-09-20 grew it; rows that grew it since named
// automation-harness-surface-remainder as their paydown.
const preSessionHarnessNodes = 181732

// TestHarnessSurfaceAtPreSessionBaseline holds the live agent harness at or
// below its pre-session production size: every growth since was a loan the
// paydown row owes back, and the harness may not regrow past it.
func TestHarnessSurfaceAtPreSessionBaseline(t *testing.T) {
	t.Parallel()
	snapshot, err := repoanalysis.DiscoverGo(testutil.RepoRoot(t), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	surface, err := BuildAgentHarnessSurface(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("harness production nodes %d, pre-session %d", surface.ProductionNodes, preSessionHarnessNodes)
	if surface.ProductionNodes > preSessionHarnessNodes {
		t.Fatalf("the harness holds %d production nodes, above its pre-session %d", surface.ProductionNodes, preSessionHarnessNodes)
	}
}
