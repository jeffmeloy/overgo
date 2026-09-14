package gate

import (
	"testing"
)

// TestTestutilReachAcceptance pins the split of the shared test helpers:
// testutil neither parses Go nor runs a program, so the packages whose
// tests import it inherit no Go-source reach; the vocabulary guard and the
// process measurer keep that reach in their own leaf packages. The live
// gate-only scope is the measured witness.
func TestTestutilReachAcceptance(t *testing.T) {
	t.Parallel()
	live := liveRepositoryFixture(t)
	g := live.context("internal/gate/preflight.go")
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range graph.nodes {
		if node.ForTest != "" || len(node.Match) == 0 {
			continue
		}
		switch node.ImportPath {
		case "overgo/internal/testutil":
			if node.sourceReader || node.testSourceReader {
				t.Errorf("testutil is a source reader again: source=%v testSource=%v reason=%s", node.sourceReader, node.testSourceReader, node.runtimeReason)
			}
		case "overgo/internal/testvocab", "overgo/internal/testprocess":
			if !node.sourceReader {
				t.Errorf("%s parses Go or runs a program yet is not a source reader", node.ImportPath)
			}
		}
	}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	total := len(scope.selected()) + scope.excluded
	// Measured 2026-09-14: a gate-only change selected 134 uncertain
	// packages before the split and 107 after; the ratchet holds that count,
	// with excluded 112 to 141 of 260, and the log line records the live values.
	if len(scope.uncertain) > 107 {
		t.Fatalf("gate-only change selected %d uncertain packages of %d; the test-helper reach is back", len(scope.uncertain), total)
	}
	t.Logf("gate-only change: direct=%d uncertain=%d dependent=%d excluded=%d of %d", len(scope.direct), len(scope.uncertain), len(scope.dependent), scope.excluded, total)
}
