package gate

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestSurfaceRefusesAnUnreconciledRuntimeMove holds the preflight to naming a
// move of the pinned media runtime before the commit records it. The case is
// the one that cost a landing and a revert: a file under a pinned path
// changed, nothing answered for the move, and the media acceptance refused
// the source only in the deferred lanes.
func TestSurfaceRefusesAnUnreconciledRuntimeMove(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	moved := "internal/cuda/testutil/require.go"
	g := &gateContext{repo: repo, paths: []string{moved}}
	skipped, err := g.stepSurface()
	if err == nil {
		t.Fatalf("an unreconciled runtime move was admitted (skipped=%t)", skipped)
	}
	if !strings.Contains(err.Error(), moved) {
		t.Fatalf("the refusal does not name the path that moved: %v", err)
	}

	// The same move with a reviewed reconciliation beside it is admitted.
	reconciled := &gateContext{repo: repo, paths: []string{moved, "docs/image_video_q10prefill_reconciliation.json"}}
	if _, err := reconciled.stepSurface(); err != nil {
		t.Fatalf("a reconciled runtime move was refused: %v", err)
	}

	// A change outside every pinned path is not this check's business.
	outside := &gateContext{repo: repo, paths: []string{"docs/plan.json"}}
	skipped, err = outside.stepSurface()
	if err != nil || !skipped {
		t.Fatalf("a change outside the pinned paths was judged: skipped=%t err=%v", skipped, err)
	}
}
