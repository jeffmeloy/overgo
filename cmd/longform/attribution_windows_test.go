package main

import (
	"testing"

	"overgo/internal/cuda/executor"
)

// TestRetainedBudgetReconciliation exercises the retained-byte budget that
// governs the E4B device guard, without a device. Every named owner is summed
// into the attribution, the budget is the loaded footprint plus those owners
// plus the unsurfaced share, retention within that share is accepted as fully
// attributed, and retention past it -- an untracked context-scaled buffer --
// is rejected. The device test proves the same budget on the real artifact.
func TestRetainedBudgetReconciliation(t *testing.T) {
	metrics := executor.ExecutionMetrics{
		ArenaCommittedBytes: 7 << 30,
		BLASStagingBytes:    1 << 28,
		BLASScoreBytes:      1 << 28,
		Q8StagingBytes:      1 << 27,
		PoolTotalBytes:      4 << 30,
		PoolRetainedBytes:   4 << 30,
		PoolRetainedLimit:   6 << 30,
	}
	owners := attributedOwners(metrics)
	wantOwners := uint64(7<<30) + (1 << 28) + (1 << 28) + (1 << 27) + (4 << 30)
	if owners != wantOwners {
		t.Fatalf("attributedOwners=%d want %d", owners, wantOwners)
	}
	const footprint = 14 << 30
	budget := retainedBudget(footprint, metrics)
	if want := uint64(footprint) + owners + footprint/reconcileFootprintShare; budget != want {
		t.Fatalf("retainedBudget=%d want %d", budget, want)
	}
	// Fully attributed retention (footprint plus owners, no unaccounted bytes)
	// fits, and the unsurfaced share is headroom above it.
	if within := uint64(footprint) + owners + footprint/reconcileFootprintShare/2; within > budget {
		t.Fatalf("within-share retention %d rejected by budget %d", within, budget)
	}
	// A buffer beyond the unsurfaced share is untracked and must be rejected.
	if beyond := uint64(footprint) + owners + footprint/reconcileFootprintShare + (1 << 30); beyond <= budget {
		t.Fatalf("untracked retention %d accepted by budget %d", beyond, budget)
	}
	if metrics.PoolRetainedBytes > metrics.PoolRetainedLimit {
		t.Fatalf("pool retained %d exceeded limit %d", metrics.PoolRetainedBytes, metrics.PoolRetainedLimit)
	}
}
