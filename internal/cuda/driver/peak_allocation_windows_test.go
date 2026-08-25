//go:build windows

package driver

import "testing"

// TestMemoryPeakAllocationNamesCrossing pins the diagnostic
// instrumentation: MemoryStats reports the size of the allocation that
// last raised the peak and the largest live allocation, so a
// peak-over-budget measurement points at one nameable buffer rather
// than a diffuse total. Freeing below the peak leaves the crossing
// size intact; a reset clears it.
func TestMemoryPeakAllocationNamesCrossing(t *testing.T) {
	lib := &Library{}
	lib.accountAllocation(0x1000, 4096)  // baseline
	lib.accountAllocation(0x2000, 1<<20) // crosses: sets peak at 4096+1MiB
	lib.accountFree(0x1000)              // drop below the peak
	lib.accountAllocation(0x3000, 512)   // stays under the high-water
	stats := lib.MemoryStats()
	if stats.PeakAllocationBytes != 1<<20 {
		t.Fatalf("peak allocation = %d, want the 1MiB buffer that crossed the high-water", stats.PeakAllocationBytes)
	}
	if stats.PeakBytes != 4096+(1<<20) {
		t.Fatalf("peak = %d, want 4096+1MiB set at the crossing", stats.PeakBytes)
	}
	if stats.CurrentBytes != (1<<20)+512 {
		t.Fatalf("current = %d, want 1MiB+512 after the free", stats.CurrentBytes)
	}
	if stats.LargestLiveBytes != 1<<20 {
		t.Fatalf("largest live = %d, want 1MiB", stats.LargestLiveBytes)
	}
	// The peak ledger is the size histogram AT the crossing: the freed
	// 4096 baseline is still in it, and the post-peak 512 allocation is
	// not -- the snapshot is peak-time evidence, not the live view.
	if len(stats.PeakLedger) != 2 ||
		stats.PeakLedger[0] != (AllocationSizeClass{Bytes: 1 << 20, Count: 1}) ||
		stats.PeakLedger[1] != (AllocationSizeClass{Bytes: 4096, Count: 1}) {
		t.Fatalf("peak ledger = %+v, want the crossing-time histogram largest-first", stats.PeakLedger)
	}
	lib.ResetPeakBytes()
	reset := lib.MemoryStats()
	if reset.PeakAllocationBytes != 0 || reset.PeakBytes != reset.CurrentBytes || len(reset.PeakLedger) != 0 {
		t.Fatalf("reset stats = %+v, want the crossing and its ledger cleared to the live floor", reset)
	}
}
