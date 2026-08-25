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
	lib.ResetPeakBytes()
	if reset := lib.MemoryStats(); reset.PeakAllocationBytes != 0 || reset.PeakBytes != reset.CurrentBytes {
		t.Fatalf("reset stats = %+v, want the crossing cleared to the live floor", reset)
	}
}
