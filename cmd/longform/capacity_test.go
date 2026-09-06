package main

import (
	"math"
	"testing"

	"overgo/internal/cuda/driver"
)

// The 6ca79e2c E4B run completed 32768, then the retained-only forecast
// admitted 65536 and filled the 51526500352-byte device. Temporary peaks
// must participate in the same capacity decision before dispatch.
func TestLadderCapacityIncludesTemporaryPeak(t *testing.T) {
	rungs := []driver.MemoryStats{
		{CurrentBytes: 24361777320, PeakBytes: 25406159016},
		{CurrentBytes: 27079686312, PeakBytes: 31240435880},
		{CurrentBytes: 32532281512, PeakBytes: 40853780648},
	}
	limit := uint64(51526500352) / deviceFitDenominator * deviceFitNumerator
	if next := nextRungMemory(rungs[0], rungs[1]); next > limit {
		t.Fatalf("32768 incorrectly refused: predicted %d, limit %d", next, limit)
	}
	if next := nextRungMemory(rungs[1], rungs[2]); next <= limit {
		t.Fatalf("unsafe 65536 admitted: predicted %d, limit %d; measured 32768 peak %d", next, limit, rungs[2].PeakBytes)
	}
}

func TestLadderCapacityDoesNotWrapOrInventGrowth(t *testing.T) {
	for _, test := range []struct {
		name              string
		previous, current driver.MemoryStats
		want              uint64
	}{
		{"first peak", driver.MemoryStats{}, driver.MemoryStats{CurrentBytes: 10, PeakBytes: 30}, 30},
		{"released buffers", driver.MemoryStats{CurrentBytes: 20, PeakBytes: 30}, driver.MemoryStats{CurrentBytes: 10, PeakBytes: 30}, 30},
		{"retained growth", driver.MemoryStats{CurrentBytes: 10, PeakBytes: 40}, driver.MemoryStats{CurrentBytes: 30, PeakBytes: 40}, 70},
		{"overflow", driver.MemoryStats{CurrentBytes: 1, PeakBytes: 1}, driver.MemoryStats{CurrentBytes: math.MaxUint64 / 2, PeakBytes: math.MaxUint64 / 2}, math.MaxUint64},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := nextRungMemory(test.previous, test.current); got != test.want {
				t.Fatalf("predicted %d, want %d", got, test.want)
			}
		})
	}
}
