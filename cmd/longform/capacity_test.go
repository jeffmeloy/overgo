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

// TestALadderCountsTheExpertsThatYield holds the ladder's fit to the card the expert residency gives up: Flash-Next's
// guard, after rung 1024 on its 51526500352-byte card, held 47,185 MiB at its peak, about 38 GiB of it experts on the card
// that yield their slots to a longer prompt; counted as held they stopped the climb before its required rung 2048. A
// rung the yielding experts cannot make room for still stops, and yielding bytes past the prediction admit nothing more.
func TestALadderCountsTheExpertsThatYield(t *testing.T) {
	t.Parallel()
	const card, mebibyte = uint64(51526500352), uint64(1) << 20
	peak, experts := 47185*mebibyte, 38<<30
	if rungFits(peak, 0, card) {
		t.Fatal("a peak of 47,185 MiB fits 9/10 of the card without the experts yielding")
	}
	if !rungFits(peak, uint64(experts), card) {
		t.Fatal("the rung after 1024 is refused although the experts it would displace yield their slots")
	}
	if rungFits(card, card/deviceFitDenominator, card) {
		t.Fatal("a rung of the whole card fits with a tenth of it yielding")
	}
	if !rungFits(peak, 2*peak, card) {
		t.Fatal("yielding bytes past the prediction refuse the rung")
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
