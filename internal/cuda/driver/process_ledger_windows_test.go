//go:build windows

package driver

import (
	"sync"
	"testing"
)

// processLedgerBase restarts the process's high-water mark at what the
// process holds now, as ResetPeakBytes restarts a library's, and returns
// it. The test binary is one process with one ledger, which earlier tests
// have moved, so each test reads the ledger from its own base, and none of
// them runs in parallel.
func processLedgerBase() uint64 {
	base := processCurrentBytes.Load()
	processPeakBytes.Store(base)
	return base
}

// TestProcessLedgerSumsLibraries holds the process's device memory to the
// sum of what its libraries hold: two libraries resident together peak at
// their sum, which neither one's own ledger reports; a free lowers the
// total and never the peak; and a library whose context is destroyed while
// it still holds memory gives that back once, so the total cannot drift.
func TestProcessLedgerSumsLibraries(t *testing.T) {
	const large, small = 1 << 20, 1 << 10
	base := processLedgerBase()
	held := func(want, wantPeak uint64, when string) {
		t.Helper()
		if current, peak := processCurrentBytes.Load()-base, ProcessPeakBytes()-base; current != want || peak != wantPeak {
			t.Fatalf("%s: the process holds %d bytes over its base and peaked %d over it, want %d and %d", when, current, peak, want, wantPeak)
		}
	}
	first, second := &Library{}, &Library{}
	first.accountAllocation(0x1000, large)
	held(large, large, "one library allocated")
	second.accountAllocation(0x1000, small)
	held(large+small, large+small, "a second library allocated beside it")
	if own := max(first.MemoryStats().PeakBytes, second.MemoryStats().PeakBytes); own != large {
		t.Fatalf("the larger library peak = %d, want %d: the process's is the sum no library's ledger holds", own, large)
	}

	first.accountFree(0x1000)
	held(small, large+small, "the first library freed")
	first.accountAllocation(0x2000, large/2)
	held(small+large/2, large+small, "a smaller allocation under the peak")

	// The second library's context is destroyed with its allocation live.
	second.forgetAllocations()
	held(large/2, large+small, "a context destroyed with memory held")
	if stats := second.MemoryStats(); stats.CurrentBytes != 0 || stats.Allocations != 0 || stats.PeakBytes != small {
		t.Fatalf("the destroyed library's ledger = %+v, want nothing live and its peak kept", stats)
	}
	second.forgetAllocations()
	second.accountFree(0x1000)
	held(large/2, large+small, "what the destroyed context freed is not given back twice")

	first.accountFree(0x2000)
	held(0, large+small, "everything freed")
}

// TestProcessLedgerHoldsUnderConcurrentLibraries runs many libraries at
// once, each on its own goroutine as each worker owns one. Round after
// round every library allocates once at the same moment: the total after a
// round is the most the process has held, and the peak is exactly it, so no
// library's raise is lost to another's, which a peak written without
// compare-and-swap loses within these rounds. When they free and destroy
// at once the total returns to its base and the peak stays.
func TestProcessLedgerHoldsUnderConcurrentLibraries(t *testing.T) {
	const libraries, rounds, size = 32, 16384, 1 << 12
	base := processLedgerBase()
	owned := make([]*Library, libraries)
	for index := range owned {
		owned[index] = &Library{}
	}
	each := func(work func(index int, library *Library)) {
		var group sync.WaitGroup
		start := make(chan struct{})
		for index := range owned {
			group.Go(func() {
				<-start
				work(index, owned[index])
			})
		}
		close(start)
		group.Wait()
	}
	for round := range rounds {
		each(func(_ int, library *Library) { library.accountAllocation(DevicePtr(round+1), size) })
		held := uint64(round+1) * libraries * size
		if current, peak := processCurrentBytes.Load()-base, ProcessPeakBytes()-base; current != held || peak != held {
			t.Fatalf("after %d libraries allocated at once %d times the process holds %d bytes over its base and peaked %d over it, want %d for both",
				libraries, round+1, current, peak, held)
		}
	}
	each(func(index int, library *Library) {
		// Half free each allocation; the rest lose them with their context.
		if index%2 == 0 {
			library.forgetAllocations()
			return
		}
		for round := range rounds {
			library.accountFree(DevicePtr(round + 1))
		}
	})
	if current, peak := processCurrentBytes.Load()-base, ProcessPeakBytes()-base; current != 0 || peak != libraries*rounds*size {
		t.Fatalf("after every library released, the process holds %d bytes over its base and peaked %d over it, want 0 and the peak kept at %d",
			current, peak, uint64(libraries*rounds*size))
	}
}
