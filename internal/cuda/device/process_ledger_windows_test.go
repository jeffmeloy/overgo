//go:build windows

package device

import (
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

// TestProcessLedgerFollowsDeviceWorkers holds the process's device ledger
// to real allocations on real workers, read through the peak alone as its
// consumers read it: two workers resident together peak at the sum of what
// they hold, which neither worker's own ledger reports; freeing gives the
// memory back, so allocating it again raises nothing; and a worker closed
// with memory still allocated gives it back when its context is destroyed,
// so a worker opened after it allocates the same without raising the peak.
func TestProcessLedgerFollowsDeviceWorkers(t *testing.T) {
	cudatest.Require(t)
	// More than the process has peaked at, so the first allocation raises
	// the peak whatever earlier tests of the process held.
	size := driver.ProcessPeakBytes() + 1<<20
	open := func() *Worker {
		t.Helper()
		worker, err := New(0)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = worker.Close() })
		return worker
	}
	allocate := func(worker *Worker) driver.DevicePtr {
		t.Helper()
		var pointer driver.DevicePtr
		if err := worker.Do(t.Context(), func(state *State) (err error) {
			pointer, err = state.Driver.MemAlloc(size)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return pointer
	}
	free := func(worker *Worker, pointer driver.DevicePtr) {
		t.Helper()
		if err := worker.Do(t.Context(), func(state *State) error { return state.Driver.MemFree(pointer) }); err != nil {
			t.Fatal(err)
		}
	}

	worker, beside := open(), open()
	first := allocate(worker)
	alone := driver.ProcessPeakBytes()
	if alone < size {
		t.Fatalf("process peak = %d after a worker allocated %d bytes, want the allocation counted", alone, size)
	}
	allocate(beside)
	together := driver.ProcessPeakBytes()
	if together != alone+size {
		t.Fatalf("process peak = %d with two workers resident together, want their sum %d", together, alone+size)
	}
	for _, resident := range []*Worker{worker, beside} {
		stats, err := resident.MemoryStats(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if stats.PeakBytes != size {
			t.Fatalf("a worker's own peak = %d, want its one allocation of %d: the sum is the process's alone", stats.PeakBytes, size)
		}
	}

	free(worker, first)
	again := allocate(worker)
	if peak := driver.ProcessPeakBytes(); peak != together {
		t.Fatalf("process peak = %d after freeing and allocating the same bytes again, want it kept at %d: the free was not given back", peak, together)
	}

	// The second worker closes with its allocation live: destroying its
	// context frees it on the device, and the ledger gives it back.
	if err := beside.Close(); err != nil {
		t.Fatal(err)
	}
	reopened := open()
	third := allocate(reopened)
	if peak := driver.ProcessPeakBytes(); peak != together {
		t.Fatalf("process peak = %d after a worker opened in place of one closed with memory held, want it kept at %d: the destroyed context's memory is still counted", peak, together)
	}
	free(reopened, third)
	free(worker, again)
}
