package server

import (
	"context"
	"testing"
)

// TestCancelledDownloadHoldsItsSlotUntilItFinishes holds the registry's transfer bound and history to
// what is still running: a cancel reports the job cancelled at once but the transfer goroutine, its
// connection and its files live until it unwinds, so at one transfer the next is refused until then,
// and a job that has not unwound is not evicted to make room for history; once it has unwound, the
// slot and the place are free.
func TestCancelledDownloadHoldsItsSlotUntilItFinishes(t *testing.T) {
	t.Parallel()
	registry := newDownloadRegistry(1, 2, nil)
	job := func(id uint64) *downloadJob { return &downloadJob{ID: id, State: downloadStateRunning} }
	first, second, third := job(1), job(2), job(3)
	_, cancel := context.WithCancel(t.Context())
	if err := registry.admit(first, cancel); err != nil {
		t.Fatal(err)
	}
	if state, found := registry.cancel(first.ID); !found || state != downloadStateCancelled {
		t.Fatalf("cancel reported (%q, %t), want the job cancelled at once", state, found)
	}
	if err := registry.admit(second, cancel); err == nil {
		t.Fatal("a cancelled transfer that had not unwound released its slot")
	}
	registry.unwound(first)
	if err := registry.admit(second, cancel); err != nil {
		t.Fatalf("the slot stayed taken after the transfer unwound: %v", err)
	}
	// The history holds two jobs: a third is admitted only by evicting a job whose transfer has unwound,
	// and the oldest here (the first) has; a job still transferring is never the one evicted.
	registry.unwound(second)
	if err := registry.admit(third, cancel); err != nil {
		t.Fatalf("a third job was refused with the first unwound: %v", err)
	}
	if jobs := registry.snapshot(); len(jobs) != 2 || jobs[0].ID != third.ID || jobs[1].ID != second.ID {
		t.Fatalf("the history after eviction = %+v, want the third and second", jobs)
	}
	registry.unwound(third)
	// A running transfer's place is never taken for history even when the retention bound is full.
	fourth := job(4)
	registry.mu.Lock()
	second.transferring = true
	third.transferring = true
	registry.mu.Unlock()
	if err := registry.admit(fourth, cancel); err == nil {
		t.Fatal("a job was admitted over a full set of transfers that had not unwound")
	}
}
