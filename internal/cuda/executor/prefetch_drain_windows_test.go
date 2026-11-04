//go:build windows

package executor

import (
	"errors"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// prefetchDrainBytes is a copy long enough to be writing when the prefetch ends: an eighth of a GiB from pinned memory.
const prefetchDrainBytes = 128 << 20

// TestAFailedPrefetchDrainsItsCopies holds a prompt read's prefetch to the copies it submitted whether or not their
// completion mark was recorded: a prefetch whose copy is writing its arena and whose mark never was, as when a later
// copy or the mark itself fails, ends with its copy stream drained, so its arena can go back to a request or to the
// experts' loan.
func TestAFailedPrefetchDrainsItsCopies(t *testing.T) {
	// Serial: device fixtures share the one card.
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	if err := worker.Do(t.Context(), func(state *device.State) (err error) {
		host, _, err := state.Driver.MemHostAlloc(prefetchDrainBytes)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, state.Driver.MemFreeHost(host)) }()
		arena, err := state.Driver.MemAlloc(prefetchDrainBytes)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, state.Driver.MemFree(arena)) }()
		copies, err := state.Driver.StreamCreate(0)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, state.Driver.StreamDestroy(copies)) }()
		prefetch := moePrefetch{base: arena}
		if err := state.Driver.MemcpyHtoDAsync(arena, host, copies); err != nil {
			return err
		}
		if err := prefetch.end(state, copies); err != nil {
			return err
		}
		done, err := state.Driver.StreamQuery(copies)
		if err != nil {
			return err
		}
		if !done {
			t.Error("a prefetch without its mark ended with its copy still writing its arena")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
