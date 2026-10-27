//go:build windows

package executor

import (
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// TestDeviceBufferPoolReusesClasses: capacity requests of many distinct
// sizes share a bounded set of size classes. A pass whose every prompt had
// a new length used to leave one exact-size page set per length in the
// lists until the device filled.
func TestDeviceBufferPoolReusesClasses(t *testing.T) {
	cuda := newFixtureExecutor(t)
	ctx := t.Context()
	// Two hundred sizes between 1.1 and 3 MiB, no two alike, each held
	// briefly: with exact keys every one would be a fresh allocation.
	for round := range 3 {
		for index := range 200 {
			size := uint64(1<<20) + uint64(100_000+index*9_500)
			buffer, err := cuda.AllocateDeviceBuffer(ctx, size)
			if err != nil {
				t.Fatal(err)
			}
			if err := buffer.Release(ctx); err != nil {
				t.Fatal(err)
			}
		}
		cuda.mu.RLock()
		allocations := len(cuda.resources.buffers.allocations)
		cuda.mu.RUnlock()
		// Sizes span the 1-2 MiB and 2-4 MiB octaves: at most eight classes
		// each, and the sequential hold-and-release keeps one buffer per
		// class live at a time.
		if allocations > 2*int(deviceBufferClassSteps) {
			t.Fatalf("round %d: %d pooled allocations for 200 distinct sizes", round, allocations)
		}
	}
	// The free share is derived from the device; TestDeviceBufferPoolReleaseBound
	// holds the trim to it.
	cuda.mu.RLock()
	limit := cuda.resources.buffers.freeLimit
	cuda.mu.RUnlock()
	if limit == 0 {
		t.Fatal("free limit was not derived from the device")
	}
}

// TestDeviceBufferPoolLendsItsFreeBuffers: the pool lends the expert residency its free buffers of
// deviceBufferLendBytes or more, once each, and takes one back before it reuses it and every other before it frees
// them.
func TestDeviceBufferPoolLendsItsFreeBuffers(t *testing.T) {
	cuda := newFixtureExecutor(t)
	ctx := t.Context()
	var lent, reclaimed []driver.DevicePtr
	err := cuda.worker.Do(ctx, func(state *device.State) error {
		pool := &cuda.resources.buffers
		pool.reclaim = func(_ *device.State, pointer driver.DevicePtr, _ uint64) error {
			reclaimed = append(reclaimed, pointer)
			return nil
		}
		var leases []deviceBufferLease
		for _, size := range []uint64{deviceBufferLendBytes, deviceBufferLendBytes, deviceAllocationAlignment} {
			lease, err := pool.acquireSized(state, size)
			if err != nil {
				return err
			}
			leases = append(leases, lease)
		}
		if err := pool.release(state, leases...); err != nil {
			return err
		}
		lend := func(pointer driver.DevicePtr, _ uint64) error {
			lent = append(lent, pointer)
			return nil
		}
		for range 2 {
			if err := pool.lendFree(lend); err != nil {
				return err
			}
		}
		if want := []driver.DevicePtr{min(leases[0].pointer, leases[1].pointer), max(leases[0].pointer, leases[1].pointer)}; !slices.Equal(lent, want) {
			t.Errorf("the pool lent %v, want its two large free buffers once each in address order %v", lent, want)
		}
		reused, err := pool.acquireSized(state, deviceBufferLendBytes)
		if err != nil {
			return err
		}
		if !slices.Equal(reclaimed, []driver.DevicePtr{reused.pointer}) {
			t.Errorf("reusing %v took back %v, want it alone", reused.pointer, reclaimed)
		}
		reclaimed = nil
		if err := pool.close(state); err != nil {
			return err
		}
		if len(reclaimed) != 1 || reclaimed[0] == reused.pointer {
			t.Errorf("closing took back %v, want the other lent buffer", reclaimed)
		}
		pool.reclaim = nil
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestFixedDeviceBufferKeepsItsSize: a buffer kept for the runner's life takes its aligned size where a pooled one of
// the same request takes its class.
func TestFixedDeviceBufferKeepsItsSize(t *testing.T) {
	cuda := newFixtureExecutor(t)
	ctx := t.Context()
	const request = 9<<20 + 1
	for _, testCase := range []struct {
		allocate func() (*DeviceBuffer, error)
		want     uint64
	}{
		{func() (*DeviceBuffer, error) { return cuda.AllocateDeviceBuffer(ctx, request) }, 10 << 20},
		{func() (*DeviceBuffer, error) { return cuda.AllocateFixedDeviceBuffer(ctx, request) }, 9<<20 + 256},
	} {
		buffer, err := testCase.allocate()
		if err != nil {
			t.Fatal(err)
		}
		if buffer.lease.size != testCase.want {
			t.Errorf("a request of %d bytes took %d, want %d", request, buffer.lease.size, testCase.want)
		}
		if err := buffer.Release(ctx); err != nil {
			t.Fatal(err)
		}
	}
}

// TestDeviceBufferClass: the exact range keeps alignment, the classed
// range rounds up by an eighth of its octave and never below the request.
func TestDeviceBufferClass(t *testing.T) {
	for _, testCase := range []struct {
		size, want uint64
	}{
		{1, 256},
		{1000, 1024},
		{1 << 20, 1 << 20},
		{1<<20 + 1, 1<<20 + 1<<18},
		{1<<20 + 1<<18, 1<<20 + 1<<18},
		{3 << 20, 3 << 20},
		{3<<20 + 1, 3<<20 + 1<<19},
		{1 << 30, 1 << 30},
		{1<<30 + 5, 1<<30 + 1<<28},
	} {
		got, ok := deviceBufferClass(testCase.size)
		if !ok || got != testCase.want || got < testCase.size {
			t.Fatalf("class(%d) = %d, %t; want %d", testCase.size, got, ok, testCase.want)
		}
	}
	if _, ok := deviceBufferClass(0); ok {
		t.Fatal("zero size classed")
	}
}
