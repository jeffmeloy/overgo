//go:build windows

package executor

import (
	"fmt"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

func TestDeviceBufferPoolCapacityRefusalReclaimsIdle(t *testing.T) {
	cuda := newFixtureExecutor(t)
	err := cuda.worker.Do(t.Context(), func(state *device.State) error {
		pool := &cuda.resources.buffers
		_, total, err := state.Driver.MemInfo()
		if err != nil {
			return err
		}
		// Keep the cache ceiling nonbinding: physical admission must reclaim it.
		pool.freeLimit = total + total
		live, err := pool.acquireExact(state, deviceAllocationAlignment)
		if err != nil {
			return err
		}
		idle, err := pool.acquireExact(state, deviceAllocationAlignment)
		if err != nil {
			return err
		}
		if err := pool.release(state, idle); err != nil {
			return err
		}
		before := state.Driver.MemoryStats().CurrentBytes
		oversized, err := pool.acquireBucket(state, total+1)
		if !driver.IsOutOfMemory(err) || oversized.pointer != 0 {
			return fmt.Errorf("physical capacity refusal: %+v: %v", oversized, err)
		}
		if pool.freeBytes != 0 || len(pool.allocations) != 1 || pool.allocations[0] != live {
			return fmt.Errorf("pressure damaged live leases or retained idle buffers: %+v", pool)
		}
		if after := state.Driver.MemoryStats().CurrentBytes; before-after != idle.size {
			return fmt.Errorf("reclaimed bytes: before=%d after=%d idle=%d", before, after, idle.size)
		}
		// The surviving lease remains writable after failure and reclamation.
		if err := state.Driver.MemsetD32Async(live.pointer, 0, live.size/4, state.Stream); err != nil {
			return err
		}
		return state.Driver.StreamSynchronize(state.Stream)
	})
	if err != nil {
		t.Fatal(err)
	}
}
