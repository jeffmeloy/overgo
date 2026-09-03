package executor

import (
	"errors"
	"slices"
	"sync/atomic"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

type graphExecEntry struct {
	exec     driver.GraphExec
	compiled *CompiledGraph
	serial   uint64
	frame    []driver.DevicePtr
	used     uint64
}

// compiledSerials numbers every compiled graph once. The exec cache compares
// the serial beside the pointer: a released graph's address can be reused by
// a later compile, and a stale exec matched through that address would
// replay kernels over memory the earlier graph owned.
var compiledSerials atomic.Uint64

func nextCompiledSerial() uint64 { return compiledSerials.Add(1) }

// graphExecCache: retained instantiated graphs keyed by indexed replay frame. Retained
// buffer leases alternate between a small set of pool slots, so steady-state
// decode cycles through a handful of byte-identical traces; a match replays
// the instantiated exec with zero per-token kernel re-issue.
type graphExecCache struct {
	entries         []graphExecEntry
	tick            uint64
	hits            uint64
	misses          uint64
	captures        uint64
	updates         uint64
	instantiations  uint64
	evictions       uint64
	updateFallbacks uint64
	drops           uint64
}

func (c *graphExecCache) match(
	compiled *CompiledGraph,
	frame []driver.DevicePtr,
) (driver.GraphExec, bool) {
	for index := range c.entries {
		entry := &c.entries[index]
		if entry.exec != 0 && entry.compiled == compiled && entry.serial == compiled.serial &&
			slices.Equal(entry.frame, frame) {
			c.tick++
			c.hits++
			entry.used = c.tick
			return entry.exec, true
		}
	}
	c.misses++
	return 0, false
}

// store: captures into the least-recently-used slot; exec update preferred
// over re-instantiation.
func (c *graphExecCache) store(
	state *device.State,
	graph driver.Graph,
	compiled *CompiledGraph,
	frame []driver.DevicePtr,
) (driver.GraphExec, error) {
	c.captures++
	var entry *graphExecEntry
	if len(c.entries) < graphExecCacheCapacity {
		c.entries = append(c.entries, graphExecEntry{})
		entry = &c.entries[len(c.entries)-1]
	} else {
		c.evictions++
		entry = &c.entries[0]
		for index := range c.entries {
			if c.entries[index].used < entry.used {
				entry = &c.entries[index]
			}
		}
	}
	if entry.exec != 0 {
		updated, err := state.Driver.GraphExecUpdate(entry.exec, graph)
		if err != nil || !updated {
			c.updateFallbacks++
			_ = state.Driver.GraphExecDestroy(entry.exec)
			entry.exec = 0
		} else {
			c.updates++
		}
	}
	if entry.exec == 0 {
		exec, err := state.Driver.GraphInstantiate(graph)
		if err != nil {
			entry.compiled = nil
			entry.frame = nil
			return 0, err
		}
		entry.exec = exec
		c.instantiations++
	}
	entry.compiled = compiled
	entry.serial = compiled.serial
	entry.frame = append(entry.frame[:0], frame...)
	c.tick++
	entry.used = c.tick
	return entry.exec, nil
}

func (c *graphExecCache) close(state *device.State) error {
	var errs []error
	for index := range c.entries {
		if c.entries[index].exec != 0 {
			errs = append(errs, state.Driver.GraphExecDestroy(c.entries[index].exec))
		}
	}
	c.entries = nil
	return errors.Join(errs...)
}

// drop destroys every instantiated exec because device memory a captured
// graph may reference has been freed: a replayed exec carries its kernel
// parameters by value, so it would run over the freed range. The next
// execution captures afresh; the counters keep the history.
func (c *graphExecCache) drop(state *device.State) error {
	if len(c.entries) == 0 {
		return nil
	}
	c.drops++
	return c.close(state)
}
