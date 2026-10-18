package executor

import (
	"context"
	"errors"
	"fmt"
	"unsafe"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
)

// MemoryCapacityError refuses a graph before execution when its working set cannot fit.
// Allocation errors after admission remain distinct because availability can change.
type MemoryCapacityError struct {
	Required, Available uint64
}

// Error reports the required and available byte counts.
func (e *MemoryCapacityError) Error() string {
	return fmt.Sprintf("CUDA graph requires %d additional bytes, available %d", e.Required, e.Available)
}

// CheckRetainedMemory admits a graph against current memory and reclaimable experts.
// Targets name external output slots and capacities name new caller-owned buffers.
// Existing live buffers are already excluded from available memory.
// Free pool buffers receive no credit because scratch allocation cannot reclaim them.
func (e *Executor) CheckRetainedMemory(ctx context.Context, compiled *CompiledGraph, inputs *DeviceInputs, targets []OutputSlot, capacities []uint64) error {
	if e == nil || compiled == nil {
		return errors.New("CUDA memory admission requires an executor and compiled graph")
	}
	arena, pooled, capacity, err := compiled.retainedAllocations(inputs, targets, capacities)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return errors.New("CUDA executor is closed")
	}
	return e.worker.Do(ctx, func(state *device.State) error {
		// Initialize library-owned resources before asking for available bytes.
		resources, err := e.ensureResources(state, compiled.needBlas, 0, 0, 0)
		if err != nil {
			return err
		}
		free, _, err := state.Driver.MemInfo()
		if err != nil {
			return err
		}
		// Card experts move to host memory to make room, and a routed node
		// reads its experts where they are, so experts need no room of their own.
		if resources.experts != nil {
			reclaimable, err := resources.experts.ExpertMemory(state)
			if err != nil {
				return err
			}
			if free, err = addMemoryBytes(free, reclaimable); err != nil {
				return err
			}
		}
		// Staging grows before its predecessor is freed, unlike the arena.
		var staging, scores uint64
		if resources.blas != nil {
			staging, scores = resources.blas.stagingBytes, resources.blas.scoreBytes
		}
		var growth, peak uint64
		for _, pair := range [][tensor.PairedExtent]uint64{{compiled.matmulStagingBytes, staging}, {compiled.attentionScoreBytes, scores}, {compiled.q8InputBytes, resources.q8Input.stagingBytes}} {
			if pair[0] <= pair[1] {
				continue
			}
			transient, err := addMemoryBytes(growth, pair[0])
			if err != nil {
				return err
			}
			peak = max(peak, transient)
			if growth, err = addMemoryBytes(growth, pair[0]-pair[1]); err != nil {
				return err
			}
		}
		if arena > resources.arenaSize {
			if growth, err = addMemoryBytes(growth, arena-resources.arenaSize); err != nil {
				return err
			}
		}
		if growth, err = addMemoryBytes(growth, pooled); err != nil {
			return err
		}
		required, err := addMemoryBytes(capacity, max(peak, growth))
		if err != nil {
			return err
		}
		if required > free {
			return &MemoryCapacityError{Required: required, Available: free}
		}
		return nil
	})
}

func addMemoryBytes(left, right uint64) (uint64, error) {
	value, ok := checked.Add64(left, right)
	if !ok {
		return 0, errors.New("CUDA working set size overflows")
	}
	return value, nil
}

func (c *CompiledGraph) retainedAllocations(inputs *DeviceInputs, slots []OutputSlot, capacities []uint64) (arena, pooled, capacity uint64, err error) {
	if c == nil || inputs != nil && (inputs.compiled != c || len(inputs.Pointers) != int(c.inputCount)) {
		return 0, 0, 0, errors.New("CUDA memory plan inputs belong to another graph")
	}
	targeted := make([]bool, len(c.outputs))
	for _, slot := range slots {
		if uint64(slot) >= uint64(len(targeted)) || targeted[slot] {
			return 0, 0, 0, errors.New("CUDA memory plan output slot is invalid or duplicated")
		}
		targeted[slot] = true
	}
	isTarget := func(index int) bool { return targeted[index] }
	arena = c.retainedExecutionMemory(isTarget).ArenaSize
	storage := make([]bool, len(c.order))
	offsets := make([]uint64, len(c.order))
	bytes, err := c.retainedLayout(isTarget, storage, offsets)
	if err != nil {
		return 0, 0, 0, err
	}
	addPooled := func(bytes uint64) error {
		if bytes == 0 {
			return nil
		}
		bucket, err := deviceBufferBucket(bytes)
		if err == nil {
			pooled, err = addMemoryBytes(pooled, bucket)
		}
		return err
	}
	if err = addPooled(bytes); err != nil {
		return 0, 0, 0, err
	}
	for index, node := range c.order {
		if node.Op != tensor.OpInput {
			continue
		}
		if _, retained := c.outputIndexes[node]; retained {
			continue
		}
		slot := -c.nodes[index].operandOffset - 1
		if inputs != nil && inputs.Pointers[slot] != 0 {
			continue
		}
		bytes, err := node.Shape.Bytes(node.Type)
		if err != nil {
			return 0, 0, 0, err
		}
		if err = addPooled(bytes); err != nil {
			return 0, 0, 0, err
		}
	}
	if err = addPooled(uint64(c.attributeWords) * uint64(unsafe.Sizeof(uint32(0)))); err != nil {
		return 0, 0, 0, err
	}
	for _, bytes := range capacities {
		bucket, ok := deviceBufferClass(bytes)
		if !ok {
			return 0, 0, 0, errors.New("CUDA memory plan capacity is invalid")
		}
		if capacity, err = addMemoryBytes(capacity, bucket); err != nil {
			return 0, 0, 0, err
		}
	}
	return arena, pooled, capacity, nil
}
