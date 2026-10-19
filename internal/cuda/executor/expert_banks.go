package executor

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// ExpertBanks owns encoded expert residency on the executor's CUDA worker.
// An expert bank's input is its table of expert pointers, each always readable by the card.
// ExpertRoutes names the flags a routed node's routes mark, in host memory the card maps, so a run's routes reach the host with no copy.
// AdvanceExperts applies the last run's routes to residency before the next run, and moves no expert after a prompt.
// ReserveExpertMemory releases only completed expert reads before another owner allocates.
// ExpertSources writes each expert's table address in a routed node's bank, and the pinned bytes of each expert in host memory.
type ExpertBanks interface {
	IsExpertBank(string) bool
	ExpertRoutes(*device.State, []*tensor.Tensor) (driver.DevicePtr, error)
	ExpertSources(*device.State, *tensor.Tensor, []driver.DevicePtr, [][]byte) error
	AdvanceExperts(context.Context, *device.State, bool) error
	ReserveExpertMemory(*device.State, uint64) error
	ExpertMemory(*device.State) (reclaimable uint64, err error)
}

// SetExpertBanks binds the existing weight owner to routed execution and memory admission.
// Rebinding waits for active execution and drops graphs that address the previous owner.
func (e *Executor) SetExpertBanks(source ExpertBanks) error {
	if e == nil {
		return errors.New("expert bank executor is unavailable")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("expert bank executor is closed")
	}
	if err := e.worker.Do(context.Background(), func(state *device.State) error { return e.resources.graphExecs.drop(state) }); err != nil {
		return err
	}
	e.resources.experts = source
	e.resources.buffers.reserve = nil
	if source != nil {
		e.resources.buffers.reserve = source.ReserveExpertMemory
	}
	return nil
}

// validateExpertUses reports whether a graph routes through file-backed expert banks and the most tokens one of its routed nodes takes.
func validateExpertUses(compiled *CompiledGraph, source ExpertBanks) (bool, uint64, error) {
	used, tokens := false, uint64(0)
	if source == nil {
		return false, 0, nil
	}
	for _, node := range compiled.order {
		for index, input := range node.Inputs {
			if input.Op != tensor.OpInput || !source.IsExpertBank(input.Name) {
				continue
			}
			used, tokens = true, max(tokens, node.Shape.Dims[1])
			attributes, ok := node.Attrs.(tensor.MoEAttributes)
			end := moeFirstBankInput + tensor.PairedExtent
			if attributes.Gated && !attributes.FusedGateUp {
				end++
			}
			if node.Op != tensor.OpMoE || !ok || index < moeFirstBankInput || index >= end {
				return false, 0, fmt.Errorf("file-backed expert bank %q has a non-MoE consumer", input.Name)
			}
			for _, bank := range node.Inputs[moeFirstBankInput:end] {
				if !source.IsExpertBank(bank.Name) {
					return false, 0, errors.New("a routed layer mixes indexed and contiguous expert banks")
				}
			}
		}
	}
	for _, output := range compiled.outputs {
		if output.Op == tensor.OpInput && source.IsExpertBank(output.Name) {
			return false, 0, errors.New("a file-backed expert bank cannot be returned as a contiguous output")
		}
	}
	return used, tokens, nil
}

func (r *executorResources) allocate(state *device.State, bytes uint64) (driver.DevicePtr, error) {
	if r.experts != nil {
		if err := r.experts.ReserveExpertMemory(state, bytes); err != nil {
			return 0, err
		}
	}
	return state.Driver.MemAlloc(bytes)
}
