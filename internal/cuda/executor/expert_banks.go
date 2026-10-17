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
// ResolveExpertBanks runs after routing and stream completion and preserves expert indices.
// ReserveExpertMemory releases only completed expert reads before another owner allocates.
type ExpertBanks interface {
	IsExpertBank(string) bool
	ResolveExpertBanks(context.Context, *device.State, []*tensor.Tensor, []uint32) ([]driver.DevicePtr, error)
	ReserveExpertMemory(*device.State, uint64) error
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

func validateExpertUses(compiled *CompiledGraph, source ExpertBanks) (bool, error) {
	used := false
	if source == nil {
		return false, nil
	}
	for _, node := range compiled.order {
		for index, input := range node.Inputs {
			if input.Op != tensor.OpInput || !source.IsExpertBank(input.Name) {
				continue
			}
			used = true
			attributes, ok := node.Attrs.(tensor.MoEAttributes)
			end := moeFirstBankInput + tensor.PairedExtent
			if attributes.Gated && !attributes.FusedGateUp {
				end++
			}
			if node.Op != tensor.OpMoE || !ok || index < moeFirstBankInput || index >= end {
				return false, fmt.Errorf("file-backed expert bank %q has a non-MoE consumer", input.Name)
			}
			for _, bank := range node.Inputs[moeFirstBankInput:end] {
				if !source.IsExpertBank(bank.Name) {
					return false, errors.New("a routed layer mixes indexed and contiguous expert banks")
				}
			}
		}
	}
	for _, output := range compiled.outputs {
		if output.Op == tensor.OpInput && source.IsExpertBank(output.Name) {
			return false, errors.New("a file-backed expert bank cannot be returned as a contiguous output")
		}
	}
	return used, nil
}

func (r *executorResources) allocate(state *device.State, bytes uint64) (driver.DevicePtr, error) {
	if r.experts != nil {
		if err := r.experts.ReserveExpertMemory(state, bytes); err != nil {
			return 0, err
		}
	}
	return state.Driver.MemAlloc(bytes)
}
