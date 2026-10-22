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
// StartExpertCopies starts the copies the last advance recorded, once the run is launched, so the run's own copies to
// the card do not wait behind them.
// ReserveExpertMemory releases only completed expert reads before another owner allocates.
// ExpertSources writes each expert's table address in a routed node's bank, and the pinned bytes of each expert in host memory.
// ExpertsOnHost names a routed node's card flags of the experts every bank of it holds in host memory, which a run reads
// as the advance before it left them, and which ExpertSources' bytes then agree with.
type ExpertBanks interface {
	IsExpertBank(string) bool
	ExpertRoutes(*device.State, []*tensor.Tensor) (driver.DevicePtr, error)
	ExpertsOnHost(*device.State, []*tensor.Tensor) (driver.DevicePtr, error)
	ExpertSources(*device.State, *tensor.Tensor, []driver.DevicePtr, [][]byte) error
	AdvanceExperts(context.Context, *device.State, bool) error
	StartExpertCopies(context.Context, *device.State) error
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

// expertUse is what validateExpertUses found of a compiled graph for one source.
type expertUse struct {
	source ExpertBanks
	used   bool
	tokens uint64
	err    error
}

// validateExpertUses reports whether a graph routes through file-backed expert banks and the most tokens one of its routed nodes takes.
//   - a graph keeps the answer for its source: a decode step that scanned its every node's inputs each run spent
//     about 0.15 ms of Flash-Next's
func validateExpertUses(compiled *CompiledGraph, source ExpertBanks) (bool, uint64, error) {
	if source == nil {
		return false, 0, nil
	}
	if found := compiled.expertUse.Load(); found != nil && found.source == source {
		return found.used, found.tokens, found.err
	}
	used, tokens, err := scanExpertUses(compiled, source)
	compiled.expertUse.Store(&expertUse{source: source, used: used, tokens: tokens, err: err})
	return used, tokens, err
}

// scanExpertUses is validateExpertUses over the graph's nodes.
func scanExpertUses(compiled *CompiledGraph, source ExpertBanks) (bool, uint64, error) {
	used, tokens := false, uint64(0)
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
