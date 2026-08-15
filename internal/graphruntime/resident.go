package graphruntime

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// ResidentGraph: compiled topology plus device-owned static F32 inputs.
type ResidentGraph struct {
	worker      *device.Worker
	executor    *executor.Executor
	allocations device.AllocationSet
	compiled    *executor.CompiledGraph
	inputs      *executor.DeviceInputs
	staticBytes uint64
}

func NewResidentGraph(
	ctx context.Context,
	ordinal int,
	static map[*tensor.Tensor]reference.Value,
	outputs ...*tensor.Tensor,
) (*ResidentGraph, error) {
	compiled, err := executor.Compile(outputs...)
	if err != nil {
		return nil, fmt.Errorf("resident graph: compile: %w", err)
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	runtime := &ResidentGraph{
		worker: worker, compiled: compiled, inputs: compiled.NewDeviceInputs(),
	}
	runtime.allocations = device.NewAllocationSet(worker)
	runtime.executor, err = executor.NewWithWorker(worker)
	if err != nil {
		_ = worker.Close()
		return nil, err
	}
	for node, value := range static {
		if node == nil || node.Type != dtype.F32 || node.Shape != value.Shape {
			err = errors.New("resident graph: static F32 input shape differs")
			break
		}
		slot, ok := compiled.InputSlot(node)
		if !ok || runtime.inputs.Pointers[slot] != 0 {
			err = fmt.Errorf("resident graph: static input %q is invalid", node.Name)
			break
		}
		var pointer driver.DevicePtr
		pointer, err = runtime.allocations.Upload(ctx, driver.Bytes(value.Data))
		if err != nil {
			break
		}
		runtime.inputs.Pointers[slot] = pointer
		runtime.staticBytes += uint64(len(value.Data)) * 4
	}
	if err == nil {
		err = runtime.executor.PrepareCompiled(ctx, compiled)
	}
	if err != nil {
		return nil, errors.Join(err, runtime.Close(context.Background()))
	}
	return runtime, nil
}

type ResidentStats struct {
	StaticBytes uint64
	Device      driver.MemoryStats
	Execution   driver.ExecutionStats
}

func (runtime *ResidentGraph) Stats(ctx context.Context) (ResidentStats, error) {
	if runtime == nil || runtime.worker == nil {
		return ResidentStats{}, errors.New("resident graph: runtime is unavailable")
	}
	stats, err := runtime.worker.MemoryStats(ctx)
	if err != nil {
		return ResidentStats{}, err
	}
	execution, err := runtime.worker.ExecutionStats(ctx)
	return ResidentStats{StaticBytes: runtime.staticBytes, Device: stats, Execution: execution}, err
}

func (runtime *ResidentGraph) Execute(
	ctx context.Context,
	dynamic map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	if runtime == nil || runtime.executor == nil || runtime.compiled == nil {
		return nil, errors.New("resident graph: runtime is unavailable")
	}
	return runtime.executor.ExecuteCompiled(ctx, runtime.compiled, dynamic, runtime.inputs)
}

func (runtime *ResidentGraph) Close(ctx context.Context) error {
	if runtime == nil {
		return nil
	}
	var result error
	if runtime.executor != nil {
		result = errors.Join(result, runtime.executor.Close())
		runtime.executor = nil
	}
	if runtime.worker != nil {
		result = errors.Join(result, runtime.allocations.Close(ctx), runtime.worker.Close())
		runtime.worker = nil
	}
	runtime.compiled = nil
	runtime.inputs = nil
	runtime.staticBytes = 0
	return result
}
