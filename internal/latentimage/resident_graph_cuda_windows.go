//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// residentRuntime owns one CUDA context, executor, and deduplicated weight set.
type residentRuntime struct {
	worker  *device.Worker
	exec    *executor.Executor
	weights map[residentWeight]driver.DevicePtr
	allocs  []driver.DevicePtr
}

type residentWeight struct {
	source string
	name   string
	type_  dtype.Type
	shape  tensor.Shape
}

type residentGraph struct {
	compiled *executor.CompiledGraph
	feeds    map[*tensor.Tensor]driver.DevicePtr
	bytes    uint64
}

func newResidentRuntime(ordinal int) (*residentRuntime, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	exec, err := executor.NewWithWorker(worker)
	if err != nil {
		_ = worker.Close()
		return nil, err
	}
	return &residentRuntime{
		worker: worker, exec: exec, weights: make(map[residentWeight]driver.DevicePtr),
	}, nil
}

func (r *residentRuntime) compile(
	ctx context.Context,
	label, sourcePath string,
	inputs map[string]*tensor.Tensor,
	outputs ...*tensor.Tensor,
) (*residentGraph, error) {
	if r == nil || r.worker == nil || r.exec == nil {
		return nil, errors.New("resident graph: runtime is unavailable")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(sourcePath)
	if err != nil {
		return nil, fmt.Errorf("%s: open weights: %w", label, err)
	}
	defer source.Close()

	graph := &residentGraph{feeds: make(map[*tensor.Tensor]driver.DevicePtr, len(inputs))}
	for name, node := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tensorValue, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("%s: missing tensor %s", label, name)
		}
		elements, err := node.Shape.Elements()
		if err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", label, name, err)
		}
		bytes := elements * uint64(storageBytes(node.Type))
		key := residentWeight{source: sourcePath, name: name, type_: node.Type, shape: node.Shape}
		pointer, found := r.weights[key]
		if !found {
			payload, payloadErr := weightPayload(tensorValue, node.Type)
			if payloadErr != nil {
				return nil, fmt.Errorf("%s: tensor %s: %w", label, name, payloadErr)
			}
			if uint64(len(payload)) != bytes {
				return nil, fmt.Errorf("%s: tensor %s payload=%d want %d", label, name, len(payload), bytes)
			}
			if err := r.worker.Do(ctx, func(state *device.State) error {
				allocated, allocErr := state.Driver.MemAlloc(bytes)
				if allocErr != nil {
					return allocErr
				}
				if copyErr := state.Driver.MemcpyHtoD(allocated, payload); copyErr != nil {
					return errors.Join(copyErr, state.Driver.MemFree(allocated))
				}
				pointer = allocated
				return nil
			}); err != nil {
				return nil, fmt.Errorf("%s: upload %s: %w", label, name, err)
			}
			r.weights[key] = pointer
			r.allocs = append(r.allocs, pointer)
		}
		graph.feeds[node] = pointer
		graph.bytes += bytes
	}
	graph.compiled, err = executor.Compile(outputs...)
	if err != nil {
		return nil, fmt.Errorf("%s: compile: %w", label, err)
	}
	return graph, nil
}

func (r *residentRuntime) execute(
	ctx context.Context,
	graph *residentGraph,
	feeds map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	if r == nil || r.exec == nil || graph == nil || graph.compiled == nil {
		return nil, errors.New("resident graph: execution is unavailable")
	}
	return r.exec.ExecuteCompiledWithDeviceFeeds(ctx, graph.compiled, feeds, graph.feeds)
}

func (r *residentRuntime) close(ctx context.Context) error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.exec != nil {
		errs = append(errs, r.exec.Close())
		r.exec = nil
	}
	if r.worker != nil && len(r.allocs) != 0 {
		errs = append(errs, r.worker.Do(ctx, func(state *device.State) error {
			var freeErrors []error
			for _, pointer := range r.allocs {
				freeErrors = append(freeErrors, state.Driver.MemFree(pointer))
			}
			return errors.Join(freeErrors...)
		}))
		r.allocs = nil
	}
	if r.worker != nil {
		errs = append(errs, r.worker.Close())
		r.worker = nil
	}
	return errors.Join(errs...)
}
