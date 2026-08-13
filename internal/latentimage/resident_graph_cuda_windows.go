//go:build windows

package latentimage

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func storageBytes(dataType dtype.Type) int {
	if dataType == dtype.BF16 {
		return 2
	}
	return 4
}

func weightPayload(value safetensors.Tensor, storage dtype.Type) ([]byte, error) {
	elements := int(value.Elements())
	if storage == dtype.BF16 && value.DType == "BF16" {
		buffer := make([]byte, elements*2)
		_, err := io.ReadFull(value.Reader(), buffer)
		return buffer, err
	}
	reader, err := safetensors.F32Reader(value)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, elements*4)
	if _, err := io.ReadFull(reader, raw); err != nil || storage == dtype.F32 {
		return raw, err
	}
	result := make([]byte, elements*2)
	for index := range elements {
		item := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*index:]))
		binary.LittleEndian.PutUint16(result[2*index:], dtype.Float32ToBF16(item))
	}
	return result, nil
}

// residentRuntime owns one CUDA context, executor, and deduplicated weight set.
type residentRuntime struct {
	worker      *device.Worker
	exec        *executor.Executor
	weights     map[residentWeight]driver.DevicePtr
	weightRefs  map[residentWeight]int
	allocs      []driver.DevicePtr
	weightBytes uint64
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
	weights  map[residentWeight]struct{}
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
		weightRefs: make(map[residentWeight]int),
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
	graph := &residentGraph{feeds: make(map[*tensor.Tensor]driver.DevicePtr, len(inputs))}
	var err error
	if len(inputs) == 0 {
		graph.compiled, err = executor.Compile(outputs...)
		if err != nil {
			return nil, fmt.Errorf("%s: compile: %w", label, err)
		}
		return graph, nil
	}
	source, err := safetensors.OpenSource(sourcePath)
	if err != nil {
		return nil, fmt.Errorf("%s: open weights: %w", label, err)
	}
	defer source.Close()
	for name, node := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tensorValue, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("%s: missing tensor %s", label, name)
		}
		key := residentWeight{source: sourcePath, name: name, type_: node.Type, shape: node.Shape}
		if err := r.bind(ctx, graph, key, node, func() ([]byte, error) {
			return weightPayload(tensorValue, node.Type)
		}); err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", label, name, err)
		}
	}
	graph.compiled, err = executor.Compile(outputs...)
	if err != nil {
		return nil, fmt.Errorf("%s: compile: %w", label, err)
	}
	return graph, nil
}

func (r *residentRuntime) compileStatic(
	ctx context.Context,
	label string,
	inputs map[*tensor.Tensor][]float32,
	outputs ...*tensor.Tensor,
) (*residentGraph, error) {
	if r == nil || r.worker == nil || r.exec == nil {
		return nil, errors.New("resident graph: runtime is unavailable")
	}
	graph := &residentGraph{feeds: make(map[*tensor.Tensor]driver.DevicePtr, len(inputs))}
	for node, values := range inputs {
		if err := r.bindStatic(ctx, graph, label, node, values); err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", label, node.Name, err)
		}
	}
	compiled, err := executor.Compile(outputs...)
	if err != nil {
		return nil, fmt.Errorf("%s: compile: %w", label, err)
	}
	graph.compiled = compiled
	return graph, nil
}

func (r *residentRuntime) bindStatic(
	ctx context.Context,
	graph *residentGraph,
	label string,
	node *tensor.Tensor,
	values []float32,
) error {
	if node == nil || node.Type != dtype.F32 {
		return fmt.Errorf("%s: static input must be F32", label)
	}
	key := residentWeight{source: "static:" + label, name: node.Name, type_: node.Type, shape: node.Shape}
	return r.bind(ctx, graph, key, node, func() ([]byte, error) {
		return driver.Bytes(values), nil
	})
}

func (r *residentRuntime) bind(
	ctx context.Context,
	graph *residentGraph,
	key residentWeight,
	node *tensor.Tensor,
	payload func() ([]byte, error),
) error {
	elements, err := node.Shape.Elements()
	if err != nil {
		return err
	}
	bytes := elements * uint64(storageBytes(node.Type))
	pointer, found := r.weights[key]
	if !found {
		data, err := payload()
		if err != nil {
			return err
		}
		if uint64(len(data)) != bytes {
			return fmt.Errorf("payload=%d want %d", len(data), bytes)
		}
		if err := r.worker.Do(ctx, func(state *device.State) error {
			allocated, allocErr := state.Driver.MemAlloc(bytes)
			if allocErr != nil {
				return allocErr
			}
			if copyErr := state.Driver.MemcpyHtoD(allocated, data); copyErr != nil {
				return errors.Join(copyErr, state.Driver.MemFree(allocated))
			}
			pointer = allocated
			return nil
		}); err != nil {
			return err
		}
		r.weights[key] = pointer
		r.allocs = append(r.allocs, pointer)
		r.weightBytes += bytes
	}
	graph.feeds[node] = pointer
	if graph.weights == nil {
		graph.weights = make(map[residentWeight]struct{})
	}
	if _, bound := graph.weights[key]; !bound {
		graph.weights[key] = struct{}{}
		r.weightRefs[key]++
		graph.bytes += bytes
	}
	return nil
}

func (r *residentRuntime) releaseGraphs(ctx context.Context, graphs ...*residentGraph) error {
	if r == nil || r.worker == nil {
		return errors.New("resident graph: runtime is unavailable")
	}
	return r.worker.Do(ctx, func(state *device.State) error {
		var errs []error
		for _, graph := range graphs {
			if graph == nil {
				continue
			}
			for key := range graph.weights {
				refs := r.weightRefs[key] - 1
				if refs > 0 {
					r.weightRefs[key] = refs
					continue
				}
				pointer := r.weights[key]
				if pointer != 0 {
					errs = append(errs, state.Driver.MemFree(pointer))
					for index, allocated := range r.allocs {
						if allocated == pointer {
							r.allocs[index] = 0
							break
						}
					}
				}
				bytes, err := key.shape.Bytes(key.type_)
				if err == nil && bytes <= r.weightBytes {
					r.weightBytes -= bytes
				}
				delete(r.weights, key)
				delete(r.weightRefs, key)
			}
			graph.feeds = nil
			graph.weights = nil
			graph.bytes = 0
		}
		return errors.Join(errs...)
	})
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

func (r *residentRuntime) executeWithDevices(
	ctx context.Context,
	graph *residentGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	feeds, err := graphDeviceFeeds(graph, deviceFeeds)
	if err != nil {
		return nil, err
	}
	return r.exec.ExecuteCompiledWithDeviceFeeds(ctx, graph.compiled, hostFeeds, feeds)
}

func (r *residentRuntime) retain(
	ctx context.Context,
	graph *residentGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*executor.RetainedOutputs, error) {
	feeds, err := graphDeviceFeeds(graph, deviceFeeds)
	if err != nil {
		return nil, err
	}
	return r.exec.ExecuteRetainedCompiledWithDeviceFeeds(ctx, graph.compiled, hostFeeds, feeds)
}

func graphDeviceFeeds(
	graph *residentGraph,
	dynamic map[*tensor.Tensor]driver.DevicePtr,
) (map[*tensor.Tensor]driver.DevicePtr, error) {
	if graph == nil || graph.compiled == nil {
		return nil, errors.New("resident graph: execution is unavailable")
	}
	if len(dynamic) == 0 {
		return graph.feeds, nil
	}
	feeds := make(map[*tensor.Tensor]driver.DevicePtr, len(graph.feeds)+len(dynamic))
	for node, pointer := range graph.feeds {
		feeds[node] = pointer
	}
	for node, pointer := range dynamic {
		if pointer == 0 {
			return nil, errors.New("resident graph: dynamic device feed is empty")
		}
		if _, duplicate := feeds[node]; duplicate {
			return nil, errors.New("resident graph: dynamic device feed replaces a weight")
		}
		feeds[node] = pointer
	}
	return feeds, nil
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
				if pointer != 0 {
					freeErrors = append(freeErrors, state.Driver.MemFree(pointer))
				}
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
