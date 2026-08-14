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
	worker        *device.Worker
	exec          *executor.Executor
	weightIndexes map[residentWeight]int
	weightSlots   []residentWeightSlot
	weightBytes   uint64
}

type residentWeight struct {
	source string
	name   string
	type_  dtype.Type
	shape  tensor.Shape
}

type residentWeightSlot struct {
	key     residentWeight
	pointer driver.DevicePtr
	bytes   uint64
	refs    int
}

type residentGraph struct {
	compiled *executor.CompiledGraph
	inputs   *executor.DeviceInputs
	weights  []int
	bytes    uint64
}

type dynamicDeviceInput struct {
	slot    executor.InputSlot
	pointer driver.DevicePtr
}

func compileDynamicDeviceInput(graph *residentGraph, node *tensor.Tensor) (dynamicDeviceInput, error) {
	if graph == nil || graph.compiled == nil {
		return dynamicDeviceInput{}, errors.New("resident graph: compiled graph is unavailable")
	}
	slot, ok := graph.compiled.InputSlot(node)
	if !ok {
		return dynamicDeviceInput{}, errors.New("resident graph: dynamic input is not compiled")
	}
	return dynamicDeviceInput{slot: slot}, nil
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
		worker: worker, exec: exec, weightIndexes: make(map[residentWeight]int),
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
	graph, err := compileResidentGraph(label, outputs)
	if err != nil || len(inputs) == 0 {
		return graph, err
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
	graph, err := compileResidentGraph(label, outputs)
	if err != nil {
		return nil, err
	}
	for node, values := range inputs {
		if err := r.bindStatic(ctx, graph, label, node, values); err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", label, node.Name, err)
		}
	}
	return graph, nil
}

func compileResidentGraph(label string, outputs []*tensor.Tensor) (*residentGraph, error) {
	compiled, err := executor.Compile(outputs...)
	if err != nil {
		return nil, fmt.Errorf("%s: compile: %w", label, err)
	}
	return &residentGraph{compiled: compiled, inputs: compiled.NewDeviceInputs()}, nil
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
	if r.weightIndexes == nil {
		return errors.New("resident graph: weight ownership is sealed")
	}
	if graph == nil || graph.compiled == nil || graph.inputs == nil {
		return errors.New("resident graph: static input graph is unavailable")
	}
	slot, ok := graph.compiled.InputSlot(node)
	if !ok || graph.inputs.Pointers[slot] != 0 {
		return errors.New("resident graph: static input is invalid")
	}
	elements, err := node.Shape.Elements()
	if err != nil {
		return err
	}
	bytes := elements * uint64(storageBytes(node.Type))
	slotIndex, found := r.weightIndexes[key]
	var pointer driver.DevicePtr
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
		slotIndex = len(r.weightSlots)
		r.weightIndexes[key] = slotIndex
		r.weightSlots = append(r.weightSlots, residentWeightSlot{
			key: key, pointer: pointer, bytes: bytes,
		})
		r.weightBytes += bytes
	} else {
		pointer = r.weightSlots[slotIndex].pointer
	}
	if pointer == 0 {
		return errors.New("resident graph: static input is invalid")
	}
	graph.inputs.Pointers[slot] = pointer
	graph.weights = append(graph.weights, slotIndex)
	r.weightSlots[slotIndex].refs++
	graph.bytes += bytes
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
			for _, slotIndex := range graph.weights {
				slot := &r.weightSlots[slotIndex]
				slot.refs--
				if slot.refs > 0 {
					continue
				}
				if slot.pointer != 0 {
					errs = append(errs, state.Driver.MemFree(slot.pointer))
				}
				if slot.bytes <= r.weightBytes {
					r.weightBytes -= slot.bytes
				}
				delete(r.weightIndexes, slot.key)
				*slot = residentWeightSlot{}
			}
			graph.inputs = nil
			graph.weights = nil
			graph.bytes = 0
		}
		return errors.Join(errs...)
	})
}

func (r *residentRuntime) sealWeights() {
	r.weightIndexes = nil
	for index := range r.weightSlots {
		r.weightSlots[index].key = residentWeight{}
	}
}

func (r *residentRuntime) execute(
	ctx context.Context,
	graph *residentGraph,
	feeds map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	if r == nil || r.exec == nil || graph == nil || graph.compiled == nil {
		return nil, errors.New("resident graph: execution is unavailable")
	}
	return r.exec.ExecuteCompiledWithDeviceInputs(ctx, graph.compiled, feeds, graph.inputs)
}

func (r *residentRuntime) executeWithDevices(
	ctx context.Context,
	graph *residentGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceInputs []dynamicDeviceInput,
) (map[*tensor.Tensor]reference.Value, error) {
	if err := bindDynamicDeviceInputs(graph, deviceInputs); err != nil {
		return nil, err
	}
	defer clearDynamicDeviceInputs(graph, deviceInputs)
	return r.exec.ExecuteCompiledWithDeviceInputs(ctx, graph.compiled, hostFeeds, graph.inputs)
}

func (r *residentRuntime) retain(
	ctx context.Context,
	graph *residentGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceInputs []dynamicDeviceInput,
) (*executor.RetainedOutputs, error) {
	if err := bindDynamicDeviceInputs(graph, deviceInputs); err != nil {
		return nil, err
	}
	defer clearDynamicDeviceInputs(graph, deviceInputs)
	return r.exec.ExecuteRetainedCompiledWithDeviceInputs(ctx, graph.compiled, hostFeeds, graph.inputs, nil)
}

func bindDynamicDeviceInputs(
	graph *residentGraph,
	dynamic []dynamicDeviceInput,
) error {
	if graph == nil || graph.compiled == nil || graph.inputs == nil {
		return errors.New("resident graph: execution is unavailable")
	}
	for index, input := range dynamic {
		if graph.inputs.Pointers[input.slot] != 0 {
			clearDynamicDeviceInputs(graph, dynamic[:index])
			return errors.New("resident graph: dynamic device input replaces a weight")
		}
		if input.pointer == 0 {
			clearDynamicDeviceInputs(graph, dynamic[:index])
			return errors.New("resident graph: dynamic device input is empty")
		}
		graph.inputs.Pointers[input.slot] = input.pointer
	}
	return nil
}

func clearDynamicDeviceInputs(graph *residentGraph, dynamic []dynamicDeviceInput) {
	if graph == nil || graph.inputs == nil {
		return
	}
	for _, input := range dynamic {
		graph.inputs.Pointers[input.slot] = 0
	}
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
	if r.worker != nil && len(r.weightSlots) != 0 {
		errs = append(errs, r.worker.Do(ctx, func(state *device.State) error {
			var freeErrors []error
			for index := range r.weightSlots {
				if pointer := r.weightSlots[index].pointer; pointer != 0 {
					freeErrors = append(freeErrors, state.Driver.MemFree(pointer))
					r.weightSlots[index] = residentWeightSlot{}
				}
			}
			return errors.Join(freeErrors...)
		}))
		r.weightSlots = nil
		r.weightIndexes = nil
	}
	if r.worker != nil {
		errs = append(errs, r.worker.Close())
		r.worker = nil
	}
	return errors.Join(errs...)
}
