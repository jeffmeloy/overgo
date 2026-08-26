package graphruntime

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// Runner executes one tensor graph.
type Runner func([]*tensor.Tensor, map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error)

// AddHostWeights resolves ordered F32 bindings into host feeds.
func AddHostWeights(
	feeds map[*tensor.Tensor]reference.Value,
	bindings tensor.WeightBindings,
	load func(string) ([]float32, error),
) error {
	for _, node := range bindings {
		if node.Type != dtype.F32 {
			return fmt.Errorf("weight %s: host feed requires F32 storage, compiled %s", node.Name, node.Type)
		}
		data, err := load(node.Name)
		if err != nil {
			return err
		}
		elements, err := node.Shape.Elements()
		if err != nil || uint64(len(data)) != elements {
			return fmt.Errorf("weight %s: have %d elements, need %d", node.Name, len(data), elements)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
	}
	return nil
}

type Feeds struct {
	host               tensor.InputBindings[reference.Value]
	device             tensor.InputBindings[driver.DevicePtr]
	graph              *executor.IndexedGraph
	reference          *reference.Program
	referenceInputs    *reference.Inputs
	referenceWorkspace *reference.Workspace
	deviceOutput       []*tensor.Tensor
	referenceOutput    []*tensor.Tensor
}

func NewFeeds() *Feeds {
	return &Feeds{}
}

func (f *Feeds) Input(builder *tensor.Builder, name string, value reference.Value) *tensor.Tensor {
	node := builder.Input(name, dtype.F32, value.Shape)
	f.host.Add(node, value)
	return node
}

func (f *Feeds) AddHost(values tensor.InputBindings[reference.Value]) {
	f.host = append(f.host, values...)
}

func (f *Feeds) SetHost(node *tensor.Tensor, value reference.Value) {
	f.host.Add(node, value)
}

func (f *Feeds) HostBindings() *tensor.InputBindings[reference.Value] {
	return &f.host
}

func (f *Feeds) AddDevice(values tensor.InputBindings[driver.DevicePtr]) {
	f.device = append(f.device, values...)
}

func (f *Feeds) SetDevice(node *tensor.Tensor, pointer driver.DevicePtr) {
	f.device.Add(node, pointer)
}

func (f *Feeds) Execute(
	ctx context.Context,
	outputs []*tensor.Tensor,
	device *executor.Executor,
) (map[*tensor.Tensor]reference.Value, error) {
	if f == nil {
		return nil, errors.New("graph runtime feeds are nil")
	}
	if device == nil {
		program, err := f.compileReference(outputs)
		if err != nil {
			return nil, err
		}
		return program.Execute(f.referenceInputs, f.referenceWorkspace)
	}
	indexed, err := f.compile(outputs)
	if err != nil {
		return nil, err
	}
	host := make(map[*tensor.Tensor]reference.Value, len(f.host))
	for _, binding := range f.host {
		host[binding.Node] = binding.Value
	}
	return device.ExecuteCompiled(ctx, indexed.Graph, host, indexed.Inputs)
}

func (f *Feeds) compileReference(outputs []*tensor.Tensor) (*reference.Program, error) {
	if f.reference != nil && slices.Equal(f.referenceOutput, outputs) {
		return f.reference, nil
	}
	program, err := reference.Compile(outputs...)
	if err != nil {
		return nil, err
	}
	inputs := program.NewInputs()
	for _, binding := range f.host {
		if program.HasInput(binding.Node) {
			if err := inputs.Set(binding.Node, binding.Value); err != nil {
				return nil, err
			}
		}
	}
	f.reference, f.referenceInputs = program, inputs
	f.referenceWorkspace, f.referenceOutput = program.NewWorkspace(), slices.Clone(outputs)
	return program, nil
}

func (f *Feeds) compile(outputs []*tensor.Tensor) (*executor.IndexedGraph, error) {
	if f.graph != nil && slices.Equal(f.deviceOutput, outputs) {
		return f.graph, nil
	}
	indexed, err := executor.CompileIndexed(outputs...)
	if err != nil {
		return nil, err
	}
	if err := indexed.Inputs.Bind(f.device); err != nil {
		return nil, err
	}
	f.graph, f.deviceOutput = indexed, slices.Clone(outputs)
	return indexed, nil
}
