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
	Host   map[*tensor.Tensor]reference.Value
	Device map[*tensor.Tensor]driver.DevicePtr
	graph  *executor.IndexedGraph
	output []*tensor.Tensor
}

func NewFeeds() *Feeds {
	return &Feeds{
		Host:   make(map[*tensor.Tensor]reference.Value),
		Device: make(map[*tensor.Tensor]driver.DevicePtr),
	}
}

func (f *Feeds) Input(builder *tensor.Builder, name string, value reference.Value) *tensor.Tensor {
	node := builder.Input(name, dtype.F32, value.Shape)
	f.Host[node] = value
	return node
}

func (f *Feeds) AddHost(values map[*tensor.Tensor]reference.Value) {
	for node, value := range values {
		f.Host[node] = value
	}
}

func (f *Feeds) AddDevice(values map[*tensor.Tensor]driver.DevicePtr) {
	for node, value := range values {
		f.Device[node] = value
	}
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
		return reference.Execute(outputs, f.Host)
	}
	indexed, err := f.compile(outputs)
	if err != nil {
		return nil, err
	}
	for node, pointer := range f.Device {
		if err := indexed.Inputs.Set(node, pointer); err != nil {
			return nil, err
		}
	}
	return device.ExecuteCompiled(ctx, indexed.Graph, f.Host, indexed.Inputs)
}

func (f *Feeds) compile(outputs []*tensor.Tensor) (*executor.IndexedGraph, error) {
	if f.graph != nil && slices.Equal(f.output, outputs) {
		return f.graph, nil
	}
	indexed, err := executor.CompileIndexed(outputs...)
	if err != nil {
		return nil, err
	}
	f.graph, f.output = indexed, slices.Clone(outputs)
	return indexed, nil
}
