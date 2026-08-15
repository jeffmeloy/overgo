package graphruntime

import (
	"context"
	"errors"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type Feeds struct {
	Host   map[*tensor.Tensor]reference.Value
	Device map[*tensor.Tensor]driver.DevicePtr
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
	compiled, err := executor.Compile(outputs...)
	if err != nil {
		return nil, err
	}
	inputs, err := compiled.BindDeviceInputs(f.Device)
	if err != nil {
		return nil, err
	}
	return device.ExecuteCompiled(ctx, compiled, f.Host, inputs)
}
