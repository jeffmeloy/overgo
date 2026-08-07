package graphruntime

import (
	"errors"

	"overgo/internal/cuda/driver"
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

type HostExecutor func([]*tensor.Tensor, map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error)

type DeviceExecutor func(
	[]*tensor.Tensor,
	map[*tensor.Tensor]reference.Value,
	map[*tensor.Tensor]driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error)

func (f *Feeds) Execute(
	outputs []*tensor.Tensor,
	device bool,
	host HostExecutor,
	deviceExecutor DeviceExecutor,
) (map[*tensor.Tensor]reference.Value, error) {
	if device {
		if deviceExecutor == nil {
			return nil, errors.New("graph runtime device executor is nil")
		}
		return deviceExecutor(outputs, f.Host, f.Device)
	}
	if host == nil {
		return nil, errors.New("graph runtime host executor is nil")
	}
	return host(outputs, f.Host)
}
