//go:build !windows

package cublas

import (
	"errors"

	"llamacpp2go/internal/cuda/driver"
)

type Library struct{}

func Open() (*Library, error) {
	return nil, errors.New("cuBLAS loading is currently supported only on Windows")
}

func (l *Library) Close() error {
	return nil
}

func (l *Library) Create() (Handle, error) {
	return 0, errors.New("cuBLAS loading is currently supported only on Windows")
}

func (l *Library) Destroy(handle Handle) error {
	return errors.New("cuBLAS loading is currently supported only on Windows")
}

func (l *Library) SetStream(handle Handle, stream driver.Stream) error {
	return errors.New("cuBLAS loading is currently supported only on Windows")
}

func (l *Library) SGEMM(
	handle Handle,
	operationA Operation,
	operationB Operation,
	m, n, k int32,
	alpha float32,
	a driver.DevicePtr,
	leadingA int32,
	b driver.DevicePtr,
	leadingB int32,
	beta float32,
	c driver.DevicePtr,
	leadingC int32,
) error {
	return errors.New("cuBLAS loading is currently supported only on Windows")
}
