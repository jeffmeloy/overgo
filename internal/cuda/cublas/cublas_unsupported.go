//go:build !windows

package cublas

import (
	"errors"

	"overgo/internal/cuda/driver"
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

func (l *Library) SetMathMode(handle Handle, mode MathMode) error {
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

func (l *Library) SGEMMStridedBatched(
	handle Handle,
	operationA, operationB Operation,
	m, n, k int32,
	alpha float32,
	a driver.DevicePtr,
	leadingA int32,
	strideA int64,
	b driver.DevicePtr,
	leadingB int32,
	strideB int64,
	beta float32,
	c driver.DevicePtr,
	leadingC int32,
	strideC int64,
	batch int32,
) error {
	return errors.New("cuBLAS loading is currently supported only on Windows")
}

func (l *Library) GEMMEx(
	handle Handle,
	operationA, operationB Operation,
	m, n, k int32,
	alpha float32,
	a driver.DevicePtr,
	typeA DataType,
	leadingA int32,
	b driver.DevicePtr,
	typeB DataType,
	leadingB int32,
	beta float32,
	c driver.DevicePtr,
	typeC DataType,
	leadingC int32,
	compute ComputeType,
	algorithm GemmAlgorithm,
) error {
	return errors.New("cuBLAS loading is currently supported only on Windows")
}
