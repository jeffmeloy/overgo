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

// CreateLt creates a cuBLASLt context.
func (l *Library) CreateLt() (LtHandle, error) {
	return 0, errors.New("cuBLAS loading is currently supported only on Windows")
}

// DestroyLt releases a cuBLASLt context.
func (l *Library) DestroyLt(handle LtHandle) error {
	return errors.New("cuBLAS loading is currently supported only on Windows")
}

// PlanFP8 builds an FP8 product plan.
func (l *Library) PlanFP8(handle LtHandle, m, n, k, workspaceLimit uint64) (*FP8Plan, error) {
	return nil, errors.New("cuBLAS loading is currently supported only on Windows")
}

// MatmulFP8 enqueues an FP8 product.
func (l *Library) MatmulFP8(
	handle LtHandle,
	plan *FP8Plan,
	a, b, d, workspace driver.DevicePtr,
	workspaceBytes uint64,
	stream driver.Stream,
) error {
	return errors.New("cuBLAS loading is currently supported only on Windows")
}

// DestroyFP8Plan releases a plan's descriptors.
func (l *Library) DestroyFP8Plan(plan *FP8Plan) error {
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
