package cublas

import (
	"fmt"

	"overgo/internal/cuda/driver"
)

type Handle uintptr

type Operation int32

type DataType int32

type ComputeType int32

type GemmAlgorithm int32

type MathMode int32

const (
	OperationNone Operation = iota
	OperationTranspose
)

const (
	DataF32 DataType = 0
	// DataF16 is cuBLAS CUDA_R_16F: IEEE half-precision operands.
	DataF16              DataType      = 2
	DataBF16             DataType      = 14
	ComputeF32           ComputeType   = 68
	GemmDefault          GemmAlgorithm = -1
	MathTF32TensorOp     MathMode      = 3
	gemmProductScale                   = float32(1)
	gemmAccumulatorScale               = float32(0)
)

// RowMajorGEMMF32: row-major C[m,n] = A[m,k]*B[k,n].
func (l *Library) RowMajorGEMMF32(handle Handle, m, k, n int32, a, b, c driver.DevicePtr) error {
	return l.RowMajorGEMMExF32(handle, false, false, m, k, n, a, b, c)
}

// RowMajorGEMMExF32: row-major C[m,n] = op(A)*op(B).
// Transpose flags avoid materialized X^T*X and X*X^T inputs.
// cuBLAS uses the equivalent swapped column-major product.
func (l *Library) RowMajorGEMMExF32(handle Handle, transA, transB bool, m, k, n int32, a, b, c driver.DevicePtr) error {
	return l.rowMajorGEMMEx(handle, transA, transB, m, k, n, a, DataF32, b, DataF32, c)
}

// RowMajorGEMMExBF16 computes from BF16 operands into FP32 output.
func (l *Library) RowMajorGEMMExBF16(handle Handle, transA, transB bool, m, k, n int32, a, b, c driver.DevicePtr) error {
	return l.rowMajorGEMMEx(handle, transA, transB, m, k, n, a, DataBF16, b, DataBF16, c)
}

func (l *Library) rowMajorGEMMEx(handle Handle, transA, transB bool, m, k, n int32, a driver.DevicePtr, typeA DataType, b driver.DevicePtr, typeB DataType, c driver.DevicePtr) error {
	opA, lda := OperationNone, k
	if transA {
		opA, lda = OperationTranspose, m
	}
	opB, ldb := OperationNone, n
	if transB {
		opB, ldb = OperationTranspose, k
	}
	return l.GEMMEx(
		handle, opB, opA,
		n, m, k, gemmProductScale,
		b, typeB, ldb,
		a, typeA, lda,
		gemmAccumulatorScale, c, DataF32, n,
		ComputeF32, GemmDefault,
	)
}

// RowMajorGEMMStridedBatchedF32 computes independent row-major products.
func (l *Library) RowMajorGEMMStridedBatchedF32(
	handle Handle,
	transA, transB bool,
	m, k, n int32,
	a driver.DevicePtr,
	strideA int64,
	b driver.DevicePtr,
	strideB int64,
	c driver.DevicePtr,
	strideC int64,
	batch int32,
) error {
	opA, lda := OperationNone, k
	if transA {
		opA, lda = OperationTranspose, m
	}
	opB, ldb := OperationNone, n
	if transB {
		opB, ldb = OperationTranspose, k
	}
	return l.SGEMMStridedBatched(
		handle, opB, opA,
		n, m, k, gemmProductScale,
		b, ldb, strideB,
		a, lda, strideA,
		gemmAccumulatorScale, c, n, strideC,
		batch,
	)
}

// StatusError: reports cuBLAS API failure
type StatusError struct {
	Operation string
	Status    int32
}

func (e *StatusError) Error() string {
	name := statusName(e.Status)
	if e.Operation == "" {
		return name
	}
	return fmt.Sprintf("%s: %s", e.Operation, name)
}

func statusName(status int32) string {
	switch status {
	case 0:
		return "CUBLAS_STATUS_SUCCESS"
	case 1:
		return "CUBLAS_STATUS_NOT_INITIALIZED"
	case 3:
		return "CUBLAS_STATUS_ALLOC_FAILED"
	case 7:
		return "CUBLAS_STATUS_INVALID_VALUE"
	case 8:
		return "CUBLAS_STATUS_ARCH_MISMATCH"
	case 11:
		return "CUBLAS_STATUS_MAPPING_ERROR"
	case 13:
		return "CUBLAS_STATUS_EXECUTION_FAILED"
	case 14:
		return "CUBLAS_STATUS_INTERNAL_ERROR"
	case 15:
		return "CUBLAS_STATUS_NOT_SUPPORTED"
	case 16:
		return "CUBLAS_STATUS_LICENSE_ERROR"
	default:
		return fmt.Sprintf("cuBLAS status %d", status)
	}
}
