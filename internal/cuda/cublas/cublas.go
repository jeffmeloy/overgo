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

const (
	OperationNone      Operation     = 0
	OperationTranspose Operation     = 1
	DataF32            DataType      = 0
	DataBF16           DataType      = 14
	ComputeF32         ComputeType   = 68
	GemmDefault        GemmAlgorithm = -1
)

// RowMajorGEMMF32 computes row-major C[m,n] = A[m,k] · B[k,n] in fp32 on device
// pointers (no transposes). Convenience over RowMajorGEMMExF32.
func (l *Library) RowMajorGEMMF32(handle Handle, m, k, n int32, a, b, c driver.DevicePtr) error {
	return l.RowMajorGEMMExF32(handle, false, false, m, k, n, a, b, c)
}

// RowMajorGEMMExF32 computes row-major C[m,n] = op(A) · op(B) in fp32 on device
// pointers, where op(A) is [m,k] and op(B) is [k,n]. transA/transB select
// whether the stored row-major A/B are used transposed (so the gram products
// XᵀX and XXᵀ need no materialized transpose).
//
// cuBLAS is column-major; a row-major matrix is seen as its transpose. The
// row-major product is therefore obtained by computing Cᵀ = op(B)ᵀ·op(A)ᵀ with
// the operands swapped: GEMM(opB, opA, n, m, k, B, A, C). Leading dims are the
// stored row-major widths (lda = k or m under transA; ldb = n or k under
// transB; ldc = n). With both flags off this reduces to the verified
// GEMM(N,N, n,m,k, B(n), A(k), C(n)) formula.
func (l *Library) RowMajorGEMMExF32(handle Handle, transA, transB bool, m, k, n int32, a, b, c driver.DevicePtr) error {
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
		n, m, k, 1,
		b, DataF32, ldb,
		a, DataF32, lda,
		0, c, DataF32, n,
		ComputeF32, GemmDefault,
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
