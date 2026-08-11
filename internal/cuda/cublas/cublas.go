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
// pointers. cuBLAS is column-major, so the row-major product is computed
// transposed (Cᵀ = Bᵀ·Aᵀ): GEMM(N,N, n, m, k, B, A, C) with leading dims n, k,
// n. Encapsulates the layout convention once so device Newton-Schulz and other
// matmul-composed callers do not re-derive it. Calls GEMMEx, so it resolves to
// the real or unsupported backend like any other cuBLAS op.
func (l *Library) RowMajorGEMMF32(handle Handle, m, k, n int32, a, b, c driver.DevicePtr) error {
	return l.GEMMEx(
		handle, OperationNone, OperationNone,
		n, m, k, 1,
		b, DataF32, n,
		a, DataF32, k,
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
