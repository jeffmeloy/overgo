package cublas

import "fmt"

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
