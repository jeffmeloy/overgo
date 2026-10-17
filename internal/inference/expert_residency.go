package inference

import (
	"errors"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// splitExpertResidency excludes eligible banks only when a native preload cannot fit.
// Runtime cache and scratch allocations reclaim completed experts through the executor.
func splitExpertResidency(raw, converted []gguf.TensorInfo, banks map[string]struct{}, available uint64) ([]gguf.TensorInfo, []gguf.TensorInfo, error) {
	var total uint64
	for _, info := range raw {
		next, ok := checked.Add64(total, info.Size)
		if !ok {
			return nil, nil, errors.New("native weight extent overflows")
		}
		total = next
	}
	for _, info := range converted {
		shape, err := tensor.NewShape(info.Shape[:info.Dimensions]...)
		if err != nil {
			return nil, nil, err
		}
		bytes, err := shape.Bytes(dtype.F32)
		if err != nil {
			return nil, nil, err
		}
		next, ok := checked.Add64(total, bytes)
		if !ok {
			return nil, nil, errors.New("converted weight extent overflows")
		}
		total = next
	}
	if total <= available {
		return raw, nil, nil
	}
	var resident, backed []gguf.TensorInfo
	for _, info := range raw {
		if _, eligible := banks[info.Name]; eligible {
			backed = append(backed, info)
		} else {
			resident = append(resident, info)
		}
	}
	return resident, backed, nil
}
