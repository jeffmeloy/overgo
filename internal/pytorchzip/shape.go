package pytorchzip

import (
	"fmt"

	"overgo/internal/checked"
)

// HostShape converts an exact-rank tensor shape to positive host extents.
func HostShape(tensor TensorMeta, rank int) ([]int, error) {
	if rank <= 0 || len(tensor.Shape) != rank {
		return nil, fmt.Errorf("pytorchzip: tensor %s rank %d, want %d", tensor.Name, len(tensor.Shape), rank)
	}
	shape := make([]int, rank)
	for axis, extent := range tensor.Shape {
		unsigned, ok := checked.Uint64(extent)
		if !ok {
			return nil, fmt.Errorf("pytorchzip: tensor %s axis %d is invalid", tensor.Name, axis)
		}
		value, ok := checked.Int(unsigned)
		if !ok || value <= 0 {
			return nil, fmt.Errorf("pytorchzip: tensor %s axis %d is invalid", tensor.Name, axis)
		}
		shape[axis] = value
	}
	return shape, nil
}
