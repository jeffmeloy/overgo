//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// SoftmaxCrossEntropyBackwardResident replaces logits with mean CE gradients.
func SoftmaxCrossEntropyBackwardResident(
	worker *device.Worker,
	logits, targets, losses driver.DevicePtr,
	rows, classes int,
) error {
	if worker == nil || logits == 0 || targets == 0 || losses == 0 || rows <= 0 || classes <= 0 || uint64(rows) > math.MaxUint32 || uint64(classes) > math.MaxUint32 {
		return fmt.Errorf("SoftmaxCrossEntropyBackwardResident: invalid buffer or geometry")
	}
	return withCUDA(worker, func(scope *cudaScope) error {
		function, err := scope.function("softmax_ce_grad_rows_f32")
		if err != nil {
			return err
		}
		rowsU, classesU, scale := uint32(rows), uint32(classes), float32(1)/float32(rows)
		if err := scope.launch1D(function, rowsU,
			unsafe.Pointer(&logits), unsafe.Pointer(&targets), unsafe.Pointer(&losses),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&classesU), unsafe.Pointer(&scale),
		); err != nil {
			return err
		}
		return scope.finish()
	})
}
