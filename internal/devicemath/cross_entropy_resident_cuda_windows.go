//go:build windows

package devicemath

import (
	"fmt"
	"math"

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
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.SoftmaxCrossEntropy(logits, targets, losses, rows, classes)
	})
}
