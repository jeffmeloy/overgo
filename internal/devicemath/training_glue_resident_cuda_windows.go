//go:build windows

package devicemath

import (
	"context"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// ZeroResidentF32 clears a persistent f32 buffer.
func ZeroResidentF32(worker *device.Worker, buffer driver.DevicePtr, count int) error {
	if worker == nil || buffer == 0 || count <= 0 {
		return fmt.Errorf("ZeroResidentF32: invalid buffer or count")
	}
	return worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemsetD32Async(buffer, 0, uint64(count), state.Stream)
	})
}

// AddResident computes output = left + right on resident buffers.
func AddResident(worker *device.Worker, left, right, output driver.DevicePtr, count int) error {
	if worker == nil || left == 0 || right == 0 || output == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return fmt.Errorf("AddResident: invalid buffer or count")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.Add(left, right, output, count)
	})
}

// ScaleResident computes output = input * scale on resident buffers.
func ScaleResident(worker *device.Worker, input, output driver.DevicePtr, scale float32, count int) error {
	if worker == nil || input == 0 || output == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return fmt.Errorf("ScaleResident: invalid buffer or count")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.Scale(input, output, scale, count)
	})
}

// ScaleByResidentScalar multiplies a resident vector by a resident scalar.
func ScaleByResidentScalar(worker *device.Worker, input, scale, output driver.DevicePtr, count int) error {
	if worker == nil || input == 0 || scale == 0 || output == 0 || count <= 0 {
		return fmt.Errorf("ScaleByResidentScalar: invalid buffer or count")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.ScaleByScalar(input, scale, output, count)
	})
}

// ReLUBackwardResident writes the ReLU input VJP.
func ReLUBackwardResident(worker *device.Worker, incoming, input, gradient driver.DevicePtr, count int) error {
	if worker == nil || incoming == 0 || input == 0 || gradient == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return fmt.Errorf("ReLUBackwardResident: invalid buffer or count")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.ReLUBackward(incoming, input, gradient, count)
	})
}

// StridedRowCopyResident copies one rectangular row view between device buffers.
func StridedRowCopyResident(
	worker *device.Worker,
	source, destination driver.DevicePtr,
	rows, width, sourceStride, sourceOffset, destinationStride, destinationOffset int,
) error {
	if worker == nil || source == 0 || destination == 0 || rows <= 0 || width <= 0 || sourceStride < sourceOffset+width || destinationStride < destinationOffset+width {
		return fmt.Errorf("StridedRowCopyResident: invalid buffer or geometry")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.StridedRowCopy(source, destination, rows, width, sourceStride, sourceOffset, destinationStride, destinationOffset)
	})
}

// IndexedRowScatterAddResident accumulates source rows into a device table.
func IndexedRowScatterAddResident(
	worker *device.Worker,
	source, rows, destination driver.DevicePtr,
	count, width int,
) error {
	if worker == nil || source == 0 || rows == 0 || destination == 0 || count <= 0 || width <= 0 {
		return fmt.Errorf("IndexedRowScatterAddResident: invalid buffer or geometry")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.IndexedRowScatterAdd(source, rows, destination, count, width)
	})
}

// AttentionScoreAffineBackwardResident accumulates lag-bias and scale gradients.
func AttentionScoreAffineBackwardResident(
	worker *device.Worker,
	dScores, query, key, dLagBias, dScale driver.DevicePtr,
	sequence, headDim, lagCount int,
) error {
	if worker == nil || dScores == 0 || query == 0 || key == 0 || dLagBias == 0 || dScale == 0 || sequence <= 0 || headDim <= 0 || lagCount <= 0 {
		return fmt.Errorf("AttentionScoreAffineBackwardResident: invalid buffer or geometry")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.AttentionScoreAffineBackward(dScores, query, key, dLagBias, dScale, sequence, headDim, lagCount)
	})
}
