//go:build windows

package devicemath

import (
	"context"
	"fmt"
	"math"
	"unsafe"

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
	return withCUDA(worker, func(scope *cudaScope) error {
		function, err := scope.function("add_f32")
		if err != nil {
			return err
		}
		if err := scope.launchVector3(function, left, right, output, count); err != nil {
			return err
		}
		return scope.finish()
	})
}

// ReLUBackwardResident writes the ReLU input VJP.
func ReLUBackwardResident(worker *device.Worker, incoming, input, gradient driver.DevicePtr, count int) error {
	if worker == nil || incoming == 0 || input == 0 || gradient == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return fmt.Errorf("ReLUBackwardResident: invalid buffer or count")
	}
	return withCUDA(worker, func(scope *cudaScope) error {
		function, err := scope.function("relu_backward_f32")
		if err != nil {
			return err
		}
		countU := uint32(count)
		if err := scope.launch1D(function, countU,
			unsafe.Pointer(&incoming), unsafe.Pointer(&input), unsafe.Pointer(&gradient), unsafe.Pointer(&countU),
		); err != nil {
			return err
		}
		return scope.finish()
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
	return withCUDA(worker, func(scope *cudaScope) error {
		function, err := scope.function("strided_row_copy_f32")
		if err != nil {
			return err
		}
		rowsU, widthU := uint32(rows), uint32(width)
		sourceStrideU, sourceOffsetU := uint32(sourceStride), uint32(sourceOffset)
		destinationStrideU, destinationOffsetU := uint32(destinationStride), uint32(destinationOffset)
		count := rowsU * widthU
		if err := scope.launch1D(function, count,
			unsafe.Pointer(&source), unsafe.Pointer(&destination),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU),
			unsafe.Pointer(&sourceStrideU), unsafe.Pointer(&sourceOffsetU),
			unsafe.Pointer(&destinationStrideU), unsafe.Pointer(&destinationOffsetU),
		); err != nil {
			return err
		}
		return scope.finish()
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
	return withCUDA(worker, func(scope *cudaScope) error {
		function, err := scope.function("indexed_row_scatter_add_f32")
		if err != nil {
			return err
		}
		countU, widthU := uint32(count), uint32(width)
		if err := scope.launch1D(function, countU*widthU,
			unsafe.Pointer(&source), unsafe.Pointer(&rows), unsafe.Pointer(&destination),
			unsafe.Pointer(&countU), unsafe.Pointer(&widthU),
		); err != nil {
			return err
		}
		return scope.finish()
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
	return withCUDA(worker, func(scope *cudaScope) error {
		function, err := scope.function("attention_score_affine_backward_f32")
		if err != nil {
			return err
		}
		sequenceU, headDimU, lagCountU := uint32(sequence), uint32(headDim), uint32(lagCount)
		if err := scope.launch1D(function, sequenceU*sequenceU,
			unsafe.Pointer(&dScores), unsafe.Pointer(&query), unsafe.Pointer(&key),
			unsafe.Pointer(&dLagBias), unsafe.Pointer(&dScale),
			unsafe.Pointer(&sequenceU), unsafe.Pointer(&headDimU), unsafe.Pointer(&lagCountU),
		); err != nil {
			return err
		}
		return scope.finish()
	})
}
