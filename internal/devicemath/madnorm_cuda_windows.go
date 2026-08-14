//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// MADNormForward applies row-wise rank/MAD normalization on device.
func MADNormForward(worker *device.Worker, input []float32, rows, width int, epsilon float64) ([]float32, error) {
	if rows <= 0 || width <= 0 || len(input) != rows*width || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return nil, fmt.Errorf("MADNormForward: invalid geometry or epsilon")
	}
	output := make([]float32, len(input))
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("mad_norm_f32")
		if err != nil {
			return err
		}
		inputPtr, err := scope.upload(input)
		if err != nil {
			return err
		}
		outputPtr, err := scope.alloc(len(output))
		if err != nil {
			return err
		}
		widthU, rowsU, epsilonF := uint32(width), uint32(rows), float32(epsilon)
		if err := scope.launch1D(fn, rowsU,
			unsafe.Pointer(&inputPtr), unsafe.Pointer(&outputPtr), unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&epsilonF),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{output, outputPtr})
	})
	return output, err
}

// MADNormBackwardResident writes the rank/MAD input VJP between device buffers.
func MADNormBackwardResident(
	worker *device.Worker,
	incoming, input, output, gradient driver.DevicePtr,
	rows, width int,
	epsilon float64,
) error {
	if worker == nil || incoming == 0 || input == 0 || output == 0 || gradient == 0 || rows <= 0 || width <= 0 || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return fmt.Errorf("MADNormBackwardResident: invalid buffer, geometry, or epsilon")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.MADNormBackward(incoming, input, output, gradient, rows, width, epsilon)
	})
}

// MADNormBackward computes the rank/MAD input VJP on device.
func MADNormBackward(worker *device.Worker, incoming, input []float32, rows, width int, epsilon float64) ([]float32, error) {
	if rows <= 0 || width <= 0 || len(input) != rows*width || len(incoming) != len(input) || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return nil, fmt.Errorf("MADNormBackward: invalid geometry or epsilon")
	}
	output, err := MADNormForward(worker, input, rows, width, epsilon)
	if err != nil {
		return nil, err
	}
	gradient := make([]float32, len(input))
	err = withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("mad_norm_backward_f32")
		if err != nil {
			return err
		}
		incomingPtr, err := scope.upload(incoming)
		if err != nil {
			return err
		}
		inputPtr, err := scope.upload(input)
		if err != nil {
			return err
		}
		outputPtr, err := scope.upload(output)
		if err != nil {
			return err
		}
		gradientPtr, err := scope.alloc(len(gradient))
		if err != nil {
			return err
		}
		widthU, rowsU, epsilonF := uint32(width), uint32(rows), float32(epsilon)
		if err := scope.launch1D(fn, rowsU,
			unsafe.Pointer(&incomingPtr), unsafe.Pointer(&inputPtr), unsafe.Pointer(&outputPtr), unsafe.Pointer(&gradientPtr),
			unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&epsilonF),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{gradient, gradientPtr})
	})
	return gradient, err
}
