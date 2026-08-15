//go:build windows

package main

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/densecausal"
)

// runTrainingState executes the selected backend only.
func runTrainingState(m *densecausal.Model, batches [][]int, baseLR, mu float64, preferDevice, freezeLexical bool, resume *densecausal.TrainState) ([]float64, string, densecausal.TrainState, error) {
	if preferDevice {
		if ok, reason := densecausal.DeviceTrainingSupported(m.Dims); !ok {
			return nil, "", densecausal.TrainState{}, fmt.Errorf("CUDA training unsupported: %s", reason)
		}
		worker, err := device.New(0)
		if err != nil {
			return nil, "", densecausal.TrainState{}, fmt.Errorf("CUDA training unavailable: %w", err)
		}
		defer worker.Close()
		var trajectory []float64
		var state densecausal.TrainState
		if freezeLexical {
			trajectory, state, err = m.TrainDeviceResidentFrozenLexicalBatches(worker, batches, baseLR, mu, resume)
		} else {
			trajectory, state, err = m.TrainDeviceResidentBatches(worker, batches, baseLR, mu, resume)
		}
		return trajectory, "cuda", state, err
	}
	trajectory, state, err := runHostTrainingState(m, batches, baseLR, mu, resume)
	return trajectory, "host", state, err
}
