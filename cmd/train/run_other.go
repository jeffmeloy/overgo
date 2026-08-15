//go:build !windows

package main

import (
	"fmt"

	"overgo/internal/densecausal"
)

// runTrainingState executes the selected backend only.
func runTrainingState(m *densecausal.Model, batches [][]int, baseLR, mu float64, preferDevice, freezeLexical bool, resume *densecausal.TrainState) ([]float64, string, densecausal.TrainState, error) {
	if preferDevice {
		return nil, "", densecausal.TrainState{}, fmt.Errorf("CUDA training requires Windows")
	}
	if freezeLexical {
		return nil, "", densecausal.TrainState{}, fmt.Errorf("frozen lexical training requires Windows CUDA")
	}
	trajectory, state, err := runHostTrainingState(m, batches, baseLR, mu, resume)
	return trajectory, "host", state, err
}
