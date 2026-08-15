//go:build !windows

package main

import (
	"fmt"

	"overgo/internal/densecausal"
)

// runTraining on non-Windows builds runs the host path only; the device Muon
// training lane is Windows+CUDA (nvcuda.dll + committed PTX). preferDevice is
// accepted for a uniform signature but has no device to prefer here.
func runTrainingState(m *densecausal.Model, batches [][]int, baseLR, mu float64, preferDevice, freezeLexical bool, resume *densecausal.TrainState) ([]float64, string, densecausal.TrainState, error) {
	if freezeLexical {
		return nil, "", densecausal.TrainState{}, fmt.Errorf("frozen lexical training requires Windows CUDA")
	}
	trajectory, state, err := runHostTrainingState(m, batches, baseLR, mu, resume)
	return trajectory, "host", state, err
}
