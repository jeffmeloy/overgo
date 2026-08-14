//go:build !windows

package main

import (
	"fmt"

	"overgo/internal/densecausal"
)

// runTraining on non-Windows builds runs the host path only; the device Muon
// training lane is Windows+CUDA (nvcuda.dll + committed PTX). preferDevice is
// accepted for a uniform signature but has no device to prefer here.
func runTraining(m *densecausal.Model, tokens []int, steps int, baseLR, mu float64, preferDevice, freezeLexical bool) ([]float64, string, error) {
	if freezeLexical {
		return nil, "", fmt.Errorf("frozen lexical training requires Windows CUDA")
	}
	return runHostTraining(m, tokens, steps, baseLR, mu)
}
