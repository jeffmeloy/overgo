//go:build !windows

package main

import "overgo/internal/densecausal"

// runTraining on non-Windows builds runs the host path only; the device Muon
// training lane is Windows+CUDA (nvcuda.dll + committed PTX). preferDevice is
// accepted for a uniform signature but has no device to prefer here.
func runTraining(m *densecausal.Model, tokens []int, steps int, baseLR, mu float64, preferDevice bool) ([]float64, string, error) {
	traj, err := m.Train(tokens, steps, baseLR, mu)
	if err != nil {
		return nil, "", err
	}
	return traj, "host", nil
}
