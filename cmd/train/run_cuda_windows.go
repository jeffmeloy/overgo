//go:build windows

package main

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/densecausal"
)

// runTraining runs the device Muon path when preferDevice is set, the model's
// architecture is supported by the device training backend, and a CUDA device
// opens; otherwise it falls back to the host path. The capability decision runs
// at SELECTION time -- before any device session is constructed -- so a model
// using a device-unsupported trait (e.g. attention bias) is routed to host up
// front instead of failing mid-session. It returns the loss trajectory and the
// backend label actually used ("cuda" or "host").
func runTraining(m *densecausal.Model, tokens []int, steps int, baseLR, mu float64, preferDevice bool) ([]float64, string, error) {
	if preferDevice {
		if ok, reason := densecausal.DeviceTrainingSupported(m.Dims); !ok {
			fmt.Printf("train: device training unsupported (%s); using host path\n", reason)
			return runHostTraining(m, tokens, steps, baseLR, mu)
		}
		worker, err := device.New(0)
		if err == nil {
			defer worker.Close()
			traj, terr := m.TrainDeviceFull(worker, tokens, steps, baseLR, mu)
			if terr != nil {
				return nil, "", terr
			}
			return traj, "cuda", nil
		}
		fmt.Printf("train: CUDA unavailable (%v); using host path\n", err)
	}
	return runHostTraining(m, tokens, steps, baseLR, mu)
}
