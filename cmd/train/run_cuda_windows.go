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
func runTraining(m *densecausal.Model, batches [][]int, baseLR, mu float64, preferDevice, freezeLexical bool) ([]float64, string, error) {
	if preferDevice {
		if ok, reason := densecausal.DeviceTrainingSupported(m.Dims); !ok {
			if freezeLexical {
				return nil, "", fmt.Errorf("frozen lexical CUDA training unsupported: %s", reason)
			}
			fmt.Printf("train: device training unsupported (%s); using host path\n", reason)
			return runHostTraining(m, batches, baseLR, mu)
		}
		worker, err := device.New(0)
		if err == nil {
			defer worker.Close()
			var traj []float64
			var terr error
			if freezeLexical {
				traj, terr = m.TrainDeviceResidentFrozenLexicalBatches(worker, batches, baseLR, mu)
			} else {
				traj, terr = m.TrainDeviceResidentBatches(worker, batches, baseLR, mu)
			}
			if terr != nil {
				return nil, "", terr
			}
			return traj, "cuda", nil
		}
		if freezeLexical {
			return nil, "", fmt.Errorf("frozen lexical CUDA training unavailable: %w", err)
		}
		fmt.Printf("train: CUDA unavailable (%v); using host path\n", err)
	}
	return runHostTraining(m, batches, baseLR, mu)
}
