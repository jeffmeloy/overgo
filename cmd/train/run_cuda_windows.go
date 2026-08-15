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
func runTrainingState(m *densecausal.Model, batches [][]int, baseLR, mu float64, preferDevice, freezeLexical bool, resume *densecausal.TrainState) ([]float64, string, densecausal.TrainState, error) {
	if preferDevice {
		if ok, reason := densecausal.DeviceTrainingSupported(m.Dims); !ok {
			if freezeLexical {
				return nil, "", densecausal.TrainState{}, fmt.Errorf("frozen lexical CUDA training unsupported: %s", reason)
			}
			fmt.Printf("train: device training unsupported (%s); using host path\n", reason)
			trajectory, state, err := runHostTrainingState(m, batches, baseLR, mu, resume)
			return trajectory, "host", state, err
		}
		worker, err := device.New(0)
		if err == nil {
			defer worker.Close()
			var trajectory []float64
			var state densecausal.TrainState
			var trainErr error
			if freezeLexical {
				trajectory, state, trainErr = m.TrainDeviceResidentFrozenLexicalBatches(worker, batches, baseLR, mu, resume)
			} else {
				trajectory, state, trainErr = m.TrainDeviceResidentBatches(worker, batches, baseLR, mu, resume)
			}
			if trainErr != nil {
				return nil, "", densecausal.TrainState{}, trainErr
			}
			return trajectory, "cuda", state, nil
		}
		if freezeLexical {
			return nil, "", densecausal.TrainState{}, fmt.Errorf("frozen lexical CUDA training unavailable: %w", err)
		}
		fmt.Printf("train: CUDA unavailable (%v); using host path\n", err)
	}
	trajectory, state, err := runHostTrainingState(m, batches, baseLR, mu, resume)
	return trajectory, "host", state, err
}
