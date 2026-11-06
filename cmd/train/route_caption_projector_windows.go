package main

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/densecausal"
	"overgo/internal/densecausaltrain"
)

// The frozen text model runs on the card through the CUDA worker, which only the Windows build carries, so the
// caption projector's device registers itself here.
func init() { captionDevice = captionFrozenSession }

// captionFrozenSession opens the frozen text model on the first device for sequences of length tokens.
func captionFrozenSession(model *densecausal.Model, length int) (captionModel, func() error, error) {
	worker, err := device.New(0)
	if err != nil {
		return nil, nil, fmt.Errorf("CUDA unavailable: %w", err)
	}
	frozen, err := densecausaltrain.OpenFrozenSession(model, worker, length)
	if err != nil {
		return nil, nil, errors.Join(err, worker.Close())
	}
	return frozen.Run, func() error { return errors.Join(frozen.Close(), worker.Close()) }, nil
}
