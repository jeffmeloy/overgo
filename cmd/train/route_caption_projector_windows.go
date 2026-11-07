package main

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/cuda/device"
	"overgo/internal/densecausal"
	"overgo/internal/densecausaltrain"
	"overgo/internal/hybridtrain"
)

// The frozen text model runs on the card through the CUDA worker, which only the Windows build carries, so the
// caption projector's device registers itself here.
func init() { captionDevice = captionFrozenSession }

// captionFrozenSession opens the frozen text model at path on the first device for sequences of length tokens: a
// GGUF file's hybrid model through its frozen hybrid walk, a directory's dense model through its own.
func captionFrozenSession(ctx context.Context, path string, length int) (captionModel, int, func() error, error) {
	worker, err := device.New(0)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("CUDA unavailable: %w", err)
	}
	if strings.EqualFold(filepath.Ext(path), ".gguf") {
		model, err := hybridtrain.LoadFrozenLanguageModel(ctx, path, length)
		if err != nil {
			return nil, 0, nil, errors.Join(err, worker.Close())
		}
		frozen, err := hybridtrain.OpenFrozenSession(ctx, model, worker, length)
		if err != nil {
			return nil, 0, nil, errors.Join(err, model.Close(), worker.Close())
		}
		run := func(tokens []int, trained []bool, positions []int, rows []float32) (float64, []float32, error) {
			return frozen.Run(ctx, tokens, trained, positions, rows)
		}
		return run, model.Geometry.Hidden, func() error { return errors.Join(frozen.Close(), model.Close(), worker.Close()) }, nil
	}
	model, err := densecausal.Load(path)
	if err != nil {
		return nil, 0, nil, errors.Join(err, worker.Close())
	}
	frozen, err := densecausaltrain.OpenFrozenSession(model, worker, length)
	if err != nil {
		return nil, 0, nil, errors.Join(err, worker.Close())
	}
	return frozen.Run, model.Dims.Hidden, func() error { return errors.Join(frozen.Close(), worker.Close()) }, nil
}
