package main

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/densecausal"
	"overgo/internal/densecausaltrain"
)

// The dense-causal device trainer steps through the CUDA worker, which only
// the Windows build carries, so it registers itself here.
func init() { denseDevice = trainDenseResident }

// trainDenseResident trains the workflow's dense-causal model resident on
// the first device, refusing a model the resident backend cannot serve
// before it opens the device.
func trainDenseResident(model *densecausal.Model, batches [][]int, learningRate, momentum float64, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver) ([]float64, densecausal.TrainState, error) {
	if ok, reason := densecausal.DeviceTrainingSupported(model.Dims); !ok {
		return nil, densecausal.TrainState{}, fmt.Errorf("CUDA training unsupported: %s", reason)
	}
	worker, err := device.New(0)
	if err != nil {
		return nil, densecausal.TrainState{}, fmt.Errorf("CUDA training unavailable: %w", err)
	}
	defer worker.Close()
	result, err := densecausaltrain.TrainDeviceResident(model, worker, batches, learningRate, momentum, densecausaltrain.Options{
		FrozenLexical: freezeLexical, Resume: resume, Observe: observe,
	})
	return result.Losses, result.State, err
}
