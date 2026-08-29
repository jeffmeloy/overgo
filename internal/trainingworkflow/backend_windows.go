//go:build windows

package trainingworkflow

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/densecausal"
)

func runTrainingState(model *densecausal.Model, batches [][]int, learningRate, momentum float64, preferDevice, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver, observeRouter densecausal.MoERouterObserver) ([]float64, string, densecausal.TrainState, error) {
	if !preferDevice {
		losses, state, err := model.TrainWithRouterObservations(batches, learningRate, momentum, resume, observe, observeRouter)
		return losses, "host", state, err
	}
	if ok, reason := densecausal.DeviceTrainingSupported(model.Dims); !ok {
		return nil, "", densecausal.TrainState{}, fmt.Errorf("CUDA training unsupported: %s", reason)
	}
	worker, err := newDeviceWorker()
	if err != nil {
		return nil, "", densecausal.TrainState{}, fmt.Errorf("CUDA training unavailable: %w", err)
	}
	defer worker.Close()
	result, err := model.TrainDeviceResident(worker, batches, learningRate, momentum, densecausal.DeviceTrainingOptions{
		FrozenLexical: freezeLexical,
		Resume:        resume,
		Observe:       observe,
	})
	return result.Losses, "cuda", result.State, err
}

func newDeviceWorker() (*device.Worker, error) {
	var ordinal int
	return device.New(ordinal)
}
