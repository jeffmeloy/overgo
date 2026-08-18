//go:build windows

package trainingworkflow

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/densecausal"
)

func runTrainingState(model *densecausal.Model, batches [][]int, learningRate, momentum float64, preferDevice, freezeLexical bool, resume *densecausal.TrainState) ([]float64, string, densecausal.TrainState, error) {
	if !preferDevice {
		losses, state, err := model.TrainBatchesResume(batches, learningRate, momentum, resume)
		return losses, "host", state, err
	}
	if ok, reason := densecausal.DeviceTrainingSupported(model.Dims); !ok {
		return nil, "", densecausal.TrainState{}, fmt.Errorf("CUDA training unsupported: %s", reason)
	}
	worker, err := device.New(0)
	if err != nil {
		return nil, "", densecausal.TrainState{}, fmt.Errorf("CUDA training unavailable: %w", err)
	}
	defer worker.Close()
	result, err := model.TrainDeviceResident(worker, batches, learningRate, momentum, densecausal.DeviceTrainingOptions{
		FrozenLexical: freezeLexical,
		Resume:        resume,
	})
	return result.Losses, "cuda", result.State, err
}
