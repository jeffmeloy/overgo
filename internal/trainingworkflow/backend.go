package trainingworkflow

import (
	"errors"

	"overgo/internal/densecausal"
)

// DeviceTrainer trains a dense-causal model resident on the device, one
// Muon step per batch, from the resume state and under the step observer.
// The training command supplies it, so the serving packages that run the
// workflow on the host never link the device backend.
type DeviceTrainer func(model *densecausal.Model, batches [][]int, learningRate, momentum float64, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver) ([]float64, densecausal.TrainState, error)

func runTrainingState(model *densecausal.Model, batches [][]int, learningRate, momentum float64, device DeviceTrainer, preferDevice, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver, observeRouter densecausal.MoERouterObserver) ([]float64, string, densecausal.TrainState, error) {
	if !preferDevice {
		losses, state, err := model.TrainWithRouterObservations(batches, learningRate, momentum, resume, observe, observeRouter)
		return losses, "host", state, err
	}
	if device == nil {
		return nil, "", densecausal.TrainState{}, errors.New("training workflow: CUDA training needs the device trainer a training command supplies")
	}
	losses, state, err := device(model, batches, learningRate, momentum, freezeLexical, resume, observe)
	return losses, "cuda", state, err
}
