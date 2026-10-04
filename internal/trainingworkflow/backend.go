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

// runTrainingState trains the model on the batches. trained, when it is not
// nil, holds each batch's flags for the tokens it learns, which only the host
// trainer, and no router observation beside it, takes.
func runTrainingState(model *densecausal.Model, batches [][]int, trained [][]bool, learningRate, momentum float64, device DeviceTrainer, preferDevice, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver, observeRouter densecausal.MoERouterObserver) ([]float64, string, densecausal.TrainState, error) {
	if trained != nil && (preferDevice || observeRouter != nil) {
		return nil, "", densecausal.TrainState{}, errors.New("training workflow: a run that learns only some tokens trains on the host, without router observation")
	}
	if !preferDevice {
		if trained != nil {
			losses, state, err := model.TrainMasked(batches, trained, learningRate, momentum, resume, observe)
			return losses, "host", state, err
		}
		losses, state, err := model.TrainWithRouterObservations(batches, learningRate, momentum, resume, observe, observeRouter)
		return losses, "host", state, err
	}
	if device == nil {
		return nil, "", densecausal.TrainState{}, errors.New("training workflow: CUDA training needs the device trainer a training command supplies")
	}
	losses, state, err := device(model, batches, learningRate, momentum, freezeLexical, resume, observe)
	return losses, "cuda", state, err
}
