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

// runTrainingState trains the model on the batches. The options, which only
// the host trainer takes, say which tokens each batch learns, whether the
// trained model is a moving average of the weights, and who observes the
// router; a device run refuses them rather than quietly train without.
func runTrainingState(model *densecausal.Model, batches [][]int, options densecausal.TrainOptions, learningRate, momentum float64, device DeviceTrainer, preferDevice, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver) ([]float64, string, densecausal.TrainState, error) {
	if !preferDevice {
		losses, state, err := model.TrainWith(batches, learningRate, momentum, resume, observe, options)
		return losses, "host", state, err
	}
	if options.Trained != nil || options.Average != 0 || options.ObserveRouter != nil {
		return nil, "", densecausal.TrainState{}, errors.New("training workflow: only the host trainer learns some tokens, averages the weights or observes the router")
	}
	if device == nil {
		return nil, "", densecausal.TrainState{}, errors.New("training workflow: CUDA training needs the device trainer a training command supplies")
	}
	losses, state, err := device(model, batches, learningRate, momentum, freezeLexical, resume, observe)
	return losses, "cuda", state, err
}
