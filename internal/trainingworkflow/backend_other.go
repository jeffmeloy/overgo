//go:build !windows

package trainingworkflow

import (
	"errors"

	"overgo/internal/densecausal"
)

func runTrainingState(model *densecausal.Model, batches [][]int, learningRate, momentum float64, preferDevice, freezeLexical bool, resume *densecausal.TrainState) ([]float64, string, densecausal.TrainState, error) {
	if preferDevice || freezeLexical {
		return nil, "", densecausal.TrainState{}, errors.New("training workflow: CUDA training requires Windows")
	}
	losses, state, err := model.TrainBatchesResume(batches, learningRate, momentum, resume)
	return losses, "host", state, err
}
