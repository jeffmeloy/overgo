package trainingworkflow

import (
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/densecausal"
	"overgo/internal/densecausaltrain"
)

// TestDeviceSamplerDrawsWhatTheHostSamplerDraws holds the sampler a reward
// session on the device draws a GGUF model's answers by to the model in
// training.
//
// The weights are opened in the inference runtime on the device, and what it
// draws is what the host sampler draws from the model in training: the same
// ids from the same seeds, each at the host's probability to within
// servedAgreement; and moved and released, the weights opened next are the
// moved ones (heldToHostSampler).
//
// A session that trains on the device, through the resident trainer's one
// weighted step, samples each group there from the weights as they stand: the
// runtime's model and the trainer's take the device in turn within one
// process, group after group, and the session stopped and resumed ends at the
// bytes the uninterrupted one ends at (rewardSessionOverGGUF).
func TestDeviceSamplerDrawsWhatTheHostSamplerDraws(t *testing.T) {
	cudatest.Require(t)
	request := rewardGGUFFixture(t)
	heldToHostSampler(t, request, false, servedAgreement)
	request.Host, request.Device = false, func(model *densecausal.Model, groups [][]densecausal.WeightedSequence, learningRate, momentum float64, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver) ([]float64, int, densecausal.TrainState, error) {
		worker, err := device.New(0)
		if err != nil {
			return nil, 0, densecausal.TrainState{}, err
		}
		defer worker.Close()
		result, err := densecausaltrain.TrainDeviceResidentWeighted(model, worker, groups, learningRate, momentum, densecausaltrain.Options{FrozenLexical: freezeLexical, Resume: resume, Observe: observe})
		return result.Losses, result.Skipped, result.State, err
	}
	rewardSessionOverGGUF(t, request)
}
