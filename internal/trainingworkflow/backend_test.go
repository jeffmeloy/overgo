package trainingworkflow

import (
	"strings"
	"testing"

	"overgo/internal/densecausal"
	"overgo/internal/optimizer"
)

// TestDeviceRunRefusesWhatOnlyTheHostTrains holds the device path to the options it can honour:
// a run that learns some tokens, averages the weights, continues a resumed average, saves
// checkpoints at an interval or observes the router is refused rather than trained without; and a
// run that carries a checkpoint callback it never calls (an interval of zero, which every
// text run passes) trains on the device as it did.
func TestDeviceRunRefusesWhatOnlyTheHostTrains(t *testing.T) {
	t.Parallel()
	device := func(*densecausal.Model, [][]int, float64, float64, bool, *densecausal.TrainState, densecausal.TrainObserver) ([]float64, densecausal.TrainState, error) {
		return []float64{1}, densecausal.TrainState{}, nil
	}
	callback := func(int, densecausal.TrainState, *optimizer.Average) error { return nil }
	average, err := optimizer.NewAverage([]float32{1, 2}, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	for name, test := range map[string]struct {
		options densecausal.TrainOptions
		refused bool
	}{
		"an unused checkpoint callback": {densecausal.TrainOptions{Checkpoint: callback}, false},
		"no options":                    {densecausal.TrainOptions{}, false},
		"learned tokens":                {densecausal.TrainOptions{Trained: [][]bool{{true}}}, true},
		"an average":                    {densecausal.TrainOptions{Average: 0.9}, true},
		"a resumed average":             {densecausal.TrainOptions{ResumeAverage: average}, true},
		"a checkpoint interval":         {densecausal.TrainOptions{CheckpointEvery: 4, Checkpoint: callback}, true},
	} {
		_, backend, _, err := runTrainingState(nil, [][]int{{1, 2}}, test.options, 0.01, 0.9, device, true, false, nil, nil)
		switch {
		case test.refused && (err == nil || !strings.Contains(err.Error(), "only the host trainer")):
			t.Errorf("%s: err = %v, want the device refusal", name, err)
		case !test.refused && (err != nil || backend != "cuda"):
			t.Errorf("%s: ran on %q with %v, want the device to train it", name, backend, err)
		}
	}
}
