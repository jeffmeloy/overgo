package trainingworkflow

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/densecausal"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// TestTextRunOnTheDeviceReachesTheDeviceTrainer holds a held-out text run whose request names a
// device trainer to training on it: the options a run passes whatever its request (a checkpoint
// callback at an interval of zero) are not what only the host honours, so the run reaches the
// trainer, which here is the host's behind the device's signature, and publishes its verdict.
// A change to the device path's guards that the landing's tests could not reach once refused every
// device-trained run.
func TestTextRunOnTheDeviceReachesTheDeviceTrainer(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	reached := 0
	device := func(model *densecausal.Model, batches [][]int, learningRate, momentum float64, _ bool, resume *densecausal.TrainState, observe densecausal.TrainObserver) ([]float64, densecausal.TrainState, error) {
		reached++
		return model.TrainWith(batches, learningRate, momentum, resume, observe, densecausal.TrainOptions{})
	}
	result, err := RunTextHeldout(t.Context(), text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "device"), Steps: 4, Device: device,
	})
	if err != nil || !result.Verdict.Valid() || reached != 1 {
		t.Fatalf("a run on the device trainer = (%+v, %v) with %d trainer calls, want a verdict from one", result, err, reached)
	}
}

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
