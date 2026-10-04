package densecausal

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/optimizer"
)

// averageDecay is the decay an averaged training test runs under; with six
// updates the warm-up, not the decay, sets every rate.
const averageDecay = 0.99

// TestAveragedTrainingEqualsTheAverageOfItsSteps trains one model six updates
// with a moving average of its weights, and another identical model the same six
// one update at a time, resuming its optimizer between them, taking each update's
// weights: the first model ends holding what an average of those weights is by the
// recurrence, written out here, and the second, with no average, ends holding the
// last of them.
func TestAveragedTrainingEqualsTheAverageOfItsSteps(t *testing.T) {
	const rate, momentum = 0.01, 0.9
	batches := [][]int{
		{3, 9, 14, 5, 21, 7, 18, 2}, {5, 1, 9, 9, 14, 2, 21, 7}, {2, 2, 7, 18, 3, 9, 14, 5},
		{21, 7, 18, 2, 3, 9, 14, 5}, {9, 14, 5, 21, 7, 18, 2, 3}, {14, 5, 21, 7, 18, 2, 3, 9},
	}
	start := tinyMuonModel(t)
	expected := map[string][]float64{}
	for name, values := range start.Weights {
		expected[name] = make([]float64, len(values))
		for index, value := range values {
			expected[name][index] = float64(value)
		}
	}

	stepped := tinyMuonModel(t)
	var state *TrainState
	for update, batch := range batches {
		_, next, err := stepped.Train([][]int{batch}, rate, momentum, state, nil)
		if err != nil {
			t.Fatal(err)
		}
		state = &next
		keep := min(averageDecay, float64(1+update+1)/float64(10+update+1))
		for name, values := range stepped.Weights {
			for index, value := range values {
				expected[name][index] = keep*expected[name][index] + (1-keep)*float64(value)
			}
		}
	}

	averaged := tinyMuonModel(t)
	if _, _, err := averaged.TrainWith(batches, rate, momentum, nil, nil, TrainOptions{Average: averageDecay}); err != nil {
		t.Fatal(err)
	}
	differs := false
	for name, values := range averaged.Weights {
		for index, value := range values {
			if math.Abs(float64(value)-expected[name][index]) > 1e-5 {
				t.Fatalf("%s[%d] = %v, the average of the steps is %v", name, index, value, expected[name][index])
			}
			differs = differs || value != stepped.Weights[name][index]
		}
	}
	if !differs {
		t.Fatal("the averaged model equals the last step's weights: the average changed nothing")
	}

	// A run with no average is untouched by the option existing.
	plain := tinyMuonModel(t)
	if _, _, err := plain.TrainWith(batches, rate, momentum, nil, nil, TrainOptions{}); err != nil {
		t.Fatal(err)
	}
	for name, values := range plain.Weights {
		for index, value := range values {
			if value != stepped.Weights[name][index] {
				t.Fatalf("%s[%d] = %v without an average, %v stepping one batch at a time", name, index, value, stepped.Weights[name][index])
			}
		}
	}
}

// TestAveragedTrainingRefusesADecayOutsideTheOpenUnitInterval checks a decay of
// one or more is an error, where zero means no average.
func TestAveragedTrainingRefusesADecayOutsideTheOpenUnitInterval(t *testing.T) {
	batches := [][]int{{3, 9, 14, 5, 21, 7, 18, 2}}
	if _, _, err := tinyMuonModel(t).TrainWith(batches, 0.01, 0.9, nil, nil, TrainOptions{Average: 1}); err == nil {
		t.Error("a decay of one was accepted")
	}
	if _, _, err := tinyMuonModel(t).TrainWith(batches, 0.01, 0.9, nil, nil, TrainOptions{Average: 0}); err != nil {
		t.Errorf("no average: %v", err)
	}
}

// TestAveragedTrainingResumesFromACheckpoint trains eight updates under an
// average, and again with a checkpoint after the fourth: a fresh model given the
// checkpoint's weights, optimizer state and average and the last four batches
// ends holding exactly the weights the uninterrupted run did, and the checkpoint
// is not taken after the last update.
func TestAveragedTrainingResumesFromACheckpoint(t *testing.T) {
	const rate, momentum = 0.01, 0.9
	batches := make([][]int, 8)
	for index := range batches {
		batches[index] = []int{(3 + index) % 30, 9, (14 + 2*index) % 30, 5, 21, (7 + index) % 30, 18, 2}
	}
	uninterrupted := tinyMuonModel(t)
	if _, _, err := uninterrupted.TrainWith(batches, rate, momentum, nil, nil, TrainOptions{Average: averageDecay}); err != nil {
		t.Fatal(err)
	}

	type saved struct {
		done    int
		state   TrainState
		average *optimizer.Average
		weights map[string][]float32
	}
	var checkpoints []saved
	interrupted := tinyMuonModel(t)
	if _, _, err := interrupted.TrainWith(batches, rate, momentum, nil, nil, TrainOptions{
		Average: averageDecay, CheckpointEvery: 4,
		Checkpoint: func(done int, state TrainState, average *optimizer.Average) error {
			weights := map[string][]float32{}
			for name, values := range interrupted.Weights {
				weights[name] = slices.Clone(values)
			}
			restored, err := optimizer.RestoreAverage(slices.Clone(average.Values()), average.Updates(), average.Decay())
			checkpoints = append(checkpoints, saved{done, state, restored, weights})
			return err
		},
	}); err != nil {
		t.Fatal(err)
	}
	if len(checkpoints) != 1 || checkpoints[0].done != 4 || checkpoints[0].average.Updates() != 4 {
		t.Fatalf("%d checkpoints taken, want one, after the fourth of eight updates", len(checkpoints))
	}

	resumed := tinyMuonModel(t)
	for name, values := range checkpoints[0].weights {
		copy(resumed.Weights[name], values)
	}
	state := checkpoints[0].state
	if _, _, err := resumed.TrainWith(batches[4:], rate, momentum, &state, nil, TrainOptions{ResumeAverage: checkpoints[0].average}); err != nil {
		t.Fatal(err)
	}
	for name, values := range uninterrupted.Weights {
		if !slices.Equal(values, resumed.Weights[name]) {
			t.Fatalf("%s differs between the uninterrupted run and the one resumed from its fourth update", name)
		}
	}
}
