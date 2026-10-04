package densecausal

import (
	"math"
	"testing"
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
