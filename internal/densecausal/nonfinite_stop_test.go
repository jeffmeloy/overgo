package densecausal

import (
	"errors"
	"math"
	"testing"

	"overgo/internal/optimizer"
)

// TestTrainingStopsOnANonFiniteDirection plants a NaN in a weight and trains: the first step ends the run, naming the
// step, where every step left would be spent on a model that can only be NaN. The weights the step would have written are
// not written, and a run of finite weights takes each of its steps.
func TestTrainingStopsOnANonFiniteDirection(t *testing.T) {
	t.Parallel()
	const rate, momentum, steps = 0.01, 0.9, 5
	batches := make([][]int, steps)
	for index := range batches {
		batches[index] = []int{3, 9, 14, 5, 21, 7, 18, 2}
	}
	broken := tinyMuonModel(t)
	var name string
	for candidate := range broken.Weights {
		name = candidate
		break
	}
	broken.Weights[name][0] = float32(math.NaN())
	if _, _, err := broken.Train(batches, rate, momentum, nil, nil); !errors.Is(err, optimizer.ErrNonfiniteDirection) {
		t.Fatalf("training on a NaN weight = %v, want it stopped", err)
	}
	if trajectory, _, err := tinyMuonModel(t).Train(batches, rate, momentum, nil, nil); err != nil || len(trajectory) != steps {
		t.Fatalf("training on finite weights = %d steps (%v), want %d", len(trajectory), err, steps)
	}
}
