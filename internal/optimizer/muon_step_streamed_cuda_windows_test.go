//go:build windows

package optimizer

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestStreamedMuonSessionPreparesOnce runs three streamed steps through one session and through a session opened and
// closed around each step, as every step once did: the weights, momentum and cleared gradients agree bit for bit, a
// frozen group's gradients clear without a product, and the one session keeps the operations it opened.
func TestStreamedMuonSessionPreparesOnce(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	shapes := [][2]int{{32, 16}, {16, 64}, {8, 8}}
	specs := make([]GroupSpec, 0, len(shapes))
	start := 0
	for index, shape := range shapes {
		end := start + shape[0]*shape[1]
		specs = append(specs, GroupSpec{Name: string(rune('a' + index)), Start: start, End: end, Rows: shape[0], Cols: shape[1], Frozen: index == len(shapes)-1})
		start = end
	}
	plan, err := CompilePlan(start, specs)
	if err != nil {
		t.Fatal(err)
	}
	config := Config{BaseLearningRate: 0.05, Momentum: 0.9, Steps: 3, Schedule: ScheduleLinearDecay}
	rng := rand.New(rand.NewSource(5))
	initial := make([]float32, start)
	for index := range initial {
		initial[index] = float32(rng.NormFloat64())
	}
	gradientsAt := func(step int) []float32 {
		gradients := make([]float32, start)
		for index := range gradients {
			gradients[index] = float32(math.Sin(float64(index*(step+1)))) * 0.1
		}
		return gradients
	}
	type run struct{ weights, momentum, gradients []float32 }
	steps := func(step func(weights, gradients, momentum []float32, number int) error) run {
		result := run{weights: slices.Clone(initial), momentum: make([]float32, start)}
		for number := 1; number <= config.Steps; number++ {
			result.gradients = gradientsAt(number)
			if err := step(result.weights, result.gradients, result.momentum, number); err != nil {
				t.Fatal(err)
			}
		}
		return result
	}
	session, err := NewStreamedMuon(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	opened := session.ops
	kept := steps(func(weights, gradients, momentum []float32, number int) error {
		return session.Step(weights, gradients, momentum, plan, number, config)
	})
	if session.ops != opened {
		t.Fatal("the session opened its operations again")
	}
	reopened := steps(func(weights, gradients, momentum []float32, number int) error {
		each, err := NewStreamedMuon(worker)
		if err != nil {
			return err
		}
		defer each.Close()
		return each.Step(weights, gradients, momentum, plan, number, config)
	})
	for _, compared := range [][2][]float32{{kept.weights, reopened.weights}, {kept.momentum, reopened.momentum}, {kept.gradients, reopened.gradients}} {
		for index := range compared[0] {
			if math.Float32bits(compared[0][index]) != math.Float32bits(compared[1][index]) {
				t.Fatalf("value %d = %v through one session, %v through one a step", index, compared[0][index], compared[1][index])
			}
		}
	}
	if slices.Equal(kept.weights, initial) || slices.ContainsFunc(kept.gradients, func(value float32) bool { return value != 0 }) {
		t.Fatal("the steps left the weights unchanged or the gradients uncleared")
	}
}
