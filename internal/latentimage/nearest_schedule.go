package latentimage

import (
	"fmt"
	"math"
	"overgo/internal/checked"
	"sort"
)

// nearestGridHalves follows the two-ended linspace in pinned PyTorch RangeFactoriesKernel.cpp.
// It changes only when the reference linspace contract changes.
const nearestGridHalves = 2

// compileNearestGridFlowSchedule snaps shifted float32 knots onto a training grid.
// Ties select the lower sigma and consecutive duplicates collapse before the terminal zero.
func compileNearestGridFlowSchedule(steps, total int, shift, trainingShift float64) (FlowSchedule, error) {
	if !checked.PositiveInts(steps, total) || !checked.PositiveFinite64(shift) || !checked.PositiveFinite64(trainingShift) || !checked.PositiveFinite32(float32(shift)) || !checked.PositiveFinite32(float32(trainingShift)) {
		return FlowSchedule{}, fmt.Errorf("nearest-grid flow: positive finite extents and shifts required, got %d/%d/%g/%g", steps, total, shift, trainingShift)
	}
	train := make([]float32, total)
	for i := range train {
		train[total-i-1] = nearestGridSigma(i, total, total, trainingShift)
	}
	sigmas := make([]float64, 0, min(steps, total))
	for i := range steps {
		sigma := nearestGridSigma(i, steps, total, shift)
		upper := min(sort.Search(len(train), func(j int) bool { return train[j] >= sigma }), len(train)-1)
		lower := max(upper-1, 0)
		if math.Abs(float64(sigma-train[lower])) <= math.Abs(float64(train[upper]-sigma)) {
			upper = lower
		}
		selected := float64(train[upper])
		if len(sigmas) == 0 || sigmas[len(sigmas)-1] != selected {
			sigmas = append(sigmas, selected)
		}
	}
	result := FlowSchedule{Steps: len(sigmas), NumTrainTimesteps: total, Shift: shift, Sigmas: append(sigmas, 0), Timesteps: make([]float64, len(sigmas)), Deltas: make([]float64, len(sigmas))}
	for i := range result.Steps {
		result.Timesteps[i] = float64(float32(result.Sigmas[i]) * float32(total))
		result.Deltas[i] = float64(float32(result.Sigmas[i+1]) - float32(result.Sigmas[i]))
	}
	return result, nil
}

// nearestGridSigma preserves the pinned CPU linspace and unfused float32 shift arithmetic.
func nearestGridSigma(index, count, total int, shift float64) float32 {
	lower := float32(shift / (float64(total) + shift - 1))
	value := float32(1)
	if count > 1 {
		step := float32(lower-1) / float32(count-1)
		if index < count/nearestGridHalves {
			value = 1 + float32(step*float32(index))
		} else {
			value = lower - float32(step*float32(count-index-1))
		}
	}
	s := float32(shift)
	return float32(s*value) / float32(1+float32(float32(s-1)*value))
}
