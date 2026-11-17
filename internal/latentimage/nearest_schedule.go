package latentimage

import (
	"context"
	"fmt"
	"math"
	"overgo/internal/checked"
	"sort"
)

// nearestGridHalves follows the two-ended linspace in pinned PyTorch RangeFactoriesKernel.cpp.
// It changes only when the reference linspace contract changes.
const nearestGridHalves = 2

// TrainingGrid is the noise levels a model trained on, least first, which inference snaps its schedule to and a trainer draws
// from: total of them, bent by training_shift.
func TrainingGrid(total int, trainingShift float64) []float32 {
	train := make([]float32, total)
	for i := range train {
		train[total-i-1] = nearestGridSigma(i, total, total, trainingShift)
	}
	return train
}

// compileNearestGridFlowSchedule snaps shifted float32 knots onto a training grid.
// Ties select the lower sigma and consecutive duplicates collapse before the terminal zero.
// Steps are the request's and the work grows with them though the schedule does not, so
// each one answers to the request's context.
func compileNearestGridFlowSchedule(ctx context.Context, steps, total int, shift, trainingShift float64) (FlowSchedule, error) {
	if !checked.PositiveInts(steps, total) || !checked.PositiveFinite64(shift) || !checked.PositiveFinite64(trainingShift) || !checked.PositiveFinite32(float32(shift)) || !checked.PositiveFinite32(float32(trainingShift)) {
		return FlowSchedule{}, fmt.Errorf("nearest-grid flow: positive finite extents and shifts required, got %d/%d/%g/%g", steps, total, shift, trainingShift)
	}
	train := TrainingGrid(total, trainingShift)
	sigmas := make([]float64, 0, min(steps, total))
	for i := range steps {
		if err := ctx.Err(); err != nil {
			return FlowSchedule{}, err
		}
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

// sigmaFlowSchedule is the Euler walk over sigmas as given: each timestep the sigma at total levels, each delta the
// float32 difference to the next sigma, as compileNearestGridFlowSchedule makes them from its snapped knots.
func sigmaFlowSchedule(sigmas []float32, total int) FlowSchedule {
	steps := len(sigmas) - 1
	result := FlowSchedule{Steps: steps, NumTrainTimesteps: total, Sigmas: make([]float64, len(sigmas)), Timesteps: make([]float64, steps), Deltas: make([]float64, steps)}
	for i, sigma := range sigmas {
		result.Sigmas[i] = float64(sigma)
	}
	for i := range steps {
		result.Timesteps[i] = float64(sigmas[i] * float32(total))
		result.Deltas[i] = float64(sigmas[i+1] - sigmas[i])
	}
	return result
}

// cutSchedule is the last steps of a schedule, its last steps + 1 sigmas with their timesteps and deltas, as ComfyUI's
// KSampler keeps of a schedule drawn for a denoise below 1; a schedule of steps or fewer is kept whole.
func cutSchedule(schedule FlowSchedule, steps int) (FlowSchedule, error) {
	if steps <= 0 || len(schedule.Sigmas) != schedule.Steps+1 || len(schedule.Timesteps) != schedule.Steps || len(schedule.Deltas) != schedule.Steps {
		return FlowSchedule{}, fmt.Errorf("flow schedule: cannot keep the last %d steps of %d", steps, schedule.Steps)
	}
	if schedule.Steps <= steps {
		return schedule, nil
	}
	from := schedule.Steps - steps
	schedule.Sigmas, schedule.Timesteps, schedule.Deltas = schedule.Sigmas[from:], schedule.Timesteps[from:], schedule.Deltas[from:]
	schedule.Steps = steps
	return schedule, nil
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
