package sampling

import (
	"errors"
	"math"

	"overgo/internal/checked"
)

// EditFlowConfig bounds compilation of a shifted flow-matching schedule. Warp names a requested timestep t by the
// schedule's own entry t places from its end, a final zero after it, as Self-Forcing's warp_denoising_step reads its
// denoising_step_list (pipeline/causal_inference.py: timesteps[1000 - list]); without it a requested timestep takes the
// sigma of the schedule's timestep nearest it, as LiveEdit's pipeline, which leaves the warp out, does.
type EditFlowConfig struct {
	InferenceSteps int
	TrainTimesteps int
	Shift          float32
	SigmaMin       float32
	SigmaMax       float32
	ExtraStep      bool
	Warp           bool
}

// CompileEditFlowSigmas compiles requested source-order timesteps from a
// shifted FP32 flow schedule.
func CompileEditFlowSigmas(config EditFlowConfig, requested []int64) ([]float32, error) {
	if !checked.PositiveInts(config.InferenceSteps, config.TrainTimesteps) || !checked.PositiveFinite32(config.Shift) ||
		!checked.NonNegativeFinite32(config.SigmaMin) || !checked.NonNegativeFinite32(config.SigmaMax) || config.SigmaMax < config.SigmaMin {
		return nil, errors.New("sampling: invalid edit flow config")
	}
	if _, ok := checked.First(requested); !ok {
		return nil, errors.New("sampling: edit flow has no requested timesteps")
	}
	linspace := config.InferenceSteps
	if config.ExtraStep {
		linspace++
	}
	if linspace < 2 {
		return nil, errors.New("sampling: edit flow schedule is too short")
	}
	count := linspace
	if config.ExtraStep {
		count--
	}
	// The warped schedule ends in a zero after its last entry.
	sigmas := make([]float32, count, count+1)
	timesteps := make([]float32, count)
	for index := range count {
		raw := float32(float64(config.SigmaMax) + float64(index)*float64(config.SigmaMin-config.SigmaMax)/float64(linspace-1))
		sigmas[index] = config.Shift * raw / (1 + (config.Shift-1)*raw)
		timesteps[index] = sigmas[index] * float32(config.TrainTimesteps)
	}
	selected := make([]float32, len(requested))
	for requestIndex, target := range requested {
		if config.Warp {
			place := int64(config.TrainTimesteps) - target
			if !checked.NonNegativeInt64(place) || !checked.AtMostInt64(place, int64(count)) {
				return nil, errors.New("sampling: a warped timestep lies outside the schedule")
			}
			selected[requestIndex] = append(sigmas, 0)[place]
			continue
		}
		best := 0
		distance := math.Abs(float64(timesteps[0]) - float64(target))
		for index := 1; index < len(timesteps); index++ {
			candidate := math.Abs(float64(timesteps[index]) - float64(target))
			if candidate < distance {
				best, distance = index, candidate
			}
		}
		selected[requestIndex] = sigmas[best]
	}
	return selected, nil
}
