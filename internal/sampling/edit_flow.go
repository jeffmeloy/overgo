package sampling

import (
	"errors"
	"math"

	"overgo/internal/checked"
)

// EditFlowConfig bounds compilation of a shifted flow-matching schedule.
type EditFlowConfig struct {
	InferenceSteps int
	TrainTimesteps int
	Shift          float32
	SigmaMin       float32
	SigmaMax       float32
	ExtraStep      bool
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
	sigmas := make([]float32, count)
	timesteps := make([]float32, count)
	for index := range count {
		raw := float32(float64(config.SigmaMax) + float64(index)*float64(config.SigmaMin-config.SigmaMax)/float64(linspace-1))
		sigmas[index] = config.Shift * raw / (1 + (config.Shift-1)*raw)
		timesteps[index] = sigmas[index] * float32(config.TrainTimesteps)
	}
	selected := make([]float32, len(requested))
	for requestIndex, target := range requested {
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
