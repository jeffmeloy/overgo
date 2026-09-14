package hostoptimizer

import (
	"errors"
	"math"

	"overgo/internal/checked"
)

// Schedule selects the learning-rate policy.
type Schedule uint8

const (
	// ScheduleConstant keeps the base learning rate.
	ScheduleConstant Schedule = iota
	// ScheduleLinearDecay decays the rate linearly over the horizon.
	ScheduleLinearDecay
)

// Config is the adaptive optimizer policy.
type Config struct {
	BaseLearningRate float64  `json:"base_learning_rate"`
	Momentum         float64  `json:"momentum"`
	Steps            int      `json:"steps"`
	Schedule         Schedule `json:"schedule"`
}

// Validate checks optimizer policy bounds.
func (c Config) Validate() error {
	if !checked.NonNegativeFinite64(c.BaseLearningRate) {
		return errors.New("optimizer config: learning rate must be finite and non-negative")
	}
	if !checked.NonNegativeFinite64(c.Momentum) || c.Momentum >= 1 {
		return errors.New("optimizer config: momentum must be finite and in [0,1)")
	}
	if c.Steps < 0 {
		return errors.New("optimizer config: negative step count")
	}
	switch c.Schedule {
	case ScheduleConstant:
	case ScheduleLinearDecay:
		if c.Steps == 0 {
			return errors.New("optimizer config: linear decay requires a positive step count")
		}
	default:
		return errors.New("optimizer config: unsupported schedule")
	}
	return nil
}

// LearningRate returns the rate for a one-based optimizer step.
func (c Config) LearningRate(step int) float64 {
	if c.Schedule == ScheduleConstant {
		return c.BaseLearningRate
	}
	progress := min(float64(max(step, 0))/float64(c.Steps), 1)
	return c.BaseLearningRate * (1 - progress)
}

// StepRMS is the update root-mean-square scale for a momentum coefficient.
func StepRMS(momentum float64) float64 {
	return math.Sqrt(1 - momentum*momentum)
}
