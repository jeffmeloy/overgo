//go:build windows

package optimizer

import (
	"overgo/internal/cuda/device"
	"overgo/internal/tensor/dtype"
)

type deviceStepper struct {
	worker             *device.Worker
	weights, gradients []float32
	momentum           []float32
	plan               Plan
	config             Config
	step               int
}

// NewStepper binds flat host storage to the shared CUDA Muon implementation.
func NewStepper(weights, gradients []float32, plan Plan, config Config) (Stepper, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	return &deviceStepper{
		worker: worker, weights: weights, gradients: gradients, momentum: make([]float32, len(weights)),
		plan: plan, config: config,
	}, nil
}

func (s *deviceStepper) Step() error {
	s.step++
	return DeviceMuonStepPlan(s.worker, s.weights, s.gradients, s.momentum, s.plan, s.step, s.config)
}

// Snapshot widens the f32 momentum the device steps, exactly, to the host
// State.
func (s *deviceStepper) Snapshot() State {
	return State{PlanIdentity: s.plan.Identity(), Config: s.config, Step: s.step, Momentum: dtype.ConvertSlice[float64](s.momentum)}
}

func (s *deviceStepper) Close() error { return s.worker.Close() }
