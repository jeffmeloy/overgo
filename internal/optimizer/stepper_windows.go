//go:build windows

package optimizer

import "overgo/internal/cuda/device"

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

func (s *deviceStepper) Close() error { return s.worker.Close() }
