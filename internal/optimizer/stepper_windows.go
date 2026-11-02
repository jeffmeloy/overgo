//go:build windows

package optimizer

import (
	"errors"
	"slices"

	"overgo/internal/cuda/device"
)

type deviceStepper struct {
	worker             *device.Worker
	weights, gradients []float32
	momentum           []float32
	plan               Plan
	config             Config
	step               int
	received           func(Group) bool
}

// NewStepper binds flat host storage to the shared CUDA Muon implementation.
func NewStepper(weights, gradients []float32, plan Plan, config Config) (Stepper, error) {
	return newStepper(weights, gradients, plan, config, func(Group) bool { return true })
}

// newStepper is NewStepper leaving alone a group received says took no gradient.
func newStepper(weights, gradients []float32, plan Plan, config Config, received func(Group) bool) (Stepper, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	return &deviceStepper{
		worker: worker, weights: weights, gradients: gradients, momentum: make([]float32, len(weights)),
		plan: plan, config: config, received: received,
	}, nil
}

func (s *deviceStepper) Step() error {
	s.step++
	return deviceMuonStep(s.worker, s.weights, s.gradients, s.momentum, s.plan, s.step, s.config, s.received)
}

// Snapshot copies the f32 momentum the device steps.
func (s *deviceStepper) Snapshot() State {
	return State{PlanIdentity: s.plan.Identity(), Config: s.config, Step: s.step, MomentumF32: slices.Clone(s.momentum)}
}

// Restore replaces a validated checkpoint with the device's native momentum precision.
func (s *deviceStepper) Restore(state State) error {
	if state.Config != s.config || len(state.MomentumF32) != len(s.momentum) {
		return errors.New("device optimizer: restore requires matching configuration and F32 momentum")
	}
	if err := ValidateState(state, s.plan.Identity(), len(s.momentum)); err != nil {
		return err
	}
	copy(s.momentum, state.MomentumF32)
	s.step = state.Step
	clear(s.gradients)
	return nil
}

func (s *deviceStepper) Close() error { return s.worker.Close() }
