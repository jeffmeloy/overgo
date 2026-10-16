//go:build !windows

package optimizer

import "errors"

type hostStepper struct{ optimizer *Optimizer }

// NewStepper binds flat storage to the shared host Muon implementation.
func NewStepper(weights, gradients []float32, plan Plan, config Config) (Stepper, error) {
	instance, err := New(weights, gradients, plan, config)
	if err != nil {
		return nil, err
	}
	return &hostStepper{optimizer: instance}, nil
}

func (s *hostStepper) Step() error {
	s.optimizer.Step()
	return nil
}

// Snapshot is the host optimizer's own state.
func (s *hostStepper) Snapshot() State { return s.optimizer.Snapshot() }

// Restore replaces a validated checkpoint with the host's native momentum precision.
func (s *hostStepper) Restore(state State) error {
	if len(state.MomentumF32) != 0 {
		return errors.New("host optimizer: exact resume requires F64 momentum")
	}
	return s.optimizer.Restore(state)
}

func (s *hostStepper) Close() error { return nil }
