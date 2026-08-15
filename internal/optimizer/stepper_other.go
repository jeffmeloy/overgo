//go:build !windows

package optimizer

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

func (s *hostStepper) Close() error { return nil }
