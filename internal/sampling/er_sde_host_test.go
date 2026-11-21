package sampling

import "fmt"

// HostERSDE steps an ERSDE on the host, applying each Plan as samplingcuda.ERSDEDevice applies it on the card: the
// reference both are held to, ComfyUI's er_sde by TestERSDEDrawsNoiseAsTorchDoes and the card by
// TestERSDEStepsOnTheCardAsOnTheHost.
type HostERSDE struct {
	*ERSDE
	// previous is the last step's data prediction and difference its first divided difference over λ.
	previous, difference []float32
}

// NewHostERSDE is a host stepper over a flow schedule's sigmas (NewERSDE).
func NewHostERSDE(sigmas []float32, stages int) (*HostERSDE, error) {
	solver, err := NewERSDE(sigmas, stages)
	if err != nil {
		return nil, err
	}
	return &HostERSDE{ERSDE: solver}, nil
}

// Draws is whether step draws fresh noise: every step but one to σ = 0.
func (s *HostERSDE) Draws(step int) bool {
	return step >= 0 && step < s.Steps() && s.sigmas[step+1] > 0
}

// Skip passes over the next step (ERSDE.Skip), its earlier predictions let go.
func (s *HostERSDE) Skip() error {
	s.previous, s.difference = nil, nil
	return s.ERSDE.Skip()
}

// Step takes the solver's next step on the host from latent x, its data prediction denoised and fresh noise (nil where
// Draws is false), giving the next latent. Steps are taken in order, each once.
func (s *HostERSDE) Step(x, denoised, noise []float32) ([]float32, error) {
	step := s.taken
	if step >= s.Steps() || len(denoised) != len(x) || s.Draws(step) != (noise != nil) || noise != nil && len(noise) != len(x) {
		return nil, fmt.Errorf("sampling: ER-SDE step %d of %d needs a latent, its prediction and, but for a step to 0, noise of its size", step, s.Steps())
	}
	plan, err := s.Plan()
	if err != nil {
		return nil, err
	}
	defer func() { s.previous = denoised }()
	if plan.ToZero {
		return append([]float32(nil), denoised...), nil
	}
	next := make([]float32, len(x))
	for i := range x {
		next[i] = float32(plan.Keep*float64(x[i]) + plan.Data*float64(denoised[i]) + plan.Noise*float64(noise[i]))
	}
	if plan.Order < 2 {
		s.difference = nil
		return next, nil
	}
	first := make([]float32, len(x))
	for i := range first {
		first[i] = float32((float64(denoised[i]) - float64(s.previous[i])) / plan.Divisor)
		next[i] += float32(plan.FirstWeight * float64(first[i]))
	}
	if plan.Order >= 3 {
		for i := range next {
			next[i] += float32(plan.SecondWeight * (float64(first[i]) - float64(s.difference[i])) / plan.Span)
		}
	}
	s.difference = first
	return next, nil
}
