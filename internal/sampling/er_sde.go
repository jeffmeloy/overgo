package sampling

import (
	"fmt"
	"math"
)

// ER-SDE's settings as the MOSS Ultra workflow samples with it (moss-ultra-sections/er-sde).
const (
	// ERSDEStages is the solver's highest order, its third.
	ERSDEStages = 3
	// erSDEPoints is how many points each integral over 1/φ sums, at the left of each of its even widths: at 200 a
	// solver lands within float32 rounding of ComfyUI's er_sde, at the paper's 100 about 3e-4 off it.
	erSDEPoints = 200
	// erSDEExponent and erSDEOffset shape the noise scale function φ(x) = x·(exp(x^0.3) + 10), the paper's ER SDE 4.
	erSDEExponent, erSDEOffset = 0.3, 10
)

// ERSDE is the extended reverse-time SDE solver of Cui et al. (arXiv 2309.06169, ER-SDE-Solver-3 for a VP model's data
// prediction) over a flow model's sigmas: a flow latent at sigma σ is α·x0 + σ·ε with α = 1 − σ, so λ = σ/α. Each step
// from λs to λt keeps r·αt/αs of the latent, r = φ(λt)/φ(λs), takes αt·(1 − r) of the data prediction, adds the
// divided differences of the earlier predictions over λ weighted by their integrals of 1/φ (the second and third
// orders, as many as the steps taken allow), and draws fresh noise of scale αt·√(λt² − r²λs²). From σ = 1, where λs is
// unbounded, r·λs and r/αs both vanish, so the step is the prediction and noise of scale σt. A step to σ = 0 lands on
// the data prediction and draws nothing.
type ERSDE struct {
	sigmas []float64
	stages int
	// previous is the last step's data prediction, and difference its first divided difference over λ.
	previous, difference []float32
	taken                int
}

// NewERSDE is the solver over a flow schedule's sigmas (ValidateFlowSigmas), to stages orders.
func NewERSDE(sigmas []float32, stages int) (*ERSDE, error) {
	if err := ValidateFlowSigmas(sigmas); err != nil {
		return nil, err
	}
	if stages < 1 || stages > ERSDEStages {
		return nil, fmt.Errorf("sampling: ER-SDE runs 1 to %d stages, not %d", ERSDEStages, stages)
	}
	solver := &ERSDE{sigmas: make([]float64, len(sigmas)), stages: stages}
	for i, sigma := range sigmas {
		solver.sigmas[i] = float64(sigma)
	}
	return solver, nil
}

// Steps is how many steps the solver takes.
func (s *ERSDE) Steps() int { return len(s.sigmas) - 1 }

// Draws is whether step draws fresh noise: every step but one to σ = 0.
func (s *ERSDE) Draws(step int) bool { return step >= 0 && step < s.Steps() && s.sigmas[step+1] > 0 }

// erSDEScale is the noise scale function φ.
func erSDEScale(x float64) float64 { return x * (math.Exp(math.Pow(x, erSDEExponent)) + erSDEOffset) }

// lambda is a flow sigma's λ = σ/(1 − σ).
func lambda(sigma float64) float64 { return sigma / (1 - sigma) }

// Skip passes over the solver's next step, one the guidance zeroes, which leaves the latent where it is: the earlier
// predictions no longer lie one step before the next, so the order starts again from the first.
func (s *ERSDE) Skip() error {
	if s.taken >= s.Steps() {
		return fmt.Errorf("sampling: ER-SDE has taken its %d steps", s.Steps())
	}
	s.taken++
	s.previous, s.difference = nil, nil
	return nil
}

// Step takes the solver's next step from latent x, its data prediction denoised and fresh noise (nil where Draws is
// false), giving the next latent. Steps are taken in order, each once.
func (s *ERSDE) Step(x, denoised, noise []float32) ([]float32, error) {
	step := s.taken
	if step >= s.Steps() || len(denoised) != len(x) || s.Draws(step) != (noise != nil) || noise != nil && len(noise) != len(x) {
		return nil, fmt.Errorf("sampling: ER-SDE step %d of %d needs a latent, its prediction and, but for a step to 0, noise of its size", step, s.Steps())
	}
	s.taken++
	defer func() { s.previous = denoised }()
	sigmaS, sigmaT := s.sigmas[step], s.sigmas[step+1]
	if sigmaT == 0 {
		return append([]float32(nil), denoised...), nil
	}
	lambdaS, lambdaT, alphaT := lambda(sigmaS), lambda(sigmaT), 1-sigmaT
	keep, ratio, noiseScale := 0.0, 0.0, lambdaT
	if sigmaS < 1 {
		ratio = erSDEScale(lambdaT) / erSDEScale(lambdaS)
		keep = ratio * alphaT / (1 - sigmaS)
		noiseScale = math.Sqrt(math.Max(lambdaT*lambdaT-ratio*ratio*lambdaS*lambdaS, 0))
	}
	next := make([]float32, len(x))
	for i := range x {
		next[i] = float32(keep*float64(x[i]) + alphaT*(1-ratio)*float64(denoised[i]) + alphaT*noiseScale*float64(noise[i]))
	}
	if s.stages < 2 || s.previous == nil || sigmaS >= 1 {
		s.difference = nil
		return next, nil
	}
	// The integrals over [λt, λs] of 1/φ and of (λ − λs)/φ, each a left sum over erSDEPoints even widths.
	width := (lambdaS - lambdaT) / erSDEPoints
	var flat, linear float64
	for k := range erSDEPoints {
		at := lambdaT + float64(k)*width
		flat += width / erSDEScale(at)
		linear += (at - lambdaS) * width / erSDEScale(at)
	}
	scaleT := erSDEScale(lambdaT)
	lambdaBefore := lambda(s.sigmas[step-1])
	first := make([]float32, len(x))
	firstWeight := alphaT * (lambdaT - lambdaS + flat*scaleT)
	for i := range first {
		first[i] = float32((float64(denoised[i]) - float64(s.previous[i])) / (lambdaS - lambdaBefore))
		next[i] += float32(firstWeight * float64(first[i]))
	}
	if s.stages >= 3 && s.difference != nil {
		secondWeight := alphaT * ((lambdaT-lambdaS)*(lambdaT-lambdaS)/2 + linear*scaleT)
		span := (lambdaS - lambda(s.sigmas[step-2])) / 2
		for i := range next {
			next[i] += float32(secondWeight * (float64(first[i]) - float64(s.difference[i])) / span)
		}
	}
	s.difference = first
	return next, nil
}
