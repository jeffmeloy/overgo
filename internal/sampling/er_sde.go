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
// the data prediction and draws nothing. Plan gives each step's arithmetic, which samplingcuda.ERSDEDevice applies on
// the card, held bit for bit to HostERSDE, the same arithmetic on the host, beside the tests.
type ERSDE struct {
	sigmas []float64
	stages int
	// held says the last step's data prediction is kept for the next, and differenced that its first divided difference
	// over λ is.
	held, differenced bool
	taken             int
}

// ERSDEPlan is one ER-SDE step's arithmetic over each element, in F64 and rounded to F32 where it rounds: the next latent
// is Keep·x + Data·denoised + Noise·noise; from the second order the first divided difference, (denoised − previous) /
// Divisor rounded, is kept for the next step and FirstWeight times it added, rounded; at the third, SecondWeight·(first
// − difference)/Span, rounded, is added too. A step ToZero lands on the data prediction and draws nothing.
type ERSDEPlan struct {
	ToZero                                   bool
	Keep, Data, Noise                        float64
	Order                                    int
	Divisor, FirstWeight, SecondWeight, Span float64
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
	s.held, s.differenced = false, false
	return nil
}

// Plan is the solver's next step's arithmetic, which the caller then applies to the step's latent, its data
// prediction and, but for a step to 0, fresh noise; the step's prediction is kept for the next, and its first divided
// difference where the plan's order reaches the second. Steps are planned in order, each once.
func (s *ERSDE) Plan() (ERSDEPlan, error) {
	step := s.taken
	if step >= s.Steps() {
		return ERSDEPlan{}, fmt.Errorf("sampling: ER-SDE has taken its %d steps", s.Steps())
	}
	s.taken++
	held := s.held
	s.held = true
	sigmaS, sigmaT := s.sigmas[step], s.sigmas[step+1]
	if sigmaT == 0 {
		return ERSDEPlan{ToZero: true}, nil
	}
	lambdaS, lambdaT, alphaT := lambda(sigmaS), lambda(sigmaT), 1-sigmaT
	keep, ratio, noiseScale := 0.0, 0.0, lambdaT
	if sigmaS < 1 {
		ratio = erSDEScale(lambdaT) / erSDEScale(lambdaS)
		keep = ratio * alphaT / (1 - sigmaS)
		noiseScale = math.Sqrt(math.Max(lambdaT*lambdaT-ratio*ratio*lambdaS*lambdaS, 0))
	}
	plan := ERSDEPlan{Keep: keep, Data: alphaT * (1 - ratio), Noise: alphaT * noiseScale, Order: 1}
	if s.stages < 2 || !held || sigmaS >= 1 {
		s.differenced = false
		return plan, nil
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
	plan.Order, plan.Divisor = 2, lambdaS-lambda(s.sigmas[step-1])
	plan.FirstWeight = alphaT * (lambdaT - lambdaS + flat*scaleT)
	if s.stages >= 3 && s.differenced {
		plan.Order = 3
		plan.SecondWeight = alphaT * ((lambdaT-lambdaS)*(lambdaT-lambdaS)/2 + linear*scaleT)
		plan.Span = (lambdaS - lambda(s.sigmas[step-2])) / 2
	}
	s.differenced = true
	return plan, nil
}
