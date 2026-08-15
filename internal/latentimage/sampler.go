// FlowMatchEuler dynamic-shift sampling for the latent-image pipeline
// (FlowMatchEulerDiscreteScheduler with use_dynamic_shifting). The schedule is
// DERIVED: given the dynamic-shift parameter mu, the shift is exp(mu) and each
// sigma knot is shiftFlowSigma(1 - i/steps, exp(mu)); timesteps are sigma*N.
// This mirrors adaptive's compileShiftedFlowSchedule(steps, math.Exp(mu),
// shiftedFlowSigmaDescending) exactly. Pure host math, f64 throughout — the
// g2 golden knots reproduce bit-for-bit from mu alone (no hardcoded knots).
package latentimage

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// FlowSchedule: descending knots, timesteps, and Euler deltas.
type FlowSchedule struct {
	Steps             int
	NumTrainTimesteps int
	Mu                float64
	Shift             float64
	Sigmas            []float64 // Steps+1 knots, descending, terminal 0
	Timesteps         []float64 // Steps entries, sigma_current*N
	Deltas            []float64 // Steps entries, float32-rounded Sigmas[i+1]-Sigmas[i]
}

// DynamicShift maps mu to its multiplicative factor.
func DynamicShift(mu float64) float64 { return math.Exp(mu) }

// CompileFlowSchedule derives knots from declared extents and mu.
func CompileFlowSchedule(steps, numTrainTimesteps int, mu float64) (FlowSchedule, error) {
	if steps <= 0 {
		return FlowSchedule{}, fmt.Errorf("flow schedule: steps must be positive, got %d", steps)
	}
	if numTrainTimesteps <= 0 {
		return FlowSchedule{}, fmt.Errorf("flow schedule: num_train_timesteps must be positive, got %d", numTrainTimesteps)
	}
	if math.IsNaN(mu) || math.IsInf(mu, 0) {
		return FlowSchedule{}, fmt.Errorf("flow schedule: mu must be finite, got %g", mu)
	}
	shift := DynamicShift(mu)
	if shift <= 0 || math.IsInf(shift, 0) {
		return FlowSchedule{}, fmt.Errorf("flow schedule: shift exp(mu)=%g is invalid", shift)
	}
	sigmas := make([]float64, steps+1)
	for i := range sigmas {
		base := 1 - float64(i)/float64(steps)
		sigmas[i] = hostmath.ShiftFlowSigma(base, shift)
	}
	// Exact terminal knot.
	sigmas[steps] = 0
	timesteps := make([]float64, steps)
	deltas := make([]float64, steps)
	for i := 0; i < steps; i++ {
		timesteps[i] = sigmas[i] * float64(numTrainTimesteps)
		// Match device F32 delta storage.
		deltas[i] = float64(float32(sigmas[i+1] - sigmas[i]))
	}
	return FlowSchedule{
		Steps:             steps,
		NumTrainTimesteps: numTrainTimesteps,
		Mu:                mu,
		Shift:             shift,
		Sigmas:            sigmas,
		Timesteps:         timesteps,
		Deltas:            deltas,
	}, nil
}

// EulerStep advances sample in place.
func (s FlowSchedule) EulerStep(sample, velocity []float64, step int) error {
	if step < 0 || step >= s.Steps {
		return fmt.Errorf("flow euler: step %d out of range [0,%d)", step, s.Steps)
	}
	if len(sample) != len(velocity) {
		return fmt.Errorf("flow euler: sample len=%d velocity len=%d differ", len(sample), len(velocity))
	}
	delta := s.Deltas[step]
	for i := range sample {
		sample[i] += delta * velocity[i]
	}
	return nil
}
