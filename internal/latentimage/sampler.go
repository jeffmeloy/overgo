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
)

// FlowSchedule: one compiled sampler schedule. Sigmas has Steps+1 entries
// (descending from the sigma_max knot to the terminal 0). Timesteps has Steps
// entries (t = sigma*NumTrainTimesteps for each of the first Steps sigmas).
// Deltas[i] = Sigmas[i+1] - Sigmas[i] is the Euler sigma step.
type FlowSchedule struct {
	Steps             int
	NumTrainTimesteps int
	Mu                float64
	Shift             float64
	Sigmas            []float64 // Steps+1 knots, descending, terminal 0
	Timesteps         []float64 // Steps entries, sigma_current*N
	Deltas            []float64 // Steps entries, float32-rounded Sigmas[i+1]-Sigmas[i]
}

// shiftFlowSigma applies the flow-matching time shift with a given shift
// factor: sigma' = shift*sigma / (1 + (shift-1)*sigma). This is algebraically
// identical to the diffusers dynamic form exp(mu)/(exp(mu)+(1/sigma-1)) with
// shift=exp(mu).
func shiftFlowSigma(sigma, shift float64) float64 {
	return shift * sigma / (1 + (shift-1)*sigma)
}

// DynamicShift maps the dynamic-shift parameter mu to the multiplicative shift
// factor exp(mu). The diffusers use_dynamic_shifting path derives mu itself
// from the image sequence length (a linear base_shift..max_shift interpolation)
// and passes it here; this pipeline supplies mu directly as the run parameter.
func DynamicShift(mu float64) float64 { return math.Exp(mu) }

// CompileFlowSchedule builds the FlowMatchEuler schedule from config: steps,
// the train-timestep count N, and the dynamic-shift parameter mu. The shift is
// exp(mu); knots are shiftFlowSigma(1-i/steps, shift) for i in [0,steps], with
// the terminal knot forced to exactly 0 (i==steps gives sigma=0 already).
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
		sigmas[i] = shiftFlowSigma(base, shift)
	}
	// i==steps yields base 0 -> shiftFlowSigma 0; assert the terminal knot.
	sigmas[steps] = 0
	timesteps := make([]float64, steps)
	deltas := make([]float64, steps)
	for i := 0; i < steps; i++ {
		timesteps[i] = sigmas[i] * float64(numTrainTimesteps)
		// The device computes the f64 sigma difference and stores the Euler
		// step delta in float32 (the native bf16/f32 compute dtype); round to
		// float32 so the step delta matches the device sampler bit-for-bit.
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

// EulerStep advances one FlowMatchEuler discrete step in place:
// sample += (sigma_next - sigma_current) * velocity, i.e. x_{i+1} = x_i +
// delta_i * v_i (the flow-matching Euler update, velocity = model output).
// The buffers must share length; velocity is the denoiser's per-element output
// at the current sigma.
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
