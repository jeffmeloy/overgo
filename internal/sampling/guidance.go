package sampling

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
)

// Guidance is how a guider's guidance varies over a generation's steps, each part an addition to a model's reference
// preset, written from its paper and none of its code read. CFG-Zero* (arXiv 2503.18886) gives the first ZeroSteps steps
// no velocity and rescales the unconditioned output by its projection onto the conditioned one (Projected); a guidance
// interval (arXiv 2404.07724) guides only the steps whose noise level lies within it, the rest running no unconditioned
// pass. The zero value changes nothing.
type Guidance struct {
	ZeroSteps int           `json:"zero_steps,omitzero"`
	Projected bool          `json:"projected,omitzero"`
	Interval  NoiseInterval `json:"interval,omitzero"`
	// Adaptive is APG, which combines with the sample a step denoises (AdaptiveGuider).
	Adaptive Adaptive `json:"adaptive,omitzero"`
}

// projectionFloor is what CFG-Zero*'s appendix code adds to the unconditioned output's squared norm, so an output of
// none still projects.
const projectionFloor = 1e-8

// NoiseInterval is a band of noise levels, a flow sampler's sigma from one at pure noise to zero at the image: a step at
// a level within it is guided. Its zero value holds every level.
type NoiseInterval struct {
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

// holds is whether the interval holds sigma; the zero interval holds every level.
func (i NoiseInterval) holds(sigma float64) bool {
	return i == NoiseInterval{} || i.Low <= sigma && sigma <= i.High
}

// Validate refuses guidance no generation can run: negative zeroed steps, or an interval not ordered within [0, 1].
func (g Guidance) Validate() error {
	if g.ZeroSteps < 0 {
		return fmt.Errorf("guidance: %d zeroed steps", g.ZeroSteps)
	}
	if g.Interval != (NoiseInterval{}) && !(0 <= g.Interval.Low && g.Interval.Low < g.Interval.High && g.Interval.High <= unguided) {
		return fmt.Errorf("guidance: interval [%g, %g] is no ordered band of noise levels within [0, 1]", g.Interval.Low, g.Interval.High)
	}
	if g.Projected && g.Adaptive.On() {
		return errors.New("guidance: CFG-Zero*'s projection and APG are two combines, and a step takes one")
	}
	return g.Adaptive.validate()
}

// At is the guider for the step at index step, whose noise level is sigma: one that gives no velocity on a step
// CFG-Zero* zeroes, and one at a scale of one, which reads no unconditioned output, outside the guidance interval.
func (g Guider) At(step int, sigma float64) Guider {
	at := g
	at.zeroed = step < g.Guidance.ZeroSteps
	if !g.Guidance.Interval.holds(sigma) {
		at.Scale = unguided
	}
	return at
}

// Zeroed is whether the guider's step gives no velocity, so a model runs neither pass for it.
func (g Guider) Zeroed() bool { return g.zeroed }

// combineProjected writes the guided output with the unconditioned output first rescaled by its projection onto the
// conditioned one, its scale the two outputs' dot product over the unconditioned output's own squared norm, one scale
// for the whole sample as the paper's code takes it.
func (g Guider) combineProjected(out, conditional, unconditional []float32) error {
	if !checked.Equal(len(out), len(conditional)) || !checked.Equal(len(out), len(unconditional)) {
		return fmt.Errorf("guidance: buffer lengths %d/%d/%d differ", len(out), len(conditional), len(unconditional))
	}
	var across, along float64
	for index, value := range unconditional {
		across += float64(conditional[index]) * float64(value)
		along += float64(value) * float64(value)
	}
	projection := across / (along + projectionFloor)
	if math.IsNaN(projection) || math.IsInf(projection, 0) {
		return errors.New("guidance: the unconditioned output has no projection onto the conditioned one")
	}
	for index, value := range unconditional {
		rescaled := projection * float64(value)
		out[index] = float32(rescaled + g.Scale*(float64(conditional[index])-rescaled))
	}
	return nil
}
