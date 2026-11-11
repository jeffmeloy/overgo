package sampling

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
)

// Adaptive is APG, adaptive projected guidance (Sadat, Hilliges and Weber, arXiv 2410.02416), written from its paper
// and none of its code read. It guides the denoised prediction D rather than the model's output: the guidance delta
// D_c - D_u takes a negative momentum, the running average of the deltas before it times Momentum added, the sum
// becoming the average; it is clamped to the norm Threshold; and it is split into its part along D_c and the part
// orthogonal to it, the parallel part weighted by Parallel (zero in the paper's results). The guided prediction is
// D_c + (scale - 1)(orthogonal + Parallel parallel). Its zero value is off.
type Adaptive struct {
	Threshold float64 `json:"threshold"`
	Momentum  float64 `json:"momentum,omitzero"`
	Parallel  float64 `json:"parallel,omitzero"`
}

// On is whether the guidance is APG's.
func (a Adaptive) On() bool { return a != Adaptive{} }

// validate refuses APG no generation can run: a norm that is no positive finite number, a momentum outside (-1, 0],
// which the paper takes negative and which at -1 or below never decays, or a parallel weight outside [0, 1].
func (a Adaptive) validate() error {
	if !a.On() {
		return nil
	}
	if !checked.PositiveFinite64(a.Threshold) || !(a.Momentum > -1 && a.Momentum <= 0) || !(a.Parallel >= 0 && a.Parallel <= 1) {
		return fmt.Errorf("guidance: APG of norm %g, momentum %g and parallel weight %g cannot run", a.Threshold, a.Momentum, a.Parallel)
	}
	return nil
}

// AdaptiveGuider runs APG over one generation's steps, its momentum's running average carried from step to step; a
// generation makes one and combines each guided step through it, in order.
type AdaptiveGuider struct {
	Adaptive Adaptive
	// Scale is the guidance scale w.
	Scale   float64
	average []float64
}

// Combine writes into out the velocity a step steps by under APG, from the conditioned and unconditioned velocities the
// model gave at sample: a velocity v denoises the sample x to x - along v (along is the noise level for a velocity that
// points from the image to the noise, and its negative for one that points to the image), the guided prediction is
// formed as the paper forms it, and out is the velocity that denoises sample to it. out may be either velocity: each
// element is read before it is written.
func (a *AdaptiveGuider) Combine(out, conditional, unconditional, sample []float32, along float64) error {
	elements := len(out)
	if !checked.Equal(elements, len(conditional)) || !checked.Equal(elements, len(unconditional)) || !checked.Equal(elements, len(sample)) {
		return fmt.Errorf("guidance: APG buffer lengths %d/%d/%d/%d differ", elements, len(conditional), len(unconditional), len(sample))
	}
	if along == 0 || !checked.Finite64(along) {
		return fmt.Errorf("guidance: APG cannot denoise at %g", along)
	}
	if a.average == nil {
		a.average = make([]float64, elements)
	}
	if len(a.average) != elements {
		return errors.New("guidance: APG's steps differ in extent")
	}
	denoised := func(index int, velocity []float32) float64 {
		return float64(sample[index]) - along*float64(velocity[index])
	}
	// The delta with the momentum's running average, which it then becomes, and its norm.
	var squared float64
	for index := range elements {
		delta := denoised(index, conditional) - denoised(index, unconditional) + a.Adaptive.Momentum*a.average[index]
		a.average[index] = delta
		squared += delta * delta
	}
	// The delta clamped to the norm, and its projection onto the conditioned prediction.
	clamp := min(1, a.Adaptive.Threshold/math.Sqrt(squared))
	var across, length float64
	for index := range elements {
		guided := denoised(index, conditional)
		across += clamp * a.average[index] * guided
		length += guided * guided
	}
	var ratio float64
	if length > 0 {
		ratio = across / length
	}
	for index := range elements {
		guided := denoised(index, conditional)
		parallel := ratio * guided
		orthogonal := clamp*a.average[index] - parallel
		prediction := guided + (a.Scale-1)*(orthogonal+a.Adaptive.Parallel*parallel)
		value := (float64(sample[index]) - prediction) / along
		if !checked.Finite64(value) {
			return errors.New("guidance: APG gave no finite velocity")
		}
		out[index] = float32(value)
	}
	return nil
}
