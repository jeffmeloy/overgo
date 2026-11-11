package sampling

import (
	"math"
	"testing"

	"overgo/internal/testcheck"
)

// TestAPGIsAGuiderPreset holds APG (arXiv 2410.02416) to what its paper states of it, on small vectors: with the
// parallel part weighted fully, no momentum and no norm reached, it is the plain guidance of the denoised predictions,
// which for a velocity is the plain guidance of the velocities; with the parallel part dropped, the guided prediction
// moves from the conditioned one only across it; a norm reached clamps how far it moves; and the negative momentum
// carries the step before into the next. Guidance naming APG beside CFG-Zero*'s projection, or APG that cannot run, is
// refused, and the plain guider refuses to combine for it.
func TestAPGIsAGuiderPreset(t *testing.T) {
	t.Parallel()
	sample := []float32{1, 2, 0, -1}
	conditional := []float32{0.5, -1, 2, 0.25}
	unconditional := []float32{-0.5, 0.5, 1, 1}
	const along, scale = 0.5, 3.0
	denoise := func(velocity []float32) []float64 {
		out := make([]float64, len(velocity))
		for index, value := range velocity {
			out[index] = float64(sample[index]) - along*float64(value)
		}
		return out
	}
	combine := func(guider *AdaptiveGuider, conditional, unconditional []float32) []float64 {
		t.Helper()
		out := make([]float32, len(sample))
		testcheck.NoError(t, guider.Combine(out, conditional, unconditional, sample, along))
		return denoise(out)
	}
	near := func(left, right float64) bool { return math.Abs(left-right) < 1e-5 }
	conditioned, unconditioned := denoise(conditional), denoise(unconditional)

	plain := combine(&AdaptiveGuider{Adaptive: Adaptive{Threshold: 1e9, Parallel: 1}, Scale: scale}, conditional, unconditional)
	for index := range plain {
		if want := unconditioned[index] + scale*(conditioned[index]-unconditioned[index]); !near(plain[index], want) {
			t.Errorf("APG with the parallel part whole, no momentum and no norm reached gave %v at %d, want plain guidance's %v", plain[index], index, want)
		}
	}
	across := combine(&AdaptiveGuider{Adaptive: Adaptive{Threshold: 1e9}, Scale: scale}, conditional, unconditional)
	var moved, dot float64
	for index := range across {
		dot += (across[index] - conditioned[index]) * conditioned[index]
		moved += (across[index] - conditioned[index]) * (across[index] - conditioned[index])
	}
	if !near(dot, 0) || near(moved, 0) {
		t.Errorf("APG with the parallel part dropped moved the prediction %v along the conditioned one and %v in all, want only across it", dot, math.Sqrt(moved))
	}
	const threshold = 0.1
	clamped := combine(&AdaptiveGuider{Adaptive: Adaptive{Threshold: threshold}, Scale: scale}, conditional, unconditional)
	moved = 0
	for index := range clamped {
		moved += (clamped[index] - conditioned[index]) * (clamped[index] - conditioned[index])
	}
	if math.Sqrt(moved) > (scale-1)*threshold+1e-6 {
		t.Errorf("APG clamped to %v moved the prediction %v, beyond %v", threshold, math.Sqrt(moved), (scale-1)*threshold)
	}
	// Two steps alike: with momentum the second step's delta is the first's times (1 + momentum), which plain guidance
	// with the parallel part whole shows as that much guidance.
	const momentum = -0.5
	carried := &AdaptiveGuider{Adaptive: Adaptive{Threshold: 1e9, Parallel: 1, Momentum: momentum}, Scale: scale}
	combine(carried, conditional, unconditional)
	second := combine(carried, conditional, unconditional)
	for index := range second {
		if want := conditioned[index] + (scale-1)*(1+momentum)*(conditioned[index]-unconditioned[index]); !near(second[index], want) {
			t.Errorf("APG's second step gave %v at %d, want the delta carried by the momentum, %v", second[index], index, want)
		}
	}
	for name, guidance := range map[string]Guidance{
		"APG beside CFG-Zero*'s projection": {Projected: true, Adaptive: Adaptive{Threshold: 1}},
		"APG of no norm":                    {Adaptive: Adaptive{Momentum: momentum}},
		"APG whose momentum never decays":   {Adaptive: Adaptive{Threshold: 1, Momentum: -1}},
		"APG weighting the parallel part 2": {Adaptive: Adaptive{Threshold: 1, Parallel: 2}},
	} {
		if guidance.Validate() == nil {
			t.Errorf("%s was admitted", name)
		}
	}
	if err := (Guider{Scale: scale, Guidance: Guidance{Adaptive: Adaptive{Threshold: 1}}}).Combine(make([]float32, len(sample)), conditional, unconditional); err == nil {
		t.Error("the plain guider combined for APG, which needs the sample")
	}
}
