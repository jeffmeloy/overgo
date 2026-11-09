package sampling

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheGuiderTakesCFGZeroAndAGuidanceInterval holds a guider's step to the guidance it was given. The zero guidance guides every
// step as the reference does. CFG-Zero* gives its first steps no velocity, reading neither output, and rescales the
// unconditioned output by its projection onto the conditioned one, s = <c, u> / <u, u>, before guiding with it. A
// guidance interval leaves a step outside it unguided, the conditioned output passed through with no unconditioned pass.
// Guidance no generation can run is refused.
func TestTheGuiderTakesCFGZeroAndAGuidanceInterval(t *testing.T) {
	t.Parallel()
	conditional, unconditional := []float32{1, 2, 3}, []float32{2, 0, 1}
	reference := Guider{Scale: 5}
	want := make([]float32, len(conditional))
	testcheck.NoError(t, reference.Combine(want, conditional, unconditional))
	got := make([]float32, len(conditional))
	testcheck.NoError(t, reference.At(0, 0.9).Combine(got, conditional, unconditional))
	if !slices.Equal(got, want) || !reference.At(0, 0.9).Unconditional() {
		t.Fatalf("the zero guidance guides a step to %v, want the reference's %v", got, want)
	}

	zeroStar := Guider{Scale: 5, Guidance: Guidance{ZeroSteps: 1, Projected: true}}
	first := zeroStar.At(0, 1)
	testcheck.NoError(t, first.Combine(got, conditional, nil))
	if !first.Zeroed() || first.Unconditional() || slices.ContainsFunc(got, func(value float32) bool { return value != 0 }) {
		t.Fatalf("CFG-Zero*'s first step gives %v, zeroed %t, reading the unconditioned output %t; want no velocity and no pass", got, first.Zeroed(), first.Unconditional())
	}
	second := zeroStar.At(1, 0.9)
	testcheck.NoError(t, second.Combine(got, conditional, unconditional))
	projection := (1.0*2 + 2*0 + 3*1) / (2.0*2 + 0 + 1*1 + projectionFloor)
	for index := range got {
		rescaled := projection * float64(unconditional[index])
		if expected := rescaled + 5*(float64(conditional[index])-rescaled); math.Abs(float64(got[index])-expected) > 1e-6 {
			t.Fatalf("CFG-Zero* guides element %d to %v, want %v from the projection %v", index, got[index], expected, projection)
		}
	}

	interval := Guider{Scale: 5, Guidance: Guidance{Interval: NoiseInterval{Low: 0.2, High: 0.8}}}
	for sigma, guided := range map[float64]bool{0.95: false, 0.5: true, 0.1: false} {
		at := interval.At(3, sigma)
		testcheck.NoError(t, at.Combine(got, conditional, map[bool][]float32{true: unconditional}[guided]))
		if at.Unconditional() != guided || !guided && !slices.Equal(got, conditional) {
			t.Fatalf("a step at noise %v reads the unconditioned output %t and gives %v; want %t", sigma, at.Unconditional(), got, guided)
		}
	}

	for _, invalid := range []Guidance{{ZeroSteps: -1}, {Interval: NoiseInterval{Low: 0.8, High: 0.2}}, {Interval: NoiseInterval{Low: 0.2, High: 1.5}}} {
		if invalid.Validate() == nil {
			t.Errorf("guidance %+v was admitted", invalid)
		}
	}
}
