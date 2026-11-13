package sampling

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
)

// Named schedules a flow request may sample by, as ComfyUI's KSampler names them (moss-ultra-sections/schedules).
const (
	// ScheduleBongTangent names BongTangent, the schedule the MOSS Ultra workflow samples MOSS with.
	ScheduleBongTangent = "bong_tangent"
	// ScheduleBeta57 names the beta schedule at alpha 0.5 and beta 0.7, which RES4LYF offers beside it.
	ScheduleBeta57 = "beta57"
)

// bong_tangent's shape: its first segment falls from bongStart to bongMiddle and its second from there to bongEnd,
// each an arctangent pivoting at bongPivot of the schedule's points with slope bongSlope at bongSlopeSteps points.
const (
	bongStart, bongMiddle, bongEnd = 1.0, 0.5, 0.0
	bongPivot                      = 0.6
	bongSlope                      = 0.2
	bongSlopeSteps                 = 40
	// bongExtraPoints is how many points the schedule's segments hold beyond its steps: each segment holds both its
	// ends, so the point where they join is counted twice, and the terminal sigma once more.
	bongExtraPoints = 2
)

// beta57's distribution (Lee et al., "Beta Sampling is All You Need", 2024, at the pack's alpha 0.5 and beta 0.7).
const beta57Alpha, beta57Beta = 0.5, 0.7

// fullDenoise is the denoise at and above which a schedule is drawn whole, as ComfyUI's KSampler draws it.
const fullDenoise = 0.9999

// BongTangent is the bong_tangent schedule of steps steps: steps + 1 sigmas from 1 to 0 in two arctangent segments
// joined at 0.5, written from its formula (RES4LYF names it; its code, AGPL-3.0 with a clause against use in a
// commercial AI image service without its author's licence, is not taken). The schedule's steps + 2 points, the
// joining one counted in both segments, split where the pivot falls: the first segment holds ⌊0.6·(steps + 2)⌋ points
// and pivots at its end, the second the rest and pivots at its start, both with slope 0.2 scaled by 40 / (steps + 2).
// Each segment is the arctangent rescaled so its first and last points are its ends. No flow shift applies. One step
// leaves the first segment a single point, which has no slope to rescale: ComfyUI fails there dividing by zero, and
// this refuses it.
func BongTangent(steps int) ([]float32, error) {
	if steps < 2 {
		return nil, fmt.Errorf("sampling: bong_tangent draws at least 2 steps, not %d: its first segment would be one point", steps)
	}
	points := steps + bongExtraPoints
	slope := bongSlope / (float64(points) / bongSlopeSteps)
	pivot := int(bongPivot * float64(points))
	first := arctangentSegment(pivot, slope, float64(pivot), bongStart, bongMiddle)
	second := arctangentSegment(points-pivot, slope, 0, bongMiddle, bongEnd)
	sigmas := make([]float32, 0, steps+1)
	for _, sigma := range append(first, second[1:]...) {
		sigmas = append(sigmas, float32(sigma))
	}
	return sigmas, nil
}

// arctangentSegment is n points x = 0 … n-1 of the falling arctangent ½(1 + (2/π)·atan(−slope·(x − pivot))), rescaled
// so its first point is from and its last is to.
func arctangentSegment(n int, slope, pivot, from, to float64) []float64 {
	curve := func(x int) float64 { return ((2/math.Pi)*math.Atan(-slope*(float64(x)-pivot)) + 1) / 2 }
	high, low := curve(0), curve(n-1)
	points := make([]float64, n)
	for x := range points {
		points[x] = (curve(x)-low)/(high-low)*(from-to) + to
	}
	return points
}

// ShiftedFlowTable is the noise levels a discrete flow model's sampling holds, least first, as ComfyUI's discrete flow
// sampling makes them: σ = shift·t / (1 + (shift − 1)·t) at t = i / total for i = 1 … total, each operation rounded to
// float32 as torch computes it. A schedule that indexes the model's levels, as beta57 does, reads them here.
func ShiftedFlowTable(total int, shift float64) ([]float32, error) {
	if total < 2 || !checked.PositiveFinite64(shift) {
		return nil, fmt.Errorf("sampling: a flow table needs at least 2 levels and a positive shift, not %d and %g", total, shift)
	}
	table := make([]float32, total)
	scale, bend := float32(shift), float32(shift-1)
	for i := range table {
		t := float32(i+1) / float32(total)
		// Each conversion rounds, so no multiply and add fuse where torch rounds between them.
		table[i] = float32(scale*t) / float32(1+float32(bend*t))
	}
	return table, nil
}

// BetaSchedule is the beta schedule of steps steps over a model's noise levels (table, least first): the levels at
// the beta distribution's quantiles of 1, 1 − 1/steps, …, 1/steps, each quantile scaled to the table's last index
// and rounded half to even, a level repeated in turn taken once, and a terminal 0. Repeats make it shorter than
// steps + 1 where the quantiles crowd one level.
func BetaSchedule(table []float32, steps int, alpha, beta float64) ([]float32, error) {
	if len(table) < 2 || steps < 1 || !checked.PositiveFinite64(alpha) || !checked.PositiveFinite64(beta) {
		return nil, fmt.Errorf("sampling: a beta schedule needs a table of levels, steps and a positive alpha and beta, not %d, %d, %g and %g", len(table), steps, alpha, beta)
	}
	last := float64(len(table) - 1)
	sigmas := make([]float32, 0, steps+1)
	previous := -1.0
	for i := range steps {
		index := math.RoundToEven(betaQuantile(1-float64(i)*(1/float64(steps)), alpha, beta) * last)
		if index != previous {
			sigmas = append(sigmas, table[int(index)])
		}
		previous = index
	}
	return append(sigmas, 0), nil
}

// betaQuantile is the p-quantile of the beta distribution of alpha and beta: the x in [0, 1] whose regularized
// incomplete beta function is p, found by bisection to the precision of a float64.
func betaQuantile(p, alpha, beta float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return 1
	}
	low, high := 0.0, 1.0
	for range 200 {
		middle := (low + high) / 2
		if middle == low || middle == high {
			break
		}
		if regularizedBeta(middle, alpha, beta) < p {
			low = middle
		} else {
			high = middle
		}
	}
	return (low + high) / 2
}

// regularizedBeta is I_x(a, b), the beta distribution's cumulative probability at x, from its continued fraction
// (evaluated by the modified Lentz method), taken on the side of x where it converges and reflected on the other.
func regularizedBeta(x, a, b float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	if x > (a+1)/(a+b+2) {
		return 1 - regularizedBeta(1-x, b, a)
	}
	lgammaA, _ := math.Lgamma(a)
	lgammaB, _ := math.Lgamma(b)
	lgammaAB, _ := math.Lgamma(a + b)
	front := math.Exp(a*math.Log(x)+b*math.Log1p(-x)-(lgammaA+lgammaB-lgammaAB)) / a
	const tiny, epsilon = 1e-300, 1e-16
	f, c, d := tiny, tiny, 0.0
	for i := range 1000 {
		m := float64(i / 2)
		var numerator float64
		switch {
		case i == 0:
			numerator = 1
		case i%2 == 0:
			numerator = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
		default:
			numerator = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
		}
		d = 1 + numerator*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		d = 1 / d
		c = 1 + numerator/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		step := c * d
		f *= step
		if math.Abs(1-step) < epsilon {
			break
		}
	}
	return front * f
}

// Denoised is a schedule of steps steps at denoise below 1, as ComfyUI's KSampler draws one: the full schedule of
// ⌊steps / denoise⌋ steps, its last steps + 1 sigmas, so the sampling starts part way down. At a denoise of 0.9999 or
// more it is the full schedule of steps.
func Denoised(full func(steps int) ([]float32, error), steps int, denoise float64) ([]float32, error) {
	if steps < 1 || !checked.PositiveFinite64(denoise) || denoise > 1 {
		return nil, fmt.Errorf("sampling: a schedule draws a step or more at a denoise above 0 and at most 1, not %d at %g", steps, denoise)
	}
	if denoise >= fullDenoise {
		return full(steps)
	}
	sigmas, err := full(int(float64(steps) / denoise))
	if err != nil {
		return nil, err
	}
	return sigmas[max(len(sigmas)-(steps+1), 0):], nil
}

// NamedSchedule is the schedule a name gives at steps and denoise, over a model's noise levels (table, least first),
// which beta57 reads and bong_tangent does not.
func NamedSchedule(name string, steps int, denoise float64, table []float32) ([]float32, error) {
	var full func(steps int) ([]float32, error)
	switch name {
	case ScheduleBongTangent:
		full = BongTangent
	case ScheduleBeta57:
		full = func(steps int) ([]float32, error) { return BetaSchedule(table, steps, beta57Alpha, beta57Beta) }
	default:
		return nil, fmt.Errorf("sampling: no schedule is named %q (%s or %s)", name, ScheduleBongTangent, ScheduleBeta57)
	}
	sigmas, err := Denoised(full, steps, denoise)
	if err != nil {
		return nil, err
	}
	return sigmas, ValidateFlowSigmas(sigmas)
}

// Samplers a flow request may sample by, as ComfyUI's KSampler names them.
const (
	// SamplerEuler names the flow Euler step a model samples by unless a request names another.
	SamplerEuler = "euler"
	// SamplerERSDE names ERSDE, which the MOSS Ultra workflow and ComfyUI's MOSS template sample with.
	SamplerERSDE = "er_sde"
)

// Schedule is how a flow request samples in place of its model's own way, as ComfyUI's KSampler takes a sampler and a
// scheduler by name or a list of custom sigmas: its sampler (Sampler, flow Euler where it names none), and its sigmas,
// a named schedule (Name) drawn at the request's steps or the request's own list (Sigmas), one more than its steps, or
// neither, where the model's own schedule serves.
type Schedule struct {
	Sampler string    `json:"sampler,omitzero"`
	Name    string    `json:"name,omitzero"`
	Sigmas  []float32 `json:"sigmas,omitempty"`
}

// Validate holds a schedule to a known sampler, and to a known name or a valid list of its own, not both.
func (s Schedule) Validate() error {
	if s.Sampler != "" && s.Sampler != SamplerEuler && s.Sampler != SamplerERSDE {
		return fmt.Errorf("sampling: no sampler is named %q (%s or %s)", s.Sampler, SamplerEuler, SamplerERSDE)
	}
	switch {
	case s.Name != "" && len(s.Sigmas) != 0:
		return errors.New("sampling: a schedule names a scheduler or lists its sigmas, not both")
	case s.Name == "" && len(s.Sigmas) == 0:
		if s.Sampler == "" {
			return errors.New("sampling: a schedule names a sampler, a scheduler or its sigmas")
		}
		return nil
	case s.Name != "":
		if s.Name != ScheduleBongTangent && s.Name != ScheduleBeta57 {
			return fmt.Errorf("sampling: no schedule is named %q (%s or %s)", s.Name, ScheduleBongTangent, ScheduleBeta57)
		}
		return nil
	}
	return ValidateFlowSigmas(s.Sigmas)
}

// OwnSigmas is whether the schedule gives sigmas of its own, by name or as a list, rather than the model's.
func (s Schedule) OwnSigmas() bool { return s.Name != "" || len(s.Sigmas) != 0 }

// Draw is the schedule's sigmas at steps over a model's noise levels (table, least first): a named schedule drawn
// whole, or the request's own list, whose steps are its own, which steps names again or leaves 0.
func (s Schedule) Draw(steps int, table []float32) ([]float32, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if !s.OwnSigmas() {
		return nil, errors.New("sampling: the schedule gives no sigmas of its own; the model's serve")
	}
	if s.Name != "" {
		return NamedSchedule(s.Name, steps, 1, table)
	}
	if steps != 0 && steps != len(s.Sigmas)-1 {
		return nil, fmt.Errorf("sampling: %d sigmas make %d steps, not the %d the request names", len(s.Sigmas), len(s.Sigmas)-1, steps)
	}
	return slices.Clone(s.Sigmas), nil
}

// ValidateFlowSigmas holds a flow schedule a request may carry as its own, as ComfyUI's custom sigmas allow: two or more
// finite sigmas in [0, 1], each below the one before.
func ValidateFlowSigmas(sigmas []float32) error {
	if len(sigmas) < 2 {
		return errors.New("sampling: a schedule holds two sigmas or more")
	}
	for i, sigma := range sigmas {
		if !checked.NonNegativeFinite32(sigma) || sigma > 1 {
			return fmt.Errorf("sampling: sigma %d is %g, outside [0, 1]", i, sigma)
		}
		if i > 0 && sigma >= sigmas[i-1] {
			return fmt.Errorf("sampling: sigma %d is %g, not below the %g before it", i, sigma, sigmas[i-1])
		}
	}
	return nil
}
