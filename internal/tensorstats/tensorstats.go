// Package tensorstats computes distribution-free characterizations of tensor
// value populations: robust L-moments and streaming energy/value descriptors.
//
// It assumes nothing about the shape of the data (no Gaussianity, no metric
// prior). Every quantity is a measured functional of the empirical
// distribution: L-moments (unbiased probability-weighted order statistics),
// order-statistic quantiles, and bounded-memory energy descriptors. L-moments
// are the primary shape summary; mean/std/RMS are reported as measured
// functionals, not as parameters of an assumed distribution.
//
// Ported from adaptive_new statistics.ExactLMoments and extmodel's tensor value
// statistics; the math is neutral and family-free.
package tensorstats

import (
	"math"
	"slices"

	"overgo/internal/checked"
)

const medianQuantile = 0.5

// LMoments are unbiased sample L-moments. L1 is location, L2 is scale (a robust
// dispersion), Tau3 is L-skewness and Tau4 is L-kurtosis, each bounded in
// [-1, 1]. L-moments exist whenever the mean exists, so they characterize
// heavy-tailed populations where central moments do not.
type LMoments struct {
	L1, L2, L3, L4 float64
	Tau3, Tau4     float64
}

// ExactLMoments computes L-moments from finite values. The input is cloned and
// sorted; callers must pass only finite values. Fewer than four values yields
// the lower-order moments that are defined and leaves the rest zero.
func ExactLMoments(finite []float64) LMoments {
	return exactLMomentsSorted(slices.Clone(finite))
}

// exactLMomentsSorted sorts caller-owned finite values in place and computes the
// unbiased probability-weighted L-moments.
func exactLMomentsSorted(values []float64) LMoments {
	slices.Sort(values)
	n := len(values)
	if n == 0 {
		return LMoments{}
	}
	fn := float64(n)
	var b0, b1, b2, b3 float64
	for i, value := range values {
		fi := float64(i)
		b0 += value
		b1 += fi * value
		b2 += fi * (fi - 1) * value
		b3 += fi * (fi - 1) * (fi - 2) * value
	}
	b0 /= fn
	out := LMoments{L1: b0}
	if n < 2 {
		return out
	}
	b1 /= fn * (fn - 1)
	out.L2 = 2*b1 - b0
	if n < 3 {
		return out
	}
	b2 /= fn * (fn - 1) * (fn - 2)
	out.L3 = 6*b2 - 6*b1 + b0
	if out.L2 > 0 {
		out.Tau3 = clampUnitSigned(out.L3 / out.L2)
	}
	if n < 4 {
		return out
	}
	b3 /= fn * (fn - 1) * (fn - 2) * (fn - 3)
	out.L4 = 20*b3 - 30*b2 + 12*b1 - b0
	if out.L2 > 0 {
		out.Tau4 = clampUnitSigned(out.L4 / out.L2)
	}
	return out
}

// Quantile returns the value at probability p in [0, 1] by linear interpolation
// between adjacent order statistics. The input must be sorted ascending and
// non-empty.
func Quantile(sorted []float64, p float64) float64 {
	position := p * float64(len(sorted)-1)
	lower := int(position)
	fraction := position - float64(lower)
	return sorted[lower] + fraction*(sorted[min(lower+1, len(sorted)-1)]-sorted[lower])
}

// ValueStats are shape- and order-independent scalar descriptors over a value
// population. Fractions and the energy descriptors are bounded in [0, 1];
// MeanRMSRatio is in [-1, 1]. NormalizedEnergyEntropy measures how spread the
// squared magnitudes are across nonzero elements (0 = one element carries all
// energy, 1 = uniform), a distribution-free concentration read.
type ValueStats struct {
	FiniteFraction          float64 `json:"finite_fraction"`
	ZeroFraction            float64 `json:"zero_fraction"`
	Mean                    float64 `json:"mean"`
	StdDev                  float64 `json:"standard_deviation"`
	RMS                     float64 `json:"rms"`
	MeanAbsolute            float64 `json:"mean_absolute"`
	MaxAbsolute             float64 `json:"max_absolute"`
	MeanRMSRatio            float64 `json:"mean_rms_ratio"`
	NormalizedL1L2          float64 `json:"normalized_l1_l2"`
	MaxEnergyFraction       float64 `json:"max_energy_fraction"`
	NormalizedEnergyEntropy float64 `json:"normalized_energy_entropy"`
}

// ValueAccumulator computes ValueStats in bounded memory over any number of
// value chunks. Non-finite values are excluded and counted. The energy
// descriptors use a running maximum-magnitude rescaling so a wide dynamic range
// does not lose precision.
type ValueAccumulator struct {
	total, finite, nonzero uint64
	mean, m2, sumAbs       float64
	maxAbs                 float64
	scaledEnergy           float64
	scaledEnergyLog        float64
}

// Add folds one chunk of values into the accumulator.
func (a *ValueAccumulator) Add(values []float64) {
	for _, x := range values {
		a.total++
		if !checked.Finite64(x) {
			continue
		}
		a.finite++
		n := float64(a.finite)
		delta := x - a.mean
		a.mean += delta / n
		a.m2 += delta * (x - a.mean)

		absolute := math.Abs(x)
		a.sumAbs += absolute
		if absolute == 0 {
			continue
		}
		a.nonzero++
		unitEnergy := absolute / absolute
		if a.maxAbs == 0 {
			a.maxAbs, a.scaledEnergy = absolute, unitEnergy
			continue
		}
		if absolute > a.maxAbs {
			ratio := a.maxAbs / absolute
			scale := ratio * ratio
			a.scaledEnergyLog = scale * (a.scaledEnergyLog + a.scaledEnergy*math.Log(scale))
			a.scaledEnergy = scale*a.scaledEnergy + unitEnergy
			a.maxAbs = absolute
			continue
		}
		weight := absolute / a.maxAbs
		weight *= weight
		a.scaledEnergy += weight
		a.scaledEnergyLog += weight * math.Log(weight)
	}
}

// Total and Finite report the counts folded so far.
func (a *ValueAccumulator) Total() uint64  { return a.total }
func (a *ValueAccumulator) Finite() uint64 { return a.finite }

// Stats returns the descriptors and reports whether any finite value was seen;
// when false the returned ValueStats is the zero value.
func (a *ValueAccumulator) Stats() (ValueStats, bool) {
	if a.finite == 0 {
		return ValueStats{}, false
	}
	n := float64(a.finite)
	var zero float64
	variance := max(zero, a.m2/n)
	rms := a.rms()
	var normalizedL1L2, maxEnergyFraction, normalizedEnergyEntropy float64
	if a.maxAbs > 0 {
		normalizedL1L2 = clampUnit(a.sumAbs / (math.Sqrt(n) * a.maxAbs * math.Sqrt(a.scaledEnergy)))
		maxEnergyFraction = clampUnit(1 / a.scaledEnergy)
		if a.nonzero > 1 {
			entropy := math.Log(a.scaledEnergy) - a.scaledEnergyLog/a.scaledEnergy
			normalizedEnergyEntropy = clampUnit(entropy / math.Log(float64(a.nonzero)))
		}
	}
	var meanRMSRatio float64
	if rms > 0 {
		meanRMSRatio = a.mean / rms
	}
	return ValueStats{
		FiniteFraction:          float64(a.finite) / float64(a.total),
		ZeroFraction:            float64(a.finite-a.nonzero) / n,
		Mean:                    a.mean,
		StdDev:                  math.Sqrt(variance),
		RMS:                     rms,
		MeanAbsolute:            a.sumAbs / n,
		MaxAbsolute:             a.maxAbs,
		MeanRMSRatio:            meanRMSRatio,
		NormalizedL1L2:          normalizedL1L2,
		MaxEnergyFraction:       maxEnergyFraction,
		NormalizedEnergyEntropy: normalizedEnergyEntropy,
	}, true
}

func (a *ValueAccumulator) rms() (value float64) {
	if a.finite == 0 || a.maxAbs == 0 {
		return value
	}
	return a.maxAbs * math.Sqrt(a.scaledEnergy/float64(a.finite))
}

// Characterization is the full distribution-free profile of a sampled tensor
// value population: robust order statistics, L-moments, and value descriptors.
// Elements is the tensor's full element count; Samples is how many values this
// profile observed (equal to Elements for an exhaustive scan).
type Characterization struct {
	Elements           uint64     `json:"elements"`
	Samples            uint64     `json:"samples"`
	FiniteSamples      uint64     `json:"finite_samples"`
	LowerQuartile      float64    `json:"lower_quartile"`
	Median             float64    `json:"median"`
	UpperQuartile      float64    `json:"upper_quartile"`
	InterquartileRange float64    `json:"interquartile_range"`
	LMoments           LMoments   `json:"l_moments"`
	Values             ValueStats `json:"values"`
}

// Valid reports the measured population contract.
func (c Characterization) Valid() bool {
	return c.Elements > 0 && c.Samples > 0 && c.Samples <= c.Elements &&
		c.FiniteSamples > 0 && c.FiniteSamples <= c.Samples &&
		checked.Finite64(c.LowerQuartile) && checked.Finite64(c.Median) && checked.Finite64(c.UpperQuartile) &&
		c.InterquartileRange >= 0 && c.LowerQuartile <= c.Median && c.Median <= c.UpperQuartile
}

// Characterize profiles a value population. samples may contain non-finite
// values, which are excluded and reflected in FiniteSamples and the finite
// fraction. It reports false when no finite value is present; four finite
// values are required for the higher L-moments (fewer leaves them zero).
func Characterize(elements uint64, samples []float64) (Characterization, bool) {
	var acc ValueAccumulator
	acc.Add(samples)
	values, ok := acc.Stats()
	if !ok {
		return Characterization{}, false
	}
	finite := make([]float64, 0, acc.finite)
	for _, x := range samples {
		if checked.Finite64(x) {
			finite = append(finite, x)
		}
	}
	slices.Sort(finite)
	lower := Quantile(finite, 0.25)
	upper := Quantile(finite, 0.75)
	return Characterization{
		Elements:           elements,
		Samples:            uint64(len(samples)),
		FiniteSamples:      acc.finite,
		LowerQuartile:      lower,
		Median:             Quantile(finite, medianQuantile),
		UpperQuartile:      upper,
		InterquartileRange: upper - lower,
		LMoments:           exactLMomentsSorted(finite),
		Values:             values,
	}, true
}

func clampUnit(value float64) float64       { return max(0, min(1, value)) }
func clampUnitSigned(value float64) float64 { return max(-1, min(1, value)) }
