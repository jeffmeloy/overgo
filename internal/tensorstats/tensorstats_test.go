package tensorstats

import (
	"math"
	"testing"
)

const tol = 1e-9

func approx(t *testing.T, label string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-6 {
		t.Errorf("%s = %.9g, want %.9g", label, got, want)
	}
}

func TestExactLMomentsHandComputed(t *testing.T) {
	// [1,2,3,4] is symmetric and evenly spaced: L1=mean, L2=5/6, all higher
	// L-moments and ratios are zero.
	m := ExactLMoments([]float64{4, 1, 3, 2}) // unsorted input must not matter
	approx(t, "L1", m.L1, 2.5)
	approx(t, "L2", m.L2, 5.0/6.0)
	approx(t, "L3", m.L3, 0)
	approx(t, "L4", m.L4, 0)
	approx(t, "Tau3", m.Tau3, 0)
	approx(t, "Tau4", m.Tau4, 0)
}

func TestLMomentsScaleEquivariance(t *testing.T) {
	base := []float64{2, 3, 5, 7, 11, 13}
	m := ExactLMoments(base)
	const c = 4.0
	scaled := make([]float64, len(base))
	for i, v := range base {
		scaled[i] = c*v + 100 // affine: L1 shifts+scales, L2 scales, ratios invariant
	}
	s := ExactLMoments(scaled)
	approx(t, "L1 affine", s.L1, c*m.L1+100)
	approx(t, "L2 scale", s.L2, c*m.L2)
	approx(t, "Tau3 invariant", s.Tau3, m.Tau3)
	approx(t, "Tau4 invariant", s.Tau4, m.Tau4)
}

func TestLMomentSkewnessSign(t *testing.T) {
	right := ExactLMoments([]float64{1, 1, 1, 10}) // long right tail
	if right.Tau3 <= 0 {
		t.Errorf("right-skewed Tau3 = %.4f, want > 0", right.Tau3)
	}
	left := ExactLMoments([]float64{1, 10, 10, 10}) // long left tail
	if left.Tau3 >= 0 {
		t.Errorf("left-skewed Tau3 = %.4f, want < 0", left.Tau3)
	}
}

func TestLMomentsBelowFourElements(t *testing.T) {
	// Fewer than four values: only defined lower moments are filled.
	m := ExactLMoments([]float64{2, 4})
	approx(t, "L1", m.L1, 3)
	approx(t, "L2", m.L2, 1) // 2*b1-b0, b1=(0*2+1*4)/(2*1)=2 -> 2*2-3=1
	if m.L3 != 0 || m.L4 != 0 || m.Tau3 != 0 || m.Tau4 != 0 {
		t.Errorf("higher moments must stay zero below n=3: %+v", m)
	}
}

func TestValueStatsHandComputed(t *testing.T) {
	var a ValueAccumulator
	a.Add([]float64{1, 2, 3, 4})
	s, ok := a.Stats()
	if !ok {
		t.Fatal("expected finite stats")
	}
	approx(t, "FiniteFraction", s.FiniteFraction, 1)
	approx(t, "ZeroFraction", s.ZeroFraction, 0)
	approx(t, "Mean", s.Mean, 2.5)
	approx(t, "StdDev", s.StdDev, math.Sqrt(1.25))
	approx(t, "RMS", s.RMS, math.Sqrt(7.5))
	approx(t, "MeanAbsolute", s.MeanAbsolute, 2.5)
	approx(t, "MaxAbsolute", s.MaxAbsolute, 4)
	approx(t, "MaxEnergyFraction", s.MaxEnergyFraction, 1.0/1.875)
	approx(t, "MeanRMSRatio", s.MeanRMSRatio, 2.5/math.Sqrt(7.5))
}

func TestEnergyEntropyUniformIsOne(t *testing.T) {
	// Equal magnitudes spread energy uniformly -> normalized entropy 1.
	var a ValueAccumulator
	a.Add([]float64{1, 1, 1, 1})
	s, _ := a.Stats()
	approx(t, "uniform entropy", s.NormalizedEnergyEntropy, 1)
	approx(t, "uniform maxEnergyFraction", s.MaxEnergyFraction, 0.25)
}

func TestEnergyEntropyConcentratedIsLow(t *testing.T) {
	var dominant, spread ValueAccumulator
	dominant.Add([]float64{100, 1, 1, 1})
	spread.Add([]float64{3, 2, 2, 3})
	d, _ := dominant.Stats()
	p, _ := spread.Stats()
	if d.NormalizedEnergyEntropy >= p.NormalizedEnergyEntropy {
		t.Errorf("dominant entropy %.4f should be below spread entropy %.4f",
			d.NormalizedEnergyEntropy, p.NormalizedEnergyEntropy)
	}
	if d.MaxEnergyFraction <= p.MaxEnergyFraction {
		t.Errorf("dominant maxEnergyFraction %.4f should exceed spread %.4f",
			d.MaxEnergyFraction, p.MaxEnergyFraction)
	}
	for _, s := range []ValueStats{d, p} {
		if s.NormalizedEnergyEntropy < 0 || s.NormalizedEnergyEntropy > 1 {
			t.Errorf("entropy out of [0,1]: %.4f", s.NormalizedEnergyEntropy)
		}
	}
}

func TestNonFiniteExcludedAndCounted(t *testing.T) {
	var a ValueAccumulator
	a.Add([]float64{1, math.NaN(), 2, math.Inf(1), 3, math.Inf(-1), 4})
	s, ok := a.Stats()
	if !ok {
		t.Fatal("expected finite stats")
	}
	if a.Total() != 7 || a.Finite() != 4 {
		t.Fatalf("counts total=%d finite=%d, want 7/4", a.Total(), a.Finite())
	}
	approx(t, "FiniteFraction", s.FiniteFraction, 4.0/7.0)
	approx(t, "Mean over finite", s.Mean, 2.5)
}

func TestAllNonFiniteReportsNoStats(t *testing.T) {
	var a ValueAccumulator
	a.Add([]float64{math.NaN(), math.Inf(1)})
	if _, ok := a.Stats(); ok {
		t.Error("all-non-finite must report no finite stats")
	}
	if _, ok := Characterize(2, []float64{math.NaN(), math.Inf(1)}); ok {
		t.Error("Characterize must report false when no finite value exists")
	}
}

func TestZeroFraction(t *testing.T) {
	var a ValueAccumulator
	a.Add([]float64{0, 0, 3, 0, 6})
	s, _ := a.Stats()
	approx(t, "ZeroFraction", s.ZeroFraction, 3.0/5.0)
}

func TestStreamingInvariance(t *testing.T) {
	values := []float64{5, math.NaN(), -3, 0, 12, 0.5, -7, 100, 2}
	var whole ValueAccumulator
	whole.Add(values)
	w, _ := whole.Stats()

	var chunked ValueAccumulator
	chunked.Add(values[:3])
	chunked.Add(values[3:5])
	chunked.Add(values[5:])
	c, _ := chunked.Stats()

	if w != c {
		t.Errorf("chunked stats diverged from whole:\n whole=%+v\n chunk=%+v", w, c)
	}
}

func TestCharacterizeCombined(t *testing.T) {
	samples := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	c, ok := Characterize(1000, samples)
	if !ok {
		t.Fatal("expected characterization")
	}
	if c.Elements != 1000 || c.Samples != 8 || c.FiniteSamples != 8 {
		t.Fatalf("counts = elements %d samples %d finite %d", c.Elements, c.Samples, c.FiniteSamples)
	}
	approx(t, "Median", c.Median, 4.5)
	approx(t, "LowerQuartile", c.LowerQuartile, Quantile([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 0.25))
	if c.InterquartileRange <= 0 {
		t.Errorf("IQR must be positive for a spread sample, got %.4f", c.InterquartileRange)
	}
	approx(t, "LMoment L1", c.LMoments.L1, 4.5)
}
