package loop

import (
	"math"
	"strings"
	"testing"
)

// TestCompositionDriverLearningCurve pins the instrument contract: the
// curve aggregates recorded attempts in order into candidate hit-rate,
// fitness delta per unit compute, and adapter-training share, with
// cumulative points and the early-versus-late hit-rate split the autonomy
// ratchet judges trend on; every attempt must carry a coherent measured
// cost, a promotion must be fit with a positive measured delta, a refusal
// carries none, and inconsistent records refuse rather than blending in.
func TestCompositionDriverLearningCurve(t *testing.T) {
	measurements := []DriverAttemptMeasurement{
		{Fit: false, Promoted: false, ComputeNS: 100, AdapterTrainNS: 40},
		{Fit: false, Promoted: false, ComputeNS: 120, AdapterTrainNS: 50},
		{Fit: true, Promoted: true, FitnessDelta: 0.04, ComputeNS: 200, AdapterTrainNS: 120},
		{Fit: true, Promoted: true, FitnessDelta: 0.06, ComputeNS: 180, AdapterTrainNS: 100},
	}
	curve, err := DeriveDriverLearningCurve(measurements)
	if err != nil {
		t.Fatal(err)
	}
	if curve.Attempts != 4 || curve.Promotions != 2 || curve.HitRate != 0.5 {
		t.Fatalf("curve totals = %+v", curve)
	}
	totalCompute := uint64(100 + 120 + 200 + 180)
	if curve.FitnessPerCompute != (0.04+0.06)/float64(totalCompute) {
		t.Fatalf("fitness per compute = %v", curve.FitnessPerCompute)
	}
	if curve.AdapterTrainShare != float64(40+50+120+100)/float64(totalCompute) {
		t.Fatalf("adapter-train share = %v", curve.AdapterTrainShare)
	}
	if curve.EarlyHitRate != 0 || curve.LateHitRate != 1 {
		t.Fatalf("trend split = early %v late %v", curve.EarlyHitRate, curve.LateHitRate)
	}
	if len(curve.Points) != 4 {
		t.Fatalf("curve points = %d", len(curve.Points))
	}
	last := curve.Points[3]
	if last.Attempt != 4 || last.Promotions != 2 || last.CumulativeComputeNS != totalCompute ||
		last.CumulativeFitness != 0.04+0.06 {
		t.Fatalf("final point = %+v", last)
	}
	for index := 1; index < len(curve.Points); index++ {
		if curve.Points[index].CumulativeComputeNS <= curve.Points[index-1].CumulativeComputeNS {
			t.Fatalf("cumulative compute is not monotone at %d", index)
		}
	}

	refusals := []struct {
		name   string
		mutate func(*DriverAttemptMeasurement)
		want   string
	}{
		{"unmeasured compute", func(m *DriverAttemptMeasurement) { m.ComputeNS = 0 }, "measured compute cost"},
		{"incoherent adapter share", func(m *DriverAttemptMeasurement) {
			m.AdapterTrainNS = m.ComputeNS + 1
		}, "measured compute cost"},
		{"non-finite delta", func(m *DriverAttemptMeasurement) {
			m.Promoted, m.Fit = true, true
			m.FitnessDelta = math.NaN()
		}, "not finite"},
		{"unfit promotion", func(m *DriverAttemptMeasurement) {
			m.Promoted, m.Fit = true, false
			m.FitnessDelta = 0.02
		}, "without a fit verdict"},
		{"deltaless promotion", func(m *DriverAttemptMeasurement) {
			m.Promoted, m.Fit = true, true
			m.FitnessDelta = 0
		}, "without a fit verdict and a positive measured delta"},
		{"phantom delta", func(m *DriverAttemptMeasurement) {
			m.Promoted = false
			m.FitnessDelta = 0.02
		}, "without a promotion"},
	}
	for _, refusal := range refusals {
		mutated := append([]DriverAttemptMeasurement(nil), measurements...)
		refusal.mutate(&mutated[1])
		if _, err := DeriveDriverLearningCurve(mutated); err == nil ||
			!strings.Contains(err.Error(), refusal.want) {
			t.Fatalf("%s aggregated: %v", refusal.name, err)
		}
	}
	if _, err := DeriveDriverLearningCurve(nil); err == nil {
		t.Fatal("empty record aggregated")
	}
}
