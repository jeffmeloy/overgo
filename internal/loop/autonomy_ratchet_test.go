package loop

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func ratchetInstruments(t *testing.T) (DriverLearningCurve, CandidateCalibration) {
	t.Helper()
	curve, err := DeriveDriverLearningCurve([]DriverAttemptMeasurement{
		{ComputeNS: 100, AdapterTrainNS: 40},
		{Fit: true, Promoted: true, FitnessDelta: 0.04, ComputeNS: 200, AdapterTrainNS: 120},
		{Fit: true, Promoted: true, FitnessDelta: 0.06, ComputeNS: 180, AdapterTrainNS: 100},
		{Fit: true, Promoted: true, FitnessDelta: 0.05, ComputeNS: 150, AdapterTrainNS: 90},
	})
	if err != nil {
		t.Fatal(err)
	}
	predicted := make([]float64, 30)
	measured := make([]float64, 30)
	for index := range predicted {
		predicted[index] = float64(index) / 30
		measured[index] = predicted[index] / 2
	}
	calibration, err := CalibrateCandidatePredictions(predicted, measured)
	if err != nil {
		t.Fatal(err)
	}
	return curve, calibration
}

// TestCompositionAutonomyRatchetGatesOnCurveAndSafety pins the ratchet:
// the budget widens by exactly one attempt only when the learning curve
// shows positive fitness per compute with a non-falling hit rate, the
// ranker calibration holds, and every provided breaker state is at or
// below advisory; any failing condition rolls the budget back to the
// conservative floor, every condition's outcome is recorded either way,
// and an unmeasured instrument refuses instead of deciding.
func TestCompositionAutonomyRatchetGatesOnCurveAndSafety(t *testing.T) {
	curve, calibration := ratchetInstruments(t)
	window := testutil.ArtifactID(t, artifact.KindEvidence, "ratchet-window")
	quiet := []CircuitBreakerTransition{
		{Metric: "exact-match", Level: BreakerClear, Window: window},
		{Metric: "latency", Level: BreakerAdvisory, Window: window},
	}
	decision, err := RatchetDriverAutonomy(3, curve, calibration, quiet)
	if err != nil {
		t.Fatal(err)
	}
	if !decision.Widened || decision.RolledBack || decision.NextBudget != 4 {
		t.Fatalf("positive review = %+v", decision)
	}
	recorded := strings.Join(decision.Reasons, "; ")
	for _, condition := range []string{"learning curve positive", "calibration holds", "live safety quiet"} {
		if !strings.Contains(recorded, condition) {
			t.Fatalf("decision lost condition %q: %v", condition, decision.Reasons)
		}
	}

	loud := append([]CircuitBreakerTransition(nil), quiet...)
	loud[1].Level = BreakerQuarantine
	contained, err := RatchetDriverAutonomy(3, curve, calibration, loud)
	if err != nil {
		t.Fatal(err)
	}
	if contained.Widened || !contained.RolledBack || contained.NextBudget != AutonomyBudgetFloor ||
		!strings.Contains(strings.Join(contained.Reasons, "; "), "not quiet") {
		t.Fatalf("contained review = %+v", contained)
	}

	uncalibrated := calibration
	uncalibrated.Trustworthy = false
	untrusted, err := RatchetDriverAutonomy(3, curve, uncalibrated, quiet)
	if err != nil {
		t.Fatal(err)
	}
	if untrusted.Widened || untrusted.NextBudget != AutonomyBudgetFloor ||
		!strings.Contains(strings.Join(untrusted.Reasons, "; "), "calibration does not hold") {
		t.Fatalf("uncalibrated review = %+v", untrusted)
	}

	falling := curve
	falling.EarlyHitRate, falling.LateHitRate = 1, 0.5
	regressed, err := RatchetDriverAutonomy(3, falling, calibration, quiet)
	if err != nil {
		t.Fatal(err)
	}
	if regressed.Widened || regressed.NextBudget != AutonomyBudgetFloor ||
		!strings.Contains(strings.Join(regressed.Reasons, "; "), "hit rate fell") {
		t.Fatalf("falling review = %+v", regressed)
	}

	flatline := curve
	flatline.Promotions, flatline.FitnessPerCompute = 0, 0
	stalled, err := RatchetDriverAutonomy(1, flatline, calibration, quiet)
	if err != nil {
		t.Fatal(err)
	}
	if stalled.Widened || stalled.RolledBack || stalled.NextBudget != AutonomyBudgetFloor {
		t.Fatalf("floor review rolled back below the floor: %+v", stalled)
	}

	if _, err := RatchetDriverAutonomy(0, curve, calibration, quiet); err == nil {
		t.Fatal("budgetless ratchet decided")
	}
	if _, err := RatchetDriverAutonomy(1, DriverLearningCurve{}, calibration, quiet); err == nil {
		t.Fatal("curveless ratchet decided")
	}
}
