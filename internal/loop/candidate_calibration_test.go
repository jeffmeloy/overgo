package loop

import (
	"testing"
)

// TestCompositionCandidateCalibration pins the calibration contract: the
// ranker is trustworthy only when the predicted fitness ordering matches
// the measured outcomes with a violation fraction under the bound derived
// from the comparison count — the same distribution-free concordance
// audit the alignment residual uses, with no second calibration owner. A
// ranker ordered against its outcomes is refused, too few samples refuse
// trust, and mismatched inputs refuse outright.
func TestCompositionCandidateCalibration(t *testing.T) {
	predicted := make([]float64, 30)
	measured := make([]float64, 30)
	for index := range predicted {
		predicted[index] = float64(index) / 30
		measured[index] = 0.1 + predicted[index]/2
	}
	calibration, err := CalibrateCandidatePredictions(predicted, measured)
	if err != nil {
		t.Fatal(err)
	}
	if !calibration.Trustworthy || calibration.Audit.Violations != 0 || calibration.Audit.Comparisons == 0 {
		t.Fatalf("well-ordered ranker refused: %+v", calibration)
	}

	inverted := make([]float64, len(measured))
	for index := range measured {
		inverted[index] = 1 - measured[index]
	}
	miscalibrated, err := CalibrateCandidatePredictions(predicted, inverted)
	if err != nil {
		t.Fatal(err)
	}
	if miscalibrated.Trustworthy || miscalibrated.Audit.Violations != miscalibrated.Audit.Comparisons {
		t.Fatalf("anti-ordered ranker trusted: %+v", miscalibrated)
	}

	tiny, err := CalibrateCandidatePredictions([]float64{0.2, 0.5, 0.9}, []float64{0.1, 0.3, 0.8})
	if err != nil {
		t.Fatal(err)
	}
	if tiny.Trustworthy || tiny.Audit.Bound > 0 {
		t.Fatalf("three samples earned trust: %+v", tiny)
	}

	if _, err := CalibrateCandidatePredictions(predicted, measured[:10]); err == nil {
		t.Fatal("mismatched inputs calibrated")
	}
	if _, err := CalibrateCandidatePredictions([]float64{0.5}, []float64{0.4}); err == nil {
		t.Fatal("single sample calibrated")
	}
}
