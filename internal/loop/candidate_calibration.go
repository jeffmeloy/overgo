package loop

import (
	"errors"

	"overgo/internal/composition"
)

// CandidateCalibration is the measured verdict on the enumeration ranker:
// over candidates with both a predicted fitness and a measured outcome,
// how often the predicted ordering was wrong, and whether that violation
// fraction stays under the bound derived from the comparison count. Only
// a trustworthy ranker justifies widening the driver budget.
type CandidateCalibration struct {
	Audit       composition.AlignmentBiasAudit `json:"audit"`
	Trustworthy bool                           `json:"trustworthy"`
}

// CalibrateCandidatePredictions scores predicted-versus-actual fitness
// per candidate with the same distribution-free concordance audit the
// alignment residual uses — there is no second calibration owner. A
// higher predicted fitness must come with a measured outcome at least as
// high; each strict pair ordered the other way is a violation, the
// admissible violation fraction derives from the comparison count alone,
// and too few comparisons refuse trust rather than guessing. Predictions
// are negated before delegation because the shared audit is stated for a
// residual, where lower is better.
func CalibrateCandidatePredictions(predicted, measured []float64) (CandidateCalibration, error) {
	if len(predicted) != len(measured) {
		return CandidateCalibration{}, errors.New("loop: calibration requires paired predicted and measured fitness")
	}
	negated := make([]float64, len(predicted))
	for index, value := range predicted {
		negated[index] = -value
	}
	audit, err := composition.AlignmentResidualBiasAudit(negated, measured)
	if err != nil {
		return CandidateCalibration{}, err
	}
	return CandidateCalibration{Audit: audit, Trustworthy: audit.Admitted}, nil
}
