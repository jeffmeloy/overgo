package recipe

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

// SteeringPrediction is the falsifiable benefit, cost, and uncertainty claim
// behind a candidate, measured in a named metric and unit.
type SteeringPrediction struct {
	Metric      string  `json:"metric"`
	Benefit     float64 `json:"benefit"`
	Cost        uint64  `json:"cost"`
	Unit        string  `json:"unit"`
	Uncertainty float64 `json:"uncertainty"`
}

// Validate checks the shared quantified benefit, cost, and uncertainty claim.
func (prediction SteeringPrediction) Validate() error {
	if !boundedStatement(prediction.Metric) || !boundedStatement(prediction.Unit) ||
		!checked.Finite64(prediction.Benefit) || prediction.Benefit <= 0 || prediction.Cost == 0 ||
		!checked.Finite64(prediction.Uncertainty) || prediction.Uncertainty < 0 ||
		prediction.Uncertainty > float64(artifact.InitialDocumentVersion) {
		return errors.New("recipe: invalid steering prediction")
	}
	return nil
}
