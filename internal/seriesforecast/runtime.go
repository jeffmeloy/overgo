package seriesforecast

import (
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var forecastContract = artifact.JSONContract(artifact.KindOutput, "overgo.forecast-output.v1")

func ValidateRequest(series []float32) error {
	if len(series) == 0 {
		return errors.New("seriesforecast: input series is empty")
	}
	for _, value := range series {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.New("seriesforecast: input series contains a non-finite value")
		}
	}
	return nil
}

// RegisterRuntime binds one loaded model to its forecast stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("seriesforecast: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONStage[[]float32, []float32](
		runtime, modelrecipe.ModuleForecastSeries, modelID, forecastContract,
		func(series []float32) ([]float32, error) {
			if err := ValidateRequest(series); err != nil {
				return nil, err
			}
			return model.Forecast(series)
		},
	)
}
