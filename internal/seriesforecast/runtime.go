package seriesforecast

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var forecastContract = artifact.JSONContract(artifact.KindOutput, "overgo.forecast-output.v1")

type forecaster interface {
	Forecast([]float32) ([]float32, error)
}

// RegisterRuntime binds one loaded model to its forecast stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("seriesforecast: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model forecaster) error {
	return workflowruntime.RegisterJSONStage[[]float32, []float32](
		runtime, modelrecipe.ModuleForecastSeries, modelID, forecastContract, model.Forecast,
	)
}
