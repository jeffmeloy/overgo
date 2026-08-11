package seriesforecast

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

var forecastContract = artifact.JSONContract(artifact.KindOutput, "overgo.forecast-output.v1")

type forecaster interface {
	Forecast([]float32) ([]float32, error)
}

// RegisterRuntime binds one loaded model to its forecast stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model forecaster) error {
	if runtime == nil || model == nil || modelID.Kind() != artifact.KindModel {
		return errors.New("seriesforecast: incomplete runtime binding")
	}
	return runtime.Register(modelrecipe.ModuleForecastSeries, workflowruntime.AdapterFunc(
		func(ctx context.Context, request workflowruntime.StepRequest) (map[recipe.PortName]workflowruntime.Value, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			bound, ok := request.Dependency(recipe.DependencyModel, 0)
			if !ok || bound != modelID {
				return nil, errors.New("seriesforecast: recipe model differs from runtime binding")
			}
			input, ok := request.Inputs["series"].Single()
			if !ok {
				return nil, errors.New("seriesforecast: runtime requires one series")
			}
			series, ok := input.Value.([]float32)
			if !ok {
				return nil, errors.New("seriesforecast: runtime series has invalid value type")
			}
			forecast, err := model.Forecast(series)
			if err != nil {
				return nil, err
			}
			content, err := artifact.JSONContent(forecastContract, forecast)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]workflowruntime.Value{
				"forecast": workflowruntime.ArtifactValue(recipe.DataTensor, forecast, content),
			}, nil
		},
	))
}
