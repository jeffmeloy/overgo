package seriesforecast

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

var forecastContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: "application/json", Schema: "overgo.forecast-output.v1",
}

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
			bound, ok := requestModel(request.Dependencies)
			if !ok || bound != modelID {
				return nil, errors.New("seriesforecast: recipe model differs from runtime binding")
			}
			input := request.Inputs["series"]
			if len(input.Items) != 1 {
				return nil, errors.New("seriesforecast: runtime requires one series")
			}
			series, ok := input.Items[0].Value.([]float32)
			if !ok {
				return nil, errors.New("seriesforecast: runtime series has invalid value type")
			}
			forecast, err := model.Forecast(series)
			if err != nil {
				return nil, err
			}
			content, err := forecastContent(forecast)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]workflowruntime.Value{
				"forecast": {
					Kind: recipe.DataTensor,
					Items: []workflowruntime.Datum{{
						Artifact: content.Descriptor, Content: &content, Value: forecast,
					}},
				},
			}, nil
		},
	))
}

func requestModel(dependencies []recipe.Dependency) (artifact.ID, bool) {
	for _, dependency := range dependencies {
		if dependency.Role == recipe.DependencyModel && dependency.Slot == 0 {
			return dependency.Artifact, true
		}
	}
	return artifact.ID{}, false
}

func forecastContent(forecast []float32) (artifact.Content, error) {
	payload, err := json.Marshal(forecast)
	if err != nil {
		return artifact.Content{}, fmt.Errorf("seriesforecast: encode forecast: %w", err)
	}
	id, err := forecastContract.Identify(payload)
	if err != nil {
		return artifact.Content{}, err
	}
	return forecastContract.Content(id, payload)
}
