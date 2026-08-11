package tabularicl

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

var predictionContract = artifact.JSONContract(artifact.KindOutput, "overgo.tabular-output.v1")

// Prediction: shaped tabular runtime output.
type Prediction struct {
	Values []float32 `json:"predictions"`
	Rows   int       `json:"rows"`
	OutDim int       `json:"output_dim"`
}

type predictor interface {
	Predict(Request) ([]float32, int, error)
}

// RegisterRuntime binds one loaded model to its tabular stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model predictor) error {
	if runtime == nil || model == nil || modelID.Kind() != artifact.KindModel {
		return errors.New("tabularicl: incomplete runtime binding")
	}
	return runtime.Register(modelrecipe.ModuleTabularPredict, workflowruntime.AdapterFunc(
		func(ctx context.Context, request workflowruntime.StepRequest) (map[recipe.PortName]workflowruntime.Value, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			bound, ok := request.Dependency(recipe.DependencyModel, 0)
			if !ok || bound != modelID {
				return nil, errors.New("tabularicl: recipe model differs from runtime binding")
			}
			input, ok := request.Inputs["table"].Single()
			if !ok {
				return nil, errors.New("tabularicl: runtime requires one table")
			}
			table, ok := input.Value.(Request)
			if !ok {
				return nil, errors.New("tabularicl: runtime table has invalid value type")
			}
			values, outDim, err := model.Predict(table)
			if err != nil {
				return nil, err
			}
			if outDim <= 0 || table.Rows <= 0 || len(values) != table.Rows*outDim {
				return nil, errors.New("tabularicl: predictor returned invalid output geometry")
			}
			prediction := Prediction{Values: values, Rows: table.Rows, OutDim: outDim}
			content, err := artifact.JSONContent(predictionContract, prediction)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]workflowruntime.Value{
				"predictions": workflowruntime.ArtifactValue(recipe.DataTensor, prediction, content),
			}, nil
		},
	))
}
