package tabularicl

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var predictionContract = artifact.JSONContract(artifact.KindOutput, "overgo.tabular-output.v1")

// Prediction: shaped tabular runtime output.
type Prediction struct {
	Values []float32 `json:"predictions"`
	Rows   int       `json:"rows"`
	OutDim int       `json:"output_dim"`
}

// RegisterRuntime binds one loaded model to its tabular stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("tabularicl: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONStage[Request, Prediction](
		runtime, modelrecipe.ModuleTabularPredict, modelID, predictionContract,
		func(table Request) (Prediction, error) {
			values, outDim, err := model.Predict(table)
			if err != nil {
				return Prediction{}, err
			}
			if outDim <= 0 || table.Rows <= 0 || len(values) != table.Rows*outDim {
				return Prediction{}, errors.New("tabularicl: predictor returned invalid output geometry")
			}
			return Prediction{Values: values, Rows: table.Rows, OutDim: outDim}, nil
		},
	)
}
