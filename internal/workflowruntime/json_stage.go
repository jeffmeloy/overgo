package workflowruntime

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// RegisterJSONStage binds typed computation to a catalog-owned scalar module.
func RegisterJSONStage[Input, Output any](
	runtime *Runtime,
	moduleID recipe.ModuleID,
	modelID artifact.ID,
	contract artifact.DocumentContract,
	execute func(Input) (Output, error),
) error {
	return RegisterScalarStage(runtime, moduleID, modelID, execute, func(value Output) (artifact.Content, error) {
		return artifact.JSONContent(contract, value)
	})
}

// RegisterScalarStage binds typed computation with optional artifact output.
func RegisterScalarStage[Input, Output any](
	runtime *Runtime,
	moduleID recipe.ModuleID,
	modelID artifact.ID,
	execute func(Input) (Output, error),
	encode func(Output) (artifact.Content, error),
) error {
	if runtime == nil || runtime.catalog == nil || execute == nil || modelID.Kind() != artifact.KindModel {
		return fmt.Errorf("workflow runtime: incomplete scalar stage")
	}
	module, ok := runtime.catalog.Module(moduleID)
	if !ok {
		return fmt.Errorf("workflow runtime: unknown module %q", moduleID)
	}
	if len(module.Inputs) != 1 || len(module.Outputs) != 1 ||
		module.Inputs[0].Cardinality != recipe.CardinalityOne ||
		module.Outputs[0].Cardinality != recipe.CardinalityOne {
		return fmt.Errorf("workflow runtime: module %q is not scalar", moduleID)
	}
	inputPort, outputPort := module.Inputs[0], module.Outputs[0]
	return runtime.Register(moduleID, AdapterFunc(
		func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			if request.Model != modelID {
				return nil, fmt.Errorf("workflow runtime: recipe model differs from JSON stage")
			}
			datum, ok := request.Inputs[inputPort.Name].Single()
			if !ok {
				return nil, fmt.Errorf("workflow runtime: input %q is not scalar", inputPort.Name)
			}
			input, ok := datum.Value.(Input)
			if !ok {
				return nil, fmt.Errorf("workflow runtime: input %q has invalid value type", inputPort.Name)
			}
			value, err := execute(input)
			if err != nil {
				return nil, err
			}
			output := Value{Kind: outputPort.Data, Items: []Datum{{Value: value}}}
			if encode != nil {
				content, err := encode(value)
				if err != nil {
					return nil, err
				}
				output = ArtifactValue(outputPort.Data, value, content)
			}
			return map[recipe.PortName]Value{outputPort.Name: output}, nil
		},
	))
}
