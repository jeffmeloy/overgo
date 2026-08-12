package workflowruntime

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// ExecuteScalar runs a one-input, one-output program.
func ExecuteScalar[Output any](
	ctx context.Context,
	runtime *Runtime,
	key string,
	program recipe.Program,
	inputValue any,
	inputContent artifact.Content,
) (Output, error) {
	var zero Output
	definition := program.Definition()
	if len(definition.Inputs) != 1 || len(definition.Outputs) != 1 {
		return zero, errors.New("workflow runtime: scalar program requires one input and one output")
	}
	input := definition.Inputs[0]
	result, err := runtime.ExecuteProgram(ctx, key, program, map[recipe.PortName]Value{
		input.Name: ArtifactValue(input.Data, inputValue, inputContent),
	})
	if err != nil {
		return zero, err
	}
	output := definition.Outputs[0]
	datum, ok := result.Outputs[output.Name].Single()
	if !ok {
		return zero, fmt.Errorf("workflow runtime: output %q is not scalar", output.Name)
	}
	value, ok := datum.Value.(Output)
	if !ok {
		return zero, fmt.Errorf("workflow runtime: output %q has invalid value type", output.Name)
	}
	return value, nil
}

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

// ScalarInput: typed single datum from a step port.
func ScalarInput[Input any](request StepRequest, name recipe.PortName) (Input, error) {
	var zero Input
	datum, ok := request.Inputs[name].Single()
	if !ok {
		return zero, fmt.Errorf("workflow runtime: input %q is not scalar", name)
	}
	input, ok := datum.Value.(Input)
	if !ok {
		return zero, fmt.Errorf("workflow runtime: input %q has invalid value type", name)
	}
	return input, nil
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
			input, err := ScalarInput[Input](request, inputPort.Name)
			if err != nil {
				return nil, err
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
