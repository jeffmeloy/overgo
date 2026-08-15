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

// RegisterJSONPipeline binds a typed prepare-integrate-decode pipeline.
func RegisterJSONPipeline[Input, Plan, Features, Output any](
	runtime *Runtime,
	modelID artifact.ID,
	contract artifact.DocumentContract,
	prepareID recipe.ModuleID,
	prepare func(Input) (Plan, error),
	integrateID recipe.ModuleID,
	integrate func(Plan) (Features, error),
	decodeID recipe.ModuleID,
	decode func(Features) (Output, error),
) error {
	if err := RegisterScalarStage(runtime, prepareID, modelID, prepare, nil); err != nil {
		return err
	}
	if err := RegisterScalarStage(runtime, integrateID, modelID, integrate, nil); err != nil {
		return err
	}
	return RegisterJSONStage(runtime, decodeID, modelID, contract, decode)
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
	if execute == nil {
		return fmt.Errorf("workflow runtime: incomplete scalar stage")
	}
	return RegisterContextStage(runtime, moduleID, modelID,
		func(_ context.Context, input Input) (Output, error) { return execute(input) }, encode,
	)
}

// RegisterContextStage binds context-aware scalar computation.
func RegisterContextStage[Input, Output any](
	runtime *Runtime,
	moduleID recipe.ModuleID,
	modelID artifact.ID,
	execute func(context.Context, Input) (Output, error),
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
	inputPort := module.Inputs[0]
	return RegisterResolvedStage(runtime, moduleID, modelID,
		func(ctx context.Context, request StepRequest) (Output, error) {
			input, err := ScalarInput[Input](request, inputPort.Name)
			if err != nil {
				var zero Output
				return zero, err
			}
			return execute(ctx, input)
		}, encode,
	)
}

// RegisterResolvedStage binds request resolution and one scalar output.
func RegisterResolvedStage[Output any](
	runtime *Runtime,
	moduleID recipe.ModuleID,
	modelID artifact.ID,
	execute func(context.Context, StepRequest) (Output, error),
	encode func(Output) (artifact.Content, error),
) error {
	if runtime == nil || runtime.catalog == nil || execute == nil || modelID.Kind() != artifact.KindModel {
		return fmt.Errorf("workflow runtime: incomplete resolved stage")
	}
	module, ok := runtime.catalog.Module(moduleID)
	if !ok {
		return fmt.Errorf("workflow runtime: unknown module %q", moduleID)
	}
	if len(module.Outputs) != 1 || module.Outputs[0].Cardinality != recipe.CardinalityOne {
		return fmt.Errorf("workflow runtime: module %q has no scalar output", moduleID)
	}
	outputPort := module.Outputs[0]
	return runtime.Register(moduleID, AdapterFunc(
		func(ctx context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			if request.Model != modelID {
				return nil, fmt.Errorf("workflow runtime: recipe model differs from resolved stage")
			}
			value, err := execute(ctx, request)
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
