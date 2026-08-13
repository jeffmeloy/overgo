package capabilityruntime

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

type Executor func(context.Context, artifact.Repository, string, artifact.ID, recipe.Program, string) (any, error)

func JSONScalar[Input, Model, Output any](
	name string,
	validate func(Input) error,
	load func(context.Context, string, Input) (Model, error),
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
) Executor {
	return func(
		ctx context.Context,
		store artifact.Repository,
		path string,
		modelID artifact.ID,
		program recipe.Program,
		raw string,
	) (any, error) {
		input, content, err := decodeInput(name, validate, raw)
		if err != nil {
			return nil, err
		}
		model, err := load(ctx, path, input)
		if err != nil {
			return nil, err
		}
		output, executeErr := executeScalar[Input, Model, Output](
			ctx, store, modelID, program, content, input, model, bind,
		)
		executeErr = errors.Join(executeErr, closeModel(context.WithoutCancel(ctx), model))
		return output, executeErr
	}
}

func IgnoreInput[Input, Model any](load func(string) (Model, error)) func(context.Context, string, Input) (Model, error) {
	return func(_ context.Context, path string, _ Input) (Model, error) { return load(path) }
}

// Execute binds adapters and runs a program with one typed output.
func Execute[Output any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	inputs map[recipe.PortName]workflowruntime.Value,
	bind func(*workflowruntime.Runtime) error,
) (Output, error) {
	var zero Output
	definition := program.Definition()
	if definition.Model != modelID {
		return zero, fmt.Errorf("capability runtime: program model differs from binding")
	}
	if len(definition.Outputs) != 1 {
		return zero, fmt.Errorf("capability runtime: program requires one output")
	}
	runtime, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		return zero, err
	}
	if err := bind(runtime); err != nil {
		return zero, err
	}
	result, err := runtime.ExecuteProgram(ctx, key, program, inputs)
	if err != nil {
		return zero, err
	}
	output := definition.Outputs[0]
	datum, ok := result.Outputs[output.Name].Single()
	if !ok {
		return zero, fmt.Errorf("capability runtime: output %q is not scalar", output.Name)
	}
	value, ok := datum.Value.(Output)
	if !ok {
		return zero, fmt.Errorf("capability runtime: output %q has invalid value type", output.Name)
	}
	return value, nil
}
