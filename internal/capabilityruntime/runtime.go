package capabilityruntime

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

type Executor func(
	context.Context,
	artifact.Repository,
	string,
	artifact.ID,
	recipe.Program,
	string,
) (any, error)

func JSONScalar[Input, Model, Output any](
	name string,
	validate func(Input) error,
	load func(string, Input) (Model, error),
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
		var input Input
		if err := strictjson.DecodeBytes([]byte(raw), &input); err != nil {
			return nil, fmt.Errorf("decode %s input: %w", name, err)
		}
		if err := validate(input); err != nil {
			return nil, err
		}
		content, err := artifact.JSONContent(
			artifact.JSONContract(artifact.KindFile, "overgo."+name+"-input.v1"), input,
		)
		if err != nil {
			return nil, err
		}
		model, err := load(path, input)
		if err != nil {
			return nil, err
		}
		return executeScalar[Output](ctx, store, modelID, program, input, content, func(runtime *workflowruntime.Runtime) error {
			return bind(runtime, modelID, model)
		})
	}
}

func IgnoreInput[Input, Model any](load func(string) (Model, error)) func(string, Input) (Model, error) {
	return func(path string, _ Input) (Model, error) { return load(path) }
}

func executeScalar[Output any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	inputValue any,
	inputContent artifact.Content,
	bind func(*workflowruntime.Runtime) error,
) (Output, error) {
	var zero Output
	definition := program.Definition()
	if definition.Model != modelID {
		return zero, errors.New("capability runtime: program model differs from binding")
	}
	if len(definition.Inputs) != 1 || len(definition.Outputs) != 1 {
		return zero, errors.New("capability runtime: scalar execution requires one input and output")
	}
	input, output := definition.Inputs[0], definition.Outputs[0]
	runtime, err := workflowruntime.NewWithCatalog(store, modelrecipe.Catalog())
	if err != nil {
		return zero, err
	}
	if err := bind(runtime); err != nil {
		return zero, err
	}
	result, err := runtime.ExecuteProgram(ctx,
		"recipe/run/"+definition.ID.String()+"/"+inputContent.Descriptor.ID.String(), program,
		map[recipe.PortName]workflowruntime.Value{
			input.Name: workflowruntime.ArtifactValue(input.Data, inputValue, inputContent),
		})
	if err != nil {
		return zero, err
	}
	datum, ok := result.Outputs[output.Name].Single()
	if !ok {
		return zero, fmt.Errorf("capability runtime: output %q has invalid cardinality", output.Name)
	}
	decoded, ok := datum.Value.(Output)
	if !ok {
		return zero, fmt.Errorf("capability runtime: output %q has invalid value type", output.Name)
	}
	return decoded, nil
}
