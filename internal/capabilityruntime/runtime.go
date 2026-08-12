package capabilityruntime

import (
	"context"
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
		definition := program.Definition()
		if definition.Model != modelID {
			return nil, fmt.Errorf("capability runtime: program model differs from binding")
		}
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
		runtime, err := workflowruntime.NewWithCatalog(store, modelrecipe.Catalog())
		if err != nil {
			return nil, err
		}
		if err := bind(runtime, modelID, model); err != nil {
			return nil, err
		}
		return workflowruntime.ExecuteScalar[Output](
			ctx, runtime,
			"recipe/run/"+definition.ID.String()+"/"+content.Descriptor.ID.String(),
			program, input, content,
		)
	}
}

func IgnoreInput[Input, Model any](load func(string) (Model, error)) func(string, Input) (Model, error) {
	return func(path string, _ Input) (Model, error) { return load(path) }
}
