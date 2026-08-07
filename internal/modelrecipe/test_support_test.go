package modelrecipe

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
)

func inferenceWithProfileFixture(
	modelID artifact.ID,
	profileID artifact.ID,
	placement recipe.Placement,
) (recipe.Definition, error) {
	return inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: profileID},
	}, placement)
}

func inferenceFixture(
	modelID artifact.ID,
	placement recipe.Placement,
) (recipe.Definition, error) {
	return inference(
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}}, placement,
	)
}

func compileActiveFixture(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	spec model.Spec,
	weights model.Weights,
) (Plan, bool, error) {
	definition, ok, err := Active(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok {
		return Plan{}, ok, err
	}
	document, bound, err := activeBoundProfile(ctx, store, definition)
	if err != nil {
		return Plan{}, false, err
	}
	if bound {
		plan, compileErr := CompileWithProfile(definition, document, spec, weights)
		return plan, compileErr == nil, compileErr
	}
	plan, err := Compile(definition, spec, weights)
	return plan, err == nil, err
}
