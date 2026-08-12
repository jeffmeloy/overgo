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
	session DecodeSessionPolicy,
) (recipe.Definition, error) {
	return inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: profileID},
	}, placement, session)
}

func inferenceFixture(
	modelID artifact.ID,
	placement recipe.Placement,
	session DecodeSessionPolicy,
) (recipe.Definition, error) {
	return inference(
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}}, placement, session,
	)
}

func compileInferenceFixture(
	definition recipe.Definition,
	spec model.Spec,
	weights model.Weights,
) (Plan, error) {
	return compileDefinition(definition, func() (model.ModelPlan, error) {
		return model.CompileModelPlan(spec, weights)
	})
}

func compileActiveFixture(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	spec model.Spec,
	weights model.Weights,
) (Plan, bool, error) {
	activation, ok, err := ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok {
		return Plan{}, ok, err
	}
	definition := activation.Definition
	if profileID, bound := definition.Dependency(recipe.DependencyProfile, 0); bound {
		document, loadErr := loadProfile(ctx, store, profileID)
		if loadErr != nil {
			return Plan{}, false, loadErr
		}
		plan, compileErr := CompileWithProfile(definition, document, spec, weights)
		return plan, compileErr == nil, compileErr
	}
	plan, err := compileInferenceFixture(definition, spec, weights)
	return plan, err == nil, err
}
