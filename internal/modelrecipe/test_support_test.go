package modelrecipe

import (
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
	}, placement, session, fixtureResidency(placement))
}

func inferenceFixture(
	modelID artifact.ID,
	placement recipe.Placement,
	session DecodeSessionPolicy,
) (recipe.Definition, error) {
	return inference(
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}}, placement, session,
		fixtureResidency(placement),
	)
}

func fixtureResidency(placement recipe.Placement) recipe.ResidencyPolicy {
	if placement == recipe.PlacementHost {
		return recipe.ResidencyHostCache
	}
	if placement == recipe.PlacementDevice {
		return recipe.ResidencyDeviceNative
	}
	return recipe.ResidencyHybridNative
}

func compileInferenceFixture(
	definition recipe.Definition,
	spec model.Spec,
	weights model.Weights,
) (Plan, error) {
	return compileDefinition(definition, func() (model.ModelPlan, error) {
		profile, ok := model.LookupArchitecture(spec.Architecture)
		if !ok {
			return model.ModelPlan{}, &model.UnsupportedArchitectureError{Architecture: spec.Architecture}
		}
		return model.CompileModelPlanWithProfile(spec, weights, profile)
	})
}
