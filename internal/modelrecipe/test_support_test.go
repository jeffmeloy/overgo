package modelrecipe

import (
	"context"
	"errors"

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
		plan, compileErr := compileProfileFixture(definition, document, spec, weights)
		return plan, compileErr == nil, compileErr
	}
	plan, err := compileInferenceFixture(definition, spec, weights)
	return plan, err == nil, err
}

func compileProfileFixture(
	definition recipe.Definition,
	document ProfileDocument,
	spec model.Spec,
	weights model.Weights,
) (Plan, error) {
	if err := document.ValidateIdentity(); err != nil {
		return Plan{}, err
	}
	if document.Architecture != spec.Architecture {
		return Plan{}, errors.New("model recipe: profile does not match inference recipe")
	}
	if definition.Version != recipe.LegacyVersion {
		profileID, ok := definition.Dependency(recipe.DependencyProfile, 0)
		if !ok || profileID != document.ID {
			return Plan{}, errors.New("model recipe: definition profile dependency mismatch")
		}
	}
	return compileDefinition(definition, func() (model.ModelPlan, error) {
		return model.CompileModelPlanWithProfile(spec, weights, document.Policy)
	})
}
