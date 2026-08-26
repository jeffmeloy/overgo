package modelrecipe

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// ResolveProfileDependency loads one recipe-bound typed profile.
func ResolveProfileDependency[T any](
	ctx context.Context,
	reader artifact.Reader,
	definition recipe.Definition,
	role recipe.DependencyRole,
	codec artifact.DocumentCodec[T],
) (T, error) {
	var zero T
	if !componentProfileRole(role) {
		return zero, errors.New("model recipe: dependency role is not a component profile")
	}
	if err := definition.ValidateIdentity(); err != nil {
		return zero, err
	}
	id, ok := definition.PrimaryDependency(role)
	if !ok {
		return zero, errors.New("model recipe: required component profile is absent")
	}
	return codec.Require(ctx, reader, id)
}

func componentProfileRole(role recipe.DependencyRole) bool {
	switch role {
	case recipe.DependencyProfile, recipe.DependencyProcessorProfile,
		recipe.DependencyFlowProfile, recipe.DependencyDerivationProfile:
		return true
	default:
		return false
	}
}
