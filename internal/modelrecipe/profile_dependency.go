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
	return resolveProfileDependency(ctx, reader, definition, role, codec.Require)
}

func resolveProfileDocumentDependency(
	ctx context.Context,
	reader artifact.Reader,
	definition recipe.Definition,
	role recipe.DependencyRole,
) (ProfileDocument, error) {
	return resolveProfileDependency(ctx, reader, definition, role, loadProfile)
}

func resolveProfileDependency[T any](
	ctx context.Context,
	reader artifact.Reader,
	definition recipe.Definition,
	role recipe.DependencyRole,
	require func(context.Context, artifact.Reader, artifact.ID) (T, error),
) (T, error) {
	var zero T
	if require == nil || !componentProfileRole(role) {
		return zero, errors.New("model recipe: dependency role is not a component profile")
	}
	if err := definition.ValidateIdentity(); err != nil {
		return zero, err
	}
	id, ok := definition.PrimaryDependency(role)
	if !ok {
		return zero, errors.New("model recipe: required component profile is absent")
	}
	return require(ctx, reader, id)
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
