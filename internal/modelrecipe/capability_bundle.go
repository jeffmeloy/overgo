// Package modelrecipe binds capability bundles to promoted recipes.
package modelrecipe

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// CapabilityBundle binds one manifest to active recipe evidence.
type CapabilityBundle struct {
	Manifest artifact.Manifest `json:"manifest"`
	Evidence artifact.ID       `json:"evidence"`
}

func resolveCapabilityBundles(
	ctx context.Context,
	reader artifact.Reader,
	activation Activation,
) ([]CapabilityBundle, error) {
	var bundles []CapabilityBundle
	for _, dependency := range activation.Definition.Dependencies {
		if dependency.Role != recipe.DependencyCapabilityBundle {
			continue
		}
		if !activation.Event.ID.Valid() {
			return nil, errors.New("model recipe: capability bundle lacks promotion evidence")
		}
		manifest, found, err := reader.Manifest(ctx, dependency.Artifact)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, errors.New("model recipe: capability bundle manifest is absent")
		}
		if err := artifact.ValidateCapabilityBundle(manifest); err != nil {
			return nil, err
		}
		bundles = append(bundles, CapabilityBundle{Manifest: manifest, Evidence: activation.Event.ID})
	}
	return bundles, nil
}

func validateCapabilityBundles(activation Activation, bundles []CapabilityBundle) ([]artifact.ID, error) {
	if !activation.Event.ID.Valid() {
		if len(bundles) != 0 {
			return nil, errors.New("model recipe: candidate execution carries capability bundles")
		}
		return nil, nil
	}
	expected := make([]artifact.ID, 0, len(activation.Definition.Dependencies))
	for _, dependency := range activation.Definition.Dependencies {
		if dependency.Role == recipe.DependencyCapabilityBundle {
			expected = append(expected, dependency.Artifact)
		}
	}
	if len(expected) != len(bundles) {
		return nil, errors.New("model recipe: capability bundle bindings differ")
	}
	for index, bundle := range bundles {
		if err := artifact.ValidateCapabilityBundle(bundle.Manifest); err != nil {
			return nil, err
		}
		if bundle.Manifest.ID != expected[index] || bundle.Evidence != activation.Event.ID || !bundle.Evidence.Valid() {
			return nil, errors.New("model recipe: capability bundle authority differs")
		}
	}
	return expected, nil
}

func sameCapabilityBundles(left, right []CapabilityBundle) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].Manifest.ID != right[index].Manifest.ID || left[index].Evidence != right[index].Evidence {
			return false
		}
	}
	return true
}

func cloneCapabilityBundles(values []CapabilityBundle) []CapabilityBundle {
	result := slices.Clone(values)
	for index := range result {
		result[index].Manifest = result[index].Manifest.Clone()
	}
	return result
}
