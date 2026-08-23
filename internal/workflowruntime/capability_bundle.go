package workflowruntime

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// CapabilityBundles returns recipe-admitted bundle identities.
func (executor *ToolExecutor) CapabilityBundles() []artifact.ID {
	if executor == nil {
		return nil
	}
	definition := executor.program.Definition()
	bundles := make([]artifact.ID, 0)
	for _, dependency := range definition.Dependencies {
		if dependency.Role == recipe.DependencyCapabilityBundle {
			bundles = append(bundles, dependency.Artifact)
		}
	}
	return slices.Clip(bundles)
}

// LoadCapabilityComponent resolves one admitted bundle member on demand.
func (executor *ToolExecutor) LoadCapabilityComponent(
	ctx context.Context,
	bundle artifact.ID,
	role artifact.ComponentRole,
	name string,
) (artifact.Content, error) {
	if executor == nil || executor.runtime == nil || ctx == nil {
		return artifact.Content{}, errors.New("workflow runtime: capability resource authority is absent")
	}
	if role != artifact.ComponentInstruction && role != artifact.ComponentResource {
		return artifact.Content{}, errors.New("workflow runtime: invalid capability component role")
	}
	definition := executor.program.Definition()
	if !capabilityBundleAdmitted(definition, bundle) {
		return artifact.Content{}, errors.New("workflow runtime: capability bundle is not admitted")
	}
	manifest, found, err := executor.runtime.store.Manifest(ctx, bundle)
	if err != nil || !found {
		return artifact.Content{}, errors.Join(errors.New("workflow runtime: capability bundle is absent"), err)
	}
	if err := artifact.ValidateCapabilityBundle(manifest); err != nil {
		return artifact.Content{}, err
	}
	return artifact.LoadManifestComponent(ctx, executor.runtime.store, manifest, role, name)
}

func capabilityBundleAdmitted(definition recipe.Definition, bundle artifact.ID) bool {
	for _, dependency := range definition.Dependencies {
		if dependency.Role == recipe.DependencyCapabilityBundle && dependency.Artifact == bundle {
			return true
		}
	}
	return false
}
