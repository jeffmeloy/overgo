package speechactivity

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// Detector owns an exact stored recipe and immutable CPU execution plans.
// Workspaces, stream serialization and publication are separate caller-owned
// concerns; this owner creates neither a scheduler nor a background worker.
type Detector struct {
	repository  artifact.Repository
	definition  recipe.Definition
	profile     Profile
	network     *Network
	frontend    *audiodsp.Frontend
	stream      *audiodsp.StreamFrontend
	boundary    *BoundaryPolicy
	memoryBytes uint64
}

// LoadDetector validates recipe topology, dependency lineage, every physical
// model component and the recorded license before binding weights. memoryBytes
// is a per-component numeric ceiling, as in the existing transcription owner;
// it is not a process-RSS or aggregate-concurrency bound.
func LoadDetector(ctx context.Context, repository artifact.Repository, definitionID artifact.ID, memoryBytes uint64) (*Detector, error) {
	if ctx == nil || repository == nil || memoryBytes == 0 {
		return nil, errors.New("speech activity: incomplete detector load")
	}
	definition, err := recipe.RequireDefinition(ctx, repository, definitionID)
	if err != nil {
		return nil, err
	}
	profileID, ok := definition.PrimaryDependency(recipe.DependencyProcessorProfile)
	if !ok {
		return nil, errors.New("speech activity: recipe has no processor")
	}
	profile, err := RequireProfile(ctx, repository, profileID)
	if err != nil {
		return nil, err
	}
	expected, err := modelrecipe.ActivityDefinition(profile.Model, profile.ID, profile.Inventory)
	if err != nil || expected.ID != definition.ID {
		return nil, errors.Join(errors.New("speech activity: recipe topology differs"), err)
	}
	for _, binding := range []struct {
		child   artifact.ID
		parents []artifact.ID
	}{
		{definition.ID, []artifact.ID{profile.Model, profile.ID, profile.Inventory}},
		{profile.ID, []artifact.ID{profile.Model, profile.Inventory, profile.License}},
		{profile.Model, []artifact.ID{profile.License}},
	} {
		parents, err := repository.Parents(ctx, binding.child)
		if err != nil {
			return nil, err
		}
		for _, parent := range binding.parents {
			if !slices.Contains(parents, artifact.Lineage{Child: binding.child, Parent: parent, Relation: artifact.RelationDependsOn}) {
				return nil, errors.New("speech activity: missing dependency lineage")
			}
			if _, found, err := repository.Artifact(ctx, parent); err != nil || !found {
				return nil, errors.Join(errors.New("speech activity: dependency absent"), err)
			}
		}
	}
	license, found, err := artifact.ReadContent(ctx, repository, profile.License)
	if err != nil || !found || len(license.Data) == 0 {
		return nil, errors.Join(errors.New("speech activity: license content absent"), err)
	}
	if err := license.Validate(); err != nil {
		return nil, err
	}
	inventory, err := modelartifact.ReinspectFiles(ctx, repository, profile.Model)
	if err != nil || inventory.TensorInventory.ID != profile.Inventory {
		return nil, errors.Join(errors.New("speech activity: physical tensor identity differs"), err)
	}
	checkpoint := ""
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentWeights {
			if checkpoint != "" {
				return nil, errors.New("speech activity: declaration requires one checkpoint component")
			}
			checkpoint, err = artifact.AvailablePath(ctx, repository, component.Artifact, artifact.LocationFile)
			if err != nil {
				return nil, err
			}
		}
	}
	if checkpoint == "" {
		return nil, errors.New("speech activity: weight component absent")
	}
	network, err := LoadNetwork(ctx, checkpoint, profile.Network, memoryBytes)
	if err != nil {
		return nil, err
	}
	if uint64(network.InputWidth()) != uint64(profile.Frontend.Geometry.FeatureBins) {
		return nil, errors.New("speech activity: frontend and classifier widths differ")
	}
	d := &Detector{repository: repository, definition: definition, profile: profile, network: network, memoryBytes: memoryBytes}
	if profile.Streaming != nil {
		if !network.Causal() {
			return nil, errors.New("speech activity: live policy requires causal filters")
		}
		d.stream, err = audiodsp.NewStreamFrontend(profile.Frontend, memoryBytes)
		if err == nil {
			d.boundary, err = NewBoundaryPolicy(*profile.Streaming, memoryBytes)
		}
	} else {
		d.frontend, err = audiodsp.NewFrontend(profile.Frontend, memoryBytes)
		if err == nil {
			_, err = OfflineDecisions(ctx, nil, *profile.Offline, memoryBytes)
		}
	}
	if err != nil {
		return nil, err
	}
	return d, nil
}
