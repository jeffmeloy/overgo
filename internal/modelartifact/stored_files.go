package modelartifact

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

// ReinspectFiles reconstructs a file-backed model inventory from its recorded
// component locations. Every file is rehashed and the resulting model identity
// must equal the stored manifest. It neither registers nor relocates artifacts.
func ReinspectFiles(ctx context.Context, repository artifact.Repository, modelID artifact.ID) (Inventory, error) {
	if ctx == nil || repository == nil || modelID.Kind() != artifact.KindModel {
		return Inventory{}, errors.New("model artifact: incomplete reinspection")
	}
	if err := ctx.Err(); err != nil {
		return Inventory{}, err
	}
	root, err := artifact.AvailablePath(ctx, repository, modelID, artifact.LocationDirectory)
	if err != nil {
		return Inventory{}, err
	}
	manifest, found, err := repository.Manifest(ctx, modelID)
	if err != nil || !found {
		return Inventory{}, errors.Join(errors.New("model artifact: stored manifest absent"), err)
	}
	var specs []FileSpec
	for _, component := range manifest.Components {
		if err := ctx.Err(); err != nil {
			return Inventory{}, err
		}
		path, err := artifact.AvailablePath(ctx, repository, component.Artifact, artifact.LocationFile)
		if err != nil {
			return Inventory{}, err
		}
		specs = append(specs, FileSpec{Path: path, Name: component.Name, Role: component.Role})
	}
	inventory, err := FromFiles(root, specs)
	if err != nil || inventory.Manifest.ID != modelID {
		return Inventory{}, errors.Join(errors.New("model artifact: physical files differ from stored manifest"), err)
	}
	return inventory, ctx.Err()
}
