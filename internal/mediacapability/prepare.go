package mediacapability

import (
	"context"
	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// Prepare publishes a capability's source facts and resolves its definition.
// It neither registers nor activates the candidate recipe.
func Prepare(
	ctx context.Context,
	store artifact.Repository,
	path string,
	task recipe.Task,
	capability Capability,
) (artifact.ID, recipe.Definition, error) {
	if err := ctx.Err(); err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	source, err := capability.Resolve(path)
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	modelID := source.Inventory.Manifest.ID
	var definition recipe.Definition
	var facts []artifact.Content
	if source.Define != nil {
		definition, facts, err = source.Define(modelID)
	} else {
		definition, err = modelrecipe.CapabilityDefinition(task, modelID)
	}
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	batch, err := source.Inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	for _, related := range source.Related {
		relatedBatch, err := related.Batch(batch.Key)
		if err != nil {
			return artifact.ID{}, recipe.Definition{}, err
		}
		batch.Artifacts = append(batch.Artifacts, relatedBatch.Artifacts...)
		batch.Contents = append(batch.Contents, relatedBatch.Contents...)
		batch.Manifests = append(batch.Manifests, relatedBatch.Manifests...)
		batch.Lineage = append(batch.Lineage, relatedBatch.Lineage...)
		batch.Locations = append(batch.Locations, relatedBatch.Locations...)
	}
	batch.Contents = append(batch.Contents, facts...)
	batch.Manifests = append(batch.Manifests, source.Manifests...)
	if err := modelartifact.PreserveRawTextDescriptors(ctx, store, &batch); err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	if err := modelrecipe.PublishFacts(ctx, store, batch); err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	return modelID, definition, nil
}
