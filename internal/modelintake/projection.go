package modelintake

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
)

// ProjectionCandidate is one model paired with the projector that gives it
// image, audio or video input: both inventories, the media the projector
// declares, its preprocessing profile when the catalog names one, and the
// projection recipe definition binding them.
type ProjectionCandidate struct {
	Model         modelartifact.Inventory
	Projector     modelartifact.Inventory
	ProjectorPath string
	Media         []recipe.DataKind
	Processor     *projector.MediaPreprocessProfile
	Config        *modelartifact.ModelConfigDocument
	Definition    recipe.Definition
}

// PrepareProjectionCandidate reads the model and the projector GGUF and
// defines the projection recipe binding them, publishing nothing.
func PrepareProjectionCandidate(ctx context.Context, store overgodb.DocumentReader, modelPath, projectorPath string) (ProjectionCandidate, error) {
	file, err := gguf.Open(modelPath)
	if err != nil {
		return ProjectionCandidate{}, err
	}
	model, inventoryErr := modelartifact.FromGGUF(file, artifact.KindModel)
	if err := errors.Join(inventoryErr, file.Close()); err != nil {
		return ProjectionCandidate{}, err
	}
	projectorInventory, media, processor, err := projector.InspectProjection(ctx, projectorPath)
	if err != nil {
		return ProjectionCandidate{}, err
	}
	config, declared, err := modelrecipe.ResolveModelConfig(ctx, store, model.Manifest.ID)
	if err != nil {
		return ProjectionCandidate{}, err
	}
	var boundConfig *modelartifact.ModelConfigDocument
	if declared && config.Generation != nil && config.Generation.ImageAttention != "" {
		processor = cmp.Or(processor, &projector.MediaPreprocessProfile{Version: artifact.InitialDocumentVersion})
		bound, err := processor.BindModelConfig(config)
		if err != nil {
			return ProjectionCandidate{}, err
		}
		processor, boundConfig = &bound, &config
	}
	var processorID artifact.ID
	if processor != nil {
		processorID = processor.ID
	}
	definition, err := modelrecipe.ProjectionDefinition(model.Manifest.ID, projectorInventory.Manifest.ID, processorID, media...)
	if err != nil {
		return ProjectionCandidate{}, err
	}
	return ProjectionCandidate{
		Model: model, Projector: projectorInventory, ProjectorPath: projectorPath,
		Media: media, Processor: processor, Config: boundConfig, Definition: definition,
	}, nil
}

// RegisterProjectionCandidate publishes the model's and the projector's
// facts (artifacts, manifests, locations, the preprocessing profile) and,
// when the projection recipe is not yet published, the candidate itself.
func RegisterProjectionCandidate(ctx context.Context, store artifact.Repository, candidate ProjectionCandidate) error {
	batch, err := candidate.Model.Batch("recipe/facts/" + candidate.Model.Manifest.ID.String())
	if err != nil {
		return err
	}
	related, err := candidate.Projector.Batch(batch.Key)
	if err != nil {
		return err
	}
	batch.Artifacts = append(batch.Artifacts, related.Artifacts...)
	batch.Contents = append(batch.Contents, related.Contents...)
	batch.Manifests = append(batch.Manifests, related.Manifests...)
	batch.Lineage = append(batch.Lineage, related.Lineage...)
	batch.Locations = append(batch.Locations, related.Locations...)
	if candidate.Processor != nil {
		content, err := candidate.Processor.Content()
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, content)
	}
	if candidate.Config != nil {
		content, err := candidate.Config.Content()
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(candidate.Config.ID, candidate.Config.Model)...)
		batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(candidate.Processor.ID, candidate.Config.ID)...)
	}
	// One model can acquire another projector/configuration or move on disk.
	// Idempotency must name all published facts, including their locations.
	data, err := json.Marshal(batch)
	if err != nil {
		return err
	}
	batch.Key = fmt.Sprintf("recipe/projection-facts/%x", sha256.Sum256(data))
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return fmt.Errorf("publish projection facts: %w", err)
	}
	_, published, err := modelrecipe.Status(ctx, store, candidate.Definition.ID)
	if err != nil {
		return err
	}
	if !published {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "recipe/candidate/"+candidate.Definition.ID.String(), candidate.Definition,
		); err != nil {
			return err
		}
	}
	return nil
}
