package modelintake

import (
	"cmp"
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
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

// DraftCandidate is one registered model paired with the draft layer its
// decoding speculates through: the model's identity, the draft's inventory
// and the speculation recipe binding them.
type DraftCandidate struct {
	Model      artifact.ID
	Draft      modelartifact.Inventory
	DraftPath  string
	Definition recipe.Definition
}

// PrepareDraftCandidate reads the draft GGUF beside the model trunk names
// and defines the speculation recipe binding them, publishing nothing.
func PrepareDraftCandidate(trunk artifact.ID, draftPath string) (DraftCandidate, error) {
	file, err := gguf.Open(draftPath)
	if err != nil {
		return DraftCandidate{}, err
	}
	if !model.DescribesHyperDraft(file) {
		return DraftCandidate{}, errors.Join(fmt.Errorf("model intake: %s declares no draft layer", draftPath), file.Close())
	}
	draft, inventoryErr := modelartifact.FromGGUF(file, artifact.KindDraft)
	if err := errors.Join(inventoryErr, file.Close()); err != nil {
		return DraftCandidate{}, err
	}
	definition, err := modelrecipe.DraftDefinition(trunk, draft.Manifest.ID)
	if err != nil {
		return DraftCandidate{}, err
	}
	return DraftCandidate{Model: trunk, Draft: draft, DraftPath: draftPath, Definition: definition}, nil
}

// RegisterDraftCandidate publishes the draft's facts beside its registered
// model's and, when the speculation recipe is not yet published, the
// candidate itself.
func RegisterDraftCandidate(ctx context.Context, store artifact.Repository, candidate DraftCandidate) error {
	batch, err := candidate.Draft.Batch("recipe/facts/" + candidate.Model.String())
	if err != nil {
		return err
	}
	return publishPairCandidate(ctx, store, batch, candidate.Definition)
}

// pairBatch is the facts batch of a model and a file a capability recipe binds to it.
func pairBatch(model, component modelartifact.Inventory) (artifact.Batch, error) {
	batch, err := model.Batch("recipe/facts/" + model.Manifest.ID.String())
	if err != nil {
		return artifact.Batch{}, err
	}
	related, err := component.Batch(batch.Key)
	if err != nil {
		return artifact.Batch{}, err
	}
	batch.Artifacts = append(batch.Artifacts, related.Artifacts...)
	batch.Contents = append(batch.Contents, related.Contents...)
	batch.Manifests = append(batch.Manifests, related.Manifests...)
	batch.Lineage = append(batch.Lineage, related.Lineage...)
	batch.Locations = append(batch.Locations, related.Locations...)
	return batch, nil
}

// publishPairCandidate publishes a pair's facts and, when unpublished, the candidate recipe binding them.
func publishPairCandidate(ctx context.Context, store artifact.Repository, batch artifact.Batch, definition recipe.Definition) error {
	if err := modelrecipe.PublishFacts(ctx, store, batch); err != nil {
		return err
	}
	_, published, err := modelrecipe.Status(ctx, store, definition.ID)
	if err != nil || published {
		return err
	}
	_, _, err = modelrecipe.PublishCandidate(ctx, store, "recipe/candidate/"+definition.ID.String(), definition)
	return err
}

// RegisterProjectionCandidate publishes the model's and the projector's
// facts (artifacts, manifests, locations, the preprocessing profile) and,
// when the projection recipe is not yet published, the candidate itself.
func RegisterProjectionCandidate(ctx context.Context, store artifact.Repository, candidate ProjectionCandidate) error {
	batch, err := pairBatch(candidate.Model, candidate.Projector)
	if err != nil {
		return err
	}
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
	return publishPairCandidate(ctx, store, batch, candidate.Definition)
}
