package modelintake

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
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
	Definition    recipe.Definition
}

// PrepareProjectionCandidate reads the model and the projector GGUF and
// defines the projection recipe binding them, publishing nothing.
func PrepareProjectionCandidate(ctx context.Context, modelPath, projectorPath string) (ProjectionCandidate, error) {
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
		Media: media, Processor: processor, Definition: definition,
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

// VerifyProjection proves the projector loads on the host and declares the
// media its recipe binds, and publishes that as the candidate's
// verification evidence bound to the code revision.
func VerifyProjection(ctx context.Context, store artifact.Repository, candidate ProjectionCandidate, revision string) (modelrecipe.Verification, error) {
	started := time.Now()
	session, err := projector.OpenSession(ctx, candidate.ProjectorPath, projector.OpenOptions{})
	if err != nil {
		return modelrecipe.Verification{}, fmt.Errorf("projector did not open: %w", err)
	}
	capabilities := session.Capabilities()
	if err := session.Close(); err != nil {
		return modelrecipe.Verification{}, err
	}
	var declared []string
	for _, kind := range candidate.Media {
		declared = append(declared, string(kind))
	}
	evidence := fmt.Sprintf("projector opened on the host; media=%s; image=%v audio=%v video=%v",
		strings.Join(declared, ","), capabilities.Image, capabilities.Audio, capabilities.Video)
	return PublishVerification(ctx, store, candidate.Definition, revision, time.Since(started), "host", "go", evidence)
}

// ActivateProjection binds the verification to the projection recipe as
// its active evidence: the model then serves the projector's media.
func ActivateProjection(ctx context.Context, store artifact.Repository, candidate ProjectionCandidate, verification modelrecipe.Verification, reason string) error {
	return modelrecipe.ActivateCapability(ctx, store, candidate.Definition, verification, recipe.EvidenceVerified, reason)
}
