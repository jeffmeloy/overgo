package runrecord

import (
	"context"
	"encoding/json"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

const (
	// AttemptStimulusMediaType identifies exact admitted agent-attempt boundaries.
	AttemptStimulusMediaType = "application/vnd.overgo.attempt-stimulus+json"
	// AttemptStimulusSchema identifies the boundary document contract.
	AttemptStimulusSchema = "overgo/attempt-stimulus/v1"
	// AttemptStimulusAliasRoot scopes immutable boundaries by operation and attempt.
	AttemptStimulusAliasRoot = "attempt/stimulus/"
	// AttemptArgumentMediaType identifies exact raw tool-argument bytes.
	AttemptArgumentMediaType = "application/json"
	// AttemptArgumentSchema identifies the raw argument contract.
	AttemptArgumentSchema = "overgo/attempt-arguments/v1"
)

var attemptArgumentContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: AttemptArgumentMediaType, Schema: AttemptArgumentSchema,
}

// AttemptStimulusBoundary fixes the store head and complete cited selection an
// agent attempt could observe before execution begins.
type AttemptStimulusBoundary struct {
	Version   uint16                       `json:"version"`
	Operation artifact.ID                  `json:"operation"`
	Attempt   uint32                       `json:"attempt"`
	Manual    artifact.ID                  `json:"manual"`
	Prior     artifact.ID                  `json:"prior,omitzero"`
	Selection dataset.InteractionSelection `json:"selection"`
	ID        artifact.ID                  `json:"-"`
}

var attemptStimulusCodec = artifact.JSONDocumentCodec(
	"attempt stimulus boundary", artifact.KindEvidence, AttemptStimulusMediaType, AttemptStimulusSchema,
	canonicalizeAttemptStimulus,
	func(value AttemptStimulusBoundary) artifact.ID { return value.ID },
	func(value *AttemptStimulusBoundary, id artifact.ID) { value.ID = id },
	cloneAttemptStimulus,
)

// AttemptArgumentContent binds exact raw argument bytes for source citation.
func AttemptArgumentContent(arguments []byte) (artifact.Content, error) {
	if !json.Valid(arguments) {
		return artifact.Content{}, errors.New("run record: attempt arguments are not valid JSON")
	}
	return attemptArgumentContract.ContentBytes(arguments)
}

// ResolveAttemptStimulus returns an admitted boundary by operation and ordinal.
func ResolveAttemptStimulus(ctx context.Context, reader artifact.Reader, operation artifact.ID, attempt uint32) (AttemptStimulusBoundary, bool, error) {
	return attemptStimulusCodec.Resolve(ctx, reader, attemptStimulusAlias(operation, attempt))
}

// PublishAttemptStimulus commits the immutable boundary and its new cited content.
func PublishAttemptStimulus(ctx context.Context, repository artifact.Repository, value AttemptStimulusBoundary, contents []artifact.Content) (AttemptStimulusBoundary, error) {
	if ctx == nil || repository == nil {
		return AttemptStimulusBoundary{}, errors.New("run record: attempt stimulus repository is absent")
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	identified, err := attemptStimulusCodec.New(value)
	if err != nil {
		return AttemptStimulusBoundary{}, err
	}
	if existing, found, resolveErr := ResolveAttemptStimulus(ctx, repository, identified.Operation, identified.Attempt); resolveErr != nil {
		return AttemptStimulusBoundary{}, resolveErr
	} else if found {
		if existing.ID != identified.ID {
			return AttemptStimulusBoundary{}, errors.New("run record: attempt stimulus boundary already differs")
		}
		return existing, nil
	}
	boundaryContent, err := attemptStimulusCodec.Content(identified)
	if err != nil {
		return AttemptStimulusBoundary{}, err
	}
	parents := []artifact.ID{identified.Operation, identified.Manual, identified.Prior}
	for _, source := range identified.Selection.Sources {
		parents = append(parents, source.Source, source.CausalRoot)
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	batch, err := artifact.NewDocumentBatch(
		"attempt/stimulus/"+identified.ID.String(), append(slices.Clone(contents), boundaryContent),
		artifact.DependencyLineage(identified.ID, parents...),
		[]artifact.AliasBinding{{Name: attemptStimulusAlias(identified.Operation, identified.Attempt), Target: identified.ID}},
	)
	if err != nil {
		return AttemptStimulusBoundary{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: identified.Operation})
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return AttemptStimulusBoundary{}, err
	}
	return identified, nil
}

func canonicalizeAttemptStimulus(value *AttemptStimulusBoundary) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Operation.Kind() != artifact.KindEvidence ||
		value.Attempt == 0 || !value.Manual.Valid() || value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid attempt stimulus boundary")
	}
	if err := value.Selection.Validate(); err != nil {
		return errors.Join(errors.New("run record: invalid attempt stimulus selection"), err)
	}
	return nil
}

func cloneAttemptStimulus(value AttemptStimulusBoundary) AttemptStimulusBoundary {
	value.Selection.Sources = slices.Clone(value.Selection.Sources)
	if value.Selection.Cursor != nil {
		cursor := *value.Selection.Cursor
		value.Selection.Cursor = &cursor
	}
	return value
}

func attemptStimulusAlias(operation artifact.ID, attempt uint32) string {
	return indexedAlias(AttemptStimulusAliasRoot, operation, uint64(attempt))
}
