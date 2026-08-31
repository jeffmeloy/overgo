package runrecord

import (
	"context"
	"encoding/json"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/invocation"
)

const (
	// AttemptStimulusMediaType identifies exact admitted agent-attempt boundaries.
	AttemptStimulusMediaType = "application/vnd.overgo.attempt-stimulus+json"
	// AttemptStimulusSchema identifies the boundary document contract.
	AttemptStimulusSchema = "overgo/attempt-stimulus/v2"
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
	Version          uint16                       `json:"version"`
	Operation        artifact.ID                  `json:"operation"`
	Attempt          uint32                       `json:"attempt"`
	Manual           artifact.ID                  `json:"manual"`
	Class            invocation.Class             `json:"class"`
	Arguments        artifact.ID                  `json:"arguments"`
	Effect           artifact.ID                  `json:"effect"`
	Inspection       artifact.ID                  `json:"inspection,omitzero"`
	InspectionEffect artifact.ID                  `json:"inspection_effect,omitzero"`
	Ceiling          artifact.ID                  `json:"ceiling"`
	CausalContext    artifact.ID                  `json:"causal_context,omitzero"`
	Prior            artifact.ID                  `json:"prior,omitzero"`
	Selection        dataset.InteractionSelection `json:"selection"`
	ID               artifact.ID                  `json:"-"`
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

// RequireAttemptStimulus returns one boundary by immutable identity.
func RequireAttemptStimulus(ctx context.Context, reader artifact.Reader, id artifact.ID) (AttemptStimulusBoundary, error) {
	return attemptStimulusCodec.RequireExactLineage(ctx, reader, id, attemptStimulusLineage)
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
	batch, err := artifact.NewDocumentBatch(
		"attempt/stimulus/"+identified.ID.String(), append(slices.Clone(contents), boundaryContent),
		attemptStimulusLineage(identified),
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

func attemptStimulusLineage(value AttemptStimulusBoundary) []artifact.Lineage {
	parents := []artifact.ID{value.Operation, value.Manual, value.Arguments, value.Effect,
		value.Inspection, value.InspectionEffect, value.Ceiling, value.CausalContext, value.Prior}
	for _, source := range value.Selection.Sources {
		parents = append(parents, source.Source, source.CausalRoot)
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

func canonicalizeAttemptStimulus(value *AttemptStimulusBoundary) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Operation.Kind() != artifact.KindEvidence ||
		value.Attempt == 0 || value.Manual.Kind() != artifact.KindRecipe || !value.Class.Valid() ||
		value.Arguments.Kind() != artifact.KindEvidence || value.Effect.Kind() != artifact.KindEvidence ||
		value.Ceiling.Kind() != artifact.KindRecipe || !value.Selection.Head.Valid() ||
		value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence ||
		value.CausalContext.Valid() && value.CausalContext.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid attempt stimulus boundary")
	}
	if !slices.ContainsFunc(value.Selection.Sources, func(source dataset.InteractionSelectionSource) bool {
		return source.Source == value.Arguments && source.CausalRoot == value.Operation
	}) {
		return errors.New("run record: attempt stimulus omits exact arguments")
	}
	if value.Class == invocation.ClassMutation {
		if value.Inspection.Kind() != artifact.KindEvidence || value.InspectionEffect.Kind() != artifact.KindEvidence ||
			value.CausalContext.Kind() != artifact.KindEvidence || value.CausalContext != value.Prior {
			return errors.New("run record: mutation requires action-bound inspection and causal context")
		}
	} else if value.Inspection.Valid() || value.InspectionEffect.Valid() {
		return errors.New("run record: inspection cannot claim prior inspection authority")
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
