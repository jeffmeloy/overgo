// Package bridgetrain owns optimizer updates whose complete mutable extent is
// a representation bridge; source and target models are read-only evidence.
package bridgetrain

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/optimizer"
	"overgo/internal/runrecord"
)

// Request binds the two frozen model artifacts and the complete mutable bridge
// parameter slab to one compiled optimizer plan.
type Request struct {
	Reader         artifact.Reader
	Source         artifact.ID
	Target         artifact.ID
	Weights        []float32
	Gradients      []float32
	Plan           optimizer.Plan
	Config         optimizer.Config
	Dataset        artifact.ID
	Examples       []Example
	SourceForward  FrozenForward
	TargetForward  FrozenForward
	TrainingPolicy artifact.ID
	Epochs         int
	Resume         artifact.ID
}

// ModelEvidence records the immutable descriptor at both sides of the update.
type ModelEvidence struct {
	Before artifact.Descriptor
	After  artifact.Descriptor
}

// Result proves that a bridge changed while both model descriptors remained
// exactly stable.
type Result struct {
	Source       ModelEvidence
	Target       ModelEvidence
	BridgeBefore artifact.ID
	BridgeAfter  artifact.ID
	Optimizer    optimizer.StepResult
	Lineage      []artifact.Lineage
	Checkpoint   artifact.ID
	Metrics      []runrecord.Metric
	Batch        artifact.Batch
}

// Trainer has no repository mutation capability; its zero value performs one
// host optimizer step over a private bridge copy.
type Trainer struct{}

// Step commits bridge weights to the caller only after frozen-model evidence
// and a non-empty bridge update have both been verified.
func (trainer Trainer) Step(ctx context.Context, request Request) (Result, error) {
	if len(request.Examples) != 0 || request.Dataset.Valid() || request.Resume.Valid() {
		return trainer.trainDataset(ctx, request)
	}
	if ctx == nil || request.Reader == nil {
		return Result{}, errors.New("bridge training: artifact authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if request.Source.Kind() != artifact.KindModel || request.Target.Kind() != artifact.KindModel ||
		request.Source == request.Target {
		return Result{}, errors.New("bridge training: frozen model identities are invalid")
	}
	for index := range request.Plan.GroupCount() {
		group, ok := request.Plan.Group(index)
		if !ok || group.Frozen {
			return Result{}, errors.New("bridge training: optimizer plan includes a frozen or unavailable bridge group")
		}
	}
	sourceBefore, err := trainer.descriptor(ctx, request.Reader, request.Source)
	if err != nil {
		return Result{}, err
	}
	targetBefore, err := trainer.descriptor(ctx, request.Reader, request.Target)
	if err != nil {
		return Result{}, err
	}
	updated := slices.Clone(request.Weights)
	gradients := slices.Clone(request.Gradients)
	bridgeBefore, err := artifact.JSONID(artifact.KindAdapter, updated)
	if err != nil {
		return Result{}, err
	}
	stepper, err := optimizer.New(updated, gradients, request.Plan, request.Config)
	if err != nil {
		return Result{}, err
	}
	step := stepper.Step()
	bridgeAfter, err := artifact.JSONID(artifact.KindAdapter, updated)
	if err != nil {
		return Result{}, err
	}
	sourceAfter, err := trainer.descriptor(ctx, request.Reader, request.Source)
	if err != nil {
		return Result{}, err
	}
	targetAfter, err := trainer.descriptor(ctx, request.Reader, request.Target)
	if err != nil {
		return Result{}, err
	}
	if sourceBefore != sourceAfter || targetBefore != targetAfter {
		return Result{}, errors.New("bridge training: frozen model descriptor changed during update")
	}
	if bridgeBefore == bridgeAfter || !checked.PositiveFinite64(step.UpdateL2) {
		return Result{}, errors.New("bridge training: optimizer produced no bridge update")
	}
	copy(request.Weights, updated)
	return Result{
		Source:       ModelEvidence{Before: sourceBefore, After: sourceAfter},
		Target:       ModelEvidence{Before: targetBefore, After: targetAfter},
		BridgeBefore: bridgeBefore, BridgeAfter: bridgeAfter, Optimizer: step,
		Lineage: []artifact.Lineage{
			{Child: bridgeAfter, Parent: bridgeBefore, Relation: artifact.RelationDerivedFrom},
			{Child: bridgeAfter, Parent: request.Source, Relation: artifact.RelationTrainedFrom},
			{Child: bridgeAfter, Parent: request.Target, Relation: artifact.RelationTrainedFrom},
		},
	}, nil
}

func (Trainer) descriptor(ctx context.Context, reader artifact.Reader, id artifact.ID) (artifact.Descriptor, error) {
	descriptor, found, err := reader.Artifact(ctx, id)
	if err != nil {
		return artifact.Descriptor{}, err
	}
	if !found || descriptor.ID != id {
		return artifact.Descriptor{}, errors.New("bridge training: frozen model descriptor is absent")
	}
	if err := descriptor.Validate(); err != nil {
		return artifact.Descriptor{}, err
	}
	return descriptor, nil
}
