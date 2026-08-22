package bridgetrain

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/optimizer"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
)

const (
	bridgeCheckpointMediaType = "application/vnd.overgo.bridge-training-checkpoint+json"
	bridgeCheckpointSchema    = "overgo/bridge-training-checkpoint/v1"
)

// Example is one immutable dataset row supplied to both frozen forward ports.
type Example struct {
	SourceInput []float32 `json:"source_input"`
	TargetInput []float32 `json:"target_input"`
}

// FrozenForward executes one model without exposing mutable parameters.
type FrozenForward interface {
	ModelID() artifact.ID
	Forward(context.Context, []float32) ([]float32, error)
}

type bridgeTrainingCheckpoint struct {
	Version        uint16             `json:"version"`
	Source         artifact.ID        `json:"source"`
	Target         artifact.ID        `json:"target"`
	Dataset        artifact.ID        `json:"dataset"`
	TrainingPolicy artifact.ID        `json:"training_policy"`
	ParentBridge   artifact.ID        `json:"parent_bridge"`
	Bridge         artifact.ID        `json:"bridge"`
	Weights        []float32          `json:"weights"`
	Optimizer      optimizer.State    `json:"optimizer"`
	Metrics        []runrecord.Metric `json:"metrics"`
	ID             artifact.ID        `json:"-"`
}

var bridgeCheckpointCodec = artifact.JSONDocumentCodec(
	"bridge training checkpoint", artifact.KindCheckpoint,
	bridgeCheckpointMediaType, bridgeCheckpointSchema,
	canonicalizeBridgeTrainingCheckpoint,
	func(value bridgeTrainingCheckpoint) artifact.ID { return value.ID },
	func(value *bridgeTrainingCheckpoint, id artifact.ID) { value.ID = id },
	func(value bridgeTrainingCheckpoint) bridgeTrainingCheckpoint {
		value.Weights = slices.Clone(value.Weights)
		value.Optimizer.Momentum = slices.Clone(value.Optimizer.Momentum)
		value.Metrics = slices.Clone(value.Metrics)
		return value
	},
)

func (trainer Trainer) trainDataset(ctx context.Context, request Request) (Result, error) {
	if ctx == nil || request.Reader == nil || request.SourceForward == nil || request.TargetForward == nil {
		return Result{}, errors.New("bridge training: dataset execution authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if request.Source.Kind() != artifact.KindModel || request.Target.Kind() != artifact.KindModel ||
		request.Source == request.Target || request.SourceForward.ModelID() != request.Source ||
		request.TargetForward.ModelID() != request.Target || request.Dataset.Kind() != artifact.KindDataset ||
		request.TrainingPolicy.Kind() != artifact.KindProfile || request.Epochs <= 0 || len(request.Examples) == 0 {
		return Result{}, errors.New("bridge training: dataset request authority is invalid")
	}
	datasetID, err := artifact.JSONID(artifact.KindDataset, request.Examples)
	if err != nil || datasetID != request.Dataset {
		return Result{}, errors.Join(err, errors.New("bridge training: dataset content identity differs"))
	}
	for _, id := range []artifact.ID{request.Source, request.Target, request.Dataset, request.TrainingPolicy} {
		if _, err := trainer.descriptor(ctx, request.Reader, id); err != nil {
			return Result{}, err
		}
	}
	if request.Plan.GroupCount() != tensor.SingletonExtent {
		return Result{}, errors.New("bridge training: one bridge parameter group is required")
	}
	group, found := request.Plan.Group(tensor.FirstOffset)
	if !found || group.Frozen || group.Start != tensor.FirstOffset || group.End != len(request.Weights) {
		return Result{}, errors.New("bridge training: optimizer owns anything other than the bridge")
	}
	sourceBefore, err := trainer.descriptor(ctx, request.Reader, request.Source)
	if err != nil {
		return Result{}, err
	}
	targetBefore, err := trainer.descriptor(ctx, request.Reader, request.Target)
	if err != nil {
		return Result{}, err
	}
	weights := slices.Clone(request.Weights)
	parentBridge, err := artifact.JSONID(artifact.KindAdapter, weights)
	if err != nil {
		return Result{}, err
	}
	gradients := make([]float32, len(weights))
	instance, err := optimizer.New(weights, gradients, request.Plan, request.Config)
	if err != nil {
		return Result{}, err
	}
	if request.Resume.Valid() {
		checkpoint, loadErr := loadBridgeTrainingCheckpoint(ctx, request.Reader, request.Resume)
		if loadErr != nil {
			return Result{}, loadErr
		}
		if checkpoint.Source != request.Source || checkpoint.Target != request.Target ||
			checkpoint.Dataset != request.Dataset || checkpoint.TrainingPolicy != request.TrainingPolicy ||
			checkpoint.Bridge != parentBridge || checkpoint.Optimizer.PlanIdentity != request.Plan.Identity() ||
			checkpoint.Optimizer.Config != request.Config || len(checkpoint.Weights) != len(weights) {
			return Result{}, errors.New("bridge training: resume authority differs")
		}
		copy(weights, checkpoint.Weights)
		parentBridge = checkpoint.Bridge
		if err := instance.Restore(checkpoint.Optimizer); err != nil {
			return Result{}, err
		}
	}
	var finalLoss float64
	var step optimizer.StepResult
	for range request.Epochs {
		clear(gradients)
		var epochLoss float64
		for _, example := range request.Examples {
			source, forwardErr := request.SourceForward.Forward(ctx, slices.Clone(example.SourceInput))
			if forwardErr != nil {
				return Result{}, forwardErr
			}
			target, forwardErr := request.TargetForward.Forward(ctx, slices.Clone(example.TargetInput))
			if forwardErr != nil {
				return Result{}, forwardErr
			}
			inputExtent, outputExtent := group.Rows, group.Cols
			loss, gradientErr := accumulateLinearBridgeGradient(
				source, target, weights, gradients, inputExtent, outputExtent,
			)
			if gradientErr != nil {
				return Result{}, gradientErr
			}
			epochLoss += loss
		}
		scale := float32(tensor.SingletonExtent) / float32(len(request.Examples))
		for index := range gradients {
			gradients[index] *= scale
		}
		finalLoss = epochLoss / float64(len(request.Examples))
		step = instance.Step()
	}
	bridgeAfter, err := artifact.JSONID(artifact.KindAdapter, weights)
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
		return Result{}, errors.New("bridge training: frozen model descriptor changed during dataset execution")
	}
	if bridgeAfter == parentBridge || !checked.PositiveFinite64(step.UpdateL2) || !checked.Finite64(finalLoss) {
		return Result{}, errors.New("bridge training: dataset optimizer produced no valid bridge update")
	}
	metrics := []runrecord.Metric{
		{Name: "bridge_loss", Value: finalLoss, Direction: runrecord.DirectionMinimize},
		{Name: "bridge_update_l2", Value: step.UpdateL2, Direction: runrecord.DirectionMaximize},
	}
	checkpoint, err := bridgeCheckpointCodec.New(bridgeTrainingCheckpoint{
		Version: artifact.InitialDocumentVersion,
		Source:  request.Source, Target: request.Target, Dataset: request.Dataset,
		TrainingPolicy: request.TrainingPolicy, ParentBridge: parentBridge, Bridge: bridgeAfter,
		Weights: weights, Optimizer: instance.Snapshot(), Metrics: metrics,
	})
	if err != nil {
		return Result{}, err
	}
	content, err := bridgeCheckpointCodec.Content(checkpoint)
	if err != nil {
		return Result{}, err
	}
	lineage := []artifact.Lineage{
		{Child: bridgeAfter, Parent: parentBridge, Relation: artifact.RelationDerivedFrom},
		{Child: bridgeAfter, Parent: request.Source, Relation: artifact.RelationTrainedFrom},
		{Child: bridgeAfter, Parent: request.Target, Relation: artifact.RelationTrainedFrom},
		{Child: bridgeAfter, Parent: request.Dataset, Relation: artifact.RelationTrainedFrom},
		{Child: checkpoint.ID, Parent: bridgeAfter, Relation: artifact.RelationDependsOn},
		{Child: checkpoint.ID, Parent: request.TrainingPolicy, Relation: artifact.RelationDependsOn},
	}
	batch := artifact.Batch{
		Key:       "bridge-training/" + checkpoint.ID.String(),
		Artifacts: []artifact.Descriptor{{ID: bridgeAfter}}, Contents: []artifact.Content{content}, Lineage: lineage,
	}
	if err := batch.Validate(); err != nil {
		return Result{}, err
	}
	copy(request.Weights, weights)
	return Result{
		Source:       ModelEvidence{Before: sourceBefore, After: sourceAfter},
		Target:       ModelEvidence{Before: targetBefore, After: targetAfter},
		BridgeBefore: parentBridge, BridgeAfter: bridgeAfter, Optimizer: step,
		Lineage: lineage, Checkpoint: checkpoint.ID, Metrics: metrics, Batch: batch,
	}, nil
}

func accumulateLinearBridgeGradient(
	source, target, weights, gradients []float32,
	inputExtent, outputExtent int,
) (float64, error) {
	if errors.Join(
		checked.Length(source, inputExtent), checked.Length(target, outputExtent),
		checked.Length(weights, inputExtent, outputExtent),
		checked.Length(gradients, inputExtent, outputExtent),
	) != nil {
		return 0, errors.New("bridge training: forward representation geometry differs")
	}
	var loss float64
	for output := range outputExtent {
		var predicted float32
		for input := range inputExtent {
			predicted += source[input] * weights[input*outputExtent+output]
		}
		difference := predicted - target[output]
		loss += float64(difference) * float64(difference)
		for input := range inputExtent {
			gradients[input*outputExtent+output] += (difference + difference) * source[input] / float32(outputExtent)
		}
	}
	return loss / float64(outputExtent), nil
}

func loadBridgeTrainingCheckpoint(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (bridgeTrainingCheckpoint, error) {
	return bridgeCheckpointCodec.Require(ctx, reader, id)
}

func canonicalizeBridgeTrainingCheckpoint(value *bridgeTrainingCheckpoint) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Source.Kind() != artifact.KindModel || value.Target.Kind() != artifact.KindModel ||
		value.Source == value.Target || value.Dataset.Kind() != artifact.KindDataset ||
		value.TrainingPolicy.Kind() != artifact.KindProfile || value.ParentBridge.Kind() != artifact.KindAdapter ||
		value.Bridge.Kind() != artifact.KindAdapter || value.ParentBridge == value.Bridge || len(value.Weights) == 0 ||
		len(value.Metrics) == 0 {
		return errors.New("bridge training: invalid checkpoint authority")
	}
	if err := optimizer.ValidateState(value.Optimizer, value.Optimizer.PlanIdentity, len(value.Weights)); err != nil {
		return err
	}
	for _, weight := range value.Weights {
		if !checked.Finite32(weight) {
			return errors.New("bridge training: checkpoint contains non-finite weights")
		}
	}
	for _, metric := range value.Metrics {
		if metric.Name == "" || !checked.Finite64(metric.Value) ||
			metric.Direction != runrecord.DirectionMinimize && metric.Direction != runrecord.DirectionMaximize {
			return errors.New("bridge training: checkpoint contains invalid metrics")
		}
	}
	return nil
}
