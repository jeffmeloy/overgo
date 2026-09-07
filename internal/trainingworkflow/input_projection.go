package trainingworkflow

import (
	"errors"
	"slices"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
)

// PublishInputProjection publishes an adapter-only checkpoint through the
// universal format. The caller supplies actual dataset, stream, RNG and lineage
// authorities; optimizer progress is taken from the completed live adapter.
func PublishInputProjection(target string, adapter *adaptertrain.LinearCTC, binding adaptertrain.InputProjectionBinding, spec trainingprogram.CheckpointSpec) (trainingprogram.Checkpoint, error) {
	if adapter == nil || spec.Program != adapter.Program().ID() || !slices.Contains(spec.Processors, binding.TargetTransform) {
		return trainingprogram.Checkpoint{}, errors.New("training workflow: input projection authority differs")
	}
	state, err := adapter.OptimizerSnapshot()
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	spec.Optimizer, spec.ParameterCount = state, len(state.Momentum)
	spec.Lineage = slices.Clone(spec.Lineage)
	for _, id := range []artifact.ID{binding.BaseRecipe, binding.TargetTransform} {
		parent := trainingprogram.LineageParent{Artifact: id, Relation: artifact.RelationDependsOn}
		if !slices.Contains(spec.Lineage, parent) {
			spec.Lineage = append(spec.Lineage, parent)
		}
	}
	return trainingprogram.PublishCheckpoint(target, spec, func(stage string) error {
		return adapter.SaveWeights(stage, binding)
	})
}

// RestoreInputProjection validates the universal resume contract, exact RNG
// state and file binding before restoring the adapter. The returned stream
// boundary is consumed by trainingdata.NewStream, not by a second sampler.
func RestoreInputProjection(directory string, adapter *adaptertrain.LinearCTC, binding adaptertrain.InputProjectionBinding, runPlan trainingprogram.TrainingRunPlan, authority trainingprogram.ResumeAuthority, expectedRNG []trainingprogram.RNGState) (trainingprogram.Checkpoint, error) {
	if adapter == nil || runPlan.Program().ID() != adapter.Program().ID() || !slices.Contains(runPlan.Processors(), binding.TargetTransform) {
		return trainingprogram.Checkpoint{}, errors.New("training workflow: input projection resume program differs")
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(directory)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	if err := trainingprogram.ValidateResume(runPlan, checkpoint, authority); err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	if !slices.Contains(checkpoint.Lineage, trainingprogram.LineageParent{Artifact: binding.BaseRecipe, Relation: artifact.RelationDependsOn}) ||
		len(expectedRNG) != len(checkpoint.RNG) {
		return trainingprogram.Checkpoint{}, errors.New("training workflow: projection base or RNG authority differs")
	}
	for index, state := range expectedRNG {
		if !slices.Contains(checkpoint.RNG, state) || slices.Contains(expectedRNG[:index], state) {
			return trainingprogram.Checkpoint{}, errors.New("training workflow: projection RNG state differs")
		}
	}
	width := adapter.Program().Parameters()[0].Rows
	weights, err := adaptertrain.LoadInputProjection(directory, binding, checkpoint.Weights, width, adapter.StorageBytes())
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	if checkpoint.ParameterCount != len(weights) {
		return trainingprogram.Checkpoint{}, errors.New("training workflow: projection parameter count differs")
	}
	if err := adapter.Restore(weights, checkpoint.Optimizer); err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	return checkpoint, nil
}
