//overgo:runtime-inputs caller

package adaptertrain

import (
	"errors"
	"maps"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

const inputProjectionTensor = "input-projection"

// InputProjectionBinding identifies the exact base recipe and declared target
// transform for a projection immediately before that recipe's final output.
// Tensor metadata binds placement; no model-family names enter execution.
type InputProjectionBinding struct {
	BaseRecipe      artifact.ID
	TargetTransform artifact.ID
}

func (binding InputProjectionBinding) metadata() (map[string]string, error) {
	if binding.BaseRecipe.Kind() != artifact.KindRecipe || binding.TargetTransform.Kind() != artifact.KindProfile {
		return nil, errors.New("input projection: base recipe and target transform are required")
	}
	return map[string]string{
		"schema": "overgo/input-projection/v1", "placement": "before-final-output",
		"base_recipe": binding.BaseRecipe.String(), "target_transform": binding.TargetTransform.String(),
	}, nil
}

// Program returns the adapter's compiled shared objective and parameter plan.
func (m *LinearCTC) Program() trainingprogram.TrainingProgram { return m.program }

// OptimizerSnapshot returns exact optimizer/scheduler progress
// only at initialization or a successfully completed update boundary.
func (m *LinearCTC) OptimizerSnapshot() (optimizer.State, error) {
	if m == nil || !m.checkpointReady || !finiteProjectionValues(m.weights) {
		return optimizer.State{}, errors.New("linear CTC: checkpoint boundary is unavailable")
	}
	return m.optimizer.Snapshot(), nil
}

// Restore validates the entire replacement before changing parameters. It
// invalidates unfinished executions and restores the shared Muon scheduler.
func (m *LinearCTC) Restore(weights []float32, state optimizer.State) error {
	if m == nil || len(weights) != len(m.weights) || !finiteProjectionValues(weights) {
		return errors.New("linear CTC: restored weights differ in shape or are non-finite")
	}
	if err := m.optimizer.Restore(state); err != nil {
		return err
	}
	copy(m.weights, weights)
	clear(m.gradients)
	m.phase, m.checkpointReady = "", true
	return nil
}

// SaveWeights writes only the trainable projection to the shared Safetensors
// checkpoint filename. The caller uses trainingprogram.PublishCheckpoint to
// publish weights and optimizer/data state together without copying the base.
func (m *LinearCTC) SaveWeights(directory string, binding InputProjectionBinding) error {
	metadata, err := binding.metadata()
	if err != nil {
		return err
	}
	if m == nil || !m.checkpointReady || !finiteProjectionValues(m.weights) {
		return errors.New("linear CTC: checkpoint boundary is unavailable")
	}
	return safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights),
		map[string][]float32{inputProjectionTensor: m.weights},
		map[string][]int{inputProjectionTensor: {m.width, m.width}}, metadata)
}

// LoadInputProjection reads one immutable projection after exact placement,
// recipe, transform, shape, dtype and opened-shard digest checks. memoryBytes
// bounds the returned numeric allocation; no optimizer or frozen weights load.
func LoadInputProjection(directory string, binding InputProjectionBinding, weightsID artifact.ID, width int, memoryBytes uint64) ([]float32, error) {
	metadata, err := binding.metadata()
	if err != nil {
		return nil, err
	}
	count, ok := checked.MulInt(width, width)
	bytes, bytesOK := checked.Bytes(uint64(count), binaryschema.Uint32Bytes)
	if width <= 0 || !ok || !bytesOK || bytes > memoryBytes || weightsID.Kind() != artifact.KindTensorSet {
		return nil, errors.New("input projection: invalid shape, identity or memory admission")
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	tensor, found := source.Tensors[inputProjectionTensor]
	if !source.ContainsOnlyShard(trainingprogram.CheckpointWeights) || len(source.Tensors) != 1 || !found ||
		tensor.DType != "F32" || len(tensor.Shape) != 2 || tensor.Shape[0] != uint64(width) || tensor.Shape[1] != uint64(width) ||
		!maps.Equal(source.Metadata[trainingprogram.CheckpointWeights], metadata) {
		return nil, errors.New("input projection: tensor or exact placement binding differs")
	}
	digests, err := source.ShardDigests()
	if err != nil {
		return nil, err
	}
	if len(digests) != 1 {
		return nil, errors.New("input projection: opened shard count differs")
	}
	actual, err := artifact.NewID(artifact.KindTensorSet, digests[0].Digest)
	if err != nil || actual != weightsID {
		return nil, errors.New("input projection: opened weights identity differs")
	}
	weights, err := safetensors.ReadF32(tensor)
	if err != nil {
		return nil, err
	}
	if !finiteProjectionValues(weights) {
		return nil, errors.New("input projection: non-finite weights")
	}
	return weights, nil
}
