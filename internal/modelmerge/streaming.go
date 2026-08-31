package modelmerge

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/composition"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

const streamingSingleShardName = "model.safetensors"

// StreamingSource binds one local Safetensors directory to the exact model
// identity named by an offline execution plan.
type StreamingSource struct {
	Model     artifact.ID
	Directory string
	Shards    []StreamingShard
}

// StreamingShard binds one repository-relative shard name to its exact stored
// byte identity. Sources without bindings retain the generic low-level API;
// authority-bearing callers provide the complete set.
type StreamingShard struct {
	Name     string
	Artifact artifact.ID
}

// StreamingResult reports the derived execution envelope of a published
// offline model directory.
type StreamingResult struct {
	Destination       string
	TensorCount       int
	PeakResidentBytes uint64
}

// ExecuteStreaming consumes a validated offline plan tensor-by-tensor and
// atomically publishes deterministic Safetensors shards. It refuses source
// identities and catalogs that differ from the compiled plan.
func ExecuteStreaming(
	ctx context.Context,
	plan composition.OfflineTensorExecutionPlan,
	sources []StreamingSource,
	destination string,
) (StreamingResult, error) {
	if ctx == nil {
		return StreamingResult{}, errors.New("model merge: streaming context is absent")
	}
	if err := ctx.Err(); err != nil {
		return StreamingResult{}, err
	}
	if err := plan.ValidateIdentity(); err != nil {
		return StreamingResult{}, fmt.Errorf("model merge: streaming plan identity differs: %w", err)
	}
	if plan.Operator == composition.OfflineArtifactTaskArithmetic {
		for _, operation := range plan.Operations {
			if !strings.EqualFold(operation.Storage, "F32") {
				return StreamingResult{}, errors.New("model merge: task arithmetic requires F32 tensors")
			}
		}
	}
	opened, err := openStreamingSources(plan, sources)
	if err != nil {
		return StreamingResult{}, err
	}
	if err := validateOpenedStreamingShards(plan, sources, opened); err != nil {
		return StreamingResult{}, errors.Join(err, closeStreamingSources(opened))
	}
	result, executeErr := executeOpenedStreaming(ctx, plan, opened, destination)
	return result, errors.Join(executeErr, closeStreamingSources(opened))
}

func executeOpenedStreaming(
	ctx context.Context,
	plan composition.OfflineTensorExecutionPlan,
	opened []*safetensors.Source,
	destination string,
) (StreamingResult, error) {
	outputPlan, err := compileStreamingOutput(plan, opened)
	if err != nil {
		return StreamingResult{}, err
	}
	writer, err := safetensors.NewStreamingWriter(destination, outputPlan)
	if err != nil {
		return StreamingResult{}, err
	}
	tracker := residencyTracker{limit: plan.PeakResidentBytes}
	for _, operation := range plan.Operations {
		if err := ctx.Err(); err != nil {
			return StreamingResult{}, err
		}
		if writer.Completed(operation.Name) {
			continue
		}
		switch plan.Operator {
		case composition.OfflineArtifactExactPassthrough:
			payload, allocateErr := tracker.allocate(operation.OutputBytes)
			if allocateErr != nil {
				return StreamingResult{}, allocateErr
			}
			_, readErr := io.ReadFull(opened[tensor.FirstOffset].Tensors[operation.Name].Reader(), payload)
			if readErr == nil {
				readErr = writer.WriteTensor(operation.Name, operation.OutputBytes, bytes.NewReader(payload))
			}
			tracker.release(operation.OutputBytes)
			if readErr != nil {
				return StreamingResult{}, readErr
			}
		case composition.OfflineArtifactTaskArithmetic:
			payload, combineErr := combineStreamingF32(plan, opened, operation, &tracker)
			if combineErr != nil {
				return StreamingResult{}, combineErr
			}
			if err := writer.WriteTensor(operation.Name, operation.OutputBytes, bytes.NewReader(payload)); err != nil {
				return StreamingResult{}, err
			}
			tracker.release(operation.OutputBytes)
		default:
			return StreamingResult{}, errors.New("model merge: streaming operator is unsupported")
		}
	}
	if err := writer.Finalize(); err != nil {
		return StreamingResult{}, err
	}
	return StreamingResult{
		Destination: destination, TensorCount: len(plan.Operations), PeakResidentBytes: tracker.peak,
	}, nil
}

func openStreamingSources(
	plan composition.OfflineTensorExecutionPlan,
	provided []StreamingSource,
) ([]*safetensors.Source, error) {
	if len(provided) != len(plan.Inputs) {
		return nil, errors.New("model merge: streaming source count differs from plan")
	}
	byModel := make(map[artifact.ID]string, len(provided))
	for _, source := range provided {
		if source.Model.Kind() != artifact.KindModel || strings.TrimSpace(source.Directory) == "" {
			return nil, errors.New("model merge: streaming source is invalid")
		}
		if _, duplicate := byModel[source.Model]; duplicate {
			return nil, errors.New("model merge: streaming source model is duplicated")
		}
		byModel[source.Model] = source.Directory
	}
	opened := make([]*safetensors.Source, len(plan.Inputs))
	for index, input := range plan.Inputs {
		directory, found := byModel[input.Model]
		if !found {
			_ = closeStreamingSources(opened)
			return nil, fmt.Errorf("model merge: streaming source %d identity is absent", index)
		}
		source, err := safetensors.OpenSource(directory)
		if err != nil {
			_ = closeStreamingSources(opened)
			return nil, fmt.Errorf("model merge: open streaming source %d: %w", index, err)
		}
		opened[index] = source
	}
	return opened, nil
}

func validateOpenedStreamingShards(
	plan composition.OfflineTensorExecutionPlan,
	provided []StreamingSource,
	opened []*safetensors.Source,
) error {
	byModel := make(map[artifact.ID][]StreamingShard, len(provided))
	for _, source := range provided {
		bindings := slices.Clone(source.Shards)
		slices.SortFunc(bindings, func(left, right StreamingShard) int {
			return strings.Compare(left.Name, right.Name)
		})
		byModel[source.Model] = bindings
	}
	for index, input := range plan.Inputs {
		expected := byModel[input.Model]
		if len(expected) == 0 {
			continue
		}
		actual, err := opened[index].ShardDigests()
		if err != nil {
			return err
		}
		if len(actual) != len(expected) {
			return fmt.Errorf("model merge: source %d shard closure differs", index)
		}
		for shardIndex, digest := range actual {
			binding := expected[shardIndex]
			id, err := artifact.NewID(artifact.KindTensorSet, digest.Digest)
			if err != nil || binding.Name != digest.Name || binding.Artifact != id {
				return errors.Join(fmt.Errorf("model merge: source %d shard %q bytes differ", index, digest.Name), err)
			}
			if shardIndex > 0 && binding.Name == expected[shardIndex-1].Name {
				return fmt.Errorf("model merge: source %d shard binding is duplicated", index)
			}
		}
	}
	return nil
}

func compileStreamingOutput(
	plan composition.OfflineTensorExecutionPlan,
	sources []*safetensors.Source,
) (safetensors.StreamingPlan, error) {
	if len(sources) != len(plan.Inputs) {
		return safetensors.StreamingPlan{}, errors.New("model merge: opened source count differs")
	}
	for sourceIndex, source := range sources {
		if source == nil || len(source.Tensors) != len(plan.Operations) {
			return safetensors.StreamingPlan{}, fmt.Errorf("model merge: source %d tensor inventory differs", sourceIndex)
		}
	}
	shards := make([]safetensors.StreamingShardSpec, len(plan.Shards))
	for shardIndex, shard := range plan.Shards {
		shards[shardIndex].Name = streamingShardName(shardIndex, len(plan.Shards))
		shards[shardIndex].Tensors = make([]safetensors.StreamingTensorSpec, len(shard.Tensors))
		for tensorIndex, name := range shard.Tensors {
			operationIndex, found := tensorIndexForName(plan.Operations, name)
			if !found {
				return safetensors.StreamingPlan{}, fmt.Errorf("model merge: shard tensor %q is absent from operations", name)
			}
			operation := plan.Operations[operationIndex]
			reference, err := validateStreamingTensor(operation, sources)
			if err != nil {
				return safetensors.StreamingPlan{}, err
			}
			shards[shardIndex].Tensors[tensorIndex] = safetensors.StreamingTensorSpec{
				Name: name, DType: reference.DType, Shape: slices.Clone(reference.Shape),
			}
		}
	}
	return safetensors.StreamingPlan{Shards: shards}, nil
}

func validateStreamingTensor(
	operation composition.OfflineTensorOperation,
	sources []*safetensors.Source,
) (safetensors.Tensor, error) {
	var reference safetensors.Tensor
	for sourceIndex, source := range sources {
		candidate, found := source.Tensors[operation.Name]
		if !found || uint64(candidate.Size()) != operation.SourceBytes[sourceIndex] ||
			!strings.EqualFold(candidate.DType, operation.Storage) {
			return safetensors.Tensor{}, fmt.Errorf("model merge: tensor %q source %d differs from plan", operation.Name, sourceIndex)
		}
		if sourceIndex == tensor.FirstOffset {
			reference = candidate
			continue
		}
		if candidate.DType != reference.DType || !slices.Equal(candidate.Shape, reference.Shape) {
			return safetensors.Tensor{}, fmt.Errorf("model merge: tensor %q source geometry differs", operation.Name)
		}
	}
	if uint64(reference.Size()) != operation.OutputBytes {
		return safetensors.Tensor{}, fmt.Errorf("model merge: tensor %q output extent differs", operation.Name)
	}
	return reference, nil
}

func combineStreamingF32(
	plan composition.OfflineTensorExecutionPlan,
	sources []*safetensors.Source,
	operation composition.OfflineTensorOperation,
	tracker *residencyTracker,
) ([]byte, error) {
	output, err := tracker.allocate(operation.OutputBytes)
	if err != nil {
		return nil, err
	}
	if _, err := io.ReadFull(sources[tensor.FirstOffset].Tensors[operation.Name].Reader(), output); err != nil {
		return nil, fmt.Errorf("model merge: read tensor %q source 0: %w", operation.Name, err)
	}
	if err := scaleStreamingF32(output, plan.Inputs[tensor.FirstOffset].Coefficient); err != nil {
		return nil, fmt.Errorf("model merge: tensor %q: %w", operation.Name, err)
	}
	for sourceIndex := tensor.SingletonExtent; sourceIndex < len(sources); sourceIndex++ {
		scratch, allocateErr := tracker.allocate(operation.SourceBytes[sourceIndex])
		if allocateErr != nil {
			return nil, allocateErr
		}
		_, readErr := io.ReadFull(sources[sourceIndex].Tensors[operation.Name].Reader(), scratch)
		if readErr == nil {
			readErr = accumulateStreamingF32(output, scratch, plan.Inputs[sourceIndex].Coefficient)
		}
		tracker.release(operation.SourceBytes[sourceIndex])
		if readErr != nil {
			return nil, fmt.Errorf("model merge: tensor %q source %d: %w", operation.Name, sourceIndex, readErr)
		}
	}
	return output, nil
}

func scaleStreamingF32(values []byte, coefficient float64) error {
	for offset := range len(values) / binaryschema.Uint32Bytes {
		position := offset * binaryschema.Uint32Bytes
		value := float64(math.Float32frombits(binary.LittleEndian.Uint32(values[position:]))) * coefficient
		converted := float32(value)
		if !checked.Finite32(converted) {
			return errors.New("task arithmetic produced a non-finite value")
		}
		binary.LittleEndian.PutUint32(values[position:], math.Float32bits(converted))
	}
	return nil
}

func accumulateStreamingF32(destination, source []byte, coefficient float64) error {
	if len(destination) != len(source) {
		return errors.New("task arithmetic source extent differs")
	}
	for offset := range len(destination) / binaryschema.Uint32Bytes {
		position := offset * binaryschema.Uint32Bytes
		left := math.Float32frombits(binary.LittleEndian.Uint32(destination[position:]))
		right := math.Float32frombits(binary.LittleEndian.Uint32(source[position:]))
		converted := float32(float64(left) + coefficient*float64(right))
		if !checked.Finite32(converted) {
			return errors.New("task arithmetic produced a non-finite value")
		}
		binary.LittleEndian.PutUint32(destination[position:], math.Float32bits(converted))
	}
	return nil
}

type residencyTracker struct {
	current uint64
	peak    uint64
	limit   uint64
}

func (tracker *residencyTracker) allocate(size uint64) ([]byte, error) {
	next, ok := checked.Add64(tracker.current, size)
	hostSize := int(size)
	if !ok || next > tracker.limit || hostSize < 0 || uint64(hostSize) != size {
		return nil, errors.New("model merge: tensor residency exceeds compiled plan")
	}
	tracker.current = next
	if next > tracker.peak {
		tracker.peak = next
	}
	return make([]byte, hostSize), nil
}

func (tracker *residencyTracker) release(size uint64) {
	tracker.current -= size
}

func streamingShardName(index, count int) string {
	if count == tensor.SingletonExtent {
		return streamingSingleShardName
	}
	return fmt.Sprintf("model-%d-of-%d.safetensors", index+tensor.SingletonExtent, count)
}

func tensorIndexForName(operations []composition.OfflineTensorOperation, name string) (int, bool) {
	return slices.BinarySearchFunc(operations, name, func(operation composition.OfflineTensorOperation, target string) int {
		return strings.Compare(operation.Name, target)
	})
}

func closeStreamingSources(sources []*safetensors.Source) error {
	var result error
	for _, source := range sources {
		result = errors.Join(result, source.Close())
	}
	return result
}
