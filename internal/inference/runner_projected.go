package inference

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func validateVisualExpertBlocks(tokenCount int, blocks []AttentionBlock, overrides []EmbeddingOverride) ([]AttentionBlock, error) {
	ordered := slices.Clone(blocks)
	slices.SortFunc(ordered, func(a, b AttentionBlock) int { return cmp.Compare(a.Start, b.Start) })
	var expected int
	var previousEnd uint32
	for index, block := range ordered {
		if block.Start >= block.End || block.End > uint32(tokenCount) || checked.Nonzero(index) && block.Start < previousEnd {
			return nil, fmt.Errorf("inference: invalid CogVLM visual expert block [%d,%d)", block.Start, block.End)
		}
		expected += int(block.End - block.Start)
		previousEnd = block.End
	}
	if len(overrides) != expected {
		return nil, fmt.Errorf("inference: CogVLM visual blocks cover %d tokens but have %d embeddings", expected, len(overrides))
	}
	seen := make(map[uint32]struct{}, len(overrides))
	for _, override := range overrides {
		inside := false
		for _, block := range ordered {
			if override.TokenIndex >= block.Start && override.TokenIndex < block.End {
				inside = true
				break
			}
		}
		if !inside {
			return nil, fmt.Errorf("inference: CogVLM visual embedding token %d is outside visual blocks", override.TokenIndex)
		}
		if _, ok := seen[override.TokenIndex]; ok {
			return nil, fmt.Errorf("inference: duplicate CogVLM visual embedding for token index %d", override.TokenIndex)
		}
		seen[override.TokenIndex] = struct{}{}
	}
	return ordered, nil
}

func applyScaledRawEmbeddingOverrides(activation *reference.Value, overrides []EmbeddingOverride, scale float32) error {
	if activation == nil {
		return errors.New("inference: embedding activation is nil")
	}
	if !checked.Equal(scale, float32(tensor.SingletonExtent)) {
		for index := range activation.Data {
			activation.Data[index] *= scale
		}
	}
	return applyEmbeddingOverrides(activation, overrides)
}

func applyEmbeddingOverrides(activation *reference.Value, overrides []EmbeddingOverride) error {
	if !checked.Nonzero(len(overrides)) {
		return nil
	}
	if activation == nil {
		return errors.New("inference: token embeddings have invalid shape")
	}
	width, tokens, validActivation := activation.MatrixExtents()
	if !validActivation {
		return errors.New("inference: token embeddings have invalid storage")
	}
	seen := make(map[uint32]struct{}, len(overrides))
	for overrideIndex, override := range overrides {
		if !checked.Less64(uint64(override.TokenIndex), uint64(tokens)) {
			return fmt.Errorf(
				"inference: embedding override %d token index %d is out of range for %d tokens",
				overrideIndex, override.TokenIndex, tokens,
			)
		}
		if _, duplicate := seen[override.TokenIndex]; duplicate {
			return fmt.Errorf("inference: duplicate embedding override for token index %d", override.TokenIndex)
		}
		seen[override.TokenIndex] = struct{}{}
		if !checked.Equal(len(override.Embedding), width) {
			return fmt.Errorf(
				"inference: embedding override %d width %d differs from model width %d",
				overrideIndex, len(override.Embedding), width,
			)
		}
		for valueIndex, value := range override.Embedding {
			if !checked.Finite32(value) {
				return fmt.Errorf(
					"inference: embedding override %d contains non-finite value at %d",
					overrideIndex, valueIndex,
				)
			}
		}
		start := int(override.TokenIndex) * width
		copy(activation.Data[start:start+width], override.Embedding)
	}
	return nil
}

func validateCogVLMVisualOverrides(tokenCount int, overrides []EmbeddingOverride) error {
	if len(overrides) != tokenCount {
		return fmt.Errorf(
			"inference: CogVLM visual mode requires one projected embedding per token; got %d for %d tokens",
			len(overrides), tokenCount,
		)
	}
	seen := make([]bool, tokenCount)
	for _, override := range overrides {
		if int(override.TokenIndex) >= tokenCount {
			return fmt.Errorf(
				"inference: CogVLM visual embedding token index %d is out of range for %d tokens",
				override.TokenIndex, tokenCount,
			)
		}
		if seen[override.TokenIndex] {
			return fmt.Errorf(
				"inference: duplicate CogVLM visual embedding for token index %d",
				override.TokenIndex,
			)
		}
		seen[override.TokenIndex] = true
	}
	return nil
}

func (r *Runner) applyCogVLMVisualWeights(
	ctx context.Context,
	builder *tensor.Builder,
	info model.LayerWeights,
	weights *model.LayerGraphWeights,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) error {
	if r.program.Model.ProjectedInput().Overrides != model.EmbeddingOverrideVisualSpan {
		return errors.New("inference: visual expert weights require CogVLM architecture")
	}
	if weights == nil {
		return errors.New("inference: CogVLM graph weights are nil")
	}
	infos := []*gguf.TensorInfo{
		info.VisualAttentionQKV,
		info.VisualAttentionOutput,
		info.VisualFeedForwardGate,
		info.VisualFeedForwardUp,
		info.VisualFeedForwardDown,
	}
	nodes := make([]*tensor.Tensor, len(infos))
	for index, tensorInfo := range infos {
		if tensorInfo == nil {
			return errors.New("inference: CogVLM visual expert catalog is incomplete")
		}
		if r.hasPreloadedWeights() {
			node, pointer, err := r.deviceInput(builder, *tensorInfo)
			if err != nil {
				return err
			}
			nodes[index] = node
			deviceFeeds[node] = pointer
			continue
		}
		value, err := r.hostTensor(ctx, *tensorInfo)
		if err != nil {
			return err
		}
		node := builder.Input(tensorInfo.Name, dtype.F32, value.Shape)
		nodes[index] = node
		hostFeeds[node] = value
	}
	if err := selectCogVLMVisualGraphWeights(weights, nodes); err != nil {
		return err
	}
	return builder.Err()
}

func selectCogVLMVisualGraphWeights(
	weights *model.LayerGraphWeights,
	nodes []*tensor.Tensor,
) error {
	if weights == nil {
		return errors.New("inference: CogVLM graph weights are nil")
	}
	if len(nodes) != tensor.MaxDimensions+tensor.SingletonExtent {
		return errors.New("inference: CogVLM visual graph weight set is incomplete")
	}
	for _, node := range nodes {
		if node == nil {
			return errors.New("inference: CogVLM visual graph weight is nil")
		}
	}
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = nodes[tensor.FirstOffset]
	weights.AttentionOutput = nodes[tensor.SingletonExtent]
	weights.FeedForwardGate = nodes[tensor.PairedExtent]
	weights.FeedForwardUp = nodes[tensor.TripleExtent]
	weights.FeedForwardDown = nodes[tensor.MaxDimensions]
	return nil
}

func validateDeepstackInputs(
	spec model.Spec,
	streams uint32,
	tokens int,
	inputs []reference.Value,
) error {
	if !checked.Nonzero(len(inputs)) {
		return nil
	}
	if !checked.Nonzero(streams) {
		return errors.New("inference: model does not support deepstack embeddings")
	}
	if len(inputs) != int(streams) {
		return fmt.Errorf(
			"inference: received %d deepstack streams, need %d",
			len(inputs), streams,
		)
	}
	want := tensor.MustShape(uint64(spec.EmbeddingLength), uint64(tokens))
	for streamIndex, stream := range inputs {
		if !stream.Shape.Equal(want) || len(stream.Data) != int(spec.EmbeddingLength)*tokens {
			return fmt.Errorf("inference: deepstack stream %d has invalid shape", streamIndex)
		}
		for _, value := range stream.Data {
			if !checked.Finite32(value) {
				return fmt.Errorf("inference: deepstack stream %d contains non-finite value", streamIndex)
			}
		}
	}
	return nil
}

func projectedAttentionBlockIDs(
	policy model.AttentionBlockPolicy,
	tokens int,
	hasCache bool,
	blocks []AttentionBlock,
) ([]float32, error) {
	if !checked.Nonzero(len(blocks)) {
		return nil, nil
	}
	if policy != model.AttentionBlocksUncached {
		return nil, errors.New("inference: model does not support bidirectional attention blocks")
	}
	if hasCache {
		return nil, errors.New("inference: bidirectional attention blocks require an uncached prompt prefill")
	}
	ids := make([]float32, tokens)
	for index := range ids {
		ids[index] = model.AbsentAttentionBlock()
	}
	var previousEnd uint32
	for index, block := range blocks {
		if block.Start >= block.End || uint64(block.End) > uint64(tokens) {
			return nil, fmt.Errorf("inference: attention block %d range [%d,%d) is invalid for %d tokens", index, block.Start, block.End, tokens)
		}
		if checked.Nonzero(index) && block.Start < previousEnd {
			return nil, fmt.Errorf("inference: attention block %d overlaps or precedes the prior block", index)
		}
		for token := block.Start; token < block.End; token++ {
			ids[token] = float32(index)
		}
		previousEnd = block.End
	}
	return ids, nil
}

func deepstackInputForLayer(
	source model.DeepstackSource,
	base reference.Value,
	inputs []reference.Value,
) *reference.Value {
	if !checked.Nonzero(len(inputs)) || source == model.DeepstackSourceNone {
		return nil
	}
	if source == model.DeepstackSourceBase {
		return &base
	}
	index := int(source)
	if !checked.ValidIndex(index, len(inputs)) {
		return nil
	}
	return &inputs[index]
}

func addDeepstackEmbedding(activation, deepstack reference.Value) (reference.Value, error) {
	if !activation.Shape.Equal(deepstack.Shape) || len(activation.Data) != len(deepstack.Data) {
		return reference.Value{}, errors.New("activation and deepstack shapes differ")
	}
	output := reference.Value{
		Shape: activation.Shape,
		Data:  slices.Clone(activation.Data),
	}
	for index, value := range deepstack.Data {
		output.Data[index] += value
	}
	return output, nil
}
