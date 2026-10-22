package inference

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

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
	expected := 0
	previousEnd := uint32(0)
	for index, block := range ordered {
		if block.Start >= block.End || block.End > uint32(tokenCount) || index > 0 && block.Start < previousEnd {
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
	if scale != 1 {
		for index := range activation.Data {
			activation.Data[index] *= scale
		}
	}
	return applyEmbeddingOverrides(activation, overrides)
}

func applyEmbeddingOverrides(activation *reference.Value, overrides []EmbeddingOverride) error {
	if len(overrides) == 0 {
		return nil
	}
	if activation == nil {
		return errors.New("inference: token embeddings have invalid shape")
	}
	width, tokens, valid := activation.MatrixExtents()
	if !valid {
		return errors.New("inference: token embeddings have invalid storage")
	}
	seen := make(map[uint32]struct{}, len(overrides))
	for overrideIndex, override := range overrides {
		if uint64(override.TokenIndex) >= uint64(tokens) {
			return fmt.Errorf(
				"inference: embedding override %d token index %d is out of range for %d tokens",
				overrideIndex, override.TokenIndex, tokens,
			)
		}
		if _, duplicate := seen[override.TokenIndex]; duplicate {
			return fmt.Errorf("inference: duplicate embedding override for token index %d", override.TokenIndex)
		}
		seen[override.TokenIndex] = struct{}{}
		if len(override.Embedding) != width {
			return fmt.Errorf(
				"inference: embedding override %d width %d differs from model width %d",
				overrideIndex, len(override.Embedding), width,
			)
		}
		for valueIndex, value := range override.Embedding {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
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
	deviceFeeds *tensor.InputBindings[driver.DevicePtr],
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
			deviceFeeds.Add(node, pointer)
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
	if len(nodes) != 5 {
		return errors.New("inference: CogVLM visual graph weight set is incomplete")
	}
	for _, node := range nodes {
		if node == nil {
			return errors.New("inference: CogVLM visual graph weight is nil")
		}
	}
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = nodes[0]
	weights.AttentionOutput = nodes[1]
	weights.FeedForwardGate = nodes[2]
	weights.FeedForwardUp = nodes[3]
	weights.FeedForwardDown = nodes[4]
	return nil
}

func validateDeepstackInputs(
	spec model.Spec,
	streams uint32,
	tokens int,
	inputs []reference.Value,
) error {
	if len(inputs) == 0 {
		return nil
	}
	if streams == 0 {
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
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
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
	if len(blocks) == 0 {
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
		ids[index] = -1
	}
	var previousEnd uint32
	for index, block := range blocks {
		if block.Start >= block.End || uint64(block.End) > uint64(tokens) {
			return nil, fmt.Errorf("inference: attention block %d range [%d,%d) is invalid for %d tokens", index, block.Start, block.End, tokens)
		}
		if index > 0 && block.Start < previousEnd {
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
	if len(inputs) == 0 || source == model.DeepstackSourceNone {
		return nil
	}
	if source == model.DeepstackSourceBase {
		return &base
	}
	index := int(source)
	if index < 0 || index >= len(inputs) {
		return nil
	}
	return &inputs[index]
}

// deviceProjectable reports whether a device prompt reads these inputs as the host's forward does: rows put in place
// of their tokens', multi-axis positions, and deepstack streams added at their layers. Attention and visual expert
// blocks, per-layer embeddings, and rows scaled or mapped from a base are read on the host alone.
func (r *Runner) deviceProjectable(inputs ProjectedInputs) bool {
	program := r.program.Model.ProjectedInput()
	return len(inputs.BidirectionalAttentionBlocks) == 0 && len(inputs.VisualExpertBlocks) == 0 && !program.PerLayerEmbeddings &&
		program.Overrides != model.EmbeddingOverrideVisualSpan && program.Overrides != model.EmbeddingOverrideRawScaled &&
		program.Overrides != model.EmbeddingOverrideMappedBase
}

// span gives the projected inputs of count tokens from start, each indexed from start: a chunk of a prompt the
// device reads in parts. Attention and visual expert blocks, which no device prompt reads, are not carried.
func (inputs ProjectedInputs) span(start, count int) *ProjectedInputs {
	result := &ProjectedInputs{}
	for _, override := range inputs.EmbeddingOverrides {
		if index := int(override.TokenIndex) - start; index >= 0 && index < count {
			result.EmbeddingOverrides = append(result.EmbeddingOverrides, EmbeddingOverride{TokenIndex: uint32(index), Embedding: override.Embedding})
		}
	}
	if inputs.MultiAxisPositions != nil {
		var positions MultiAxisPositions
		for axis, values := range inputs.MultiAxisPositions {
			positions[axis] = values[start : start+count]
		}
		result.MultiAxisPositions = &positions
	}
	for _, stream := range inputs.DeepstackEmbeddings {
		width, _, _ := stream.MatrixExtents()
		shape := tensor.MustShape(uint64(width), uint64(count))
		result.DeepstackEmbeddings = append(result.DeepstackEmbeddings, reference.Value{Shape: shape, Data: stream.Data[start*width : (start+count)*width]})
	}
	return result
}

// overrideDeviceRows puts projected rows in place of their tokens' embedding rows on the device: the rows times a
// mask that is zero at a replaced token and one elsewhere, plus the replacing rows laid at their tokens, each exact.
func overrideDeviceRows(
	builder *tensor.Builder, prefix string, rows *tensor.Tensor, overrides []EmbeddingOverride, tokens int,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	width := int(rows.Shape.Dims[0])
	keep := make([]float32, tokens)
	for index := range keep {
		keep[index] = 1
	}
	placed := make([]float32, width*tokens)
	for _, override := range overrides {
		keep[override.TokenIndex] = 0
		copy(placed[int(override.TokenIndex)*width:], override.Embedding)
	}
	keepShape, placedShape := tensor.MustShape(1, uint64(tokens)), tensor.MustShape(uint64(width), uint64(tokens))
	keepInput := builder.Input(prefix+"projected.keep", dtype.F32, keepShape)
	placedInput := builder.Input(prefix+"projected.rows", dtype.F32, placedShape)
	hostFeeds[keepInput] = reference.Value{Shape: keepShape, Data: keep}
	hostFeeds[placedInput] = reference.Value{Shape: placedShape, Data: placed}
	return builder.Add(builder.Multiply(rows, keepInput), placedInput)
}

// addDeepstackRows adds to a layer's rows the deepstack stream source names, fed from the host; none adds nothing.
func addDeepstackRows(
	builder *tensor.Builder, name string, rows *tensor.Tensor, source model.DeepstackSource, streams []reference.Value,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	stream := deepstackInputForLayer(source, reference.Value{}, streams)
	if stream == nil || len(stream.Data) == 0 {
		return rows
	}
	input := builder.Input(name, dtype.F32, stream.Shape)
	hostFeeds[input] = *stream
	return builder.Add(rows, input)
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
