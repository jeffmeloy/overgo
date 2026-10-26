package model

import (
	"errors"
	"slices"

	"overgo/internal/tensor"
)

// QSABinding is a QSA layer's cells for one step (tensor.QSAAttention): the storage's tensors the step appends to in
// place, its capacity and the cell of the step's first token. The layer's fixed state is its cache pair: PastKey the
// indexer's tail, PastValue its spare key.
type QSABinding struct {
	State     tensor.QSAState
	Capacity  uint32
	FirstCell uint32
}

// HoldsQSA reports whether a layer is a block-sparse gated attention layer at the geometry Strata's QSA kernels are
// compiled for, which a runner may build as Strata's QSA layer over its own state (QSABinding): its heads, its
// K/V heads and their width, its indexer's heads, width, budget and block, its rotated dims, base and axes.
func (s Spec) HoldsQSA(plan LayerPlan) bool {
	rotary := plan.Rotary
	shapes := s.TensorShapes(plan.Layer)
	return plan.Attention == AttentionGatedDelta && !plan.Recurrent && !plan.SharedKV &&
		plan.AttentionGraph.BlockSize == tensor.QSABlockCells &&
		shapes.QueryHeads == tensor.QSAQueryHeads && shapes.KVHeads == tensor.QSAKeyValueHeads &&
		shapes.Key == tensor.QSAHeadWidth && shapes.Value == tensor.QSAHeadWidth &&
		s.IndexerHeadCount == tensor.QSAIndexerHeads && s.IndexerKeyLength == tensor.QSAIndexerWidth &&
		s.IndexerTopK == tensor.QSABudget && rotary.rotaryDimensions == tensor.QSARotated &&
		rotary.frequencyBase == tensor.QSARopeBase && rotary.frequencyScale == tensor.UnitScale &&
		rotary.layout == tensor.RoPELayoutNeoX && rotary.qsaAxes()
}

// qsaAxes reports whether the rotation turns each pair by the axis Strata's pooling turns it by, pair % 3 of a cell's
// (temporal, height, width) positions (mrope.hpp mrope_pos): a single-axis rotation without YaRN, or interleaved
// sections whose every pair cycles the three axes.
func (p rotaryPlan) qsaAxes() bool {
	switch {
	case p.kind == rotaryGraphSingle:
		return !p.yarn
	case p.kind != rotaryGraphMulti || !p.interleavedSections:
		return false
	}
	var sectionPairs uint32
	for _, section := range p.sections {
		sectionPairs += uint32(section)
	}
	for pair := range p.rotaryDimensions / tensor.PairedExtent {
		if sectionPairs == 0 {
			return false
		}
		sector := pair % sectionPairs
		axis := sector % tensor.QSAPositionAxes
		if axis != pair%tensor.QSAPositionAxes || sector >= tensor.QSAPositionAxes*uint32(p.sections[axis]) {
			return false
		}
	}
	return true
}

// QSACacheSchema is a QSA layer's cache: its fixed state, the indexer's tail (128, 3) as the key and its spare key
// (128) as the value, both zero before the first cell; its cells live in the runner's QSA storage.
func QSACacheSchema() LayerCacheSchema {
	return LayerCacheSchema{
		Label: "QSA", StrictStates: true,
		Primary: NewCachePair(
			CacheState[CacheValueSchema]{Mode: CacheStateFixed, Value: CacheValueSchema{
				Shape: tensor.MustShape(tensor.QSAIndexerWidth, tensor.QSATailRows), ZeroInitial: true,
			}},
			CacheState[CacheValueSchema]{Mode: CacheStateFixed, Value: CacheValueSchema{
				Shape: tensor.MustShape(tensor.QSAIndexerWidth), ZeroInitial: true,
			}},
		),
	}
}

// buildQSAAttentionMix is a block-sparse gated attention layer built as Strata's QSA layer (src/core/layer.cpp
// qsa_layer for a token, src/prefill/prefill.cpp's QSA section for a chunk): the projections, norms and rotations of
// the query, key and indexer query, the raw indexer key, one QSA step over the binding's cells, the gate and the
// output projection. A verify span's checkpoints cut the step into steps, each taking the fixed state the one before
// it left, as the gated-delta scan cuts its scan, so each checkpoint's fixed state is a step's.
func buildQSAAttentionMix(options blockDispatchOptions, normalized *tensor.Tensor, rotary rotaryPlan) (DenseBlockResult, error) {
	c, spec, weights := options.Context, options.Spec, options.Weights
	builder := c.Builder
	if builder == nil || normalized == nil || c.QSA == nil {
		return DenseBlockResult{}, errors.New("QSA attention mix input is nil")
	}
	required := graphWeights{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.AttentionQNorm, weights.AttentionKNorm,
		weights.IndexerAttentionQB, weights.IndexerAttentionK, weights.IndexerQNorm, weights.IndexerKNorm,
	}
	if err := required.validate("QSA attention mix"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(c.Positions))
	if c.Sequences != tensor.SingletonExtent || tokens == 0 ||
		!tensor.IsBatchedMatrix(normalized.Shape, uint64(spec.EmbeddingLength), tokens, tensor.SingletonExtent) {
		return DenseBlockResult{}, errors.New("QSA attention takes one sequence's tokens")
	}
	if c.PastKey == nil || c.PastValue == nil {
		return DenseBlockResult{}, errors.New("QSA attention takes the indexer's fixed state")
	}
	queryAndGate := builder.MulMat(weights.AttentionQ, normalized)
	query := builder.GroupSlice(queryAndGate, tensor.FirstOffset, tensor.QSAHeadWidth, tensor.QSAQueryHeads, tensor.PairedExtent*tensor.QSAHeadWidth)
	gate := builder.GroupSlice(queryAndGate, tensor.QSAHeadWidth, tensor.QSAHeadWidth, tensor.QSAQueryHeads, tensor.PairedExtent*tensor.QSAHeadWidth)
	key := builder.Reshape(builder.MulMat(weights.AttentionK, normalized), tensor.QSAHeadWidth, tensor.QSAKeyValueHeads, tokens)
	value := builder.Reshape(builder.MulMat(weights.AttentionV, normalized), tensor.QSAHeadWidth, tensor.QSAKeyValueHeads, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	query, key = rotary.Apply(builder, query, key, c.Positions, c.MultiPositions, nil)
	raw := builder.Reshape(builder.MulMat(weights.IndexerAttentionK, normalized), tensor.QSAIndexerWidth, tensor.SingletonExtent, tokens)
	indexerQuery := builder.Reshape(builder.MulMat(weights.IndexerAttentionQB, normalized), tensor.QSAIndexerWidth, tensor.QSAIndexerHeads, tokens)
	indexerQuery = builder.WeightedRMSNorm(indexerQuery, weights.IndexerQNorm, spec.RMSNormEpsilon)
	indexerQuery = rotary.ApplyOne(builder, indexerQuery, c.Positions, c.MultiPositions, nil)

	if c.Window != nil {
		// verify.cpp record_window's QSA section: one step over the window's cells, appending to the indexer's fixed
		// state where it sits after its tail's snapshot, the raw keys kept for the commit.
		if c.Window.Snapshot == nil {
			return DenseBlockResult{}, errors.New("a QSA window takes its tail's snapshot")
		}
		cells := make([]uint32, tokens)
		for index := range cells {
			cells[index] = c.QSA.FirstCell + uint32(index)
		}
		packed := builder.QSAAttentionInPlace(query, key, value, raw, indexerQuery, weights.IndexerKNorm, c.PastKey, c.PastValue,
			c.Window.Snapshot, c.QSA.State, tensor.QSAAttentionAttributes{Cells: cells, Capacity: c.QSA.Capacity, Epsilon: spec.RMSNormEpsilon})
		attention, _, _ := builder.QSAAttentionParts(packed, tokens)
		output := qsaGatedOutput(builder, attention, gate, weights.AttentionOutput, tokens)
		if err := builder.Err(); err != nil {
			return DenseBlockResult{}, err
		}
		// The raw keys are kept as their projection writes them, (128, tokens), where a target holds a node's own output.
		return DenseBlockResult{Output: output, Window: LayerWindow{Raw: raw.Inputs[0]}}, nil
	}
	ends := []uint64{tokens}
	if spanCheckpoints(c.Checkpoints, tokens) {
		ends = ends[:0]
		for _, at := range c.Checkpoints {
			ends = append(ends, uint64(at))
		}
		ends = append(ends, tokens)
	}
	slice := func(input *tensor.Tensor, first, count uint64) *tensor.Tensor {
		if first == tensor.FirstOffset && count == tokens {
			return input
		}
		width, heads := input.Shape.Dims[0], input.Shape.Dims[1]
		return builder.FlatSlice(input, first*width*heads, width, heads, count)
	}
	tail, spare := c.PastKey, c.PastValue
	var attention *tensor.Tensor
	var checkpoints []LayerCheckpoint
	start := uint64(tensor.FirstOffset)
	for _, end := range ends {
		count := end - start
		cells := make([]uint32, count)
		for index := range cells {
			cells[index] = c.QSA.FirstCell + uint32(start) + uint32(index)
		}
		packed := builder.QSAAttention(
			slice(query, start, count), slice(key, start, count), slice(value, start, count),
			slice(raw, start, count), slice(indexerQuery, start, count), weights.IndexerKNorm, tail, spare,
			c.QSA.State, tensor.QSAAttentionAttributes{Cells: cells, Capacity: c.QSA.Capacity, Epsilon: spec.RMSNormEpsilon},
		)
		var part *tensor.Tensor
		part, tail, spare = builder.QSAAttentionParts(packed, count)
		attention = concatTokens(builder, attention, part)
		if end < tokens {
			checkpoints = append(checkpoints, LayerCheckpoint{Key: tail, Value: spare})
		}
		start = end
	}
	attention = qsaGatedOutput(builder, attention, gate, weights.AttentionOutput, tokens)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: tail, Value: spare, Checkpoints: slices.Clip(checkpoints)}, nil
}

// qsaGatedOutput is the QSA layer's attention gated by sigmoid of its gate rows, through the output projection.
func qsaGatedOutput(builder *tensor.Builder, attention, gate, projection *tensor.Tensor, tokens uint64) *tensor.Tensor {
	width := uint64(tensor.QSAQueryHeads) * tensor.QSAHeadWidth
	attention = builder.Reshape(attention, width, tokens)
	attention = builder.Multiply(attention, builder.Sigmoid(builder.Reshape(gate, width, tokens)))
	return builder.MulMat(projection, attention)
}
