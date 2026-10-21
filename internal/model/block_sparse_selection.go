package model

import (
	"errors"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

func buildBlockSparseSelection(options blockDispatchOptions, normalized *tensor.Tensor) (*tensor.Tensor, CacheStates[*tensor.Tensor], error) {
	c, spec, weights, plan := options.Context, options.Spec, options.Weights, options.Plan
	builder := c.Builder
	if err := (graphWeights{weights.IndexerAttentionQB, weights.IndexerAttentionK, weights.IndexerQNorm, weights.IndexerKNorm, c.CurrentPositions}).validate("block sparse indexer"); err != nil {
		return nil, nil, err
	}
	if c.Sequences != tensor.SingletonExtent {
		return nil, nil, errors.New("block sparse indexer requires one sequence")
	}
	width, heads, tokens := uint64(spec.IndexerKeyLength), uint64(spec.IndexerHeadCount), uint64(len(c.Positions))
	ratio := uint64(plan.AttentionGraph.BlockSize)
	if ratio == 0 || width == 0 || heads == 0 || tokens == 0 || spec.IndexerTopK < uint32(ratio) || uint64(spec.IndexerTopK)%ratio != 0 {
		return nil, nil, errors.New("block sparse indexer dimensions are invalid")
	}
	if c.CurrentPositions.Shape != tensor.MustShape(tensor.MaxDimensions, tensor.SingletonExtent, tokens) {
		return nil, nil, errors.New("block sparse indexer position shape is invalid")
	}
	raw := builder.Reshape(builder.MulMat(weights.IndexerAttentionK, normalized), width, tensor.SingletonExtent, tokens)
	positions := c.CurrentPositions
	var start uint32
	pastRaw, pastPositions := c.PastStates[CacheStateIndexerKey].Value, c.PastStates[CacheStatePositions].Value
	if (c.PastKey != nil) != (pastRaw != nil) || (pastRaw != nil) != (pastPositions != nil) {
		return nil, nil, errors.New("block sparse indexer cache requires raw keys and positions beside attention keys")
	}
	if pastRaw != nil {
		pastTokens, axis, valid := tensor.TrailingExtent32(pastRaw.Shape, width, tensor.SingletonExtent)
		keyTokens, _, validKey := tensor.TrailingExtent32(c.PastKey.Shape, uint64(spec.KeyLength), uint64(spec.HeadCountKV))
		if !valid || !validKey || keyTokens != pastTokens || pastPositions.Shape != tensor.MustShape(tensor.MaxDimensions, tensor.SingletonExtent, uint64(pastTokens)) {
			return nil, nil, errors.New("block sparse indexer cache shape is invalid")
		}
		start = builder.CacheTokenOffset(pastTokens)
		raw = builder.WriteCache(pastRaw, raw, axis, c.CacheWrite)
		positions = builder.WriteCache(pastPositions, positions, axis, c.CacheWrite)
	}
	states := CacheStates[*tensor.Tensor]{CacheStateIndexerKey: {Mode: CacheStateToken, Value: raw}, CacheStatePositions: {Mode: CacheStateToken, Value: positions}}
	total := uint64(start) + tokens
	if total > math.MaxUint32 {
		return nil, nil, errors.New("block sparse indexer token count exceeds uint32")
	}
	if !spec.selectsAttentionBlocks(total) {
		return nil, states, builder.Err()
	}
	// The blocks span the stored keys, so an appending write scores every block of its page.
	//   - stored = total under a concatenating write, the page capacity under an appending one
	//   - BlockTopK reads only the blocks complete before each query, so the scores of a block past the active tokens are never read
	//   - so the graph of one page keeps its structure as its active tokens grow
	stored := raw.Shape.Dims[raw.Shape.Rank-tensor.SingletonExtent]
	blocks := stored / ratio
	complete := builder.FlatSlice(raw, tensor.FirstOffset, width*ratio, blocks)
	var pooled *tensor.Tensor
	for member := range ratio {
		part := builder.Reshape(builder.GroupSlice(complete, member*width, width, tensor.SingletonExtent, width), width, blocks)
		if pooled == nil {
			pooled = part
		} else {
			pooled = builder.Add(pooled, part)
		}
	}
	pooled = builder.WeightedRMSNorm(builder.Scale(pooled, tensor.UnitScale/float32(ratio)), weights.IndexerKNorm, spec.RMSNormEpsilon)
	blockRows := make([]uint32, blocks)
	for block := range blockRows {
		blockRows[block] = uint32(uint64(block) * ratio)
	}
	positionMatrix := builder.Reshape(builder.FlatSlice(positions, tensor.FirstOffset, tensor.MaxDimensions, tensor.SingletonExtent, blocks*ratio), tensor.MaxDimensions, blocks*ratio)
	blockPositions := builder.Reshape(builder.GetRows(positionMatrix, blockRows), tensor.MaxDimensions, tensor.SingletonExtent, blocks)
	rotary := plan.Rotary
	pooled = builder.RoPEWithOptions(builder.Reshape(pooled, width, tensor.SingletonExtent, blocks), tensor.RoPEOptions{
		Layout: rotary.layout, PositionInput: blockPositions, Sections: rotary.sections, InterleavedSections: rotary.interleavedSections,
		RotaryDimensions: rotary.rotaryDimensions, FrequencyBase: rotary.frequencyBase, FrequencyScale: rotary.frequencyScale,
	})
	query := builder.Reshape(builder.MulMat(weights.IndexerAttentionQB, normalized), width, heads, tokens)
	query = builder.WeightedRMSNorm(query, weights.IndexerQNorm, spec.RMSNormEpsilon)
	query = rotary.ApplyOne(builder, query, c.Positions, c.MultiPositions, nil)
	keys := builder.Reshape(pooled, width, blocks)
	chunk := max(indexerChunkScores/(blocks*heads), tensor.SingletonExtent)
	indices := selectBlocksByChunk(builder, keys, query, chunk, blockSelection{budget: spec.IndexerTopK, ratio: uint32(ratio), start: start, stored: uint32(stored)})
	return indices, states, builder.Err()
}

// indexerChunkScores is the scores a chunk of the block-sparse indexer's queries holds at once, every stored block
// for each of their heads: 512 MiB in f32. A 20K prompt's queries score in four chunks, a 262K cache's in chunks of
// 512 queries; a decode step's one query is always one chunk. indexerTokenAxis is the selections' query axis.
const (
	indexerChunkScores        = 1 << 27
	indexerTokenAxis   uint32 = 1
)

// blockSelection is what the indexer selects by: each query's budget of tokens in blocks of ratio tokens, the first
// query's position in the cache and the cache's stored tokens.
type blockSelection struct {
	budget, ratio, start, stored uint32
}

// selectBlocksByChunk selects each query's blocks from the pooled block keys (width x blocks) and the queries (width x
// heads x tokens), chunk queries at a time. A query's scores span every stored block for each head, so a long
// prompt's queries are scored a chunk at a time, each chunk's scores released once its blocks are selected, and only
// the selections are joined; a query's selection does not depend on the chunk it is scored in.
func selectBlocksByChunk(builder *tensor.Builder, keys, query *tensor.Tensor, chunk uint64, selection blockSelection) *tensor.Tensor {
	width, blocks := keys.Shape.Dims[0], keys.Shape.Dims[1]
	heads, tokens := query.Shape.Dims[1], query.Shape.Dims[query.Shape.Rank-tensor.SingletonExtent]
	var selections []*tensor.Tensor
	for first := uint64(0); first < tokens; first += chunk {
		count := min(chunk, tokens-first)
		queries := builder.FlatSlice(query, first*width*heads, width, heads*count)
		scores := builder.Reshape(builder.ReLU(builder.MulMat(keys, queries)), blocks*heads, count)
		var summed *tensor.Tensor
		for head := range heads {
			part := builder.Reshape(builder.GroupSlice(scores, head*blocks, blocks, tensor.SingletonExtent, blocks), blocks, count)
			if summed == nil {
				summed = part
			} else {
				summed = builder.Add(summed, part)
			}
		}
		summed = builder.Scale(summed, hostmath.InvSqrt32(width))
		selections = append(selections, builder.BlockTopK(summed, selection.budget, selection.ratio, selection.start+uint32(first), selection.stored))
	}
	// The chunks' selections are joined in pairs, so each is copied once a level rather than once a chunk.
	for len(selections) > 1 {
		joined := make([]*tensor.Tensor, 0, len(selections)/tensor.PairedExtent+1)
		for index := 0; index+1 < len(selections); index += tensor.PairedExtent {
			joined = append(joined, builder.Concat(selections[index], selections[index+1], indexerTokenAxis))
		}
		if len(selections)%tensor.PairedExtent != 0 {
			joined = append(joined, selections[len(selections)-1])
		}
		selections = joined
	}
	return selections[0]
}

// selectsAttentionBlocks reports whether a block-sparse layer selects blocks for a cache of the given token count.
func (s Spec) selectsAttentionBlocks(total uint64) bool {
	return total > uint64(s.IndexerTopK)
}
