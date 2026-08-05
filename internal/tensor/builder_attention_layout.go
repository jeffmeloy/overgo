package tensor

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"llamacpp2go/internal/tensor/dtype"
)

// Reshape: changes only logical dimensions and preserves contiguous order
func (b *Builder) Reshape(input *Tensor, dimensions ...uint64) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil {
		b.setError(errors.New("reshape input is nil"))
		return nil
	}
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	inputElements, err := input.Shape.Elements()
	if err != nil {
		b.setError(err)
		return nil
	}
	outputElements, err := shape.Elements()
	if err != nil {
		b.setError(err)
		return nil
	}
	if inputElements != outputElements {
		b.setError(fmt.Errorf("reshape changes element count from %d to %d", inputElements, outputElements))
		return nil
	}
	return b.add("", input.Type, shape, OpReshape, []*Tensor{input}, nil)
}

// Transpose2D materializes transpose of contiguous rank-2 tensor
func (b *Builder) Transpose2D(input *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Shape.Rank != 2 {
		b.setError(errors.New("Transpose2D requires a rank-2 input"))
		return nil
	}
	shape, err := NewShape(input.Shape.Dims[1], input.Shape.Dims[0])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", input.Type, shape, OpTranspose2D, []*Tensor{input}, nil)
}

// GroupSlice: extracts equally-strided groups from dimension zero; input
// rank must be 2 or 3 and result prepends [width, groups] to input's
// remaining dimensions
func (b *Builder) GroupSlice(
	input *Tensor,
	offset, width, groups, stride uint64,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || (input.Shape.Rank != 2 && input.Shape.Rank != 3) {
		b.setError(errors.New("GroupSlice requires a rank-2/3 input"))
		return nil
	}
	if width == 0 || groups == 0 || stride < width ||
		groups-1 > (math.MaxUint64-offset-width)/stride ||
		offset+(groups-1)*stride+width > input.Shape.Dims[0] {
		b.setError(errors.New("GroupSlice range exceeds input dimension zero"))
		return nil
	}
	dimensions := []uint64{width, groups, input.Shape.Dims[1]}
	if input.Shape.Rank == 3 {
		dimensions = append(dimensions, input.Shape.Dims[2])
	}
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add(
		"",
		input.Type,
		shape,
		OpGroupSlice,
		[]*Tensor{input},
		GroupSliceAttributes{Offset: offset, Width: width, Groups: groups, Stride: stride},
	)
}

// FlatSlice: copies contiguous element range and gives it requested
// logical shape
func (b *Builder) FlatSlice(input *Tensor, offset uint64, dimensions ...uint64) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil {
		b.setError(errors.New("FlatSlice input is nil"))
		return nil
	}
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	inputElements, inputErr := input.Shape.Elements()
	outputElements, outputErr := shape.Elements()
	if inputErr != nil || outputErr != nil {
		b.setError(errors.Join(inputErr, outputErr))
		return nil
	}
	if offset > inputElements || outputElements > inputElements-offset {
		b.setError(errors.New("FlatSlice range exceeds input storage"))
		return nil
	}
	return b.add(
		"",
		input.Type,
		shape,
		OpFlatSlice,
		[]*Tensor{input},
		FlatSliceAttributes{Offset: offset},
	)
}

// Attention: computes grouped-query scaled dot-product attention; Q has shape
// [key width, query heads, tokens], K is [key width, KV heads, tokens], and V
// [value width, KV heads, tokens]
func (b *Builder) Attention(query, key, value *Tensor, scale float32, causal bool) *Tensor {
	return b.AttentionWithOffset(query, key, value, scale, causal, 0)
}

// DeepSeek4Attention: raw plus reconstructed compressed attention.
func (b *Builder) DeepSeek4Attention(
	query, cacheKV, cachePositions, sinks, compressorKV, compressorScore, compressorNorm,
	indexerQuery, indexerWeights, indexerKV, indexerScore, indexerNorm *Tensor,
	attributes DeepSeek4AttentionAttributes,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || cacheKV == nil || cachePositions == nil || sinks == nil ||
		query.Type != dtype.F32 || cacheKV.Type != dtype.F32 || cachePositions.Type != dtype.F32 || sinks.Type != dtype.F32 ||
		query.Shape.Rank != 3 || cacheKV.Shape.Rank != 3 || sinks.Shape.Rank != 1 ||
		cacheKV.Shape.Dims[0] != query.Shape.Dims[0] || cacheKV.Shape.Dims[1] != 1 ||
		cachePositions.Shape != MustShape(1, 1, cacheKV.Shape.Dims[2]) ||
		sinks.Shape.Dims[0] != query.Shape.Dims[1] ||
		len(attributes.Positions) != int(query.Shape.Dims[2]) ||
		attributes.Heads != uint32(query.Shape.Dims[1]) || attributes.Window == 0 ||
		attributes.RotaryDimensions == 0 || attributes.RotaryDimensions%2 != 0 ||
		uint64(attributes.RotaryDimensions) > query.Shape.Dims[0] ||
		attributes.FrequencyBase <= 0 || attributes.FrequencyScale <= 0 || attributes.NormEpsilon <= 0 ||
		!attributes.Ratio.Valid() {
		b.setError(errors.New("DeepSeek 4 attention metadata or base inputs are invalid"))
		return nil
	}
	inputs := []*Tensor{query, cacheKV, cachePositions, sinks}
	tokens := cacheKV.Shape.Dims[2]
	if attributes.Ratio.Enabled() {
		coefficient := attributes.Ratio.KVWidthMultiplier()
		if compressorKV == nil || compressorScore == nil || compressorNorm == nil ||
			compressorKV.Type != dtype.F32 || compressorScore.Type != dtype.F32 || compressorNorm.Type != dtype.F32 ||
			compressorKV.Shape != MustShape(coefficient*query.Shape.Dims[0], 1, tokens) ||
			!compressorScore.Shape.Equal(compressorKV.Shape) ||
			compressorNorm.Shape != MustShape(query.Shape.Dims[0]) {
			b.setError(errors.New("DeepSeek 4 compressor inputs are invalid"))
			return nil
		}
		inputs = append(inputs, compressorKV, compressorScore, compressorNorm)
	}
	if attributes.Ratio.UsesIndexer() {
		if attributes.IndexerHeads == 0 || attributes.IndexerTopK == 0 ||
			indexerQuery == nil || indexerWeights == nil || indexerKV == nil || indexerScore == nil || indexerNorm == nil ||
			indexerQuery.Type != dtype.F32 || indexerWeights.Type != dtype.F32 || indexerKV.Type != dtype.F32 ||
			indexerScore.Type != dtype.F32 || indexerNorm.Type != dtype.F32 ||
			indexerQuery.Shape.Rank != 3 || indexerQuery.Shape.Dims[1] != uint64(attributes.IndexerHeads) ||
			indexerQuery.Shape.Dims[2] != query.Shape.Dims[2] ||
			indexerWeights.Shape != MustShape(uint64(attributes.IndexerHeads), query.Shape.Dims[2]) ||
			indexerKV.Shape != MustShape(2*indexerQuery.Shape.Dims[0], 1, tokens) ||
			!indexerScore.Shape.Equal(indexerKV.Shape) || indexerNorm.Shape != MustShape(indexerQuery.Shape.Dims[0]) {
			b.setError(errors.New("DeepSeek 4 indexer inputs are invalid"))
			return nil
		}
		inputs = append(inputs, indexerQuery, indexerWeights, indexerKV, indexerScore, indexerNorm)
	}
	attributes.Positions = slices.Clone(attributes.Positions)
	return b.add("", dtype.F32, query.Shape, OpDeepSeek4Attention, inputs, attributes)
}

// SparseAttention: top-k indexed grouped-query attention.
func (b *Builder) SparseAttention(
	query, key, value, indices *Tensor,
	scale float32,
) *Tensor {
	return b.SparseAttentionWithOffset(query, key, value, indices, scale, false, 0)
}

// SparseAttentionWithOffset: optional causal suffix attention.
func (b *Builder) SparseAttentionWithOffset(
	query, key, value, indices *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || key == nil || value == nil || indices == nil {
		b.setError(errors.New("SparseAttention input is nil"))
		return nil
	}
	if query.Type != dtype.F32 || key.Type != dtype.F32 || value.Type != dtype.F32 || indices.Type != dtype.F32 {
		b.setError(errors.New("SparseAttention requires F32 inputs"))
		return nil
	}
	if query.Shape.Rank != 3 || key.Shape.Rank != 3 || value.Shape.Rank != 3 || indices.Shape.Rank != 2 {
		b.setError(errors.New("SparseAttention requires rank-3 Q/K/V and rank-2 indices"))
		return nil
	}
	if query.Shape.Dims[0] != key.Shape.Dims[0] ||
		key.Shape.Dims[1] != value.Shape.Dims[1] ||
		key.Shape.Dims[2] != value.Shape.Dims[2] ||
		query.Shape.Dims[1]%key.Shape.Dims[1] != 0 ||
		indices.Shape.Dims[0] == 0 || indices.Shape.Dims[0] > key.Shape.Dims[2] ||
		indices.Shape.Dims[1] != query.Shape.Dims[2] {
		b.setError(errors.New("SparseAttention input shapes are incompatible"))
		return nil
	}
	if uint64(queryStart) > key.Shape.Dims[2] ||
		query.Shape.Dims[2] > key.Shape.Dims[2]-uint64(queryStart) {
		b.setError(errors.New("SparseAttention query range exceeds KV tokens"))
		return nil
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		b.setError(errors.New("SparseAttention scale must be finite"))
		return nil
	}
	shape, err := NewShape(value.Shape.Dims[0], query.Shape.Dims[1], query.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add(
		"", dtype.F32, shape, OpSparseAttention,
		[]*Tensor{query, key, value, indices},
		SparseAttentionAttributes{Scale: scale, Causal: causal, QueryStart: queryStart},
	)
}

// IndexerScore: causal DSA head-reduced scores.
func (b *Builder) IndexerScore(
	query, key, weights *Tensor,
	scale float32,
	queryStart uint32,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || key == nil || weights == nil || query.Type != dtype.F32 ||
		key.Type != dtype.F32 || weights.Type != dtype.F32 {
		b.setError(errors.New("IndexerScore requires F32 inputs"))
		return nil
	}
	if query.Shape.Rank != 3 || key.Shape.Rank != 3 || weights.Shape.Rank != 2 ||
		query.Shape.Dims[0] != key.Shape.Dims[0] || key.Shape.Dims[1] != 1 ||
		weights.Shape.Dims[0] != query.Shape.Dims[1] ||
		weights.Shape.Dims[1] != query.Shape.Dims[2] ||
		uint64(queryStart) > key.Shape.Dims[2] ||
		query.Shape.Dims[2] > key.Shape.Dims[2]-uint64(queryStart) {
		b.setError(errors.New("IndexerScore input shapes are incompatible"))
		return nil
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		b.setError(errors.New("IndexerScore scale must be finite"))
		return nil
	}
	shape, err := NewShape(key.Shape.Dims[2], query.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpIndexerScore, []*Tensor{query, key, weights},
		IndexerScoreAttributes{Scale: scale, QueryStart: queryStart})
}

type attentionOptions struct {
	bias, sinks, blockIDs *Tensor
	scale, softcap        float32
	maxALiBiBias          float32
	causal                bool
	symmetricWindow       bool
	relativeBidirectional bool
	chunkedWindow         bool
	queryStart, window    uint32
}

// AttentionOptions: attention graph controls.
type AttentionOptions struct {
	Bias                  *Tensor
	Sinks                 *Tensor
	BlockIDs              *Tensor
	Scale                 float32
	Softcap               float32
	MaxALiBiBias          float32
	Causal                bool
	SymmetricWindow       bool
	RelativeBidirectional bool
	ChunkedWindow         bool
	QueryStart            uint32
	Window                uint32
}

// AttentionWithOptions: typed attention construction.
func (b *Builder) AttentionWithOptions(
	query, key, value *Tensor,
	options AttentionOptions,
) *Tensor {
	if options.Window == 0 &&
		(options.SymmetricWindow || options.ChunkedWindow || options.BlockIDs != nil) {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		bias: options.Bias, sinks: options.Sinks, blockIDs: options.BlockIDs,
		scale: options.Scale, softcap: options.Softcap, maxALiBiBias: options.MaxALiBiBias,
		causal: options.Causal, symmetricWindow: options.SymmetricWindow,
		relativeBidirectional: options.RelativeBidirectional,
		chunkedWindow:         options.ChunkedWindow, queryStart: options.QueryStart,
		window: options.Window,
	})
}

// AttentionWithOffset: query suffix at queryStart in cached key/value sequence.
func (b *Builder) AttentionWithOffset(
	query, key, value *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.buildAttention(query, key, value, attentionOptions{
		scale: scale, causal: causal, queryStart: queryStart,
	})
}

func (b *Builder) AttentionWithSinksWithOffset(
	query, key, value, sinks *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.buildAttention(query, key, value, attentionOptions{
		sinks: sinks, scale: scale, causal: causal, queryStart: queryStart,
	})
}

// AttentionALiBiWithOffset: applies llama.cpp-compatible head slopes to
// linear relative-position mask; maxBias controls steepest slope
func (b *Builder) AttentionALiBiWithOffset(
	query, key, value *Tensor,
	scale, maxBias float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.buildAttention(query, key, value, attentionOptions{
		scale: scale, maxALiBiBias: maxBias, causal: causal, queryStart: queryStart,
	})
}

// AttentionSoftcappedWithOffset: applies cap*tanh(score/cap) before softmax
func (b *Builder) AttentionSoftcappedWithOffset(
	query, key, value *Tensor,
	scale float32,
	softcap float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.buildAttention(query, key, value, attentionOptions{
		scale: scale, softcap: softcap, causal: causal, queryStart: queryStart,
	})
}

// AttentionWithRelativeBias: computes full bidirectional attention and adds
// T5-style bucketed relative-position bias; Bias has shape [heads, buckets]
func (b *Builder) AttentionWithRelativeBias(
	query, key, value, bias *Tensor,
	scale float32,
) *Tensor {
	return b.buildAttention(query, key, value, attentionOptions{
		bias: bias, scale: scale, relativeBidirectional: true,
	})
}

// AttentionWithRelativeBiasAndOffset: causal T5 decoder attention.
func (b *Builder) AttentionWithRelativeBiasAndOffset(
	query, key, value, bias *Tensor,
	scale float32,
	queryStart uint32,
) *Tensor {
	return b.buildAttention(query, key, value, attentionOptions{
		bias: bias, scale: scale, causal: true, queryStart: queryStart,
	})
}

func (b *Builder) AttentionWindowWithOffset(
	query, key, value *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		scale: scale, causal: causal, queryStart: queryStart, window: window,
	})
}

// AttentionWindowWithBlockMaskWithOffset: causal window plus bidirectional
// nonnegative block IDs. Negative IDs remain causal.
func (b *Builder) AttentionWindowWithBlockMaskWithOffset(
	query, key, value, blockIDs *Tensor,
	scale float32,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		blockIDs: blockIDs, scale: scale, causal: true, queryStart: queryStart, window: window,
	})
}

func (b *Builder) AttentionWindowWithSinksWithOffset(
	query, key, value, sinks *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		sinks: sinks, scale: scale, causal: causal, queryStart: queryStart, window: window,
	})
}

func (b *Builder) AttentionChunkedWindowWithOffset(
	query, key, value *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention chunk must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		scale: scale, causal: causal, chunkedWindow: true, queryStart: queryStart, window: window,
	})
}

func (b *Builder) AttentionWindowSoftcappedWithOffset(
	query, key, value *Tensor,
	scale float32,
	softcap float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		scale: scale, softcap: softcap, causal: causal, queryStart: queryStart, window: window,
	})
}

func (b *Builder) AttentionSymmetricWindow(
	query, key, value *Tensor,
	scale float32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		scale: scale, symmetricWindow: true, window: window,
	})
}

func (b *Builder) AttentionSymmetricWindowWithSinks(
	query, key, value, sinks *Tensor,
	scale float32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.buildAttention(query, key, value, attentionOptions{
		sinks: sinks, scale: scale, symmetricWindow: true, window: window,
	})
}

func (b *Builder) buildAttention(
	query, key, value *Tensor,
	options attentionOptions,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || key == nil || value == nil {
		b.setError(errors.New("attention input is nil"))
		return nil
	}
	if query.Type != key.Type || query.Type != value.Type {
		b.setError(errors.New("attention input types differ"))
		return nil
	}
	if query.Shape.Rank != 3 || key.Shape.Rank != 3 || value.Shape.Rank != 3 {
		b.setError(errors.New("attention inputs must have rank 3"))
		return nil
	}
	if query.Shape.Dims[0] != key.Shape.Dims[0] {
		b.setError(errors.New("attention query/key widths differ"))
		return nil
	}
	if key.Shape.Dims[1] != value.Shape.Dims[1] {
		b.setError(errors.New("attention key/value head counts differ"))
		return nil
	}
	if key.Shape.Dims[2] != value.Shape.Dims[2] {
		b.setError(errors.New("attention key/value token counts differ"))
		return nil
	}
	queryRangeExceeds := uint64(options.queryStart) > key.Shape.Dims[2] ||
		uint64(options.queryStart) <= key.Shape.Dims[2] &&
			query.Shape.Dims[2] > key.Shape.Dims[2]-uint64(options.queryStart)
	if queryRangeExceeds && (options.causal || options.queryStart != 0) {
		b.setError(fmt.Errorf(
			"attention query range [%d,%d) exceeds key/value token count %d",
			options.queryStart,
			uint64(options.queryStart)+query.Shape.Dims[2],
			key.Shape.Dims[2],
		))
		return nil
	}
	if query.Shape.Dims[1]%key.Shape.Dims[1] != 0 {
		b.setError(errors.New("attention query head count is not divisible by KV head count"))
		return nil
	}
	if options.scale <= 0 {
		b.setError(errors.New("attention scale must be positive"))
		return nil
	}
	if options.softcap < 0 || math.IsNaN(float64(options.softcap)) ||
		math.IsInf(float64(options.softcap), 0) {
		b.setError(errors.New("attention softcap must be finite and non-negative"))
		return nil
	}
	if options.maxALiBiBias < 0 || math.IsNaN(float64(options.maxALiBiBias)) ||
		math.IsInf(float64(options.maxALiBiBias), 0) {
		b.setError(errors.New("attention ALiBi bias must be finite and non-negative"))
		return nil
	}
	if options.symmetricWindow &&
		(options.causal || options.queryStart != 0 || query.Shape.Dims[2] != key.Shape.Dims[2]) {
		b.setError(errors.New("symmetric-window attention requires a full bidirectional sequence"))
		return nil
	}
	var relativeBuckets uint32
	if options.bias != nil {
		if options.maxALiBiBias > 0 {
			b.setError(errors.New("attention cannot combine learned relative bias and ALiBi"))
			return nil
		}
		if options.window != 0 {
			b.setError(errors.New("relative-bias attention cannot use a window"))
			return nil
		}
		if options.bias.Type != query.Type || options.bias.Shape.Rank != 2 ||
			options.bias.Shape.Dims[0] != query.Shape.Dims[1] ||
			options.bias.Shape.Dims[1] < 4 || options.bias.Shape.Dims[1]%2 != 0 ||
			options.bias.Shape.Dims[1] > math.MaxUint32 {
			b.setError(errors.New("attention relative bias must have shape [query heads, even buckets >= 4]"))
			return nil
		}
		relativeBuckets = uint32(options.bias.Shape.Dims[1])
	}
	if options.relativeBidirectional && options.bias == nil {
		b.setError(errors.New("bidirectional relative attention requires learned bias"))
		return nil
	}
	if options.sinks != nil {
		if options.bias != nil {
			b.setError(errors.New("attention cannot combine learned relative bias and sinks"))
			return nil
		}
		if options.sinks.Type != query.Type || options.sinks.Shape.Rank != 1 ||
			options.sinks.Shape.Dims[0] != query.Shape.Dims[1] {
			b.setError(errors.New("attention sinks must have shape [query heads]"))
			return nil
		}
	}
	if options.blockIDs != nil &&
		(options.blockIDs.Type != query.Type || options.blockIDs.Shape.Rank != 1 ||
			options.blockIDs.Shape.Dims[0] != key.Shape.Dims[2]) {
		b.setError(errors.New("attention block IDs must have shape [key tokens] and match input type"))
		return nil
	}
	shape, err := NewShape(value.Shape.Dims[0], query.Shape.Dims[1], query.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	inputs := []*Tensor{query, key, value}
	if options.bias != nil {
		inputs = append(inputs, options.bias)
	} else if options.sinks != nil {
		inputs = append(inputs, options.sinks)
	}
	if options.blockIDs != nil {
		inputs = append(inputs, options.blockIDs)
	}
	return b.add(
		"",
		query.Type,
		shape,
		OpAttention,
		inputs,
		AttentionAttributes{
			Scale: options.scale, Softcap: options.softcap, MaxALiBiBias: options.maxALiBiBias,
			Causal:                options.causal,
			HasSinks:              options.sinks != nil,
			HasBlockMask:          options.blockIDs != nil,
			SymmetricWindow:       options.symmetricWindow,
			ChunkedWindow:         options.chunkedWindow,
			QueryStart:            options.queryStart,
			Window:                options.window,
			RelativeBuckets:       relativeBuckets,
			RelativeBidirectional: options.relativeBidirectional,
		},
	)
}
