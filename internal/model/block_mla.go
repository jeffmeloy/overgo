package model

import (
	"errors"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

func compileRoPEOptions(
	spec Spec,
	positions []uint32,
	width uint32,
	layout tensor.RoPELayout,
	factors *tensor.Tensor,
	yarn bool,
) tensor.RoPEOptions {
	options := tensor.RoPEOptions{
		Layout: layout, Positions: positions, FrequencyFactors: factors,
		RotaryDimensions: width, FrequencyBase: spec.RopeFrequencyBase,
		FrequencyScale: spec.ropeFrequencyScale(),
	}
	if yarn {
		options.YaRN, options.OriginalContext = true, spec.OriginalContextLength
		options.ExtFactor, options.AttentionFactor = spec.YaRNExtFactor, spec.YaRNAttentionFactor
		options.BetaFast, options.BetaSlow = spec.YaRNBetaFast, spec.YaRNBetaSlow
	}
	return options
}

func applyRoPEPair(builder *tensor.Builder, query, key *tensor.Tensor, options tensor.RoPEOptions) (*tensor.Tensor, *tensor.Tensor) {
	return builder.RoPEWithOptions(query, options), builder.RoPEWithOptions(key, options)
}

// buildLatentAttentionMixCached: compressed-query/KV attention.
func buildLatentAttentionMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, pastIndexerKey, previousTopK *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	layerIndex := plan.Layer
	usesNeoXResidualScale := plan.LatentAttention == latentAttentionNeoXResidualScale
	usesYaRNQuery := plan.LatentYaRNQuery
	usesSparseIndexer := plan.Attention == AttentionSparseLatent
	usesSparseNeoXIndexer := plan.LatentAttention == latentAttentionSparseNeoXIndexer
	omitsRoPE := plan.LatentAttention == latentAttentionNoRoPE
	expandQuery := usesNeoXResidualScale ||
		((usesYaRNQuery || usesSparseIndexer || omitsRoPE) && spec.QLoRARank > tensor.FirstOffset)
	required := graphWeights{
		requireGraphWeight("attention Q", weights.AttentionQ),
		requireGraphWeight("attention KV-A", weights.AttentionKVAMQA),
		requireGraphWeight("attention KV-A norm", weights.AttentionKVANorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
	}
	if weights.AttentionKVB != nil {
		required.add("attention KV-B", weights.AttentionKVB)
	} else {
		required.add("attention K-B", weights.AttentionKB)
		required.add("attention V-B", weights.AttentionVB)
	}
	if expandQuery {
		required.add("attention Q-B", weights.AttentionQB)
		required.add("attention Q-A norm", weights.AttentionQNorm)
	}
	if usesSparseIndexer && spec.LayerHasFullIndexer(layerIndex) {
		if err := (graphWeights{
			requireGraphWeight("indexer K norm", weights.IndexerKNorm),
			requireGraphWeight("indexer K norm bias", weights.IndexerKNormBias),
			requireGraphWeight("indexer projection", weights.IndexerProjection),
			requireGraphWeight("indexer K", weights.IndexerAttentionK),
			requireGraphWeight("indexer Q-B", weights.IndexerAttentionQB),
		}).validate("sparse latent indexer"); err != nil {
			return DenseBlockResult{}, err
		}
	}
	if err := required.validate("latent-attention block"); err != nil {
		return DenseBlockResult{}, err
	}
	if builder == nil || normalized == nil {
		return DenseBlockResult{}, errors.New("latent-attention input shape is invalid")
	}
	tokenCount, validInput := tensor.MatrixRows32(normalized.Shape, uint64(spec.EmbeddingLength))
	if !validInput || uint64(len(positions)) != uint64(tokenCount) {
		return DenseBlockResult{}, errors.New("latent-attention input shape is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "latent-attention cache must contain both key and value"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(tokenCount)
	heads := uint64(spec.HeadCount)
	keyWidth := uint64(spec.KeyLength)
	ropeWidth := uint64(spec.RopeDimensionCount)
	nopeWidth := keyWidth - ropeWidth
	valueWidth := uint64(spec.ValueLength)
	queryMixed := builder.MulMat(weights.AttentionQ, normalized)
	if expandQuery {
		queryMixed = builder.WeightedRMSNorm(queryMixed, weights.AttentionQNorm, spec.RMSNormEpsilon)
	}
	queryRank := queryMixed
	if expandQuery {
		queryMixed = builder.MulMat(weights.AttentionQB, queryMixed)
	}
	qNoPE := builder.GroupSlice(queryMixed, tensor.FirstOffset, nopeWidth, heads, keyWidth)
	qPE := builder.GroupSlice(queryMixed, nopeWidth, ropeWidth, heads, keyWidth)
	kvPE := builder.MulMat(weights.AttentionKVAMQA, normalized)
	kvStride := uint64(spec.KVLoRARank) + ropeWidth
	kvCompressed := builder.Reshape(
		builder.GroupSlice(kvPE, tensor.FirstOffset, uint64(spec.KVLoRARank), tensor.SingletonExtent, kvStride),
		uint64(spec.KVLoRARank), tokens,
	)
	kPE := builder.Reshape(
		builder.GroupSlice(kvPE, uint64(spec.KVLoRARank), ropeWidth, tensor.SingletonExtent, kvStride),
		ropeWidth, tensor.SingletonExtent, tokens,
	)
	kvCompressed = builder.WeightedRMSNorm(kvCompressed, weights.AttentionKVANorm, spec.RMSNormEpsilon)
	if omitsRoPE {
		// No rotary transform.
	} else {
		layout := tensor.RoPELayoutNormal
		if usesNeoXResidualScale {
			layout = tensor.RoPELayoutNeoX
		}
		yarn := (usesYaRNQuery || usesSparseIndexer) && spec.RopeScalingType == ropeScalingYaRN
		factors := weights.RopeFactors
		if yarn {
			factors = nil
		}
		qPE, kPE = applyRoPEPair(builder, qPE, kPE, compileRoPEOptions(
			spec, positions, uint32(ropeWidth), layout, factors, yarn,
		))
	}
	if positiveFinite(spec.RopeAttentionFactor) && spec.RopeAttentionFactor != tensor.UnitScale {
		qPE = builder.Scale(qPE, spec.RopeAttentionFactor)
		kPE = builder.Scale(kPE, spec.RopeAttentionFactor)
	}
	var indexerKey, topK *tensor.Tensor
	if usesSparseIndexer {
		if spec.LayerHasFullIndexer(layerIndex) {
			indexerWidth := uint64(spec.IndexerKeyLength)
			indexerHeads := uint64(spec.IndexerHeadCount)
			indexerQuery := builder.MulMat(weights.IndexerAttentionQB, queryRank)
			indexerQPE := builder.GroupSlice(indexerQuery, tensor.FirstOffset, ropeWidth, indexerHeads, indexerWidth)
			indexerQNoPE := builder.GroupSlice(indexerQuery, ropeWidth, indexerWidth-ropeWidth, indexerHeads, indexerWidth)
			indexerLayout := tensor.RoPELayoutNormal
			if usesSparseNeoXIndexer && spec.RopeScalingType == ropeScalingYaRN {
				indexerLayout = tensor.RoPELayoutNeoX
			}
			indexerOptions := compileRoPEOptions(
				spec, positions, uint32(ropeWidth), indexerLayout, nil,
				spec.RopeScalingType == ropeScalingYaRN,
			)
			indexerQPE = builder.RoPEWithOptions(indexerQPE, indexerOptions)
			indexerQuery = builder.FWHT(builder.Concat(indexerQPE, indexerQNoPE, tensor.FirstOffset))

			indexerKey = builder.MulMat(weights.IndexerAttentionK, normalized)
			indexerEpsilon := spec.RMSNormEpsilon
			if usesSparseNeoXIndexer {
				indexerEpsilon = spec.LayerNormEpsilon
			}
			indexerKey = builder.AffineLayerNorm(indexerKey, weights.IndexerKNorm, weights.IndexerKNormBias, indexerEpsilon)
			indexerKPE := builder.GroupSlice(indexerKey, tensor.FirstOffset, ropeWidth, tensor.SingletonExtent, indexerWidth)
			indexerKNoPE := builder.GroupSlice(indexerKey, ropeWidth, indexerWidth-ropeWidth, tensor.SingletonExtent, indexerWidth)
			indexerKPE = builder.RoPEWithOptions(indexerKPE, indexerOptions)
			indexerKey = builder.FWHT(builder.Concat(indexerKPE, indexerKNoPE, tensor.FirstOffset))
			if pastIndexerKey != nil {
				indexerKey = builder.Concat(pastIndexerKey, indexerKey, tensor.PairedExtent)
			}
			indexerWeights := builder.MulMat(weights.IndexerProjection, normalized)
			indexerScale := hostmath.InvSqrt32(uint64(spec.IndexerKeyLength * spec.IndexerHeadCount))
			indexerTokens, _, validIndexer := tensor.TrailingExtent32(indexerKey.Shape, indexerWidth, tensor.SingletonExtent)
			if !validIndexer || indexerTokens < tokenCount {
				return DenseBlockResult{}, errors.New("sparse latent indexer cache shape is invalid")
			}
			scores := builder.IndexerScore(indexerQuery, indexerKey, indexerWeights, indexerScale, indexerTokens-tokenCount)
			selected := spec.IndexerTopK
			if selected > indexerTokens {
				selected = indexerTokens
			}
			topK = builder.TopK(scores, selected)
		} else {
			if previousTopK == nil {
				return DenseBlockResult{}, errors.New("sparse latent shared indexer has no previous top-k")
			}
			topK = previousTopK
		}
	}
	var query, key, value *tensor.Tensor
	if weights.AttentionKB != nil {
		qNoPE = builder.GroupedMulMat(weights.AttentionKB, qNoPE)
		query = builder.Concat(qNoPE, qPE, tensor.FirstOffset)
		kvCompressed = builder.Reshape(kvCompressed, uint64(spec.KVLoRARank), tensor.SingletonExtent, tokens)
		key = builder.Concat(kvCompressed, kPE, tensor.FirstOffset)
		value = kvCompressed
	} else {
		kv := builder.MulMat(weights.AttentionKVB, kvCompressed)
		stride := nopeWidth + valueWidth
		kNoPE := builder.GroupSlice(kv, tensor.FirstOffset, nopeWidth, heads, stride)
		value = builder.GroupSlice(kv, nopeWidth, valueWidth, heads, stride)
		kPEHeads := builder.RepeatHeads(kPE, spec.HeadCount)
		query = builder.Concat(qNoPE, qPE, tensor.FirstOffset)
		key = builder.Concat(kNoPE, kPEHeads, tensor.FirstOffset)
	}
	if usesYaRNQuery && weights.AttentionTemperatureScale != nil {
		query = builder.Multiply(query, weights.AttentionTemperatureScale)
	}
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		queryStart = uint32(pastKey.Shape.Dims[tensor.PairedExtent])
		cacheKey = builder.Concat(pastKey, key, tensor.PairedExtent)
		cacheValue = builder.Concat(pastValue, value, tensor.PairedExtent)
	}
	attentionScale := hostmath.InvSqrt32(uint64(spec.KeyLength))
	if (usesYaRNQuery || usesSparseIndexer) && spec.RopeScalingType == ropeScalingYaRN {
		logScale := float32(math.Log(float64(1 / spec.ropeFrequencyScale())))
		originalFactor := spec.YaRNAttentionFactor * (1 + yarnLogFactorStep*logScale)
		magnitude := originalFactor * (1 + yarnLogFactorStep*spec.RopeYaRNLogMultiplier*logScale)
		attentionScale *= magnitude * magnitude
	}
	var attention *tensor.Tensor
	if usesSparseIndexer {
		attention = builder.SparseAttentionWithOffset(query, cacheKey, cacheValue, topK, attentionScale, true, queryStart)
	} else {
		attention = builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{Scale: attentionScale, Causal: true, QueryStart: queryStart})
	}
	if weights.AttentionVB != nil {
		attention = builder.GroupedMulMat(weights.AttentionVB, attention)
	}
	attention = builder.Reshape(attention, heads*valueWidth, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if usesNeoXResidualScale {
		attention = builder.Scale(attention, spec.ResidualScale)
	}
	states := CacheStates[*tensor.Tensor](nil)
	if indexerKey != nil {
		states = CacheStates[*tensor.Tensor]{
			CacheStateIndexerKey: {Mode: CacheStateToken, Value: indexerKey},
		}
	}
	auxiliary := topK
	if usesSparseNeoXIndexer {
		auxiliary = nil
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: attention, Key: cacheKey, Value: cacheValue,
		Auxiliary: auxiliary, States: states,
	}, nil
}
