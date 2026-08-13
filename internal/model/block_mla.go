package model

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

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
	if usesNeoXResidualScale || ((usesYaRNQuery || usesSparseIndexer || omitsRoPE) && spec.QLoRARank > 0) {
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
	if builder == nil || normalized == nil || normalized.Shape.Rank != 2 || len(positions) == 0 ||
		uint64(len(positions)) != normalized.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("latent-attention input shape is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "latent-attention cache must contain both key and value"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
	heads := uint64(spec.HeadCount)
	keyWidth := uint64(spec.KeyLength)
	ropeWidth := uint64(spec.RopeDimensionCount)
	nopeWidth := keyWidth - ropeWidth
	valueWidth := uint64(spec.ValueLength)
	queryMixed := builder.MulMat(weights.AttentionQ, normalized)
	if usesNeoXResidualScale || ((usesYaRNQuery || usesSparseIndexer || omitsRoPE) && spec.QLoRARank > 0) {
		queryMixed = builder.WeightedRMSNorm(queryMixed, weights.AttentionQNorm, spec.RMSNormEpsilon)
	}
	queryRank := queryMixed
	if usesNeoXResidualScale || ((usesYaRNQuery || usesSparseIndexer || omitsRoPE) && spec.QLoRARank > 0) {
		queryMixed = builder.MulMat(weights.AttentionQB, queryMixed)
	}
	qNoPE := builder.GroupSlice(queryMixed, 0, nopeWidth, heads, keyWidth)
	qPE := builder.GroupSlice(queryMixed, nopeWidth, ropeWidth, heads, keyWidth)
	kvPE := builder.MulMat(weights.AttentionKVAMQA, normalized)
	kvCompressed := builder.Reshape(
		builder.GroupSlice(kvPE, 0, uint64(spec.KVLoRARank), 1, uint64(spec.KVLoRARank)+ropeWidth),
		uint64(spec.KVLoRARank), tokens,
	)
	kPE := builder.Reshape(
		builder.GroupSlice(kvPE, uint64(spec.KVLoRARank), ropeWidth, 1, uint64(spec.KVLoRARank)+ropeWidth),
		ropeWidth, 1, tokens,
	)
	kvCompressed = builder.WeightedRMSNorm(kvCompressed, weights.AttentionKVANorm, spec.RMSNormEpsilon)
	frequencyScale := float32(1)
	if (spec.RopeScalingType == "linear" || spec.RopeScalingType == "yarn") && spec.RopeScalingFactor > 0 {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	if omitsRoPE {
		// No rotary transform.
	} else if (usesYaRNQuery || usesSparseIndexer) && spec.RopeScalingType == "yarn" {
		qPE = builder.RoPENormalYaRN(qPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
			spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
			spec.YaRNBetaFast, spec.YaRNBetaSlow)
		kPE = builder.RoPENormalYaRN(kPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
			spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
			spec.YaRNBetaFast, spec.YaRNBetaSlow)
	} else if usesNeoXResidualScale {
		if weights.RopeFactors != nil {
			qPE = builder.RoPENeoXScaledWithFactors(qPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale, weights.RopeFactors)
			kPE = builder.RoPENeoXScaledWithFactors(kPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale, weights.RopeFactors)
		} else {
			qPE = builder.RoPENeoXScaled(qPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale)
			kPE = builder.RoPENeoXScaled(kPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale)
		}
	} else if weights.RopeFactors != nil {
		qPE = builder.RoPENormalScaledWithFactors(qPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale, weights.RopeFactors)
		kPE = builder.RoPENormalScaledWithFactors(kPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale, weights.RopeFactors)
	} else {
		qPE = builder.RoPENormalScaled(qPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale)
		kPE = builder.RoPENormalScaled(kPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale)
	}
	if spec.RopeAttentionFactor > 0 && spec.RopeAttentionFactor != 1 {
		qPE = builder.Scale(qPE, spec.RopeAttentionFactor)
		kPE = builder.Scale(kPE, spec.RopeAttentionFactor)
	}
	var indexerKey, topK *tensor.Tensor
	if usesSparseIndexer {
		if spec.LayerHasFullIndexer(layerIndex) {
			indexerWidth := uint64(spec.IndexerKeyLength)
			indexerHeads := uint64(spec.IndexerHeadCount)
			indexerQuery := builder.MulMat(weights.IndexerAttentionQB, queryRank)
			indexerQPE := builder.GroupSlice(indexerQuery, 0, ropeWidth, indexerHeads, indexerWidth)
			indexerQNoPE := builder.GroupSlice(indexerQuery, ropeWidth, indexerWidth-ropeWidth, indexerHeads, indexerWidth)
			if spec.RopeScalingType == "yarn" {
				if usesSparseNeoXIndexer {
					indexerQPE = builder.RoPENeoXYaRN(indexerQPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
						spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
						spec.YaRNBetaFast, spec.YaRNBetaSlow)
				} else {
					indexerQPE = builder.RoPENormalYaRN(indexerQPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
						spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
						spec.YaRNBetaFast, spec.YaRNBetaSlow)
				}
			} else {
				indexerQPE = builder.RoPENormalScaled(indexerQPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale)
			}
			indexerQuery = builder.FWHT(builder.Concat(indexerQPE, indexerQNoPE, 0))

			indexerKey = builder.MulMat(weights.IndexerAttentionK, normalized)
			indexerEpsilon := spec.RMSNormEpsilon
			if usesSparseNeoXIndexer {
				indexerEpsilon = spec.LayerNormEpsilon
			}
			indexerKey = builder.AffineLayerNorm(indexerKey, weights.IndexerKNorm, weights.IndexerKNormBias, indexerEpsilon)
			indexerKPE := builder.GroupSlice(indexerKey, 0, ropeWidth, 1, indexerWidth)
			indexerKNoPE := builder.GroupSlice(indexerKey, ropeWidth, indexerWidth-ropeWidth, 1, indexerWidth)
			if spec.RopeScalingType == "yarn" {
				if usesSparseNeoXIndexer {
					indexerKPE = builder.RoPENeoXYaRN(indexerKPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
						spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
						spec.YaRNBetaFast, spec.YaRNBetaSlow)
				} else {
					indexerKPE = builder.RoPENormalYaRN(indexerKPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
						spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
						spec.YaRNBetaFast, spec.YaRNBetaSlow)
				}
			} else {
				indexerKPE = builder.RoPENormalScaled(indexerKPE, positions, uint32(ropeWidth), spec.RopeFrequencyBase, frequencyScale)
			}
			indexerKey = builder.FWHT(builder.Concat(indexerKPE, indexerKNoPE, 0))
			if pastIndexerKey != nil {
				indexerKey = builder.Concat(pastIndexerKey, indexerKey, 2)
			}
			indexerWeights := builder.MulMat(weights.IndexerProjection, normalized)
			indexerScale := float32(1 / math.Sqrt(float64(spec.IndexerKeyLength*spec.IndexerHeadCount)))
			scores := builder.IndexerScore(indexerQuery, indexerKey, indexerWeights, indexerScale, uint32(indexerKey.Shape.Dims[2]-tokens))
			selected := spec.IndexerTopK
			if uint64(selected) > indexerKey.Shape.Dims[2] {
				selected = uint32(indexerKey.Shape.Dims[2])
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
		query = builder.Concat(qNoPE, qPE, 0)
		kvCompressed = builder.Reshape(kvCompressed, uint64(spec.KVLoRARank), 1, tokens)
		key = builder.Concat(kvCompressed, kPE, 0)
		value = kvCompressed
	} else {
		kv := builder.MulMat(weights.AttentionKVB, kvCompressed)
		stride := nopeWidth + valueWidth
		kNoPE := builder.GroupSlice(kv, 0, nopeWidth, heads, stride)
		value = builder.GroupSlice(kv, nopeWidth, valueWidth, heads, stride)
		kPEHeads := builder.RepeatHeads(kPE, spec.HeadCount)
		query = builder.Concat(qNoPE, qPE, 0)
		key = builder.Concat(kNoPE, kPEHeads, 0)
	}
	if usesYaRNQuery && weights.AttentionTemperatureScale != nil {
		query = builder.Multiply(query, weights.AttentionTemperatureScale)
	}
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	if (usesYaRNQuery || usesSparseIndexer) && spec.RopeScalingType == "yarn" {
		logScale := float32(math.Log(float64(1 / frequencyScale)))
		originalFactor := spec.YaRNAttentionFactor * (1 + yarnLogFactorStep*logScale)
		magnitude := originalFactor * (1 + yarnLogFactorStep*spec.RopeYaRNLogMultiplier*logScale)
		attentionScale *= magnitude * magnitude
	}
	var attention *tensor.Tensor
	if usesSparseIndexer {
		attention = builder.SparseAttentionWithOffset(query, cacheKey, cacheValue, topK, attentionScale, true, queryStart)
	} else {
		attention = builder.AttentionWithOffset(query, cacheKey, cacheValue, attentionScale, true, queryStart)
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
