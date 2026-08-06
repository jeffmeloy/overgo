package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

// BuildPLMBlockCached: PLM MLA block
func BuildPLMBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Profile().MLAVariant != mlaVariantPLM {
		return DenseBlockResult{}, errors.New("PLM block requires plm architecture")
	}
	return BuildMLABlockCached(builder, input, spec, weights, positions, pastKey, pastValue)
}

// BuildMLABlockCached: MLA block; layer-zero policy
func BuildMLABlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	return BuildMLABlockCachedForLayer(builder, input, spec, weights, positions, pastKey, pastValue, 0)
}

// BuildMLABlockCachedForLayer: MLA block with layer-dependent FFN
func BuildMLABlockCachedForLayer(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	return buildMLABlockCachedForLayer(builder, input, spec, weights, positions, pastKey, pastValue, nil, nil, layerIndex)
}

// BuildGLMDSABlockCached: GLM-DSA compatibility wrapper.
func BuildGLMDSABlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, pastIndexerKey, previousTopK *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	return BuildDSABlockCached(
		builder, input, spec, weights, positions,
		pastKey, pastValue, pastIndexerKey, previousTopK, layerIndex,
	)
}

// BuildDSABlockCached: sparse MLA plus indexer state.
func BuildDSABlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, pastIndexerKey, previousTopK *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	return buildMLABlockCachedForLayer(
		builder, input, spec, weights, positions, pastKey, pastValue,
		pastIndexerKey, previousTopK, layerIndex,
	)
}

func buildMLABlockCachedForLayer(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, pastIndexerKey, previousTopK *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	profile := spec.Profile()
	attentionPolicy := profile.Attention
	if attentionPolicy != AttentionMLA && attentionPolicy != AttentionDSA && profile.MLAVariant != mlaVariantKimi {
		return DenseBlockResult{}, errors.New("MLA block architecture is unsupported")
	}
	isMiniCPM3 := profile.MLAVariant == mlaVariantMiniCPM3
	isDeepSeek2 := profile.Has(ArchitectureDeepSeek2)
	isDSA := attentionPolicy == AttentionDSA
	isDeepSeek32 := profile.MLAVariant == mlaVariantDeepSeek32
	isKimi := profile.MLAVariant == mlaVariantKimi
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention Q", weights.AttentionQ),
		requireGraphWeight("attention KV-A", weights.AttentionKVAMQA),
		requireGraphWeight("attention KV-A norm", weights.AttentionKVANorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
	}
	if weights.AttentionKVB != nil {
		required.add("attention KV-B", weights.AttentionKVB)
	} else {
		required.add("attention K-B", weights.AttentionKB)
		required.add("attention V-B", weights.AttentionVB)
	}
	if isMiniCPM3 || ((isDeepSeek2 || isDSA || isKimi) && spec.QLoRARank > 0) {
		required.add("attention Q-B", weights.AttentionQB)
		required.add("attention Q-A norm", weights.AttentionQNorm)
	}
	if (isDeepSeek2 || isDSA || isKimi) && layerIndex >= spec.LeadingDenseBlocks {
		required.add("feed-forward router", weights.FeedForwardRouter)
		required.add("feed-forward expert down", weights.FeedForwardDownExperts)
		if weights.FeedForwardGateUpExperts == nil {
			required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		}
		required.add("feed-forward shared gate", weights.FeedForwardSharedGate)
		required.add("feed-forward shared up", weights.FeedForwardSharedUp)
		required.add("feed-forward shared down", weights.FeedForwardSharedDown)
	} else {
		required.add("feed-forward up", weights.FeedForwardUp)
		required.add("feed-forward down", weights.FeedForwardDown)
		if isMiniCPM3 || isDeepSeek2 || isKimi {
			required.add("feed-forward gate", weights.FeedForwardGate)
		}
	}
	if isMiniCPM3 {
		required.add("feed-forward gate", weights.FeedForwardGate)
	}
	if isDSA && spec.LayerHasFullIndexer(layerIndex) {
		if err := (graphWeights{
			requireGraphWeight("indexer K norm", weights.IndexerKNorm),
			requireGraphWeight("indexer K norm bias", weights.IndexerKNormBias),
			requireGraphWeight("indexer projection", weights.IndexerProjection),
			requireGraphWeight("indexer K", weights.IndexerAttentionK),
			requireGraphWeight("indexer Q-B", weights.IndexerAttentionQB),
		}).validate("DSA"); err != nil {
			return DenseBlockResult{}, err
		}
	}
	if err := required.validate("MLA block"); err != nil {
		return DenseBlockResult{}, err
	}
	if builder == nil || input == nil || input.Shape.Rank != 2 || len(positions) == 0 ||
		uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("MLA block input shape is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "MLA cache must contain both key and value"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
	heads := uint64(spec.HeadCount)
	keyWidth := uint64(spec.KeyLength)
	ropeWidth := uint64(spec.RopeDimensionCount)
	nopeWidth := keyWidth - ropeWidth
	valueWidth := uint64(spec.ValueLength)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	queryMixed := builder.MulMat(weights.AttentionQ, normalized)
	if isMiniCPM3 || ((isDeepSeek2 || isDSA || isKimi) && spec.QLoRARank > 0) {
		queryMixed = builder.WeightedRMSNorm(queryMixed, weights.AttentionQNorm, spec.RMSNormEpsilon)
	}
	queryRank := queryMixed
	if isMiniCPM3 || ((isDeepSeek2 || isDSA || isKimi) && spec.QLoRARank > 0) {
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
	if isKimi {
		// Kimi: no RoPE.
	} else if (isDeepSeek2 || isDSA) && spec.RopeScalingType == "yarn" {
		qPE = builder.RoPENormalYaRN(qPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
			spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
			spec.YaRNBetaFast, spec.YaRNBetaSlow)
		kPE = builder.RoPENormalYaRN(kPE, positions, uint32(ropeWidth), spec.OriginalContextLength,
			spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor, spec.YaRNAttentionFactor,
			spec.YaRNBetaFast, spec.YaRNBetaSlow)
	} else if isMiniCPM3 {
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
	if isDSA {
		if spec.LayerHasFullIndexer(layerIndex) {
			indexerWidth := uint64(spec.IndexerKeyLength)
			indexerHeads := uint64(spec.IndexerHeadCount)
			indexerQuery := builder.MulMat(weights.IndexerAttentionQB, queryRank)
			indexerQPE := builder.GroupSlice(indexerQuery, 0, ropeWidth, indexerHeads, indexerWidth)
			indexerQNoPE := builder.GroupSlice(indexerQuery, ropeWidth, indexerWidth-ropeWidth, indexerHeads, indexerWidth)
			if spec.RopeScalingType == "yarn" {
				if isDeepSeek32 {
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
			if isDeepSeek32 {
				indexerEpsilon = spec.LayerNormEpsilon
			}
			indexerKey = builder.AffineLayerNorm(indexerKey, weights.IndexerKNorm, weights.IndexerKNormBias, indexerEpsilon)
			indexerKPE := builder.GroupSlice(indexerKey, 0, ropeWidth, 1, indexerWidth)
			indexerKNoPE := builder.GroupSlice(indexerKey, ropeWidth, indexerWidth-ropeWidth, 1, indexerWidth)
			if spec.RopeScalingType == "yarn" {
				if isDeepSeek32 {
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
				return DenseBlockResult{}, errors.New("DSA shared indexer has no previous top-k")
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
	if isDeepSeek2 && weights.AttentionTemperatureScale != nil {
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
	if (isDeepSeek2 || isDSA) && spec.RopeScalingType == "yarn" {
		logScale := float32(math.Log(float64(1 / frequencyScale)))
		originalFactor := spec.YaRNAttentionFactor * (1 + yarnLogFactorStep*logScale)
		magnitude := originalFactor * (1 + yarnLogFactorStep*spec.RopeYaRNLogMultiplier*logScale)
		attentionScale *= magnitude * magnitude
	}
	var attention *tensor.Tensor
	if isDSA {
		attention = builder.SparseAttentionWithOffset(query, cacheKey, cacheValue, topK, attentionScale, true, queryStart)
	} else {
		attention = builder.AttentionWithOffset(query, cacheKey, cacheValue, attentionScale, true, queryStart)
	}
	if weights.AttentionVB != nil {
		attention = builder.GroupedMulMat(weights.AttentionVB, attention)
	}
	attention = builder.Reshape(attention, heads*valueWidth, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if isMiniCPM3 {
		attention = builder.Scale(attention, spec.ResidualScale)
	}
	residual := builder.Add(input, attention)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	states := map[string]*tensor.Tensor(nil)
	if indexerKey != nil {
		states = map[string]*tensor.Tensor{"indexer_key": indexerKey}
	}
	auxiliary := topK
	if isDeepSeek32 {
		auxiliary = nil
	}
	if (isDeepSeek2 || isDSA || isKimi) && layerIndex >= spec.LeadingDenseBlocks {
		moePlan := spec.moeGraphPlan(layerIndex)
		moePlan.NormalizeTopKProb = spec.ExpertWeightsNorm
		moePlan.SelectionBias = weights.FeedForwardExpertBias != nil
		feedForward := moePlan.BuildLayer(builder, normalized, nil, weights)
		shared := buildSharedSwiGLU(builder, normalized, weights)
		output := builder.Add(residual, builder.Add(feedForward, shared))
		if err := builder.Err(); err != nil {
			return DenseBlockResult{}, err
		}
		return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue, Auxiliary: auxiliary, States: states}, nil
	}
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	activated := builder.ReLUSquared(up)
	if isMiniCPM3 || isDeepSeek2 || isDSA || isKimi {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		activated = builder.SwiGLU(gate, up)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, activated)
	if isMiniCPM3 {
		feedForward = builder.Scale(feedForward, spec.ResidualScale)
	}
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue, Auxiliary: auxiliary, States: states}, nil
}
