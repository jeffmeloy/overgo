package model

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
)

const (
	rwkvHeadNormEpsilon = 64e-5
	rwkvKeyNormEpsilon  = 1e-12
)

// LayerGraphWeights: graph inputs for one dense Llama/Qwen3 block
type LayerGraphWeights struct {
	AttentionNorm               *tensor.Tensor
	AttentionNormBias           *tensor.Tensor
	AttentionNorm2              *tensor.Tensor
	AttentionNorm2Bias          *tensor.Tensor
	AttentionQ                  *tensor.Tensor
	AttentionQB                 *tensor.Tensor
	AttentionK                  *tensor.Tensor
	AttentionV                  *tensor.Tensor
	AttentionOutput             *tensor.Tensor
	AttentionQScale             *tensor.Tensor
	AttentionKScale             *tensor.Tensor
	AttentionVScale             *tensor.Tensor
	AttentionOutputScale        *tensor.Tensor
	AttentionTemperatureScale   *tensor.Tensor
	AttentionSubNorm            *tensor.Tensor
	AttentionQBias              *tensor.Tensor
	AttentionKBias              *tensor.Tensor
	AttentionVBias              *tensor.Tensor
	AttentionOutputBias         *tensor.Tensor
	AttentionQNorm              *tensor.Tensor
	AttentionKNorm              *tensor.Tensor
	AttentionQNormBias          *tensor.Tensor
	AttentionKNormBias          *tensor.Tensor
	AttentionPostNorm           *tensor.Tensor
	AttentionPostNormBias       *tensor.Tensor
	AttentionRelativeBias       *tensor.Tensor
	CrossAttentionNorm          *tensor.Tensor
	CrossAttentionQ             *tensor.Tensor
	CrossAttentionK             *tensor.Tensor
	CrossAttentionV             *tensor.Tensor
	CrossAttentionOutput        *tensor.Tensor
	AttentionOutputGate         *tensor.Tensor
	AttentionSinks              *tensor.Tensor
	AttentionBlockIDs           *tensor.Tensor
	RopeFactors                 *tensor.Tensor
	FeedForwardNorm             *tensor.Tensor
	FeedForwardNormBias         *tensor.Tensor
	FeedForwardExpertNorm       *tensor.Tensor
	FeedForwardGate             *tensor.Tensor
	FeedForwardUp               *tensor.Tensor
	FeedForwardDown             *tensor.Tensor
	FeedForwardGateScale        *tensor.Tensor
	FeedForwardUpScale          *tensor.Tensor
	FeedForwardDownScale        *tensor.Tensor
	FeedForwardActivationScale  *tensor.Tensor
	FeedForwardSubNorm          *tensor.Tensor
	FeedForwardGateBias         *tensor.Tensor
	FeedForwardUpBias           *tensor.Tensor
	FeedForwardDownBias         *tensor.Tensor
	FeedForwardPostNorm         *tensor.Tensor
	FeedForwardPostNormBias     *tensor.Tensor
	FeedForwardPreNorm2         *tensor.Tensor
	FeedForwardPostNorm1        *tensor.Tensor
	FeedForwardPostNorm2        *tensor.Tensor
	FeedForwardRouter           *tensor.Tensor
	FeedForwardRouterBias       *tensor.Tensor
	FeedForwardRouterScale      *tensor.Tensor
	FeedForwardGateUpExperts    *tensor.Tensor
	FeedForwardGateExperts      *tensor.Tensor
	FeedForwardUpExperts        *tensor.Tensor
	FeedForwardDownExperts      *tensor.Tensor
	FeedForwardDownExpertsScale *tensor.Tensor
	FeedForwardGateChunkExperts *tensor.Tensor
	FeedForwardUpChunkExperts   *tensor.Tensor
	FeedForwardDownChunkExperts *tensor.Tensor
	FeedForwardExpertBias       *tensor.Tensor
	FeedForwardLatentDown       *tensor.Tensor
	FeedForwardLatentUp         *tensor.Tensor
	FeedForwardSharedGate       *tensor.Tensor
	FeedForwardSharedUp         *tensor.Tensor
	FeedForwardSharedDown       *tensor.Tensor
	FeedForwardSharedRouter     *tensor.Tensor
	LayerOutputScale            *tensor.Tensor
	EmbeddingSkip               *tensor.Tensor
	PerLayerInput               *tensor.Tensor
	PerLayerInputGate           *tensor.Tensor
	PerLayerProjection          *tensor.Tensor
	PerLayerPostNorm            *tensor.Tensor
	AltUpCorrectCoefficient     *tensor.Tensor
	AltUpCorrectScale           *tensor.Tensor
	AltUpPredictCoefficient     *tensor.Tensor
	AltUpRouter                 *tensor.Tensor
	AltUpRouterNorm             *tensor.Tensor
	LaurelLeft                  *tensor.Tensor
	LaurelRight                 *tensor.Tensor
	LaurelPostNorm              *tensor.Tensor
	ShortConvKernel             *tensor.Tensor
	ShortConvInput              *tensor.Tensor
	ShortConvOutput             *tensor.Tensor
	AttentionKVAMQA             *tensor.Tensor
	AttentionKVANorm            *tensor.Tensor
	AttentionKVB                *tensor.Tensor
	AttentionKB                 *tensor.Tensor
	AttentionVB                 *tensor.Tensor
	IndexerKNorm                *tensor.Tensor
	IndexerKNormBias            *tensor.Tensor
	IndexerProjection           *tensor.Tensor
	IndexerAttentionK           *tensor.Tensor
	IndexerAttentionQB          *tensor.Tensor
	AttentionOutputA            *tensor.Tensor
	AttentionCompressorKV       *tensor.Tensor
	AttentionCompressorGate     *tensor.Tensor
	AttentionCompressorAPE      *tensor.Tensor
	AttentionCompressorNorm     *tensor.Tensor
	IndexerCompressorKV         *tensor.Tensor
	IndexerCompressorGate       *tensor.Tensor
	IndexerCompressorAPE        *tensor.Tensor
	IndexerCompressorNorm       *tensor.Tensor
	HyperAttentionFN            *tensor.Tensor
	HyperAttentionBase          *tensor.Tensor
	HyperAttentionScale         *tensor.Tensor
	HyperFeedForwardFN          *tensor.Tensor
	HyperFeedForwardBase        *tensor.Tensor
	HyperFeedForwardScale       *tensor.Tensor
	HyperHeadFN                 *tensor.Tensor
	HyperHeadBase               *tensor.Tensor
	HyperHeadScale              *tensor.Tensor
	FeedForwardHashExperts      *tensor.Tensor

	AttentionQKV         *tensor.Tensor
	AttentionQKVBias     *tensor.Tensor
	AttentionGate        *tensor.Tensor
	SSMConv1D            *tensor.Tensor
	SSMConv1DBias        *tensor.Tensor
	SSMInput             *tensor.Tensor
	SSMX                 *tensor.Tensor
	SSMTimeStepWeight    *tensor.Tensor
	SSMTimeStep          *tensor.Tensor
	SSMTimeStepNorm      *tensor.Tensor
	SSMA                 *tensor.Tensor
	SSMD                 *tensor.Tensor
	SSMBNorm             *tensor.Tensor
	SSMCNorm             *tensor.Tensor
	SSMBeta              *tensor.Tensor
	SSMAlpha             *tensor.Tensor
	SSMBetaAlpha         *tensor.Tensor
	SSMNorm              *tensor.Tensor
	SSMOutput            *tensor.Tensor
	SSMQueryConv         *tensor.Tensor
	SSMKeyConv           *tensor.Tensor
	SSMValueConv         *tensor.Tensor
	SSMForgetA           *tensor.Tensor
	SSMForgetB           *tensor.Tensor
	SSMOutputGateA       *tensor.Tensor
	SSMOutputGateB       *tensor.Tensor
	TimeMixW1            *tensor.Tensor
	TimeMixW2            *tensor.Tensor
	TimeMixW0            *tensor.Tensor
	TimeMixA0            *tensor.Tensor
	TimeMixA1            *tensor.Tensor
	TimeMixA2            *tensor.Tensor
	TimeMixV0            *tensor.Tensor
	TimeMixV1            *tensor.Tensor
	TimeMixV2            *tensor.Tensor
	TimeMixG1            *tensor.Tensor
	TimeMixG2            *tensor.Tensor
	TimeMixKK            *tensor.Tensor
	TimeMixKA            *tensor.Tensor
	TimeMixRK            *tensor.Tensor
	TimeMixLerpX         *tensor.Tensor
	TimeMixLerpFused     *tensor.Tensor
	TimeMixLerpW         *tensor.Tensor
	TimeMixLerpK         *tensor.Tensor
	TimeMixLerpV         *tensor.Tensor
	TimeMixLerpR         *tensor.Tensor
	TimeMixLerpG         *tensor.Tensor
	TimeMixFirst         *tensor.Tensor
	TimeMixDecay         *tensor.Tensor
	TimeMixDecayW1       *tensor.Tensor
	TimeMixDecayW2       *tensor.Tensor
	TimeMixKey           *tensor.Tensor
	TimeMixValue         *tensor.Tensor
	TimeMixReceptance    *tensor.Tensor
	TimeMixGate          *tensor.Tensor
	TimeMixLN            *tensor.Tensor
	TimeMixLNBias        *tensor.Tensor
	TimeMixOutput        *tensor.Tensor
	ChannelMixLerpK      *tensor.Tensor
	ChannelMixLerpR      *tensor.Tensor
	ChannelMixKey        *tensor.Tensor
	ChannelMixValue      *tensor.Tensor
	ChannelMixReceptance *tensor.Tensor
}

// ApplyNormalization: applies architecture's learned pre/post
// normalization; Affine LayerNorm architectures and PhiMoE's affine RMSNorm
// require learned bias
func ApplyNormalization(
	builder *tensor.Builder,
	input, weight, bias *tensor.Tensor,
	spec Spec,
) *tensor.Tensor {
	norm := spec.NormPlan()
	if norm.Operation == NormalizationUnweightedLayer {
		return builder.LayerNorm(input, spec.LayerNormEpsilon)
	}
	if norm.Operation == NormalizationUnweightedRMS {
		return builder.RMSNorm(input, spec.RMSNormEpsilon)
	}
	if norm.Operation == NormalizationWeightOnlyLayer {
		return builder.Multiply(builder.LayerNorm(input, spec.LayerNormEpsilon), weight)
	}
	if norm.Operation == NormalizationLayer {
		if bias == nil {
			return builder.Multiply(builder.LayerNorm(input, spec.LayerNormEpsilon), weight)
		}
		return builder.AffineLayerNorm(input, weight, bias, spec.LayerNormEpsilon)
	}
	normalized := builder.WeightedRMSNorm(input, weight, spec.RMSNormEpsilon)
	if (spec.Architecture == "phimoe" || spec.Architecture == "rwkv6qwen2") && bias != nil {
		return builder.Add(normalized, bias)
	}
	return normalized
}

func buildBERTEncoderBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "bert" && spec.Architecture != "jina-bert-v2" && spec.Architecture != "jina-bert-v3" && spec.Architecture != "nomic-bert" && spec.Architecture != "nomic-bert-moe" {
		return DenseBlockResult{}, errors.New("BERT-family block requires a supported encoder architecture")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("BERT-family block input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("BERT-family block position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("BERT-family block does not support a KV cache")
	}
	required := map[string]*tensor.Tensor{
		"attention output":            weights.AttentionOutput,
		"attention post norm":         weights.AttentionPostNorm,
		"attention post norm bias":    weights.AttentionPostNormBias,
		"feed-forward post norm":      weights.FeedForwardPostNorm,
		"feed-forward post norm bias": weights.FeedForwardPostNormBias,
	}
	usesExperts := (spec.Architecture == "jina-bert-v3" || spec.Architecture == "nomic-bert-moe") &&
		spec.IsInterleavedMoELayer(layerIndex)
	if usesExperts {
		required["feed-forward router"] = weights.FeedForwardRouter
		required["feed-forward expert up"] = weights.FeedForwardUpExperts
		required["feed-forward expert down"] = weights.FeedForwardDownExperts
	} else {
		required["feed-forward up"] = weights.FeedForwardUp
		required["feed-forward down"] = weights.FeedForwardDown
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("BERT-family block %s weight is nil", name)
		}
	}
	if spec.Architecture == "nomic-bert" && weights.FeedForwardGate == nil {
		return DenseBlockResult{}, errors.New("NomicBERT block feed-forward gate weight is nil")
	}
	tokens := uint64(len(positions))
	queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, input)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryLength + keyLength + valueLength
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryLength, 1, stride), queryLength, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryLength, keyLength, 1, stride), keyLength, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride), valueLength, tokens)
	} else {
		if weights.AttentionQ == nil || weights.AttentionK == nil || weights.AttentionV == nil {
			return DenseBlockResult{}, errors.New("BERT-family block Q/K/V weights are incomplete")
		}
		query = builder.MulMat(weights.AttentionQ, input)
		key = builder.MulMat(weights.AttentionK, input)
		value = builder.MulMat(weights.AttentionV, input)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
	}
	if spec.Architecture == "jina-bert-v2" {
		if weights.AttentionQNorm != nil {
			query = ApplyNormalization(builder, query, weights.AttentionQNorm, weights.AttentionQNormBias, spec)
		}
		if weights.AttentionKNorm != nil {
			key = ApplyNormalization(builder, key, weights.AttentionKNorm, weights.AttentionKNormBias, spec)
		}
	}
	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	if spec.Architecture == "jina-bert-v3" || spec.Architecture == "nomic-bert" || spec.Architecture == "nomic-bert-moe" {
		frequencyScale := float32(1)
		if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		query = builder.RoPENeoXScaled(
			query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, frequencyScale,
		)
		key = builder.RoPENeoXScaled(
			key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, frequencyScale,
		)
	}
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	var attention *tensor.Tensor
	if spec.Architecture == "jina-bert-v2" {
		attention = builder.AttentionALiBiWithOffset(
			query, key, value, attentionScale, spec.MaxALiBiBias, false, 0,
		)
	} else {
		attention = builder.Attention(query, key, value, attentionScale, false)
	}
	attention = builder.Reshape(attention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	attention = builder.AffineLayerNorm(
		builder.Add(input, attention), weights.AttentionPostNorm,
		weights.AttentionPostNormBias, spec.LayerNormEpsilon,
	)
	if spec.Architecture == "jina-bert-v2" && weights.AttentionNorm2 != nil {
		attention = ApplyNormalization(
			builder, builder.Add(attention, input),
			weights.AttentionNorm2, weights.AttentionNorm2Bias, spec,
		)
	}
	var feedForward *tensor.Tensor
	if usesExperts {
		feedForward = builder.MoEGELU(
			attention, weights.FeedForwardRouter, nil, weights.FeedForwardUpExperts,
			weights.FeedForwardDownExperts, spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
		)
	} else {
		feedForward = builder.MulMat(weights.FeedForwardUp, attention)
		if weights.FeedForwardUpBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardUpBias)
		}
	}
	if !usesExperts {
		if spec.Architecture == "jina-bert-v2" {
			width := uint64(spec.FeedForwardLength)
			if weights.FeedForwardGate != nil {
				gate := builder.MulMat(weights.FeedForwardGate, attention)
				feedForward = builder.GEGLU(gate, feedForward)
			} else if weights.FeedForwardUp.Shape.Dims[1] == 2*width {
				gate := builder.Reshape(builder.GroupSlice(feedForward, 0, width, 1, 2*width), width, tokens)
				up := builder.Reshape(builder.GroupSlice(feedForward, width, width, 1, 2*width), width, tokens)
				feedForward = builder.GEGLU(gate, up)
			} else {
				feedForward = builder.GELU(feedForward)
			}
		} else if spec.Architecture == "nomic-bert" {
			gate := builder.MulMat(weights.FeedForwardGate, attention)
			feedForward = builder.SwiGLU(gate, feedForward)
		} else {
			feedForward = builder.GELU(feedForward)
		}
		feedForward = builder.MulMat(weights.FeedForwardDown, feedForward)
		if weights.FeedForwardDownBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
		}
	}
	output := builder.AffineLayerNorm(
		builder.Add(attention, feedForward), weights.FeedForwardPostNorm,
		weights.FeedForwardPostNormBias, spec.LayerNormEpsilon,
	)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: key, Value: value}, nil
}

func buildModernBERTBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "modern-bert" {
		return DenseBlockResult{}, errors.New("ModernBERT block requires ModernBERT architecture")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("ModernBERT block input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("ModernBERT block position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("ModernBERT block does not support a KV cache")
	}
	required := map[string]*tensor.Tensor{
		"attention QKV":     weights.AttentionQKV,
		"attention output":  weights.AttentionOutput,
		"feed-forward norm": weights.FeedForwardNorm,
		"feed-forward up":   weights.FeedForwardUp,
		"feed-forward down": weights.FeedForwardDown,
	}
	if layerIndex > 0 {
		required["attention norm"] = weights.AttentionNorm
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("ModernBERT block %s weight is nil", name)
		}
	}
	tokens := uint64(len(positions))
	normalized := input
	if weights.AttentionNorm != nil {
		normalized = ApplyNormalization(builder, input, weights.AttentionNorm, nil, spec)
	}
	mixed := builder.MulMat(weights.AttentionQKV, normalized)
	if weights.AttentionQKVBias != nil {
		mixed = builder.Add(mixed, weights.AttentionQKVBias)
	}
	width := uint64(spec.EmbeddingLength)
	query := builder.Reshape(builder.GroupSlice(mixed, 0, width, 1, 3*width), width, tokens)
	key := builder.Reshape(builder.GroupSlice(mixed, width, width, 1, 3*width), width, tokens)
	value := builder.Reshape(builder.GroupSlice(mixed, 2*width, width, 1, 3*width), width, tokens)
	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	frequencyBase := spec.RopeFrequencyBase
	if spec.IsSlidingLayer(layerIndex) {
		frequencyBase = spec.RopeFrequencySWA
	}
	frequencyScale := float32(1)
	if spec.RopeScalingType == "linear" {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	query = builder.RoPENeoXScaled(query, positions, spec.RopeDimensionCount, frequencyBase, frequencyScale)
	key = builder.RoPENeoXScaled(key, positions, spec.RopeDimensionCount, frequencyBase, frequencyScale)
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	var attention *tensor.Tensor
	if spec.IsSlidingLayer(layerIndex) {
		attention = builder.AttentionSymmetricWindow(query, key, value, attentionScale, spec.SlidingWindow)
	} else {
		attention = builder.Attention(query, key, value, attentionScale, false)
	}
	attention = builder.Reshape(attention, width, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	residual := builder.Add(input, attention)
	normalized = ApplyNormalization(builder, residual, weights.FeedForwardNorm, nil, spec)
	fused := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForwardWidth := uint64(spec.FeedForwardLength)
	gate := builder.Reshape(
		builder.GroupSlice(fused, 0, feedForwardWidth, 1, 2*feedForwardWidth),
		feedForwardWidth, tokens,
	)
	up := builder.Reshape(
		builder.GroupSlice(fused, feedForwardWidth, feedForwardWidth, 1, 2*feedForwardWidth),
		feedForwardWidth, tokens,
	)
	switch spec.HiddenActivation {
	case "swiglu":
		fused = builder.SwiGLU(gate, up)
	case "reglu":
		fused = builder.ReGLU(gate, up)
	default:
		fused = builder.GEGLU(gate, up)
	}
	output := builder.Add(residual, builder.MulMat(weights.FeedForwardDown, fused))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: key, Value: value}, nil
}

func buildGemmaEmbeddingBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "gemma-embedding" {
		return DenseBlockResult{}, errors.New("Gemma embedding block requires gemma-embedding architecture")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Gemma embedding block input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Gemma embedding block position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("Gemma embedding block does not support a KV cache")
	}
	if weights.AttentionQKV != nil &&
		(weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil) {
		return DenseBlockResult{}, errors.New("Gemma embedding fused QKV cannot use separate projection biases")
	}
	if weights.AttentionQKV == nil && weights.AttentionQKVBias != nil {
		return DenseBlockResult{}, errors.New("Gemma embedding fused QKV bias has no fused projection")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":         weights.AttentionNorm,
		"attention output":       weights.AttentionOutput,
		"attention Q norm":       weights.AttentionQNorm,
		"attention K norm":       weights.AttentionKNorm,
		"attention post norm":    weights.AttentionPostNorm,
		"feed-forward norm":      weights.FeedForwardNorm,
		"feed-forward gate":      weights.FeedForwardGate,
		"feed-forward up":        weights.FeedForwardUp,
		"feed-forward down":      weights.FeedForwardDown,
		"feed-forward post norm": weights.FeedForwardPostNorm,
	}
	if weights.AttentionQKV != nil {
		required["attention QKV"] = weights.AttentionQKV
	} else {
		required["attention Q"] = weights.AttentionQ
		required["attention K"] = weights.AttentionK
		required["attention V"] = weights.AttentionV
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Gemma embedding block %s weight is nil", name)
		}
	}
	tokens := uint64(len(positions))
	headCount := uint64(spec.HeadCount)
	kvHeadCount := uint64(spec.HeadCountKV)
	queryLength := headCount * uint64(spec.KeyLength)
	keyLength := kvHeadCount * uint64(spec.KeyLength)
	valueLength := kvHeadCount * uint64(spec.ValueLength)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryLength + keyLength + valueLength
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryLength, 1, stride), queryLength, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryLength, keyLength, 1, stride), keyLength, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride), valueLength, tokens)
	} else {
		query = builder.MulMat(weights.AttentionQ, normalized)
		key = builder.MulMat(weights.AttentionK, normalized)
		value = builder.MulMat(weights.AttentionV, normalized)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
	}
	query = builder.Reshape(query, uint64(spec.KeyLength), headCount, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeadCount, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeadCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	frequencyBase := spec.RopeFrequencyBase
	if spec.IsSlidingLayer(layerIndex) {
		frequencyBase = spec.RopeFrequencySWA
	}
	frequencyScale := float32(1)
	if spec.RopeScalingType == "linear" {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	query = builder.RoPENeoXScaled(query, positions, spec.RopeDimensionCount, frequencyBase, frequencyScale)
	key = builder.RoPENeoXScaled(key, positions, spec.RopeDimensionCount, frequencyBase, frequencyScale)
	query = builder.Scale(query, float32(1/math.Sqrt(float64(spec.KeyLength))))
	var attention *tensor.Tensor
	if spec.IsSlidingLayer(layerIndex) {
		attention = builder.AttentionSymmetricWindow(query, key, value, 1, spec.SlidingWindow)
	} else {
		attention = builder.Attention(query, key, value, 1, false)
	}
	attention = builder.Reshape(attention, headCount*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	attention = builder.WeightedRMSNorm(attention, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	residual := builder.Add(input, attention)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
	feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: key, Value: value}, nil
}

func buildTalkieBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Talkie block input shape is incompatible")
	}
	if weights.EmbeddingSkip == nil || !weights.EmbeddingSkip.Shape.Equal(input.Shape) {
		return DenseBlockResult{}, errors.New("Talkie block embedding skip is missing or incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Talkie block position count is incompatible")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("Talkie block past key/value cache must both be present")
	}
	if weights.AttentionQKV != nil &&
		(weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil) {
		return DenseBlockResult{}, errors.New("Talkie fused QKV cannot use separate projection biases")
	}
	if weights.AttentionQKV == nil && weights.AttentionQKVBias != nil {
		return DenseBlockResult{}, errors.New("Talkie fused QKV bias has no fused projection")
	}
	required := map[string]*tensor.Tensor{
		"attention output":   weights.AttentionOutput,
		"attention Q norm":   weights.AttentionQNorm,
		"feed-forward gate":  weights.FeedForwardGate,
		"feed-forward up":    weights.FeedForwardUp,
		"feed-forward down":  weights.FeedForwardDown,
		"layer output scale": weights.LayerOutputScale,
	}
	if weights.AttentionQKV != nil {
		required["attention QKV"] = weights.AttentionQKV
	} else {
		required["attention Q"] = weights.AttentionQ
		required["attention K"] = weights.AttentionK
		required["attention V"] = weights.AttentionV
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Talkie block %s weight is nil", name)
		}
	}

	tokens := uint64(len(positions))
	headCount := uint64(spec.HeadCount)
	kvHeadCount := uint64(spec.HeadCountKV)
	queryLength := headCount * uint64(spec.KeyLength)
	keyLength := kvHeadCount * uint64(spec.KeyLength)
	valueLength := kvHeadCount * uint64(spec.ValueLength)
	normalized := builder.RMSNorm(input, spec.RMSNormEpsilon)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryLength + keyLength + valueLength
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryLength, 1, stride), queryLength, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryLength, keyLength, 1, stride), keyLength, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride), valueLength, tokens)
	} else {
		query = builder.MulMat(weights.AttentionQ, normalized)
		key = builder.MulMat(weights.AttentionK, normalized)
		value = builder.MulMat(weights.AttentionV, normalized)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
	}
	query = builder.Reshape(query, uint64(spec.KeyLength), headCount, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeadCount, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeadCount, tokens)
	query = builder.RoPENeoXScaled(query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
	key = builder.RoPENeoXScaled(key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.RMSNorm(key, spec.RMSNormEpsilon)

	cacheKey := key
	cacheValue := value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("Talkie KV cache token count exceeds uint32")
		}
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attention := builder.AttentionWithOffset(
		query, cacheKey, cacheValue,
		float32(1/math.Sqrt(float64(spec.KeyLength))), true, queryStart,
	)
	attention = builder.Reshape(attention, headCount*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	residual := builder.Add(input, attention)
	normalized = builder.RMSNorm(residual, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	output := builder.Add(residual, feedForward)
	output = builder.Add(output, builder.Multiply(weights.EmbeddingSkip, weights.LayerOutputScale))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}

// BuildT5EncoderBlock: full bidirectional encoder block.
func BuildT5EncoderBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if builder == nil || input == nil {
		return nil, errors.New("T5 encoder block input is nil")
	}
	if spec.Architecture != "t5" && spec.Architecture != "t5encoder" {
		return nil, errors.New("T5 encoder block requires T5 architecture")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":          weights.AttentionNorm,
		"attention Q":             weights.AttentionQ,
		"attention K":             weights.AttentionK,
		"attention V":             weights.AttentionV,
		"attention output":        weights.AttentionOutput,
		"attention relative bias": weights.AttentionRelativeBias,
		"feed-forward norm":       weights.FeedForwardNorm,
		"feed-forward up":         weights.FeedForwardUp,
		"feed-forward down":       weights.FeedForwardDown,
	}
	for name, item := range required {
		if item == nil {
			return nil, fmt.Errorf("T5 encoder block %s weight is nil", name)
		}
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("T5 encoder block input shape is incompatible")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := builder.Reshape(
		builder.MulMat(weights.AttentionQ, normalized),
		uint64(spec.KeyLength),
		uint64(spec.HeadCount),
		tokens,
	)
	key := builder.Reshape(
		builder.MulMat(weights.AttentionK, normalized),
		uint64(spec.KeyLength),
		uint64(spec.HeadCountKV),
		tokens,
	)
	value := builder.Reshape(
		builder.MulMat(weights.AttentionV, normalized),
		uint64(spec.ValueLength),
		uint64(spec.HeadCountKV),
		tokens,
	)
	attention := builder.AttentionWithRelativeBias(
		query,
		key,
		value,
		weights.AttentionRelativeBias,
		1,
	)
	attention = builder.Reshape(
		attention,
		uint64(spec.HeadCount)*uint64(spec.ValueLength),
		tokens,
	)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	residual := builder.Add(input, attention)
	normalized = builder.WeightedRMSNorm(
		residual,
		weights.FeedForwardNorm,
		spec.RMSNormEpsilon,
	)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	var activated *tensor.Tensor
	if weights.FeedForwardGate != nil {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		activated = builder.GEGLU(gate, up)
	} else {
		activated = builder.ReLU(up)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, activated)
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildT5DecoderBlockCached: causal self-attention plus fixed cross-attention.
func BuildT5DecoderBlockCached(
	builder *tensor.Builder,
	input, encoder *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastSelfKey, pastSelfValue, pastCrossKey, pastCrossValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil {
		return DenseBlockResult{}, errors.New("T5 decoder block input is nil")
	}
	if spec.Architecture != "t5" {
		return DenseBlockResult{}, errors.New("T5 decoder block requires T5 architecture")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":          weights.AttentionNorm,
		"attention Q":             weights.AttentionQ,
		"attention K":             weights.AttentionK,
		"attention V":             weights.AttentionV,
		"attention output":        weights.AttentionOutput,
		"attention relative bias": weights.AttentionRelativeBias,
		"cross-attention norm":    weights.CrossAttentionNorm,
		"cross-attention Q":       weights.CrossAttentionQ,
		"cross-attention K":       weights.CrossAttentionK,
		"cross-attention V":       weights.CrossAttentionV,
		"cross-attention output":  weights.CrossAttentionOutput,
		"feed-forward norm":       weights.FeedForwardNorm,
		"feed-forward up":         weights.FeedForwardUp,
		"feed-forward down":       weights.FeedForwardDown,
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("T5 decoder block %s weight is nil", name)
		}
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("T5 decoder block input shape is incompatible")
	}
	if (pastSelfKey == nil) != (pastSelfValue == nil) {
		return DenseBlockResult{}, errors.New("T5 decoder self cache is incomplete")
	}
	if (pastCrossKey == nil) != (pastCrossValue == nil) {
		return DenseBlockResult{}, errors.New("T5 decoder cross cache is incomplete")
	}
	if pastCrossKey == nil && encoder == nil {
		return DenseBlockResult{}, errors.New("T5 decoder encoder state is nil")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := builder.Reshape(
		builder.MulMat(weights.AttentionQ, normalized),
		uint64(spec.KeyLength), uint64(spec.HeadCount), tokens,
	)
	key := builder.Reshape(
		builder.MulMat(weights.AttentionK, normalized),
		uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens,
	)
	value := builder.Reshape(
		builder.MulMat(weights.AttentionV, normalized),
		uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens,
	)
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastSelfKey != nil {
		if pastSelfKey.Shape.Rank != 3 || pastSelfValue.Shape.Rank != 3 || pastSelfKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("T5 decoder self cache shape is incompatible")
		}
		queryStart = uint32(pastSelfKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastSelfKey, key, 2)
		cacheValue = builder.Concat(pastSelfValue, value, 2)
	}
	attention := builder.AttentionWithRelativeBiasAndOffset(
		query, cacheKey, cacheValue, weights.AttentionRelativeBias, 1, queryStart,
	)
	attention = builder.Reshape(attention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	residual := builder.Add(input, builder.MulMat(weights.AttentionOutput, attention))

	normalized = builder.WeightedRMSNorm(residual, weights.CrossAttentionNorm, spec.RMSNormEpsilon)
	crossQuery := builder.Reshape(
		builder.MulMat(weights.CrossAttentionQ, normalized),
		uint64(spec.KeyLength), uint64(spec.HeadCount), tokens,
	)
	crossKey, crossValue := pastCrossKey, pastCrossValue
	if crossKey == nil {
		if encoder.Shape.Rank != 2 || encoder.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
			return DenseBlockResult{}, errors.New("T5 decoder encoder state shape is incompatible")
		}
		encoderTokens := encoder.Shape.Dims[1]
		crossKey = builder.Reshape(
			builder.MulMat(weights.CrossAttentionK, encoder),
			uint64(spec.KeyLength), uint64(spec.HeadCountKV), encoderTokens,
		)
		crossValue = builder.Reshape(
			builder.MulMat(weights.CrossAttentionV, encoder),
			uint64(spec.ValueLength), uint64(spec.HeadCountKV), encoderTokens,
		)
	}
	crossAttention := builder.AttentionWithOffset(crossQuery, crossKey, crossValue, 1, false, 0)
	crossAttention = builder.Reshape(crossAttention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	residual = builder.Add(residual, builder.MulMat(weights.CrossAttentionOutput, crossAttention))

	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	var activated *tensor.Tensor
	if weights.FeedForwardGate != nil {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		activated = builder.GEGLU(gate, up)
	} else {
		activated = builder.ReLU(up)
	}
	output := builder.Add(residual, builder.MulMat(weights.FeedForwardDown, activated))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: output,
		Key:    cacheKey,
		Value:  cacheValue,
		FixedStates: map[string]*tensor.Tensor{
			"cross_key": crossKey, "cross_value": crossValue,
		},
	}, nil
}

type DenseBlockResult struct {
	Output      *tensor.Tensor
	Key         *tensor.Tensor
	Value       *tensor.Tensor
	Auxiliary   *tensor.Tensor
	States      map[string]*tensor.Tensor
	FixedStates map[string]*tensor.Tensor
}

type Qwen35BlockResult struct {
	Output    *tensor.Tensor
	Key       *tensor.Tensor
	Value     *tensor.Tensor
	ConvState *tensor.Tensor
	SSMState  *tensor.Tensor
	Recurrent bool
}

type LFM2BlockResult struct {
	Output    *tensor.Tensor
	Key       *tensor.Tensor
	Value     *tensor.Tensor
	Recurrent bool
}

func BuildMambaBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("Mamba block input/state is nil")
	}
	if (spec.Architecture != "mamba" && spec.Architecture != "jamba") || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Mamba block architecture/input is invalid")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":       weights.AttentionNorm,
		"SSM input":            weights.SSMInput,
		"SSM convolution":      weights.SSMConv1D,
		"SSM convolution bias": weights.SSMConv1DBias,
		"SSM X":                weights.SSMX,
		"SSM time-step weight": weights.SSMTimeStepWeight,
		"SSM time-step bias":   weights.SSMTimeStep,
		"SSM A":                weights.SSMA,
		"SSM D":                weights.SSMD,
		"SSM output":           weights.SSMOutput,
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Mamba block %s weight is nil", name)
		}
	}
	if spec.Architecture == "jamba" {
		for name, item := range map[string]*tensor.Tensor{
			"SSM time-step norm": weights.SSMTimeStepNorm,
			"SSM B norm":         weights.SSMBNorm,
			"SSM C norm":         weights.SSMCNorm,
		} {
			if item == nil {
				return DenseBlockResult{}, fmt.Errorf("Jamba block %s weight is nil", name)
			}
		}
	}
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), uint64(spec.SSMInnerSize))
	ssmShape := tensor.MustShape(uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize))
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("Mamba recurrent cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	rank := uint64(spec.SSMTimeStepRank)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	xz := builder.MulMat(weights.SSMInput, normalized)
	x := builder.Reshape(builder.GroupSlice(xz, 0, inner, 1, inner), inner, tokens)
	z := builder.Reshape(builder.GroupSlice(xz, inner, inner, 1, inner), inner, tokens)
	convInput := builder.Concat(convState, builder.Transpose2D(x), 0)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, uint64(spec.SSMConvKernel-1), 1, uint64(spec.SSMConvKernel-1)),
		uint64(spec.SSMConvKernel-1), inner,
	)
	x = builder.SiLU(builder.Add(builder.SSMConv(convInput, weights.SSMConv1D), weights.SSMConv1DBias))
	xdb := builder.MulMat(weights.SSMX, x)
	dt := builder.Reshape(builder.GroupSlice(xdb, 0, rank, 1, rank), rank, tokens)
	beta := builder.Reshape(
		builder.GroupSlice(xdb, rank, stateWidth, 1, stateWidth),
		stateWidth, 1, tokens, 1,
	)
	c := builder.Reshape(
		builder.GroupSlice(xdb, rank+stateWidth, stateWidth, 1, stateWidth),
		stateWidth, 1, tokens, 1,
	)
	if spec.SSMDtBCNorm {
		dt = builder.RMSNorm(dt, spec.RMSNormEpsilon)
		beta = builder.RMSNorm(beta, spec.RMSNormEpsilon)
		c = builder.RMSNorm(c, spec.RMSNormEpsilon)
	} else if spec.Architecture == "jamba" {
		dt = builder.WeightedRMSNorm(dt, weights.SSMTimeStepNorm, spec.RMSNormEpsilon)
		beta = builder.WeightedRMSNorm(beta, weights.SSMBNorm, spec.RMSNormEpsilon)
		c = builder.WeightedRMSNorm(c, weights.SSMCNorm, spec.RMSNormEpsilon)
	}
	dt = builder.Add(builder.MulMat(weights.SSMTimeStepWeight, dt), weights.SSMTimeStep)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, 1, inner, 1),
		builder.Reshape(x, 1, inner, tokens, 1),
		builder.Reshape(dt, inner, tokens, 1),
		weights.SSMA,
		beta,
		c,
	)
	attentionElements := inner * tokens
	attention := builder.FlatSlice(packed, 0, inner, tokens)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	attention = builder.Add(attention, builder.Multiply(x, weights.SSMD))
	attention = builder.Multiply(attention, builder.SiLU(z))
	attention = builder.MulMat(weights.SSMOutput, attention)
	output := builder.Add(input, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextConvState, Value: nextSSMState}, nil
}

func BuildJambaRecurrentBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Architecture != "jamba" {
		return DenseBlockResult{}, errors.New("Jamba recurrent block architecture is invalid")
	}
	result, err := BuildMambaBlockCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	if weights.FeedForwardNorm == nil {
		return DenseBlockResult{}, errors.New("Jamba feed-forward norm is nil")
	}
	normalized := builder.WeightedRMSNorm(result.Output, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var feedForward *tensor.Tensor
	if weights.FeedForwardRouter != nil {
		if weights.FeedForwardGateExperts == nil || weights.FeedForwardUpExperts == nil ||
			weights.FeedForwardDownExperts == nil {
			return DenseBlockResult{}, errors.New("Jamba expert catalog is incomplete")
		}
		feedForward = builder.MoE(
			normalized, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
			weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
			spec.ExpertUsedCount, false, spec.ExpertWeightsScale,
		)
	} else {
		if weights.FeedForwardGate == nil || weights.FeedForwardUp == nil || weights.FeedForwardDown == nil {
			return DenseBlockResult{}, errors.New("Jamba dense FFN catalog is incomplete")
		}
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	}
	result.Output = builder.Add(result.Output, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return result, nil
}

func buildMamba2MixerCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("Mamba2 block input/state is nil")
	}
	if (spec.Architecture != "mamba2" && spec.Architecture != "granitehybrid" && spec.Architecture != "falcon-h1" &&
		spec.Architecture != "nemotron_h" && spec.Architecture != "nemotron_h_moe") || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Mamba2 mixer architecture/input is invalid")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":     weights.AttentionNorm,
		"SSM input":          weights.SSMInput,
		"SSM convolution":    weights.SSMConv1D,
		"SSM time-step bias": weights.SSMTimeStep,
		"SSM A":              weights.SSMA,
		"SSM D":              weights.SSMD,
		"SSM output":         weights.SSMOutput,
	}
	if spec.Architecture != "falcon-h1" {
		required["SSM norm"] = weights.SSMNorm
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Mamba2 block %s weight is nil", name)
		}
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	heads := uint64(spec.SSMTimeStepRank)
	groups := uint64(spec.SSMGroupCount)
	headWidth := inner / heads
	convWidth := inner + 2*groups*stateWidth
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), convWidth)
	ssmShape := tensor.MustShape(stateWidth, inner)
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("Mamba2 recurrent cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	zxBCdt := builder.MulMat(weights.SSMInput, normalized)
	z := builder.Reshape(builder.GroupSlice(zxBCdt, 0, headWidth, heads, headWidth), headWidth, heads, tokens, 1)
	xBC := builder.Reshape(builder.GroupSlice(zxBCdt, inner, convWidth, 1, convWidth), convWidth, tokens)
	dt := builder.Reshape(
		builder.GroupSlice(zxBCdt, inner+convWidth, heads, 1, heads), heads, tokens, 1,
	)
	convInput := builder.Concat(convState, builder.Transpose2D(xBC), 0)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, uint64(spec.SSMConvKernel-1), 1, uint64(spec.SSMConvKernel-1)),
		uint64(spec.SSMConvKernel-1), convWidth,
	)
	xBC = builder.SSMConv(convInput, weights.SSMConv1D)
	if weights.SSMConv1DBias != nil {
		xBC = builder.Add(xBC, weights.SSMConv1DBias)
	}
	xBC = builder.SiLU(xBC)
	x := builder.Reshape(builder.GroupSlice(xBC, 0, headWidth, heads, headWidth), headWidth, heads, tokens, 1)
	beta := builder.Reshape(
		builder.GroupSlice(xBC, inner, stateWidth, groups, stateWidth), stateWidth, groups, tokens, 1,
	)
	c := builder.Reshape(
		builder.GroupSlice(xBC, inner+groups*stateWidth, stateWidth, groups, stateWidth),
		stateWidth, groups, tokens, 1,
	)
	dt = builder.Add(dt, weights.SSMTimeStep)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, headWidth, heads, 1),
		x,
		dt,
		weights.SSMA,
		beta,
		c,
	)
	attentionElements := inner * tokens
	attention := builder.FlatSlice(packed, 0, headWidth, heads, tokens, 1)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	attention = builder.Add(attention, builder.Multiply(x, weights.SSMD))
	attention = builder.Multiply(attention, builder.SiLU(z))
	attention = builder.Reshape(attention, inner/groups, groups, tokens, 1)
	if weights.SSMNorm != nil {
		attention = builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon)
	}
	attention = builder.Reshape(attention, inner, tokens)
	attention = builder.MulMat(weights.SSMOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextConvState, Value: nextSSMState}, nil
}

func BuildFalconH1BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("Falcon-H1 block input/state is nil")
	}
	if spec.Architecture != "falcon-h1" || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Falcon-H1 block architecture/input is invalid")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Falcon-H1 block position count is invalid")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("Falcon-H1 KV cache is incomplete")
	}
	for name, item := range map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm, "attention output": weights.AttentionOutput,
		"feed-forward norm": weights.FeedForwardNorm, "feed-forward gate": weights.FeedForwardGate,
		"feed-forward up": weights.FeedForwardUp, "feed-forward down": weights.FeedForwardDown,
	} {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Falcon-H1 block %s weight is nil", name)
		}
	}
	if weights.AttentionQKV == nil && (weights.AttentionQ == nil || weights.AttentionK == nil || weights.AttentionV == nil) {
		return DenseBlockResult{}, errors.New("Falcon-H1 attention projection catalog is incomplete")
	}

	tokens := uint64(len(positions))
	heads := uint64(spec.HeadCount)
	kvHeads := uint64(spec.HeadCountKV)
	queryWidth := heads * uint64(spec.KeyLength)
	keyWidth := kvHeads * uint64(spec.KeyLength)
	valueWidth := kvHeads * uint64(spec.ValueLength)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryWidth + keyWidth + valueWidth
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryWidth, 1, stride), queryWidth, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryWidth, keyWidth, 1, stride), keyWidth, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryWidth+keyWidth, valueWidth, 1, stride), valueWidth, tokens)
	} else {
		query = builder.MulMat(weights.AttentionQ, normalized)
		key = builder.MulMat(weights.AttentionK, normalized)
		value = builder.MulMat(weights.AttentionV, normalized)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
	}
	query = builder.Reshape(query, uint64(spec.KeyLength), heads, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeads, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeads, tokens)
	if weights.RopeFactors != nil {
		query = builder.RoPENeoXScaledWithFactors(query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1, weights.RopeFactors)
		key = builder.RoPENeoXScaledWithFactors(key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1, weights.RopeFactors)
	} else {
		query = builder.RoPENeoXScaled(query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
		key = builder.RoPENeoXScaled(key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
	}
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Rank != 3 || pastValue.Shape.Rank != 3 || pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("Falcon-H1 KV cache shape is invalid")
		}
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	if spec.AttentionScale > 0 {
		attentionScale = spec.AttentionScale
	}
	attention := builder.AttentionWithOffset(query, cacheKey, cacheValue, attentionScale, true, queryStart)
	attention = builder.Reshape(attention, heads*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}

	ssm, err := buildMamba2MixerCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	residual := builder.Add(input, builder.Add(attention, ssm.Output))
	ffnInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, ffnInput)
	up := builder.MulMat(weights.FeedForwardUp, ffnInput)
	if weights.FeedForwardGateBias != nil {
		gate = builder.Add(gate, weights.FeedForwardGateBias)
	}
	if weights.FeedForwardUpBias != nil {
		up = builder.Add(up, weights.FeedForwardUpBias)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	if weights.FeedForwardDownBias != nil {
		feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
	}
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: output, Key: cacheKey, Value: cacheValue,
		FixedStates: map[string]*tensor.Tensor{"conv_state": ssm.Key, "ssm_state": ssm.Value},
	}, nil
}

func BuildMamba2BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Architecture != "mamba2" {
		return DenseBlockResult{}, errors.New("Mamba2 block architecture is invalid")
	}
	if weights.SSMConv1DBias == nil {
		return DenseBlockResult{}, errors.New("Mamba2 block SSM convolution bias is nil")
	}
	result, err := buildMamba2MixerCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	result.Output = builder.Add(input, result.Output)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return result, nil
}

func BuildGraniteHybridRecurrentBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Architecture != "granitehybrid" {
		return DenseBlockResult{}, errors.New("Granite Hybrid recurrent block architecture is invalid")
	}
	result, err := buildMamba2MixerCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	mixer := result.Output
	if spec.ResidualScale > 0 {
		mixer = builder.Scale(mixer, spec.ResidualScale)
	}
	residual := builder.Add(input, mixer)
	if weights.FeedForwardNorm == nil {
		return DenseBlockResult{}, errors.New("Granite Hybrid feed-forward norm is nil")
	}
	normalized := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var feedForward *tensor.Tensor
	if weights.FeedForwardRouter != nil {
		if weights.FeedForwardUpExperts == nil || weights.FeedForwardDownExperts == nil {
			return DenseBlockResult{}, errors.New("Granite Hybrid expert catalog is incomplete")
		}
		if weights.FeedForwardGateExperts == nil {
			feedForward = builder.MoEUngated(
				normalized, weights.FeedForwardRouter, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		} else {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
				weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
				spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		}
		if spec.SharedExpertFF > 0 {
			if weights.FeedForwardSharedGate == nil || weights.FeedForwardSharedUp == nil ||
				weights.FeedForwardSharedDown == nil {
				return DenseBlockResult{}, errors.New("Granite Hybrid shared expert catalog is incomplete")
			}
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp))
			feedForward = builder.Add(feedForward, shared)
		}
	} else {
		if weights.FeedForwardGate == nil || weights.FeedForwardUp == nil || weights.FeedForwardDown == nil {
			return DenseBlockResult{}, errors.New("Granite Hybrid dense FFN catalog is incomplete")
		}
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		if weights.FeedForwardGateBias != nil {
			gate = builder.Add(gate, weights.FeedForwardGateBias)
		}
		if weights.FeedForwardUpBias != nil {
			up = builder.Add(up, weights.FeedForwardUpBias)
		}
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
		if weights.FeedForwardDownBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
		}
	}
	if spec.ResidualScale > 0 {
		feedForward = builder.Scale(feedForward, spec.ResidualScale)
	}
	result.Output = builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return result, nil
}

func BuildPLaMo2RecurrentBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("PLaMo2 block input/state is nil")
	}
	if spec.Architecture != "plamo2" || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("PLaMo2 block architecture/input is invalid")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":         weights.AttentionNorm,
		"attention post norm":    weights.AttentionPostNorm,
		"SSM input":              weights.SSMInput,
		"SSM convolution":        weights.SSMConv1D,
		"SSM X":                  weights.SSMX,
		"SSM time-step weight":   weights.SSMTimeStepWeight,
		"SSM time-step bias":     weights.SSMTimeStep,
		"SSM time-step norm":     weights.SSMTimeStepNorm,
		"SSM A":                  weights.SSMA,
		"SSM D":                  weights.SSMD,
		"SSM B norm":             weights.SSMBNorm,
		"SSM C norm":             weights.SSMCNorm,
		"SSM output":             weights.SSMOutput,
		"feed-forward norm":      weights.FeedForwardNorm,
		"feed-forward up":        weights.FeedForwardUp,
		"feed-forward down":      weights.FeedForwardDown,
		"feed-forward post norm": weights.FeedForwardPostNorm,
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("PLaMo2 block %s weight is nil", name)
		}
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	heads := uint64(spec.SSMTimeStepRank)
	headWidth := inner / heads
	dtWidth := uint64(64)
	if candidate := uint64(spec.EmbeddingLength / 16); candidate > dtWidth {
		dtWidth = candidate
	}
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), inner)
	ssmShape := tensor.MustShape(stateWidth, inner)
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("PLaMo2 recurrent cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	zx := builder.MulMat(weights.SSMInput, normalized)
	z := builder.Reshape(builder.GroupSlice(zx, 0, headWidth, heads, 2*headWidth), headWidth, heads, tokens, 1)
	x := builder.Reshape(builder.GroupSlice(zx, headWidth, headWidth, heads, 2*headWidth), inner, tokens)
	convInput := builder.Concat(convState, builder.Transpose2D(x), 0)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, uint64(spec.SSMConvKernel-1), 1, uint64(spec.SSMConvKernel-1)),
		uint64(spec.SSMConvKernel-1), inner,
	)
	x = builder.SiLU(builder.SSMConv(convInput, weights.SSMConv1D))
	bcdt := builder.MulMat(weights.SSMX, x)
	beta := builder.Reshape(builder.GroupSlice(bcdt, 0, stateWidth, 1, stateWidth), stateWidth, 1, tokens, 1)
	c := builder.Reshape(builder.GroupSlice(bcdt, stateWidth, stateWidth, 1, stateWidth), stateWidth, 1, tokens, 1)
	dt := builder.Reshape(builder.GroupSlice(bcdt, 2*stateWidth, dtWidth, 1, dtWidth), dtWidth, tokens)
	beta = builder.WeightedRMSNorm(beta, weights.SSMBNorm, spec.RMSNormEpsilon)
	c = builder.WeightedRMSNorm(c, weights.SSMCNorm, spec.RMSNormEpsilon)
	dt = builder.WeightedRMSNorm(dt, weights.SSMTimeStepNorm, spec.RMSNormEpsilon)
	dt = builder.Add(builder.MulMat(weights.SSMTimeStepWeight, dt), weights.SSMTimeStep)
	x = builder.Reshape(x, headWidth, heads, tokens, 1)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, headWidth, heads, 1),
		x,
		builder.Reshape(dt, heads, tokens, 1),
		builder.Reshape(weights.SSMA, 1, heads),
		beta,
		c,
	)
	attentionElements := inner * tokens
	mixer := builder.FlatSlice(packed, 0, headWidth, heads, tokens, 1)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	mixer = builder.Add(mixer, builder.Multiply(x, builder.Reshape(weights.SSMD, 1, heads)))
	mixer = builder.Multiply(mixer, builder.SiLU(z))
	mixer = builder.MulMat(weights.SSMOutput, builder.Reshape(mixer, inner, tokens))
	mixer = builder.WeightedRMSNorm(mixer, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	residual := builder.Add(input, mixer)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	fused := builder.MulMat(weights.FeedForwardUp, normalized)
	gate := builder.Reshape(builder.GroupSlice(fused, 0, uint64(spec.FeedForwardLength), 1, 2*uint64(spec.FeedForwardLength)), uint64(spec.FeedForwardLength), tokens)
	up := builder.Reshape(builder.GroupSlice(fused, uint64(spec.FeedForwardLength), uint64(spec.FeedForwardLength), 1, 2*uint64(spec.FeedForwardLength)), uint64(spec.FeedForwardLength), tokens)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextConvState, Value: nextSSMState}, nil
}

// BuildNemotronHBlockCached: attention, Mamba2, or FFN mixer.
func BuildNemotronHBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || input.Shape.Rank != 2 ||
		(spec.Architecture != "nemotron_h" && spec.Architecture != "nemotron_h_moe") {
		return DenseBlockResult{}, errors.New("Nemotron-H block architecture/input is invalid")
	}
	if weights.AttentionNorm == nil {
		return DenseBlockResult{}, errors.New("Nemotron-H block norm is nil")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Nemotron-H position count is invalid")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("Nemotron-H cache must contain both tensors")
	}
	if spec.IsRecurrentLayer(layerIndex) {
		result, err := buildMamba2MixerCached(builder, input, spec, weights, pastKey, pastValue)
		if err != nil {
			return DenseBlockResult{}, err
		}
		result.Output = builder.Add(input, result.Output)
		return result, builder.Err()
	}
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	tokens := input.Shape.Dims[1]
	if spec.LayerFeedForwardLength(layerIndex) == 0 {
		for name, item := range map[string]*tensor.Tensor{
			"attention Q": weights.AttentionQ, "attention K": weights.AttentionK,
			"attention V": weights.AttentionV, "attention output": weights.AttentionOutput,
		} {
			if item == nil {
				return DenseBlockResult{}, fmt.Errorf("Nemotron-H %s weight is nil", name)
			}
		}
		query := builder.MulMat(weights.AttentionQ, normalized)
		key := builder.MulMat(weights.AttentionK, normalized)
		value := builder.MulMat(weights.AttentionV, normalized)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
		heads := uint64(spec.LayerHeadCount(layerIndex))
		kvHeads := uint64(spec.LayerKVHeadCount(layerIndex))
		query = builder.Reshape(query, uint64(spec.KeyLength), heads, tokens)
		key = builder.Reshape(key, uint64(spec.KeyLength), kvHeads, tokens)
		value = builder.Reshape(value, uint64(spec.ValueLength), kvHeads, tokens)
		cacheKey, cacheValue := key, value
		var queryStart uint32
		if pastKey != nil {
			if pastKey.Shape.Dims[2] > math.MaxUint32 {
				return DenseBlockResult{}, errors.New("Nemotron-H cache token count exceeds uint32")
			}
			queryStart = uint32(pastKey.Shape.Dims[2])
			cacheKey = builder.Concat(pastKey, key, 2)
			cacheValue = builder.Concat(pastValue, value, 2)
		}
		scale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
		if spec.AttentionScale > 0 {
			scale = spec.AttentionScale
		}
		attention := builder.AttentionWithOffset(query, cacheKey, cacheValue, scale, true, queryStart)
		attention = builder.Reshape(attention, heads*uint64(spec.ValueLength), tokens)
		attention = builder.MulMat(weights.AttentionOutput, attention)
		if weights.AttentionOutputBias != nil {
			attention = builder.Add(attention, weights.AttentionOutputBias)
		}
		return DenseBlockResult{Output: builder.Add(input, attention), Key: cacheKey, Value: cacheValue}, builder.Err()
	}
	sentinel := builder.GroupSlice(input, 0, 1, 1, input.Shape.Dims[0])
	cacheKey, cacheValue := sentinel, sentinel
	if pastKey != nil {
		wantPrefix := tensor.MustShape(1, 1, pastKey.Shape.Dims[2])
		if !pastKey.Shape.Equal(wantPrefix) || !pastValue.Shape.Equal(wantPrefix) {
			return DenseBlockResult{}, errors.New("Nemotron-H FFN sentinel cache shape is invalid")
		}
		cacheKey = builder.Concat(pastKey, sentinel, 2)
		cacheValue = builder.Concat(pastValue, sentinel, 2)
	}
	var feedForward *tensor.Tensor
	if spec.Architecture == "nemotron_h_moe" {
		for name, item := range map[string]*tensor.Tensor{
			"router": weights.FeedForwardRouter, "expert bias": weights.FeedForwardExpertBias,
			"expert up": weights.FeedForwardUpExperts, "expert down": weights.FeedForwardDownExperts,
			"shared up": weights.FeedForwardSharedUp, "shared down": weights.FeedForwardSharedDown,
		} {
			if item == nil {
				return DenseBlockResult{}, fmt.Errorf("Nemotron-H MoE %s weight is nil", name)
			}
		}
		expertInput := normalized
		if weights.FeedForwardLatentDown != nil {
			if weights.FeedForwardLatentUp == nil {
				return DenseBlockResult{}, errors.New("Nemotron-H MoE latent projection is incomplete")
			}
			expertInput = builder.MulMat(weights.FeedForwardLatentDown, normalized)
		} else if weights.FeedForwardLatentUp != nil {
			return DenseBlockResult{}, errors.New("Nemotron-H MoE latent projection is incomplete")
		}
		feedForward = builder.MoEReLUSquaredWithRouterInput(
			expertInput, normalized, weights.FeedForwardRouter, weights.FeedForwardUpExperts,
			weights.FeedForwardDownExperts, weights.FeedForwardExpertBias, spec.ExpertUsedCount,
			spec.ExpertWeightsNorm, spec.ExpertWeightsScale, tensor.MoERoutingSigmoid,
		)
		if weights.FeedForwardLatentUp != nil {
			feedForward = builder.MulMat(weights.FeedForwardLatentUp, feedForward)
		}
		shared := builder.MulMat(weights.FeedForwardSharedUp, normalized)
		shared = builder.MulMat(weights.FeedForwardSharedDown, builder.ReLUSquared(shared))
		feedForward = builder.Add(feedForward, shared)
	} else {
		if weights.FeedForwardUp == nil || weights.FeedForwardDown == nil {
			return DenseBlockResult{}, errors.New("Nemotron-H dense FFN catalog is incomplete")
		}
		feedForward = builder.MulMat(weights.FeedForwardUp, normalized)
		if weights.FeedForwardUpBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardUpBias)
		}
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.ReLUSquared(feedForward))
		if weights.FeedForwardDownBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
		}
	}
	return DenseBlockResult{Output: builder.Add(input, feedForward), Key: cacheKey, Value: cacheValue}, builder.Err()
}

// BuildPLMBlockCached: PLM MLA block
func BuildPLMBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Architecture != "plm" {
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
	if attentionPolicy != AttentionMLA && attentionPolicy != AttentionDSA && spec.Architecture != "kimi-linear" {
		return DenseBlockResult{}, errors.New("MLA block architecture is unsupported")
	}
	isMiniCPM3 := spec.Architecture == "minicpm3"
	isDeepSeek2 := profile.Has(ArchitectureDeepSeek2)
	isDSA := attentionPolicy == AttentionDSA
	isDeepSeek32 := spec.Architecture == "deepseek32"
	isKimi := spec.Architecture == "kimi-linear"
	required := map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm, "attention Q": weights.AttentionQ,
		"attention KV-A": weights.AttentionKVAMQA, "attention KV-A norm": weights.AttentionKVANorm,
		"attention output": weights.AttentionOutput, "feed-forward norm": weights.FeedForwardNorm,
	}
	if weights.AttentionKVB != nil {
		required["attention KV-B"] = weights.AttentionKVB
	} else {
		required["attention K-B"] = weights.AttentionKB
		required["attention V-B"] = weights.AttentionVB
	}
	if isMiniCPM3 || ((isDeepSeek2 || isDSA || isKimi) && spec.QLoRARank > 0) {
		required["attention Q-B"] = weights.AttentionQB
		required["attention Q-A norm"] = weights.AttentionQNorm
	}
	if (isDeepSeek2 || isDSA || isKimi) && layerIndex >= spec.LeadingDenseBlocks {
		required["feed-forward router"] = weights.FeedForwardRouter
		required["feed-forward expert down"] = weights.FeedForwardDownExperts
		if weights.FeedForwardGateUpExperts == nil {
			required["feed-forward expert gate"] = weights.FeedForwardGateExperts
			required["feed-forward expert up"] = weights.FeedForwardUpExperts
		}
		required["feed-forward shared gate"] = weights.FeedForwardSharedGate
		required["feed-forward shared up"] = weights.FeedForwardSharedUp
		required["feed-forward shared down"] = weights.FeedForwardSharedDown
	} else {
		required["feed-forward up"] = weights.FeedForwardUp
		required["feed-forward down"] = weights.FeedForwardDown
		if isMiniCPM3 || isDeepSeek2 || isKimi {
			required["feed-forward gate"] = weights.FeedForwardGate
		}
	}
	if isMiniCPM3 {
		required["feed-forward gate"] = weights.FeedForwardGate
	}
	if isDSA && spec.LayerHasFullIndexer(layerIndex) {
		for name, item := range map[string]*tensor.Tensor{
			"indexer K norm": weights.IndexerKNorm, "indexer K norm bias": weights.IndexerKNormBias,
			"indexer projection": weights.IndexerProjection, "indexer K": weights.IndexerAttentionK,
			"indexer Q-B": weights.IndexerAttentionQB,
		} {
			if item == nil {
				return DenseBlockResult{}, fmt.Errorf("DSA %s weight is nil", name)
			}
		}
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("MLA block %s weight is nil", name)
		}
	}
	if builder == nil || input == nil || input.Shape.Rank != 2 || len(positions) == 0 ||
		uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("MLA block input shape is invalid")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("MLA cache must contain both key and value")
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
		originalFactor := spec.YaRNAttentionFactor * (1 + 0.1*logScale)
		magnitude := originalFactor * (1 + 0.1*spec.RopeYaRNLogMultiplier*logScale)
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
		var feedForward *tensor.Tensor
		if weights.FeedForwardGateUpExperts != nil {
			if spec.ExpertGatingFunc == 2 {
				feedForward = builder.MoESigmoidFusedGateUp(normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateUpExperts, weights.FeedForwardDownExperts,
					weights.FeedForwardExpertBias, spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale)
			} else {
				feedForward = builder.MoESoftmaxFusedGateUp(normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateUpExperts, weights.FeedForwardDownExperts,
					weights.FeedForwardExpertBias, spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale)
			}
		} else if spec.ExpertGatingFunc == 2 {
			feedForward = builder.MoESigmoid(normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
				weights.FeedForwardExpertBias, spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale)
		} else if weights.FeedForwardExpertBias != nil {
			feedForward = builder.MoESoftmaxWithSelectionBias(normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
				weights.FeedForwardExpertBias, spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale)
		} else {
			feedForward = builder.MoE(normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
				spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale)
		}
		sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
		sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
		shared := builder.MulMat(weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp))
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

// BuildDeepSeek4BlockCached: HC compressed-attention block.
func BuildDeepSeek4BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions, tokenRows []uint32,
	pastKV *tensor.Tensor,
	pastStates map[string]*tensor.Tensor,
	currentPositions *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "deepseek4" || builder == nil || input == nil ||
		(input.Shape.Rank != 2 && input.Shape.Rank != 3) || layerIndex >= spec.BlockCount || len(positions) == 0 ||
		uint64(len(positions)) != input.Shape.Dims[input.Shape.Rank-1] ||
		currentPositions == nil || currentPositions.Shape != tensor.MustShape(1, 1, uint64(len(positions))) {
		return DenseBlockResult{}, errors.New("DeepSeek 4 block input is invalid")
	}
	if layerIndex == 0 {
		if input.Shape.Rank != 2 {
			return DenseBlockResult{}, errors.New("DeepSeek 4 initial input must be rank 2")
		}
		input = builder.DeepSeek4HCInit(input, spec.HyperConnectionCount)
	} else if input.Shape.Rank != 3 || input.Shape.Dims[1] != uint64(spec.HyperConnectionCount) {
		return DenseBlockResult{}, errors.New("DeepSeek 4 HC input is invalid")
	}
	required := map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm, "attention Q-A": weights.AttentionQ,
		"attention Q-A norm": weights.AttentionQNorm, "attention Q-B": weights.AttentionQB,
		"attention KV": weights.AttentionK, "attention KV norm": weights.AttentionKNorm,
		"attention sinks": weights.AttentionSinks, "attention output A": weights.AttentionOutputA,
		"attention output B": weights.AttentionOutput, "attention HC function": weights.HyperAttentionFN,
		"attention HC base": weights.HyperAttentionBase, "attention HC scale": weights.HyperAttentionScale,
		"feed-forward norm": weights.FeedForwardNorm, "feed-forward router": weights.FeedForwardRouter,
		"expert gate": weights.FeedForwardGateExperts, "expert up": weights.FeedForwardUpExperts,
		"expert down": weights.FeedForwardDownExperts, "shared gate": weights.FeedForwardSharedGate,
		"shared up": weights.FeedForwardSharedUp, "shared down": weights.FeedForwardSharedDown,
		"feed-forward HC function": weights.HyperFeedForwardFN, "feed-forward HC base": weights.HyperFeedForwardBase,
		"feed-forward HC scale": weights.HyperFeedForwardScale,
	}
	if layerIndex < spec.HashLayerCount {
		required["hash routing table"] = weights.FeedForwardHashExperts
		if len(tokenRows) != len(positions) {
			return DenseBlockResult{}, errors.New("DeepSeek 4 hash routing rows are missing")
		}
	} else {
		required["router bias"] = weights.FeedForwardRouterBias
	}
	ratio := spec.CompressRatios[layerIndex]
	if ratio != 0 {
		required["compressor KV"] = weights.AttentionCompressorKV
		required["compressor gate"] = weights.AttentionCompressorGate
		required["compressor APE"] = weights.AttentionCompressorAPE
		required["compressor norm"] = weights.AttentionCompressorNorm
	}
	if ratio == 4 {
		required["indexer projection"] = weights.IndexerProjection
		required["indexer Q-B"] = weights.IndexerAttentionQB
		required["indexer compressor KV"] = weights.IndexerCompressorKV
		required["indexer compressor gate"] = weights.IndexerCompressorGate
		required["indexer compressor APE"] = weights.IndexerCompressorAPE
		required["indexer compressor norm"] = weights.IndexerCompressorNorm
	}
	if layerIndex+1 == spec.BlockCount {
		required["output HC function"] = weights.HyperHeadFN
		required["output HC base"] = weights.HyperHeadBase
		required["output HC scale"] = weights.HyperHeadScale
	}
	for name, value := range required {
		if value == nil {
			return DenseBlockResult{}, fmt.Errorf("DeepSeek 4 %s weight is nil", name)
		}
	}
	tokens := uint64(len(positions))
	heads := uint64(spec.HeadCount)
	headWidth := uint64(spec.KeyLength)
	hc := spec.HyperConnectionCount
	residual := input
	current := builder.DeepSeek4HCPre(input, weights.HyperAttentionFN, weights.HyperAttentionScale,
		weights.HyperAttentionBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	current = builder.WeightedRMSNorm(current, weights.AttentionNorm, spec.RMSNormEpsilon)
	queryRank := builder.WeightedRMSNorm(builder.MulMat(weights.AttentionQ, current), weights.AttentionQNorm, spec.RMSNormEpsilon)
	query := builder.RMSNorm(builder.Reshape(builder.MulMat(weights.AttentionQB, queryRank), headWidth, heads, tokens), spec.RMSNormEpsilon)
	kv := builder.WeightedRMSNorm(builder.MulMat(weights.AttentionK, current), weights.AttentionKNorm, spec.RMSNormEpsilon)
	kv = builder.Reshape(kv, headWidth, 1, tokens)
	frequencyBase := spec.RopeFrequencyBase
	frequencyScale := float32(1)
	extFactor := float32(0)
	attentionFactor := float32(1)
	originalContext := uint32(0)
	betaFast, betaSlow := float32(0), float32(0)
	if ratio != 0 {
		frequencyBase = spec.CompressRopeBase
		if spec.RopeScalingFactor > 0 {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		extFactor = spec.YaRNExtFactor
		attentionFactor = spec.YaRNAttentionFactor
		if attentionFactor <= 0 {
			attentionFactor = 1
		}
		originalContext = spec.OriginalContextLength
		betaFast, betaSlow = spec.YaRNBetaFast, spec.YaRNBetaSlow
	}
	cacheKV := kv
	if pastKV != nil {
		if pastKV.Shape.Rank != 3 || pastKV.Shape.Dims[0] != headWidth || pastKV.Shape.Dims[1] != 1 {
			return DenseBlockResult{}, errors.New("DeepSeek 4 raw cache shape is invalid")
		}
		cacheKV = builder.Concat(pastKV, kv, 2)
	}
	states := make(map[string]*tensor.Tensor)
	cachePositions := currentPositions
	if previous := pastStates["positions"]; previous != nil {
		cachePositions = builder.Concat(previous, currentPositions, 2)
	}
	states["positions"] = cachePositions
	appendState := func(name string, current *tensor.Tensor) *tensor.Tensor {
		if current == nil {
			return nil
		}
		current = builder.Reshape(current, current.Shape.Dims[0], 1, current.Shape.Dims[1])
		if previous := pastStates[name]; previous != nil {
			current = builder.Concat(previous, current, 2)
		}
		states[name] = current
		return current
	}
	var compressorKV, compressorScore, indexerQuery, indexerWeights, indexerKV, indexerScore *tensor.Tensor
	if ratio != 0 {
		rows := make([]uint32, len(positions))
		for index, position := range positions {
			rows[index] = position % ratio
		}
		compressorKV = appendState("compressor_kv", builder.MulMat(weights.AttentionCompressorKV, current))
		compressorScore = appendState("compressor_score", builder.Add(
			builder.MulMat(weights.AttentionCompressorGate, current), builder.GetRows(weights.AttentionCompressorAPE, rows),
		))
	}
	if ratio == 4 {
		indexerWidth := uint64(spec.IndexerKeyLength)
		indexerHeads := uint64(spec.IndexerHeadCount)
		indexerQuery = builder.Reshape(builder.MulMat(weights.IndexerAttentionQB, queryRank), indexerWidth, indexerHeads, tokens)
		indexerWeights = builder.Scale(builder.MulMat(weights.IndexerProjection, current),
			float32(1/math.Sqrt(float64(spec.IndexerKeyLength*spec.IndexerHeadCount))))
		rows := make([]uint32, len(positions))
		for index, position := range positions {
			rows[index] = position % 4
		}
		indexerKV = appendState("indexer_compressor_kv", builder.MulMat(weights.IndexerCompressorKV, current))
		indexerScore = appendState("indexer_compressor_score", builder.Add(
			builder.MulMat(weights.IndexerCompressorGate, current), builder.GetRows(weights.IndexerCompressorAPE, rows),
		))
	}
	attributes := tensor.DeepSeek4AttentionAttributes{
		Positions: positions, Ratio: ratio, Window: spec.SlidingWindow, Heads: spec.HeadCount,
		IndexerHeads: spec.IndexerHeadCount, IndexerTopK: spec.IndexerTopK,
		RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: frequencyBase, FrequencyScale: frequencyScale,
		OriginalContext: originalContext, ExtFactor: extFactor, AttentionFactor: attentionFactor,
		BetaFast: betaFast, BetaSlow: betaSlow, NormEpsilon: spec.RMSNormEpsilon,
	}
	attention := builder.DeepSeek4Attention(query, cacheKV, cachePositions, weights.AttentionSinks,
		compressorKV, compressorScore, weights.AttentionCompressorNorm,
		indexerQuery, indexerWeights, indexerKV, indexerScore, weights.IndexerCompressorNorm, attributes)
	groupDimension := uint64(spec.HeadCount/spec.AttentionOutputGroups) * headWidth
	attention = builder.Reshape(attention, groupDimension, uint64(spec.AttentionOutputGroups), tokens)
	outputA := builder.Reshape(weights.AttentionOutputA, groupDimension, uint64(spec.AttentionOutputRank), uint64(spec.AttentionOutputGroups))
	attention = builder.Reshape(builder.GroupedMulMat(outputA, attention), uint64(spec.AttentionOutputRank*spec.AttentionOutputGroups), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	input = builder.DeepSeek4HCPost(attention, residual, weights.HyperAttentionFN, weights.HyperAttentionScale,
		weights.HyperAttentionBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	residual = input
	current = builder.DeepSeek4HCPre(input, weights.HyperFeedForwardFN, weights.HyperFeedForwardScale,
		weights.HyperFeedForwardBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	current = builder.WeightedRMSNorm(current, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var selected *tensor.Tensor
	if layerIndex < spec.HashLayerCount {
		selected = builder.GetRows(weights.FeedForwardHashExperts, tokenRows)
	}
	moe := builder.MoESqrtSoftplusLimited(current, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts, weights.FeedForwardRouterBias, selected,
		spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale, spec.LayerExpertSwiGLUClamp(layerIndex))
	sharedGate := builder.MulMat(weights.FeedForwardSharedGate, current)
	sharedUp := builder.MulMat(weights.FeedForwardSharedUp, current)
	shared := builder.MulMat(weights.FeedForwardSharedDown,
		deepSeek4LimitedSwiGLU(builder, sharedGate, sharedUp, spec.LayerSharedSwiGLUClampLimit(layerIndex)))
	current = builder.Add(moe, shared)
	output := builder.DeepSeek4HCPost(current, residual, weights.HyperFeedForwardFN, weights.HyperFeedForwardScale,
		weights.HyperFeedForwardBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	if layerIndex+1 == spec.BlockCount {
		output = builder.DeepSeek4HCHead(output, weights.HyperHeadFN, weights.HyperHeadScale,
			weights.HyperHeadBase, hc, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKV, Value: cacheKV, States: states}, nil
}

// BuildRWKV6Qwen2BlockCached: RMS/SwiGLU QRWKV recurrent block.
func BuildRWKV6Qwen2BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "rwkv6qwen2" || builder == nil || input == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("RWKV6-Qwen2 block input/state is invalid")
	}
	required := map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm, "time-mix W1": weights.TimeMixW1,
		"time-mix W2": weights.TimeMixW2, "time-mix lerp X": weights.TimeMixLerpX,
		"time-mix fused lerp": weights.TimeMixLerpFused, "time decay": weights.TimeMixDecay,
		"time decay W1": weights.TimeMixDecayW1, "time decay W2": weights.TimeMixDecayW2,
		"time key": weights.TimeMixKey, "time value": weights.TimeMixValue,
		"time receptance": weights.TimeMixReceptance, "time gate": weights.TimeMixGate,
		"time output": weights.TimeMixOutput, "feed-forward norm": weights.FeedForwardNorm,
		"feed-forward gate": weights.FeedForwardGate, "feed-forward up": weights.FeedForwardUp,
		"feed-forward down": weights.FeedForwardDown,
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("RWKV6-Qwen2 %s weight is nil", name)
		}
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) || input.Shape.Dims[1] == 0 {
		return DenseBlockResult{}, errors.New("RWKV6-Qwen2 input shape is invalid")
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := embedding / width
	tokens := input.Shape.Dims[1]
	if !pastShift.Shape.Equal(tensor.MustShape(embedding)) ||
		!pastState.Shape.Equal(tensor.MustShape(width, width, heads, 1)) {
		return DenseBlockResult{}, errors.New("RWKV6-Qwen2 cache shape is invalid")
	}
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	xPrev := builder.Reshape(pastShift, embedding, 1)
	if tokens > 1 {
		xPrev = builder.Concat(xPrev, builder.FlatSlice(normalized, 0, embedding, tokens-1), 1)
	}
	nextShift := builder.FlatSlice(normalized, embedding*(tokens-1), embedding)
	sx := builder.Add(xPrev, builder.Scale(normalized, -1))
	base := builder.Add(normalized, builder.Multiply(sx, builder.Reshape(weights.TimeMixLerpX, embedding, 1)))
	adjustments := builder.GroupedMulMat(
		weights.TimeMixW2,
		builder.Reshape(
			builder.Tanh(builder.MulMat(weights.TimeMixW1, base)),
			uint64(spec.TimeMixExtraDim), 5, tokens,
		),
	)
	adjustments = builder.Reshape(adjustments, embedding*5, tokens)
	lerps := builder.Reshape(weights.TimeMixLerpFused, embedding*5, 1)
	mixed := make([]*tensor.Tensor, 5)
	for index := uint64(0); index < 5; index++ {
		adjustment := builder.Reshape(
			builder.GroupSlice(adjustments, index*embedding, embedding, 1, embedding*5),
			embedding, tokens,
		)
		lerp := builder.Reshape(
			builder.GroupSlice(lerps, index*embedding, embedding, 1, embedding*5),
			embedding,
		)
		mixed[index] = builder.Add(normalized, builder.Multiply(sx, builder.Add(adjustment, lerp)))
	}
	xw, xk, xv, xr, xg := mixed[0], mixed[1], mixed[2], mixed[3], mixed[4]
	key := builder.MulMat(weights.TimeMixKey, xk)
	value := builder.MulMat(weights.TimeMixValue, xv)
	receptance := builder.MulMat(weights.TimeMixReceptance, xr)
	if weights.AttentionKBias != nil {
		key = builder.Add(key, weights.AttentionKBias)
	}
	if weights.AttentionVBias != nil {
		value = builder.Add(value, weights.AttentionVBias)
	}
	if weights.AttentionQBias != nil {
		receptance = builder.Add(receptance, weights.AttentionQBias)
	}
	gate := builder.Sigmoid(builder.MulMat(weights.TimeMixGate, xg))
	decay := builder.MulMat(weights.TimeMixDecayW2, builder.Tanh(builder.MulMat(weights.TimeMixDecayW1, xw)))
	decay = builder.Exp(builder.Scale(builder.Exp(builder.Add(decay, weights.TimeMixDecay)), -1))
	kvHeads := uint64(spec.HeadCountKV)
	key = builder.Reshape(key, width, kvHeads, tokens, 1)
	value = builder.Reshape(value, width, kvHeads, tokens, 1)
	receptance = builder.Reshape(receptance, width, heads, tokens, 1)
	decay = builder.Reshape(decay, width, heads, tokens, 1)
	packed := builder.GatedLinearAttention(
		key, value, receptance, decay, pastState,
		1/float32(math.Sqrt(float64(width))),
	)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, 0, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, 1)
	attention = builder.MulMat(weights.TimeMixOutput, builder.Multiply(attention, gate))
	residual := builder.Add(input, attention)
	ffnInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	feedForward := builder.MulMat(
		weights.FeedForwardDown,
		builder.SwiGLU(builder.MulMat(weights.FeedForwardGate, ffnInput), builder.MulMat(weights.FeedForwardUp, ffnInput)),
	)
	output := builder.Add(residual, feedForward)
	if spec.RescaleEvery > 0 && (layerIndex+1)%spec.RescaleEvery == 0 {
		output = builder.Scale(output, 0.5)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextShift, Value: nextState}, nil
}

// BuildRWKV6BlockCached: affine-LN WKV6 recurrent block.
func BuildRWKV6BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "rwkv6" || builder == nil || input == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("RWKV6 block input/state is invalid")
	}
	required := map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm, "attention norm bias": weights.AttentionNormBias,
		"channel norm": weights.AttentionNorm2, "channel norm bias": weights.AttentionNorm2Bias,
		"time-mix W1": weights.TimeMixW1, "time-mix W2": weights.TimeMixW2,
		"time-mix lerp X": weights.TimeMixLerpX, "time first": weights.TimeMixFirst,
		"time decay": weights.TimeMixDecay, "time decay W1": weights.TimeMixDecayW1,
		"time decay W2": weights.TimeMixDecayW2, "time key": weights.TimeMixKey,
		"time value": weights.TimeMixValue, "time receptance": weights.TimeMixReceptance,
		"time gate": weights.TimeMixGate, "time-mix norm": weights.TimeMixLN,
		"time-mix norm bias": weights.TimeMixLNBias, "time output": weights.TimeMixOutput,
		"channel lerp K": weights.ChannelMixLerpK, "channel lerp R": weights.ChannelMixLerpR,
		"channel key": weights.ChannelMixKey, "channel value": weights.ChannelMixValue,
		"channel receptance": weights.ChannelMixReceptance,
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("RWKV6 %s weight is nil", name)
		}
	}
	if weights.TimeMixLerpFused == nil &&
		(weights.TimeMixLerpW == nil || weights.TimeMixLerpK == nil || weights.TimeMixLerpV == nil ||
			weights.TimeMixLerpR == nil || weights.TimeMixLerpG == nil) {
		return DenseBlockResult{}, errors.New("RWKV6 time-mix lerp catalog is incomplete")
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := uint64(spec.HeadCount)
	tokens := input.Shape.Dims[1]
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != embedding || tokens == 0 ||
		!pastShift.Shape.Equal(tensor.MustShape(embedding, 2)) ||
		!pastState.Shape.Equal(tensor.MustShape(width, width, heads, 1)) {
		return DenseBlockResult{}, errors.New("RWKV6 input/cache shape is invalid")
	}
	attNorm := ApplyNormalization(builder, input, weights.AttentionNorm, weights.AttentionNormBias, spec)
	attPrev := builder.Reshape(builder.FlatSlice(pastShift, 0, embedding), embedding, 1)
	if tokens > 1 {
		attPrev = builder.Concat(attPrev, builder.FlatSlice(attNorm, 0, embedding, tokens-1), 1)
	}
	sx := builder.Add(attPrev, builder.Scale(attNorm, -1))
	base := builder.Add(attNorm, builder.Multiply(sx, builder.Reshape(weights.TimeMixLerpX, embedding, 1)))
	adjustments := builder.GroupedMulMat(
		weights.TimeMixW2,
		builder.Reshape(builder.Tanh(builder.MulMat(weights.TimeMixW1, base)), uint64(spec.TimeMixExtraDim), 5, tokens),
	)
	adjustments = builder.Reshape(adjustments, embedding*5, tokens)
	mixed := make([]*tensor.Tensor, 5)
	separate := []*tensor.Tensor{
		weights.TimeMixLerpW, weights.TimeMixLerpK, weights.TimeMixLerpV,
		weights.TimeMixLerpR, weights.TimeMixLerpG,
	}
	for index := uint64(0); index < 5; index++ {
		adjustment := builder.Reshape(
			builder.GroupSlice(adjustments, index*embedding, embedding, 1, embedding*5), embedding, tokens,
		)
		var lerp *tensor.Tensor
		if weights.TimeMixLerpFused != nil {
			fused := builder.Reshape(weights.TimeMixLerpFused, embedding*5, 1)
			lerp = builder.Reshape(builder.GroupSlice(fused, index*embedding, embedding, 1, embedding*5), embedding)
		} else {
			lerp = builder.Reshape(separate[index], embedding)
		}
		mixed[index] = builder.Add(attNorm, builder.Multiply(sx, builder.Add(adjustment, lerp)))
	}
	xw, xk, xv, xr, xg := mixed[0], mixed[1], mixed[2], mixed[3], mixed[4]
	key := builder.Reshape(builder.MulMat(weights.TimeMixKey, xk), width, heads, tokens, 1)
	value := builder.Reshape(builder.MulMat(weights.TimeMixValue, xv), width, heads, tokens, 1)
	receptance := builder.Reshape(builder.MulMat(weights.TimeMixReceptance, xr), width, heads, tokens, 1)
	gate := builder.SiLU(builder.MulMat(weights.TimeMixGate, xg))
	decay := builder.MulMat(weights.TimeMixDecayW2, builder.Tanh(builder.MulMat(weights.TimeMixDecayW1, xw)))
	decay = builder.Reshape(builder.Exp(builder.Scale(builder.Exp(builder.Add(decay, weights.TimeMixDecay)), -1)), width, heads, tokens, 1)
	packed := builder.RWKV6(key, value, receptance, weights.TimeMixFirst, decay, pastState)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, 0, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, 1)
	attention = builder.Reshape(
		builder.LayerNorm(builder.Reshape(attention, width, heads, tokens), rwkvHeadNormEpsilon), embedding, tokens,
	)
	attention = builder.Add(builder.Multiply(attention, weights.TimeMixLN), weights.TimeMixLNBias)
	attention = builder.MulMat(weights.TimeMixOutput, builder.Multiply(attention, gate))
	ffnInput := builder.Add(input, attention)
	ffnNorm := ApplyNormalization(builder, ffnInput, weights.AttentionNorm2, weights.AttentionNorm2Bias, spec)
	ffnPrev := builder.Reshape(builder.FlatSlice(pastShift, embedding, embedding), embedding, 1)
	if tokens > 1 {
		ffnPrev = builder.Concat(ffnPrev, builder.FlatSlice(ffnNorm, 0, embedding, tokens-1), 1)
	}
	channelShift := builder.Add(ffnPrev, builder.Scale(ffnNorm, -1))
	channelKeyInput := builder.Add(ffnNorm, builder.Multiply(channelShift, builder.Reshape(weights.ChannelMixLerpK, embedding, 1)))
	channelReceptanceInput := builder.Add(ffnNorm, builder.Multiply(channelShift, builder.Reshape(weights.ChannelMixLerpR, embedding, 1)))
	channel := builder.MulMat(weights.ChannelMixValue, builder.ReLUSquared(builder.MulMat(weights.ChannelMixKey, channelKeyInput)))
	channel = builder.Multiply(builder.Sigmoid(builder.MulMat(weights.ChannelMixReceptance, channelReceptanceInput)), channel)
	output := builder.Add(ffnInput, channel)
	if spec.RescaleEvery > 0 && (layerIndex+1)%spec.RescaleEvery == 0 {
		output = builder.Scale(output, 0.5)
	}
	nextAttShift := builder.FlatSlice(attNorm, embedding*(tokens-1), embedding)
	nextFFNShift := builder.FlatSlice(ffnNorm, embedding*(tokens-1), embedding)
	nextShift := builder.Concat(builder.Reshape(nextAttShift, embedding, 1), builder.Reshape(nextFFNShift, embedding, 1), 1)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextShift, Value: nextState}, nil
}

// BuildRWKV7BlockCached: RWKV7/ARWKV7 recurrent block.
func BuildRWKV7BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if (spec.Architecture != "rwkv7" && spec.Architecture != "arwkv7") ||
		builder == nil || input == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("RWKV7 block input/state is invalid")
	}
	required := map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm, "time W0": weights.TimeMixW0,
		"time W1": weights.TimeMixW1, "time W2": weights.TimeMixW2,
		"time A0": weights.TimeMixA0, "time A1": weights.TimeMixA1, "time A2": weights.TimeMixA2,
		"time V0": weights.TimeMixV0, "time V1": weights.TimeMixV1, "time V2": weights.TimeMixV2,
		"time lerp": weights.TimeMixLerpFused, "time KK": weights.TimeMixKK,
		"time KA": weights.TimeMixKA, "time RK": weights.TimeMixRK,
		"time key": weights.TimeMixKey, "time value": weights.TimeMixValue,
		"time receptance": weights.TimeMixReceptance, "time output": weights.TimeMixOutput,
	}
	if spec.Architecture == "rwkv7" {
		required["attention norm bias"] = weights.AttentionNormBias
		required["channel norm"] = weights.AttentionNorm2
		required["channel norm bias"] = weights.AttentionNorm2Bias
		required["time norm"] = weights.TimeMixLN
		required["time norm bias"] = weights.TimeMixLNBias
		required["channel lerp"] = weights.ChannelMixLerpK
		required["channel key"] = weights.ChannelMixKey
		required["channel value"] = weights.ChannelMixValue
	} else {
		required["feed-forward norm"] = weights.FeedForwardNorm
		required["feed-forward gate"] = weights.FeedForwardGate
		required["feed-forward up"] = weights.FeedForwardUp
		required["feed-forward down"] = weights.FeedForwardDown
	}
	if spec.GateLoRARank > 0 {
		required["time G1"] = weights.TimeMixG1
		required["time G2"] = weights.TimeMixG2
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("RWKV7 %s weight is nil", name)
		}
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := uint64(spec.HeadCount)
	tokens := input.Shape.Dims[1]
	shiftCount := uint64(spec.TokenShiftCount)
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != embedding || tokens == 0 ||
		!pastShift.Shape.Equal(tensor.MustShape(embedding, shiftCount)) ||
		!pastState.Shape.Equal(tensor.MustShape(width, width, heads, 1)) {
		return DenseBlockResult{}, errors.New("RWKV7 input/cache shape is invalid")
	}
	attNorm := ApplyNormalization(builder, input, weights.AttentionNorm, weights.AttentionNormBias, spec)
	attPrev := builder.Reshape(builder.FlatSlice(pastShift, 0, embedding), embedding, 1)
	if tokens > 1 {
		attPrev = builder.Concat(attPrev, builder.FlatSlice(attNorm, 0, embedding, tokens-1), 1)
	}
	sx := builder.Add(attPrev, builder.Scale(attNorm, -1))
	lerpCount := uint64(6)
	if spec.GateLoRARank == 0 {
		lerpCount = 5
	}
	lerps := builder.Reshape(weights.TimeMixLerpFused, embedding*lerpCount, 1)
	mixed := make([]*tensor.Tensor, lerpCount)
	for index := uint64(0); index < lerpCount; index++ {
		lerp := builder.Reshape(builder.GroupSlice(lerps, index*embedding, embedding, 1, embedding*lerpCount), embedding)
		mixed[index] = builder.Add(attNorm, builder.Multiply(sx, lerp))
	}
	xr, xw, xk, xv, xa := mixed[0], mixed[1], mixed[2], mixed[3], mixed[4]
	receptance := builder.MulMat(weights.TimeMixReceptance, xr)
	decay := builder.Add(
		builder.MulMat(weights.TimeMixW2, builder.Tanh(builder.MulMat(weights.TimeMixW1, xw))),
		weights.TimeMixW0,
	)
	decay = builder.Exp(builder.Scale(builder.Sigmoid(decay), -0.606531))
	key := builder.MulMat(weights.TimeMixKey, xk)
	value := builder.MulMat(weights.TimeMixValue, xv)
	var auxiliary *tensor.Tensor
	if layerIndex == 0 {
		if weights.PerLayerInput != nil {
			return DenseBlockResult{}, errors.New("RWKV7 first layer received a value residual")
		}
		auxiliary = value
	} else {
		if weights.PerLayerInput == nil {
			return DenseBlockResult{}, errors.New("RWKV7 value residual is missing")
		}
		valueMix := builder.Sigmoid(builder.Add(
			builder.MulMat(weights.TimeMixV2, builder.MulMat(weights.TimeMixV1, xv)),
			weights.TimeMixV0,
		))
		value = builder.Add(value, builder.Multiply(builder.Add(weights.PerLayerInput, builder.Scale(value, -1)), valueMix))
	}
	a := builder.Sigmoid(builder.Add(
		builder.MulMat(weights.TimeMixA2, builder.MulMat(weights.TimeMixA1, xa)),
		weights.TimeMixA0,
	))
	kk := builder.L2Norm(
		builder.Reshape(builder.Multiply(key, weights.TimeMixKK), width, heads, tokens, 1),
		rwkvKeyNormEpsilon,
	)
	ka := builder.Multiply(key, weights.TimeMixKA)
	key = builder.Add(key, builder.Add(builder.Multiply(a, ka), builder.Scale(ka, -1)))
	receptance4 := builder.Reshape(receptance, width, heads, tokens, 1)
	decay4 := builder.Reshape(decay, width, heads, tokens, 1)
	key4 := builder.Reshape(key, width, heads, tokens, 1)
	value4 := builder.Reshape(value, width, heads, tokens, 1)
	a4 := builder.Reshape(a, width, heads, tokens, 1)
	packed := builder.RWKV7(receptance4, decay4, key4, value4, builder.Scale(kk, -1), builder.Multiply(kk, a4), pastState)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, 0, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, 1)
	if weights.TimeMixLN != nil && weights.TimeMixLNBias != nil {
		attention = builder.Reshape(
			builder.LayerNorm(builder.Reshape(attention, width, heads, tokens), rwkvHeadNormEpsilon), embedding, tokens,
		)
		attention = builder.Add(builder.Multiply(attention, weights.TimeMixLN), weights.TimeMixLNBias)
	}
	rkWeight := builder.Reshape(weights.TimeMixRK, width, heads, 1, 1)
	rk := builder.SumRows(builder.Multiply(builder.Multiply(key4, receptance4), rkWeight))
	attention = builder.Add(attention, builder.Reshape(builder.Multiply(value4, rk), embedding, tokens))
	if spec.GateLoRARank > 0 {
		xg := mixed[5]
		gate := builder.MulMat(weights.TimeMixG2, builder.Sigmoid(builder.MulMat(weights.TimeMixG1, xg)))
		attention = builder.Multiply(attention, gate)
	}
	attention = builder.MulMat(weights.TimeMixOutput, attention)
	ffnInput := builder.Add(input, attention)
	var ffnNorm, output *tensor.Tensor
	if spec.Architecture == "rwkv7" {
		ffnNorm = ApplyNormalization(builder, ffnInput, weights.AttentionNorm2, weights.AttentionNorm2Bias, spec)
		ffnPrev := builder.Reshape(builder.FlatSlice(pastShift, embedding, embedding), embedding, 1)
		if tokens > 1 {
			ffnPrev = builder.Concat(ffnPrev, builder.FlatSlice(ffnNorm, 0, embedding, tokens-1), 1)
		}
		channelShift := builder.Add(ffnPrev, builder.Scale(ffnNorm, -1))
		channelInput := builder.Add(ffnNorm, builder.Multiply(channelShift, builder.Reshape(weights.ChannelMixLerpK, embedding, 1)))
		channel := builder.MulMat(weights.ChannelMixValue, builder.ReLUSquared(builder.MulMat(weights.ChannelMixKey, channelInput)))
		output = builder.Add(ffnInput, channel)
	} else {
		ffnNorm = builder.WeightedRMSNorm(ffnInput, weights.FeedForwardNorm, spec.RMSNormEpsilon)
		ffn := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(
			builder.MulMat(weights.FeedForwardGate, ffnNorm), builder.MulMat(weights.FeedForwardUp, ffnNorm),
		))
		output = builder.Add(ffnInput, ffn)
	}
	nextAttShift := builder.FlatSlice(attNorm, embedding*(tokens-1), embedding)
	nextShift := builder.Reshape(nextAttShift, embedding, 1)
	if shiftCount == 2 {
		nextFFNShift := builder.FlatSlice(ffnNorm, embedding*(tokens-1), embedding)
		nextShift = builder.Concat(nextShift, builder.Reshape(nextFFNShift, embedding, 1), 1)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextShift, Value: nextState, Auxiliary: auxiliary}, nil
}

// BuildKimiLinearBlockCached: KDA or no-RoPE MLA block.
func BuildKimiLinearBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	recurrent bool,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "kimi-linear" {
		return DenseBlockResult{}, errors.New("Kimi Linear block requires kimi-linear architecture")
	}
	if !recurrent {
		return BuildMLABlockCachedForLayer(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	if builder == nil || input == nil || pastKey == nil || pastValue == nil {
		return DenseBlockResult{}, errors.New("Kimi Linear KDA input/state is nil")
	}
	required := map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm,
		"attention Q":    weights.AttentionQ, "attention K": weights.AttentionK,
		"attention V": weights.AttentionV, "attention output": weights.AttentionOutput,
		"Q convolution": weights.SSMQueryConv, "K convolution": weights.SSMKeyConv,
		"V convolution": weights.SSMValueConv,
		"forget A":      weights.SSMForgetA, "forget B": weights.SSMForgetB,
		"beta": weights.SSMBeta, "SSM A": weights.SSMA, "time-step bias": weights.SSMTimeStep,
		"output gate A": weights.SSMOutputGateA, "output gate B": weights.SSMOutputGateB,
		"SSM norm": weights.SSMNorm, "feed-forward norm": weights.FeedForwardNorm,
	}
	addKimiFeedForwardRequirements(required, spec, weights, layerIndex)
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Kimi Linear KDA %s weight is nil", name)
		}
	}
	if input.Shape.Rank != 2 || len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Kimi Linear KDA input shape is invalid")
	}
	headDim := uint64(spec.KDAHeadDim)
	heads := uint64(spec.HeadCount)
	inner := headDim * heads
	tokens := uint64(len(positions))
	window := uint64(spec.SSMConvKernel - 1)
	if !pastKey.Shape.Equal(tensor.MustShape(window, 3*inner)) ||
		!pastValue.Shape.Equal(tensor.MustShape(headDim, headDim, heads, 1)) {
		return DenseBlockResult{}, errors.New("Kimi Linear KDA cache shape is invalid")
	}
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	convolve := func(projection, kernel *tensor.Tensor, stateIndex uint64) (*tensor.Tensor, *tensor.Tensor) {
		state := builder.FlatSlice(pastKey, stateIndex*window*inner, window, inner)
		projected := builder.MulMat(projection, normalized)
		mixed := builder.Concat(state, builder.Transpose2D(projected), 0)
		next := builder.Reshape(builder.GroupSlice(mixed, tokens, window, 1, window), window, inner)
		kernel = builder.Reshape(kernel, uint64(spec.SSMConvKernel), inner)
		value := builder.SiLU(builder.SSMConv(mixed, kernel))
		return builder.Reshape(value, headDim, heads, tokens, 1), next
	}
	query, nextQ := convolve(weights.AttentionQ, weights.SSMQueryConv, 0)
	key, nextK := convolve(weights.AttentionK, weights.SSMKeyConv, 1)
	value, nextV := convolve(weights.AttentionV, weights.SSMValueConv, 2)
	nextConv := builder.Transpose2D(builder.Concat(
		builder.Concat(builder.Transpose2D(nextQ), builder.Transpose2D(nextK), 0),
		builder.Transpose2D(nextV), 0,
	))
	gate := builder.MulMat(weights.SSMForgetB, builder.MulMat(weights.SSMForgetA, normalized))
	gate = builder.Softplus(builder.Add(gate, weights.SSMTimeStep))
	gate = builder.Reshape(gate, headDim, heads, tokens, 1)
	gate = builder.Multiply(gate, builder.Reshape(weights.SSMA, 1, heads, 1, 1))
	beta := builder.Reshape(
		builder.Sigmoid(builder.MulMat(weights.SSMBeta, normalized)),
		1, heads, tokens, 1,
	)
	query = builder.L2Norm(query, spec.RMSNormEpsilon)
	key = builder.L2Norm(key, spec.RMSNormEpsilon)
	packed := builder.GatedDeltaNet(query, key, value, gate, beta, pastValue)
	attentionElements := headDim * heads * tokens
	attention := builder.FlatSlice(packed, 0, headDim, heads, tokens, 1)
	nextState := builder.FlatSlice(packed, attentionElements, headDim, headDim, heads, 1)
	outputGate := builder.MulMat(weights.SSMOutputGateB, builder.MulMat(weights.SSMOutputGateA, normalized))
	outputGate = builder.Reshape(outputGate, headDim, heads, tokens, 1)
	attention = builder.Multiply(
		builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon),
		builder.Sigmoid(outputGate),
	)
	attention = builder.MulMat(weights.AttentionOutput, builder.Reshape(attention, inner, tokens))
	residual := builder.Add(input, attention)
	output := buildKimiFeedForward(builder, residual, spec, weights, layerIndex)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextConv, Value: nextState}, nil
}

func buildKimiFeedForward(
	builder *tensor.Builder,
	residual *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	layerIndex uint32,
) *tensor.Tensor {
	normalized := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	if layerIndex < spec.LeadingDenseBlocks {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		return builder.Add(residual, builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up)))
	}
	var routed *tensor.Tensor
	if spec.ExpertGatingFunc == 2 {
		routed = builder.MoESigmoid(
			normalized, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
			weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
			weights.FeedForwardExpertBias, spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
		)
	} else {
		routed = builder.MoESoftmaxWithSelectionBias(
			normalized, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
			weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
			weights.FeedForwardExpertBias, spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
		)
	}
	shared := builder.MulMat(
		weights.FeedForwardSharedDown,
		builder.SwiGLU(
			builder.MulMat(weights.FeedForwardSharedGate, normalized),
			builder.MulMat(weights.FeedForwardSharedUp, normalized),
		),
	)
	return builder.Add(residual, builder.Add(routed, shared))
}

func addKimiFeedForwardRequirements(
	required map[string]*tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	layerIndex uint32,
) {
	if layerIndex < spec.LeadingDenseBlocks {
		required["feed-forward gate"] = weights.FeedForwardGate
		required["feed-forward up"] = weights.FeedForwardUp
		required["feed-forward down"] = weights.FeedForwardDown
		return
	}
	required["feed-forward router"] = weights.FeedForwardRouter
	required["feed-forward expert gate"] = weights.FeedForwardGateExperts
	required["feed-forward expert up"] = weights.FeedForwardUpExperts
	required["feed-forward expert down"] = weights.FeedForwardDownExperts
	required["feed-forward correction bias"] = weights.FeedForwardExpertBias
	required["shared expert gate"] = weights.FeedForwardSharedGate
	required["shared expert up"] = weights.FeedForwardSharedUp
	required["shared expert down"] = weights.FeedForwardSharedDown
}

// BuildLFM2BlockCached: attention or gated short-convolution block
// Recurrent cache: Key convolution window; Value reserved ABI slot
func BuildLFM2BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	recurrent bool,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (LFM2BlockResult, error) {
	if spec.Architecture != "lfm2" && spec.Architecture != "lfm2moe" {
		return LFM2BlockResult{}, errors.New("LFM2 block requires lfm2 or lfm2moe architecture")
	}
	if !recurrent {
		result, err := BuildDenseBlockCachedForLayer(
			builder, input, spec, weights, positions, pastKey, pastValue, layerIndex,
		)
		return LFM2BlockResult{
			Output: result.Output, Key: result.Key, Value: result.Value,
		}, err
	}
	if builder == nil || input == nil || pastKey == nil || pastValue == nil {
		return LFM2BlockResult{}, errors.New("LFM2 recurrent block input or state is nil")
	}
	required := map[string]*tensor.Tensor{
		"operator norm":            weights.AttentionNorm,
		"short-convolution input":  weights.ShortConvInput,
		"short-convolution kernel": weights.ShortConvKernel,
		"short-convolution output": weights.ShortConvOutput,
		"feed-forward norm":        weights.FeedForwardNorm,
	}
	usesExperts := weights.FeedForwardRouter != nil
	if usesExperts {
		required["feed-forward router"] = weights.FeedForwardRouter
		required["feed-forward expert gate"] = weights.FeedForwardGateExperts
		required["feed-forward expert up"] = weights.FeedForwardUpExperts
		required["feed-forward expert down"] = weights.FeedForwardDownExperts
		required["feed-forward expert correction bias"] = weights.FeedForwardExpertBias
	} else {
		required["feed-forward gate"] = weights.FeedForwardGate
		required["feed-forward up"] = weights.FeedForwardUp
		required["feed-forward down"] = weights.FeedForwardDown
	}
	for name, item := range required {
		if item == nil {
			return LFM2BlockResult{}, fmt.Errorf("LFM2 recurrent block %s weight is nil", name)
		}
	}
	if input.Shape.Rank != 2 || len(positions) == 0 ||
		uint64(len(positions)) != input.Shape.Dims[1] {
		return LFM2BlockResult{}, errors.New("LFM2 recurrent block input shape is invalid")
	}
	embedding := uint64(spec.EmbeddingLength)
	window := uint64(spec.ShortConvCacheLength - 1)
	if !pastKey.Shape.Equal(tensor.MustShape(window, embedding)) ||
		!pastValue.Shape.Equal(tensor.MustShape(1)) {
		return LFM2BlockResult{}, errors.New("LFM2 recurrent cache shape is invalid")
	}
	tokens := uint64(len(positions))
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	mixed := builder.MulMat(weights.ShortConvInput, normalized)
	b := builder.Reshape(builder.GroupSlice(mixed, 0, embedding, 1, 3*embedding), embedding, tokens)
	c := builder.Reshape(builder.GroupSlice(mixed, embedding, embedding, 1, 3*embedding), embedding, tokens)
	x := builder.Reshape(builder.GroupSlice(mixed, 2*embedding, embedding, 1, 3*embedding), embedding, tokens)
	projected := builder.Transpose2D(builder.Multiply(b, x))
	convInput := builder.Concat(pastKey, projected, 0)
	if spec.NonCausalAttention {
		leftPad := window / 2
		rightPad := window - leftPad
		convInput = projected
		if leftPad > 0 {
			left := builder.GroupSlice(pastKey, window-leftPad, leftPad, 1, window)
			left = builder.Reshape(left, leftPad, embedding)
			convInput = builder.Concat(left, convInput, 0)
		}
		if rightPad > 0 {
			right := builder.GroupSlice(pastKey, window-rightPad, rightPad, 1, window)
			right = builder.Scale(builder.Reshape(right, rightPad, embedding), 0)
			convInput = builder.Concat(convInput, right, 0)
		}
	}
	nextState := builder.GroupSlice(convInput, tokens, window, 1, window)
	nextState = builder.Reshape(nextState, window, embedding)
	convolved := builder.SSMConv(convInput, weights.ShortConvKernel)
	shortConv := builder.MulMat(weights.ShortConvOutput, builder.Multiply(c, convolved))
	residual := builder.Add(input, shortConv)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var feedForward *tensor.Tensor
	if usesExperts {
		if spec.ExpertGatingFunc == 2 {
			feedForward = builder.MoESigmoid(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
				spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		} else {
			feedForward = builder.MoESoftmaxWithSelectionBias(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
				spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		}
	} else {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	}
	output := builder.Add(residual, feedForward)
	nextReserved := builder.Scale(pastValue, 1)
	if err := builder.Err(); err != nil {
		return LFM2BlockResult{}, err
	}
	return LFM2BlockResult{
		Output: output, Key: nextState, Value: nextReserved, Recurrent: true,
	}, nil
}

// BuildDenseBlock: constructs one pre-normalized grouped-query transformer
// block; covers initial Llama layout and Qwen3's per-head Q/K norms
func BuildDenseBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
) (*tensor.Tensor, error) {
	result, err := BuildDenseBlockCached(builder, input, spec, weights, positions, nil, nil)
	return result.Output, err
}

// BuildDenseBlockCached additionally accepts and returns layer's rank-3
// RoPE-key/value cache; Past cache tensors must either both be nil or both be
// present
func BuildDenseBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey *tensor.Tensor,
	pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	return BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, positions, pastKey, pastValue, 0,
	)
}

func BuildDenseBlockCachedForLayer(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey *tensor.Tensor,
	pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	return buildDenseBlockCachedForLayer(
		builder, input, spec, weights, positions, nil, pastKey, pastValue, layerIndex,
	)
}

// BuildDenseBlockCachedForLayerWithMultiPositions: distinct MRoPE axes.
func BuildDenseBlockCachedForLayerWithMultiPositions(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	multiPositions [4][]uint32,
	pastKey *tensor.Tensor,
	pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	switch spec.Architecture {
	case "glm4", "glm4moe":
		if spec.RopeSections[0] <= 0 || spec.RopeSections[1] <= 0 {
			return DenseBlockResult{}, errors.New("GLM4 multi-axis positions require multimodal RoPE sections")
		}
	case "hunyuan-dense", "hunyuan_vl":
		if spec.RopeSections[0] <= 0 || spec.RopeSections[1] <= 0 {
			return DenseBlockResult{}, errors.New("Hunyuan multi-axis positions require multimodal RoPE sections")
		}
	case "paddleocr", "qwen2vl", "qwen3vl", "qwen3vlmoe":
	default:
		return DenseBlockResult{}, errors.New("dense block architecture does not support multi-axis positions")
	}
	return buildDenseBlockCachedForLayer(
		builder, input, spec, weights, multiPositions[0], &multiPositions, pastKey, pastValue, layerIndex,
	)
}

func buildDenseBlockCachedForLayer(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	multiPositions *[4][]uint32,
	pastKey *tensor.Tensor,
	pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if builder == nil {
		return DenseBlockResult{}, errors.New("dense block builder is nil")
	}
	if input == nil {
		return DenseBlockResult{}, errors.New("dense block input is nil")
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	profile := spec.Profile()
	normPlan := spec.NormPlan()
	if spec.Architecture == "bert" || spec.Architecture == "jina-bert-v2" || spec.Architecture == "jina-bert-v3" || spec.Architecture == "nomic-bert" || spec.Architecture == "nomic-bert-moe" {
		return buildBERTEncoderBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	if spec.Architecture == "modern-bert" {
		return buildModernBERTBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	if spec.Architecture == "gemma-embedding" {
		return buildGemmaEmbeddingBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	if spec.Architecture == "talkie" {
		return buildTalkieBlock(builder, input, spec, weights, positions, pastKey, pastValue)
	}
	if spec.Architecture == "gemma4" {
		return buildGemma4BlockCached(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	if spec.Architecture == "gemma3n" {
		return DenseBlockResult{}, errors.New("Gemma 3n block requires AltUp execution")
	}
	if spec.Architecture == "rwkv6qwen2" {
		return BuildRWKV6Qwen2BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	}
	if spec.Architecture == "rwkv6" {
		return BuildRWKV6BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	}
	if spec.Architecture == "rwkv7" || spec.Architecture == "arwkv7" {
		return BuildRWKV7BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	}
	isOLMo2 := spec.Architecture == "olmo2"
	isOLMoE := spec.Architecture == "olmoe"
	isPhiMoE := spec.Architecture == "phimoe"
	isEXAOneMoE := spec.Architecture == "exaone-moe"
	isPostOnlyNorm := !normPlan.PreAttention
	isCommandRQKNorm := spec.Architecture == "command-r" && spec.BlockCount >= 64
	isChameleon := spec.Architecture == "chameleon"
	isLaguna := spec.Architecture == "laguna"
	isAFMoE := spec.Architecture == "afmoe"
	isBailingMoE := spec.Architecture == "bailingmoe"
	isBailingMoE2 := spec.Architecture == "bailingmoe2"
	isCohere2MoE := spec.Architecture == "cohere2moe"
	isDeepSeek := spec.Architecture == "deepseek"
	isDeepSeek2OCR := spec.Architecture == "deepseek2-ocr"
	isDeci := spec.Architecture == "deci"
	isDBRX := spec.Architecture == "dbrx"
	isDOTS1 := spec.Architecture == "dots1"
	isErnieMoE := spec.Architecture == "ernie4_5-moe"
	isGraniteMoE := spec.Architecture == "granitemoe" || spec.Architecture == "granitehybrid" ||
		spec.Architecture == "granite" && spec.ExpertCount > 0
	isGroveMoE := spec.Architecture == "grovemoe"
	isGLM4MoE := spec.Architecture == "glm4moe"
	isGrok := spec.Architecture == "grok"
	isHunyuanMoE := spec.Architecture == "hunyuan-moe"
	isHunyuan := isHunyuanMoE || spec.Architecture == "hunyuan-dense" || spec.Architecture == "hunyuan_vl"
	isJamba := spec.Architecture == "jamba"
	isHYV3 := spec.Architecture == "hy_v3"
	isMellum := spec.Architecture == "mellum"
	isMiMo2 := spec.Architecture == "mimo2"
	isStep35 := spec.Architecture == "step35"
	isSmallThinker := spec.Architecture == "smallthinker"
	isMiniMaxM2 := spec.Architecture == "minimax-m2"
	isLFM2MoE := spec.Architecture == "lfm2moe"
	isLlama4 := spec.Architecture == "llama4"
	isMistral3MoE := spec.Architecture == "mistral3" && spec.ExpertCount > 0
	isGPTOSS := spec.Architecture == "gpt-oss"
	isArctic := spec.Architecture == "arctic"
	isLLaDAMoE := spec.Architecture == "llada-moe"
	isQwen2MoE := spec.Architecture == "qwen2moe"
	isRefactMoE := spec.Architecture == "refact" && spec.ExpertCount > 0
	if isGroveMoE && (spec.ExpertsPerGroup == 0 || spec.ExpertCount == 0 ||
		spec.ExpertCount%spec.ExpertsPerGroup != 0) {
		return DenseBlockResult{}, errors.New("GroveMoE expert grouping is invalid")
	}
	headCount := spec.LayerHeadCount(layerIndex)
	kvHeadCount := spec.LayerKVHeadCount(layerIndex)
	if isDeci && (spec.LayerFeedForwardLength(layerIndex) == 0 || headCount == 0 || kvHeadCount == 0) {
		return buildDeciSparseBlockCached(
			builder, input, spec, weights, positions, pastKey, pastValue, layerIndex,
		)
	}
	if spec.Architecture == "apertus" &&
		(int(layerIndex) >= len(spec.XIELUAlphaN) || int(layerIndex) >= len(spec.XIELUAlphaP) ||
			int(layerIndex) >= len(spec.XIELUBeta) || int(layerIndex) >= len(spec.XIELUEpsilon)) {
		return DenseBlockResult{}, errors.New("Apertus xIELU parameters are missing for layer")
	}
	required := map[string]*tensor.Tensor{
		"attention output": weights.AttentionOutput,
	}
	usesExperts := weights.FeedForwardRouter != nil
	if usesExperts {
		if !((spec.Architecture == "llama" || spec.Architecture == "llama-embed") && spec.ExpertCount > 0) && spec.Architecture != "qwen3moe" && spec.Architecture != "qwen3vlmoe" && spec.Architecture != "rnd1" && !isArctic && !isLLaDAMoE && !isBailingMoE && !isBailingMoE2 && !isCohere2MoE && !isDeepSeek && !isDeepSeek2OCR && !isDBRX && !isDOTS1 && !isErnieMoE && !isGLM4MoE && !isGraniteMoE && !isGroveMoE && !isGrok && !isHunyuanMoE && !isHYV3 && !isJamba && !isLlama4 && !isMistral3MoE && !isGPTOSS && !isMellum && !isMiMo2 && !isStep35 && !isSmallThinker && !isMiniMaxM2 && !isLFM2MoE && !isLaguna && !isAFMoE && !isQwen2MoE && !isOLMoE && !isPhiMoE && !isEXAOneMoE && !isRefactMoE {
			return DenseBlockResult{}, errors.New("dense block expert weights require a supported MoE architecture")
		}
		required["feed-forward router"] = weights.FeedForwardRouter
		if (isCohere2MoE || isDeepSeek2OCR || isHYV3) && weights.FeedForwardGateUpExperts != nil {
			required["feed-forward fused expert gate/up"] = weights.FeedForwardGateUpExperts
		} else {
			if (!isGraniteMoE && !isGrok && !isErnieMoE && !isRefactMoE) || weights.FeedForwardGateExperts != nil {
				required["feed-forward expert gate"] = weights.FeedForwardGateExperts
			}
			required["feed-forward expert up"] = weights.FeedForwardUpExperts
		}
		required["feed-forward expert down"] = weights.FeedForwardDownExperts
		if isGroveMoE {
			required["feed-forward chunk expert gate"] = weights.FeedForwardGateChunkExperts
			required["feed-forward chunk expert up"] = weights.FeedForwardUpChunkExperts
			required["feed-forward chunk expert down"] = weights.FeedForwardDownChunkExperts
		}
		if isGLM4MoE {
			required["feed-forward expert correction bias"] = weights.FeedForwardExpertBias
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isStep35 && spec.SharedExpertFF > 0 {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isLlama4 {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isGPTOSS {
			required["feed-forward router bias"] = weights.FeedForwardRouterBias
			required["feed-forward expert gate bias"] = weights.FeedForwardGateBias
			required["feed-forward expert up bias"] = weights.FeedForwardUpBias
			required["feed-forward expert down bias"] = weights.FeedForwardDownBias
		}
		if isCohere2MoE && spec.SharedExpertFF > 0 {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isErnieMoE && spec.SharedExpertFF > 0 {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isLaguna || isAFMoE {
			required["feed-forward expert correction bias"] = weights.FeedForwardExpertBias
			if spec.SharedExpertFF > 0 {
				required["feed-forward shared gate"] = weights.FeedForwardSharedGate
				required["feed-forward shared up"] = weights.FeedForwardSharedUp
				required["feed-forward shared down"] = weights.FeedForwardSharedDown
			}
		}
		if isHunyuanMoE {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isHYV3 || isDeepSeek2OCR {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isQwen2MoE {
			required["feed-forward shared router"] = weights.FeedForwardSharedRouter
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isEXAOneMoE || isBailingMoE2 || isDOTS1 {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isBailingMoE || isDeepSeek {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isGraniteMoE && spec.SharedExpertFF > 0 {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if isLFM2MoE {
			required["feed-forward expert correction bias"] = weights.FeedForwardExpertBias
		}
		if isMiniMaxM2 {
			required["feed-forward expert correction bias"] = weights.FeedForwardExpertBias
		}
		if isArctic {
			required["feed-forward expert norm"] = weights.FeedForwardExpertNorm
			required["feed-forward gate"] = weights.FeedForwardGate
			required["feed-forward up"] = weights.FeedForwardUp
			required["feed-forward down"] = weights.FeedForwardDown
		}
	} else {
		required["feed-forward up"] = weights.FeedForwardUp
		required["feed-forward down"] = weights.FeedForwardDown
	}
	if weights.AttentionQKV != nil {
		required["attention QKV"] = weights.AttentionQKV
		if weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil {
			return DenseBlockResult{}, errors.New("dense fused QKV cannot use separate projection biases")
		}
	} else {
		required["attention Q"] = weights.AttentionQ
		required["attention K"] = weights.AttentionK
		required["attention V"] = weights.AttentionV
		if weights.AttentionQKVBias != nil {
			return DenseBlockResult{}, errors.New("dense fused QKV bias has no fused projection")
		}
	}
	if spec.Architecture == "falcon" && weights.AttentionNorm2 == nil && weights.AttentionNorm2Bias != nil {
		return DenseBlockResult{}, errors.New("Falcon secondary attention norm bias has no weight")
	}
	if spec.Architecture == "bitnet" {
		required["attention sub norm"] = weights.AttentionSubNorm
		required["feed-forward sub norm"] = weights.FeedForwardSubNorm
	}
	if !usesExperts && profile.FeedForward == FeedForwardSwiGLU {
		required["feed-forward gate"] = weights.FeedForwardGate
	} else if profile.FeedForward == FeedForwardSequentialGELU {
		required["attention output bias"] = weights.AttentionOutputBias
		required["feed-forward up bias"] = weights.FeedForwardUpBias
		required["feed-forward down bias"] = weights.FeedForwardDownBias
	}
	if isPostOnlyNorm {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
		required["attention post norm"] = weights.AttentionPostNorm
		required["feed-forward post norm"] = weights.FeedForwardPostNorm
	} else if isOLMo2 {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
		required["attention post norm"] = weights.AttentionPostNorm
		required["feed-forward post norm"] = weights.FeedForwardPostNorm
	} else if !spec.UsesUnweightedLayerNorm() {
		required["attention norm"] = weights.AttentionNorm
		if profile.Residual != ResidualParallel && !isGPTOSS {
			if spec.Architecture != "stablelm" {
				required["feed-forward norm"] = weights.FeedForwardNorm
			}
		}
		if spec.RequiresLayerNormBias() {
			required["attention norm bias"] = weights.AttentionNormBias
			if profile.Residual != ResidualParallel && spec.Architecture != "stablelm" {
				required["feed-forward norm bias"] = weights.FeedForwardNormBias
			}
		}
	}
	if isGPTOSS {
		required["attention sinks"] = weights.AttentionSinks
		required["attention output bias"] = weights.AttentionOutputBias
		required["attention post norm"] = weights.AttentionPostNorm
	}
	if isCommandRQKNorm {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if spec.Architecture == "plamo2" || spec.Architecture == "plamo3" {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if isChameleon {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if isOLMoE {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if isMiniMaxM2 {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if isDOTS1 {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if isHYV3 {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if isHunyuan {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if isPhiMoE {
		required["attention output bias"] = weights.AttentionOutputBias
	}
	if spec.Architecture == "pangu-embedded" {
		required["attention output bias"] = weights.AttentionOutputBias
	}
	if isLaguna || isAFMoE {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
		required["attention output gate"] = weights.AttentionOutputGate
	}
	if isLlama4 && !spec.UsesRoPE(layerIndex) {
		required["attention temperature scale"] = weights.AttentionTemperatureScale
	}
	if spec.Architecture == "mistral3" && spec.AttentionTempScale != 0 {
		required["attention temperature scale"] = weights.AttentionTemperatureScale
	}
	if spec.Architecture == "stablelm" &&
		(weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
		return DenseBlockResult{}, errors.New("StableLM Q/K norm weights must both be present or absent")
	}
	if spec.Architecture == "mpt" &&
		(weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
		return DenseBlockResult{}, errors.New("MPT Q/K norm weights must both be present or absent")
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("dense block %s weight is nil", name)
		}
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, fmt.Errorf("dense block has %d positions for %d tokens", len(positions), input.Shape.Dims[1])
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("dense block past key/value cache must both be present")
	}
	if spec.NonCausalAttention && spec.Architecture != "dflash" && pastKey != nil {
		return DenseBlockResult{}, errors.New("non-causal dense block does not support a KV cache")
	}

	tokens := uint64(len(positions))
	normalized := input
	if !isOLMo2 && !isPostOnlyNorm && !spec.SandwichNorm {
		normalized = ApplyNormalization(
			builder, input, weights.AttentionNorm, weights.AttentionNormBias, spec,
		)
	}
	feedForwardNormalized := normalized
	var attentionGate *tensor.Tensor
	if isLaguna || isAFMoE {
		attentionGate = builder.MulMat(weights.AttentionOutputGate, normalized)
		if isLaguna {
			attentionGate = builder.Softplus(attentionGate)
		} else {
			attentionGate = builder.Sigmoid(attentionGate)
		}
	}
	if isStep35 && weights.AttentionOutputGate != nil {
		attentionGate = builder.Sigmoid(builder.MulMat(weights.AttentionOutputGate, normalized))
	}
	if spec.Architecture == "falcon" && weights.AttentionNorm2 != nil {
		if weights.AttentionNorm2Bias == nil {
			normalized = builder.Multiply(
				builder.LayerNorm(input, spec.LayerNormEpsilon), weights.AttentionNorm2,
			)
		} else {
			normalized = ApplyNormalization(
				builder, input, weights.AttentionNorm2, weights.AttentionNorm2Bias, spec,
			)
		}
	}
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		queryLength := uint64(headCount) * uint64(spec.KeyLength)
		keyLength := uint64(kvHeadCount) * uint64(spec.KeyLength)
		valueLength := uint64(kvHeadCount) * uint64(spec.ValueLength)
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		if spec.AttentionClamp > 0 {
			mixed = builder.Clamp(mixed, -spec.AttentionClamp, spec.AttentionClamp)
		}
		stride := queryLength + keyLength + valueLength
		query = builder.Reshape(
			builder.GroupSlice(mixed, 0, queryLength, 1, stride),
			queryLength,
			tokens,
		)
		key = builder.Reshape(
			builder.GroupSlice(mixed, queryLength, keyLength, 1, stride),
			keyLength,
			tokens,
		)
		value = builder.Reshape(
			builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride),
			valueLength,
			tokens,
		)
	} else {
		query = builder.MulMat(weights.AttentionQ, normalized)
		key = builder.MulMat(weights.AttentionK, normalized)
		value = builder.MulMat(weights.AttentionV, normalized)
		if weights.AttentionQScale != nil {
			query = builder.Multiply(query, weights.AttentionQScale)
		}
		if weights.AttentionKScale != nil {
			key = builder.Multiply(key, weights.AttentionKScale)
		}
		if weights.AttentionVScale != nil {
			value = builder.Multiply(value, weights.AttentionVScale)
		}
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
		if spec.AttentionClamp > 0 {
			query = builder.Clamp(query, -spec.AttentionClamp, spec.AttentionClamp)
			key = builder.Clamp(key, -spec.AttentionClamp, spec.AttentionClamp)
			value = builder.Clamp(value, -spec.AttentionClamp, spec.AttentionClamp)
		}
	}
	if isOLMo2 || isOLMoE || isMiniMaxM2 {
		query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	}
	if spec.Architecture == "mpt" && weights.AttentionQNorm != nil {
		query = ApplyNormalization(
			builder, query, weights.AttentionQNorm, weights.AttentionQNormBias, spec,
		)
		key = ApplyNormalization(
			builder, key, weights.AttentionKNorm, weights.AttentionKNormBias, spec,
		)
	}
	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(headCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(kvHeadCount), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(kvHeadCount), tokens)

	if spec.Architecture == "apertus" || isAFMoE || isBailingMoE2 || isDOTS1 || spec.Architecture == "dflash" || spec.Architecture == "exaone4" || isEXAOneMoE || isGroveMoE || isHYV3 || isLLaDAMoE || isMellum || spec.Architecture == "openelm" || spec.Architecture == "plamo2" || spec.Architecture == "plamo3" || spec.Architecture == "qwen3" || spec.Architecture == "qwen3moe" || spec.Architecture == "qwen3vl" || spec.Architecture == "qwen3vlmoe" || spec.Architecture == "rnd1" || isLaguna || spec.Architecture == "lfm2" || isLFM2MoE || spec.Architecture == "gemma3" {
		if weights.AttentionQNorm == nil || weights.AttentionKNorm == nil {
			return DenseBlockResult{}, errors.New("dense block architecture requires Q/K norm weights")
		}
		query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	}
	if isGLM4MoE {
		if (weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
			return DenseBlockResult{}, errors.New("GLM4-MoE Q/K norm weights are incomplete")
		}
		if weights.AttentionQNorm != nil {
			query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
			key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
		}
	}
	if isStep35 {
		if (weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
			return DenseBlockResult{}, errors.New("Step3.5 Q/K norm weights are incomplete")
		}
		if weights.AttentionQNorm != nil {
			query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
			key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
		}
	}
	if isCommandRQKNorm {
		query = ApplyNormalization(builder, query, weights.AttentionQNorm, nil, spec)
		key = ApplyNormalization(builder, key, weights.AttentionKNorm, nil, spec)
	}
	if isChameleon {
		query = builder.Multiply(builder.LayerNorm(query, spec.QKNormEpsilon), weights.AttentionQNorm)
		key = builder.Multiply(builder.LayerNorm(key, spec.QKNormEpsilon), weights.AttentionKNorm)
		if weights.AttentionQNormBias != nil {
			query = builder.Add(query, weights.AttentionQNormBias)
		}
		if weights.AttentionKNormBias != nil {
			key = builder.Add(key, weights.AttentionKNormBias)
		}
	}
	if spec.Architecture == "stablelm" && weights.AttentionQNorm != nil {
		query = builder.Multiply(
			builder.LayerNorm(query, spec.LayerNormEpsilon),
			weights.AttentionQNorm,
		)
		key = builder.Multiply(
			builder.LayerNorm(key, spec.LayerNormEpsilon),
			weights.AttentionKNorm,
		)
	}
	rotaryDimensions := spec.KeyLength
	if spec.RopeDimensionCount > 0 {
		rotaryDimensions = spec.LayerRopeDimensionCount(layerIndex)
	}
	if !spec.UsesRoPE(layerIndex) {
		// Periodic position-independent layer.
	} else if spec.Architecture == "paddleocr" || spec.Architecture == "qwen2vl" || spec.Architecture == "qwen3vl" || spec.Architecture == "qwen3vlmoe" || ((spec.Architecture == "glm4" || isGLM4MoE) && spec.RopeSections[0] > 0 && spec.RopeSections[1] > 0) || ((spec.Architecture == "hunyuan-dense" || spec.Architecture == "hunyuan_vl") && spec.RopeSections[0] > 0 && spec.RopeSections[1] > 0) {
		resolved := [4][]uint32{}
		if multiPositions == nil {
			for axis := range resolved {
				resolved[axis] = positions
			}
		} else {
			resolved = *multiPositions
		}
		frequencyScale := float32(1)
		if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		query = builder.RoPEMultiScaled(
			query, resolved, spec.RopeSections,
			rotaryDimensions, spec.RopeFrequencyBase, frequencyScale,
		)
		key = builder.RoPEMultiScaled(
			key, resolved, spec.RopeSections,
			rotaryDimensions, spec.RopeFrequencyBase, frequencyScale,
		)
	} else if isLaguna {
		if spec.IsSlidingLayer(layerIndex) {
			query = builder.RoPENeoX(
				query, positions, spec.RopeDimensionSWA, spec.RopeFrequencySWA,
			)
			key = builder.RoPENeoX(
				key, positions, spec.RopeDimensionSWA, spec.RopeFrequencySWA,
			)
		} else {
			frequencyScale := float32(1) / spec.RopeScalingFactor
			query = builder.RoPENeoXYaRN(
				query, positions, spec.RopeDimensionCount, spec.OriginalContextLength,
				spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
				spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
			)
			key = builder.RoPENeoXYaRN(
				key, positions, spec.RopeDimensionCount, spec.OriginalContextLength,
				spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
				spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
			)
		}
	} else if (isGrok || isMellum) && spec.RopeScalingType == "yarn" && !spec.IsSlidingLayer(layerIndex) {
		frequencyScale := float32(1) / spec.RopeScalingFactor
		query = builder.RoPENeoXYaRN(
			query, positions, rotaryDimensions, spec.OriginalContextLength,
			spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
			spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
		)
		key = builder.RoPENeoXYaRN(
			key, positions, rotaryDimensions, spec.OriginalContextLength,
			spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
			spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
		)
	} else if (spec.Architecture == "llama" || spec.Architecture == "llama-embed" ||
		spec.Architecture == "minicpm" || spec.Architecture == "mistral3") &&
		spec.RopeScalingType == "yarn" {
		frequencyScale := float32(1) / spec.RopeScalingFactor
		if weights.RopeFactors != nil {
			query = builder.RoPENormalYaRNWithFactors(
				query, positions, rotaryDimensions, spec.OriginalContextLength,
				spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
				spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
				weights.RopeFactors,
			)
			key = builder.RoPENormalYaRNWithFactors(
				key, positions, rotaryDimensions, spec.OriginalContextLength,
				spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
				spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
				weights.RopeFactors,
			)
		} else {
			query = builder.RoPENormalYaRN(
				query, positions, rotaryDimensions, spec.OriginalContextLength,
				spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
				spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
			)
			key = builder.RoPENormalYaRN(
				key, positions, rotaryDimensions, spec.OriginalContextLength,
				spec.RopeFrequencyBase, frequencyScale, spec.YaRNExtFactor,
				spec.YaRNAttentionFactor, spec.YaRNBetaFast, spec.YaRNBetaSlow,
			)
		}
	} else if profile.Position == PositionNormal {
		frequencyBase := spec.RopeFrequencyBase
		frequencyScale := float32(1)
		if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		if (spec.Architecture == "cohere2" || isCohere2MoE || isLlama4 || isGPTOSS) && spec.IsSlidingLayer(layerIndex) {
			frequencyBase = spec.RopeFrequencySWA
		}
		if weights.RopeFactors != nil {
			query = builder.RoPENormalScaledWithFactors(
				query, positions, rotaryDimensions, frequencyBase, frequencyScale, weights.RopeFactors,
			)
			key = builder.RoPENormalScaledWithFactors(
				key, positions, rotaryDimensions, frequencyBase, frequencyScale, weights.RopeFactors,
			)
		} else {
			query = builder.RoPENormalScaled(
				query, positions, rotaryDimensions, frequencyBase, frequencyScale,
			)
			key = builder.RoPENormalScaled(
				key, positions, rotaryDimensions, frequencyBase, frequencyScale,
			)
		}
	} else if profile.Has(ArchitectureGemma) {
		frequencyBase := spec.RopeFrequencyBase
		frequencyScale := float32(1)
		if spec.Architecture == "gemma3" {
			frequencyScale = 1 / spec.RopeScalingFactor
		} else if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		if spec.IsSlidingLayer(layerIndex) {
			frequencyBase = spec.RopeFrequencySWA
			if spec.Architecture == "gemma3" {
				frequencyScale = 1
			}
		}
		if weights.RopeFactors != nil {
			query = builder.RoPENeoXScaledWithFactors(
				query, positions, rotaryDimensions, frequencyBase, frequencyScale, weights.RopeFactors,
			)
			key = builder.RoPENeoXScaledWithFactors(
				key, positions, rotaryDimensions, frequencyBase, frequencyScale, weights.RopeFactors,
			)
		} else {
			query = builder.RoPENeoXScaled(
				query, positions, rotaryDimensions, frequencyBase, frequencyScale,
			)
			key = builder.RoPENeoXScaled(
				key, positions, rotaryDimensions, frequencyBase, frequencyScale,
			)
		}
	} else {
		frequencyBase := spec.RopeFrequencyBase
		frequencyScale := float32(1)
		if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		if isOLMo2 && spec.IsSlidingLayer(layerIndex) {
			frequencyScale = 1
		}
		if (isAFMoE || isEXAOneMoE || isMiMo2 || isStep35 || isSmallThinker || spec.Architecture == "plamo3") && spec.IsSlidingLayer(layerIndex) {
			frequencyBase = spec.RopeFrequencySWA
		}
		if isMellum && spec.IsSlidingLayer(layerIndex) {
			frequencyBase = spec.RopeFrequencySWA
			frequencyScale = 1
		}
		ropeFactors := weights.RopeFactors
		if isStep35 && ropeFactors != nil && ropeFactors.Shape.Dims[0] > uint64(rotaryDimensions/2) {
			ropeFactors = builder.FlatSlice(ropeFactors, 0, uint64(rotaryDimensions/2))
		}
		if ropeFactors != nil {
			query = builder.RoPENeoXScaledWithFactors(
				query, positions, rotaryDimensions, frequencyBase, frequencyScale, ropeFactors,
			)
			key = builder.RoPENeoXScaledWithFactors(
				key, positions, rotaryDimensions, frequencyBase, frequencyScale, ropeFactors,
			)
		} else {
			query = builder.RoPENeoXScaled(
				query, positions, rotaryDimensions, frequencyBase, frequencyScale,
			)
			key = builder.RoPENeoXScaled(
				key, positions, rotaryDimensions, frequencyBase, frequencyScale,
			)
		}
	}
	if profile.Has(ArchitectureLongRoPE) && spec.RopeScalingType != "yarn" && spec.RopeAttentionFactor > 0 && spec.RopeAttentionFactor != 1 {
		query = builder.Scale(query, spec.RopeAttentionFactor)
		key = builder.Scale(key, spec.RopeAttentionFactor)
	}
	if isLlama4 && spec.UsesRoPE(layerIndex) && spec.ExpertCount != 128 {
		query = builder.RMSNorm(query, spec.RMSNormEpsilon)
		key = builder.RMSNorm(key, spec.RMSNormEpsilon)
	}
	if isLlama4 && !spec.UsesRoPE(layerIndex) {
		query = builder.Multiply(query, weights.AttentionTemperatureScale)
	}
	if spec.Architecture == "mistral3" && spec.AttentionTempScale != 0 {
		query = builder.Multiply(query, weights.AttentionTemperatureScale)
	}
	if spec.Architecture == "maincoder" {
		if weights.AttentionQNorm == nil || weights.AttentionKNorm == nil {
			return DenseBlockResult{}, errors.New("dense block architecture requires Q/K norm weights")
		}
		query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	}
	if isHunyuan {
		query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	}

	cacheKey := key
	cacheValue := value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("dense block KV cache token count exceeds uint32")
		}
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	if spec.AttentionScale > 0 {
		attentionScale = spec.AttentionScale
	}
	if spec.Architecture == "phi2" || spec.Architecture == "phi3" || isPhiMoE {
		// Phi decoders scale rotated query before dot product to
		// preserve upstream precision behavior
		query = builder.Scale(query, attentionScale)
		attentionScale = 1
	}
	if profile.Has(ArchitectureGemma) {
		// Gemma scales Q before attention dot product, rather than scaling
		// accumulated score; Preserve that ordering for quantized parity
		if spec.Architecture == "gemma2" && spec.BlockCount == 46 {
			attentionScale = float32(1 / math.Sqrt(
				float64(spec.EmbeddingLength)/float64(spec.HeadCount),
			))
		}
		query = builder.Scale(query, attentionScale)
		attentionScale = 1
	}
	var attention *tensor.Tensor
	if isLlama4 && spec.IsSlidingLayer(layerIndex) {
		attention = builder.AttentionChunkedWindowWithOffset(
			query, cacheKey, cacheValue, attentionScale, true, queryStart, spec.SlidingWindow,
		)
	} else if spec.IsSlidingLayer(layerIndex) {
		causal := !spec.NonCausalAttention
		if (isMiMo2 || isGPTOSS) && weights.AttentionSinks != nil {
			attention = builder.AttentionWindowWithSinksWithOffset(
				query, cacheKey, cacheValue, weights.AttentionSinks, attentionScale,
				true, queryStart, spec.SlidingWindow,
			)
		} else if spec.AttentionSoftcap > 0 {
			attention = builder.AttentionWindowSoftcappedWithOffset(
				query, cacheKey, cacheValue, attentionScale, spec.AttentionSoftcap,
				true, queryStart, spec.SlidingWindow,
			)
		} else {
			attention = builder.AttentionWindowWithOffset(
				query, cacheKey, cacheValue, attentionScale, causal, queryStart, spec.SlidingWindow,
			)
		}
	} else {
		causal := !spec.NonCausalAttention
		if (isMiMo2 || isGPTOSS) && weights.AttentionSinks != nil {
			attention = builder.AttentionWithSinksWithOffset(
				query, cacheKey, cacheValue, weights.AttentionSinks, attentionScale, causal, queryStart,
			)
		} else if spec.MaxALiBiBias > 0 {
			attention = builder.AttentionALiBiWithOffset(
				query, cacheKey, cacheValue, attentionScale, spec.MaxALiBiBias,
				causal, queryStart,
			)
		} else if spec.AttentionSoftcap > 0 {
			attention = builder.AttentionSoftcappedWithOffset(
				query, cacheKey, cacheValue, attentionScale, spec.AttentionSoftcap,
				causal, queryStart,
			)
		} else {
			attention = builder.AttentionWithOffset(
				query, cacheKey, cacheValue, attentionScale, causal, queryStart,
			)
		}
	}
	if (isLaguna || isStep35) && attentionGate != nil && weights.AttentionOutputGate.Shape.Dims[1] == uint64(headCount) {
		attentionGate = builder.Reshape(attentionGate, 1, uint64(headCount), tokens)
		attention = builder.Multiply(attention, attentionGate)
	}
	attention = builder.Reshape(attention, uint64(headCount)*uint64(spec.ValueLength), tokens)
	if isLaguna && weights.AttentionOutputGate.Shape.Dims[1] != uint64(headCount) {
		attention = builder.Multiply(attention, attentionGate)
	}
	if isAFMoE {
		attention = builder.Multiply(attention, attentionGate)
	}
	if spec.Architecture == "bitnet" {
		attention = builder.WeightedRMSNorm(
			attention, weights.AttentionSubNorm, spec.RMSNormEpsilon,
		)
	}
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if isMiMo2 && spec.AttentionValueScale != 0 {
		attention = builder.Scale(attention, spec.AttentionValueScale)
	}
	if weights.AttentionOutputScale != nil {
		attention = builder.Multiply(attention, weights.AttentionOutputScale)
	}
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	if spec.SandwichNorm {
		attention = builder.WeightedRMSNorm(attention, weights.AttentionNorm, spec.RMSNormEpsilon)
	}
	if normPlan.PostAttention {
		if weights.AttentionPostNorm == nil || weights.FeedForwardPostNorm == nil {
			return DenseBlockResult{}, errors.New("dense post-normalized block requires post norm weights")
		}
		attention = builder.WeightedRMSNorm(
			attention, weights.AttentionPostNorm, spec.RMSNormEpsilon,
		)
	}
	if spec.ResidualScale > 0 {
		attention = builder.Scale(attention, spec.ResidualScale)
	}
	residual := builder.Add(input, attention)
	if isGPTOSS {
		normalized = builder.WeightedRMSNorm(residual, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	}
	if isArctic {
		denseInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
		denseGate := builder.MulMat(weights.FeedForwardGate, denseInput)
		denseUp := builder.MulMat(weights.FeedForwardUp, denseInput)
		dense := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(denseGate, denseUp))
		denseOutput := builder.Add(residual, dense)
		expertInput := builder.WeightedRMSNorm(input, weights.FeedForwardExpertNorm, spec.RMSNormEpsilon)
		experts := builder.MoE(
			expertInput, weights.FeedForwardRouter,
			weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
			weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
			spec.ExpertWeightsScale,
		)
		output := builder.Add(denseOutput, experts)
		if err := builder.Err(); err != nil {
			return DenseBlockResult{}, err
		}
		return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
	}

	parallelResidual := profile.Residual == ResidualParallel ||
		(spec.Architecture == "gptneox" && spec.ParallelResidual) ||
		(spec.Architecture == "stablelm" && weights.FeedForwardNorm == nil)
	if spec.Architecture == "falcon" {
		normalized = feedForwardNormalized
	} else if spec.Architecture == "gptneox" && spec.ParallelResidual {
		// GPT-NeoX parallel blocks use distinct FFN LayerNorm over
		// original residual input rather than sharing attention norm
		normalized = ApplyNormalization(
			builder, input, weights.FeedForwardNorm, weights.FeedForwardNormBias, spec,
		)
	} else if !parallelResidual && !isGPTOSS {
		normalized = residual
		if !isOLMo2 && !isPostOnlyNorm && !spec.SandwichNorm {
			if spec.Architecture == "stablelm" && weights.FeedForwardNormBias == nil {
				normalized = builder.Multiply(
					builder.LayerNorm(residual, spec.LayerNormEpsilon),
					weights.FeedForwardNorm,
				)
			} else {
				normalized = ApplyNormalization(
					builder, residual, weights.FeedForwardNorm, weights.FeedForwardNormBias, spec,
				)
			}
		}
	}
	// Cohere: shared normalized input; parallel attention/FFN.
	if usesExperts {
		var feedForward *tensor.Tensor
		if isRefactMoE {
			if weights.FeedForwardGateExperts == nil {
				feedForward = builder.MoEUngated(
					normalized, weights.FeedForwardRouter, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoE(
					normalized, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
					weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
					spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
				)
			}
		} else if isErnieMoE {
			if weights.FeedForwardGateExperts == nil {
				if weights.FeedForwardExpertBias != nil {
					feedForward = builder.MoEUngatedWithSelectionBias(
						normalized, weights.FeedForwardRouter, weights.FeedForwardUpExperts,
						weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
						spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
					)
				} else {
					feedForward = builder.MoEUngated(
						normalized, weights.FeedForwardRouter, weights.FeedForwardUpExperts,
						weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
						spec.ExpertWeightsScale,
					)
				}
			} else if weights.FeedForwardExpertBias != nil {
				feedForward = builder.MoESoftmaxWithSelectionBias(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoE(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
					spec.ExpertWeightsScale,
				)
			}
			if spec.SharedExpertFF > 0 {
				sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
				sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
				shared := builder.MulMat(
					weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
				)
				feedForward = builder.Add(feedForward, shared)
			}
		} else if isHYV3 || isDeepSeek2OCR {
			if weights.FeedForwardGateUpExperts != nil {
				if spec.ExpertGatingFunc == 2 {
					feedForward = builder.MoESigmoidFusedGateUp(
						normalized, weights.FeedForwardRouter, weights.FeedForwardGateUpExperts,
						weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
						spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
					)
				} else {
					feedForward = builder.MoESoftmaxFusedGateUp(
						normalized, weights.FeedForwardRouter, weights.FeedForwardGateUpExperts,
						weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
						spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
					)
				}
			} else if spec.ExpertGatingFunc == 2 {
				feedForward = builder.MoESigmoid(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			} else if weights.FeedForwardExpertBias != nil {
				feedForward = builder.MoESoftmaxWithSelectionBias(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoE(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, spec.ExpertUsedCount,
					spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			}
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(
				weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
			)
			feedForward = builder.Add(feedForward, shared)
		} else if isCohere2MoE {
			if weights.FeedForwardGateUpExperts != nil {
				feedForward = builder.MoESigmoidFusedGateUp(
					normalized, weights.FeedForwardRouter, weights.FeedForwardGateUpExperts,
					weights.FeedForwardDownExperts, nil, spec.ExpertUsedCount,
					spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoESigmoid(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, nil, spec.ExpertUsedCount,
					spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			}
			if spec.SharedExpertFF > 0 {
				sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
				sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
				shared := builder.MulMat(
					weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
				)
				feedForward = builder.Scale(builder.Add(feedForward, shared), 0.5)
			}
		} else if isGLM4MoE {
			if spec.ExpertGatingFunc == 2 {
				feedForward = builder.MoESigmoid(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoESoftmaxWithSelectionBias(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			}
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(
				weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
			)
			feedForward = builder.Add(feedForward, shared)
		} else if isGroveMoE {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
				spec.ExpertWeightsScale,
			)
			chunkTopK := spec.ExpertUsedCount
			chunkExperts := spec.ExpertCount / spec.ExpertsPerGroup
			if chunkTopK > chunkExperts {
				chunkTopK = chunkExperts
			}
			chunk := builder.MoEGroupedWithRouterInput(
				feedForward, normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateChunkExperts, weights.FeedForwardUpChunkExperts,
				weights.FeedForwardDownChunkExperts, chunkTopK, true,
				spec.ExpertWeightsScale, spec.ExpertsPerGroup,
			)
			feedForward = builder.Add(feedForward, builder.Scale(chunk, spec.ExpertGroupScale))
		} else if isGrok {
			feedForward = builder.MoEGELU(
				normalized, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
				weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
				spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
			if weights.FeedForwardUp != nil || weights.FeedForwardGate != nil || weights.FeedForwardDown != nil {
				if weights.FeedForwardUp == nil || weights.FeedForwardGate == nil || weights.FeedForwardDown == nil {
					return DenseBlockResult{}, errors.New("Grok dense FFN weights are incomplete")
				}
				gate := builder.MulMat(weights.FeedForwardGate, normalized)
				up := builder.MulMat(weights.FeedForwardUp, normalized)
				dense := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
				feedForward = builder.Scale(builder.Add(dense, feedForward), float32(math.Sqrt(0.5)))
			}
			if weights.FeedForwardPostNorm == nil {
				return DenseBlockResult{}, errors.New("Grok feed-forward post norm is nil")
			}
			feedForward = builder.WeightedRMSNorm(
				feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon,
			)
			output := builder.Add(residual, feedForward)
			if err := builder.Err(); err != nil {
				return DenseBlockResult{}, err
			}
			return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
		} else if isSmallThinker {
			routing := tensor.MoERoutingSoftmax
			if spec.ExpertGatingFunc == 2 {
				routing = tensor.MoERoutingSigmoid
			}
			feedForward = builder.MoEReLUWithRouterInput(
				normalized, input, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
				spec.ExpertWeightsScale, routing,
			)
		} else if isMiMo2 {
			feedForward = builder.MoESigmoid(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
				spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		} else if isStep35 {
			limit := spec.LayerExpertSwiGLUClamp(layerIndex)
			if spec.ExpertGatingFunc == 2 {
				feedForward = builder.MoESigmoidLimited(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, spec.ExpertWeightsNorm,
					spec.ExpertWeightsScale, limit,
				)
			} else {
				feedForward = builder.MoESoftmaxLimitedWithSelectionBias(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, spec.ExpertWeightsNorm,
					spec.ExpertWeightsScale, limit,
				)
			}
			if spec.SharedExpertFF > 0 {
				sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
				sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
				shared := builder.MulMat(
					weights.FeedForwardSharedDown,
					limitedSwiGLU(builder, sharedGate, sharedUp, spec.LayerSharedSwiGLUClampLimit(layerIndex)),
				)
				feedForward = builder.Add(feedForward, shared)
			}
		} else if isMiniMaxM2 {
			if spec.ExpertGatingFunc == 2 {
				feedForward = builder.MoESigmoid(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoESoftmaxWithSelectionBias(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
				)
			}
		} else if isGPTOSS {
			feedForward = builder.MoEOpenAI(
				normalized, weights.FeedForwardRouter, weights.FeedForwardRouterBias,
				weights.FeedForwardGateExperts, weights.FeedForwardGateBias,
				weights.FeedForwardUpExperts, weights.FeedForwardUpBias,
				weights.FeedForwardDownExperts, weights.FeedForwardDownBias,
				spec.ExpertUsedCount, spec.ExpertWeightsScale,
			)
		} else if isLlama4 {
			feedForward = builder.MoESigmoid(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, nil,
				spec.ExpertUsedCount, false, spec.ExpertWeightsScale,
			)
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(
				weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
			)
			feedForward = builder.Add(feedForward, shared)
		} else if isLaguna || isAFMoE || ((isEXAOneMoE || isBailingMoE2 || isLFM2MoE || isDOTS1) && spec.ExpertGatingFunc == 2) {
			feedForward = builder.MoESigmoid(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
				spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
			)
			if spec.SharedExpertFF > 0 {
				sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
				sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
				shared := builder.MulMat(
					weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
				)
				feedForward = builder.Add(feedForward, shared)
			}
		} else if isEXAOneMoE || isBailingMoE2 || isLFM2MoE || isDOTS1 {
			if weights.FeedForwardExpertBias != nil {
				feedForward = builder.MoESoftmaxWithSelectionBias(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
					spec.ExpertUsedCount, spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoE(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, spec.ExpertUsedCount,
					spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
				)
			}
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp))
			feedForward = builder.Add(feedForward, shared)
		} else if isHunyuanMoE {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
				spec.ExpertWeightsScale,
			)
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp))
			feedForward = builder.Add(feedForward, shared)
		} else if isBailingMoE {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount,
				spec.ExpertWeightsNorm, spec.ExpertWeightsScale,
			)
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(
				weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
			)
			feedForward = builder.Add(feedForward, shared)
		} else if isDeepSeek {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount, false,
				spec.ExpertWeightsScale,
			)
			sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
			sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
			shared := builder.MulMat(
				weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
			)
			feedForward = builder.Add(feedForward, shared)
		} else if isGraniteMoE {
			if weights.FeedForwardGateExperts == nil {
				feedForward = builder.MoEUngated(
					normalized, weights.FeedForwardRouter, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
					spec.ExpertWeightsScale,
				)
			} else {
				feedForward = builder.MoE(
					normalized, weights.FeedForwardRouter,
					weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
					weights.FeedForwardDownExperts, spec.ExpertUsedCount, true,
					spec.ExpertWeightsScale,
				)
			}
			if spec.SharedExpertFF > 0 {
				sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
				sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
				shared := builder.MulMat(
					weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
				)
				feedForward = builder.Add(feedForward, shared)
			}
		} else if isLLaDAMoE {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount, false,
				spec.ExpertWeightsScale,
			)
		} else if isQwen2MoE || isOLMoE {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, spec.ExpertUsedCount, false,
				spec.ExpertWeightsScale,
			)
			if isQwen2MoE {
				sharedGateWeight := builder.Reshape(
					weights.FeedForwardSharedRouter, uint64(spec.EmbeddingLength), 1,
				)
				sharedScale := builder.Sigmoid(builder.MulMat(sharedGateWeight, normalized))
				sharedGate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
				sharedUp := builder.MulMat(weights.FeedForwardSharedUp, normalized)
				shared := builder.MulMat(
					weights.FeedForwardSharedDown, builder.SwiGLU(sharedGate, sharedUp),
				)
				feedForward = builder.Add(feedForward, builder.Multiply(shared, sharedScale))
			}
		} else {
			normalizeWeights := true
			if isJamba {
				normalizeWeights = spec.ExpertWeightsNorm
			}
			feedForward = builder.MoE(
				normalized,
				weights.FeedForwardRouter,
				weights.FeedForwardGateExperts,
				weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts,
				spec.ExpertUsedCount,
				normalizeWeights,
				spec.ExpertWeightsScale,
			)
		}
		if isGraniteMoE && spec.ResidualScale > 0 {
			feedForward = builder.Scale(feedForward, spec.ResidualScale)
		}
		output := builder.Add(residual, feedForward)
		if err := builder.Err(); err != nil {
			return DenseBlockResult{}, err
		}
		return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
	}
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	if weights.FeedForwardUpScale != nil {
		up = builder.Multiply(up, weights.FeedForwardUpScale)
	}
	if weights.FeedForwardUpBias != nil {
		up = builder.Add(up, weights.FeedForwardUpBias)
	}
	var activation *tensor.Tensor
	if profile.FeedForward == FeedForwardFusedGateUp {
		width := uint64(spec.FeedForwardLength)
		stride := 2 * width
		gate := builder.Reshape(
			builder.GroupSlice(up, 0, width, 1, stride), width, tokens,
		)
		up = builder.Reshape(
			builder.GroupSlice(up, width, width, 1, stride), width, tokens,
		)
		activation = builder.SwiGLU(gate, up)
	} else if spec.Architecture == "apertus" {
		activation = builder.XIELU(
			up,
			spec.XIELUAlphaN[layerIndex],
			spec.XIELUAlphaP[layerIndex],
			spec.XIELUBeta[layerIndex],
			spec.XIELUEpsilon[layerIndex],
		)
	} else if profile.FeedForward == FeedForwardGELU || profile.FeedForward == FeedForwardSequentialGELU {
		activation = builder.GELU(up)
	} else if profile.FeedForward == FeedForwardSquaredReLU {
		activation = builder.ReLUSquared(up)
	} else {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		if weights.FeedForwardGateScale != nil {
			gate = builder.Multiply(gate, weights.FeedForwardGateScale)
		}
		if weights.FeedForwardGateBias != nil {
			gate = builder.Add(gate, weights.FeedForwardGateBias)
		}
		activation = builder.SwiGLU(gate, up)
		if profile.Has(ArchitectureGemma) {
			activation = builder.GEGLU(gate, up)
		}
	}
	if weights.FeedForwardActivationScale != nil {
		if spec.Architecture != "mpt" {
			return DenseBlockResult{}, errors.New("feed-forward activation scale requires MPT")
		}
		activation = builder.Divide(activation, weights.FeedForwardActivationScale)
	}
	if spec.Architecture == "bitnet" {
		activation = builder.WeightedRMSNorm(
			activation, weights.FeedForwardSubNorm, spec.RMSNormEpsilon,
		)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, activation)
	if weights.FeedForwardDownScale != nil {
		feedForward = builder.Multiply(feedForward, weights.FeedForwardDownScale)
	}
	if weights.FeedForwardDownBias != nil {
		feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
	}
	if spec.SandwichNorm {
		feedForward = builder.WeightedRMSNorm(
			feedForward, weights.FeedForwardNorm, spec.RMSNormEpsilon,
		)
	}
	if normPlan.PostFeedForward {
		feedForward = builder.WeightedRMSNorm(
			feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon,
		)
	}
	if spec.ResidualScale > 0 {
		feedForward = builder.Scale(feedForward, spec.ResidualScale)
	}
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}

func buildGemma4BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Gemma 4 block input shape is invalid")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("Gemma 4 cache pair is incomplete")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":         weights.AttentionNorm,
		"attention query":        weights.AttentionQ,
		"attention query norm":   weights.AttentionQNorm,
		"attention output":       weights.AttentionOutput,
		"attention post norm":    weights.AttentionPostNorm,
		"feed-forward norm":      weights.FeedForwardNorm,
		"feed-forward gate":      weights.FeedForwardGate,
		"feed-forward up":        weights.FeedForwardUp,
		"feed-forward down":      weights.FeedForwardDown,
		"feed-forward post norm": weights.FeedForwardPostNorm,
	}
	if spec.LayerHasKV(layerIndex) {
		required["attention key"] = weights.AttentionK
		required["attention key norm"] = weights.AttentionKNorm
	} else if pastKey == nil {
		return DenseBlockResult{}, errors.New("Gemma 4 shared-KV layer has no source cache")
	}
	usesExperts := weights.FeedForwardRouter != nil
	if usesExperts {
		for name, item := range map[string]*tensor.Tensor{
			"expert router":           weights.FeedForwardRouter,
			"expert router scale":     weights.FeedForwardRouterScale,
			"expert down":             weights.FeedForwardDownExperts,
			"expert pre norm":         weights.FeedForwardPreNorm2,
			"dense expert post norm":  weights.FeedForwardPostNorm1,
			"routed expert post norm": weights.FeedForwardPostNorm2,
		} {
			required[name] = item
		}
		if weights.FeedForwardGateUpExperts == nil {
			required["expert gate"] = weights.FeedForwardGateExperts
			required["expert up"] = weights.FeedForwardUpExperts
		}
	}
	if spec.EmbeddingPerLayer > 0 {
		for name, item := range map[string]*tensor.Tensor{
			"per-layer input":      weights.PerLayerInput,
			"per-layer input gate": weights.PerLayerInputGate,
			"per-layer projection": weights.PerLayerProjection,
			"per-layer post norm":  weights.PerLayerPostNorm,
		} {
			required[name] = item
		}
	}
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Gemma 4 block %s weight is nil", name)
		}
	}
	tokens := input.Shape.Dims[1]
	headCount := uint64(spec.LayerHeadCount(layerIndex))
	kvHeadCount := uint64(spec.LayerKVHeadCount(layerIndex))
	keyLength := uint64(spec.LayerKeyLength(layerIndex))
	valueLength := uint64(spec.LayerValueLength(layerIndex))
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := builder.Reshape(
		builder.MulMat(weights.AttentionQ, normalized), keyLength, headCount, tokens,
	)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	frequencyBase := spec.RopeFrequencyBase
	if spec.IsSlidingLayer(layerIndex) {
		frequencyBase = spec.RopeFrequencySWA
	}
	rotaryDimensions := spec.LayerRopeDimensionCount(layerIndex)
	if weights.RopeFactors != nil {
		query = builder.RoPENeoXScaledWithFactors(
			query, positions, rotaryDimensions, frequencyBase, 1, weights.RopeFactors,
		)
	} else {
		query = builder.RoPENeoXScaled(query, positions, rotaryDimensions, frequencyBase, 1)
	}
	cacheKey, cacheValue := pastKey, pastValue
	queryStart := uint32(0)
	if spec.LayerHasKV(layerIndex) {
		key := builder.Reshape(
			builder.MulMat(weights.AttentionK, normalized), keyLength, kvHeadCount, tokens,
		)
		value := key
		if weights.AttentionV != nil {
			value = builder.Reshape(
				builder.MulMat(weights.AttentionV, normalized), valueLength, kvHeadCount, tokens,
			)
		}
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
		value = builder.RMSNorm(value, spec.RMSNormEpsilon)
		if weights.RopeFactors != nil {
			key = builder.RoPENeoXScaledWithFactors(
				key, positions, rotaryDimensions, frequencyBase, 1, weights.RopeFactors,
			)
		} else {
			key = builder.RoPENeoXScaled(key, positions, rotaryDimensions, frequencyBase, 1)
		}
		cacheKey, cacheValue = key, value
		if pastKey != nil {
			queryStart = uint32(pastKey.Shape.Dims[2])
			cacheKey = builder.Concat(pastKey, key, 2)
			cacheValue = builder.Concat(pastValue, value, 2)
		}
	} else {
		if pastKey.Shape.Rank != 3 || pastValue.Shape.Rank != 3 ||
			pastKey.Shape.Dims[0] != keyLength || pastValue.Shape.Dims[0] != valueLength ||
			pastKey.Shape.Dims[1] != kvHeadCount || pastValue.Shape.Dims[1] != kvHeadCount ||
			pastKey.Shape.Dims[2] != pastValue.Shape.Dims[2] || pastKey.Shape.Dims[2] < tokens {
			return DenseBlockResult{}, errors.New("Gemma 4 shared-KV source shape is invalid")
		}
		queryStart = uint32(pastKey.Shape.Dims[2] - tokens)
	}
	var attention *tensor.Tensor
	if spec.IsSlidingLayer(layerIndex) {
		if weights.AttentionBlockIDs != nil {
			attention = builder.AttentionWindowWithBlockMaskWithOffset(
				query, cacheKey, cacheValue, weights.AttentionBlockIDs,
				spec.AttentionScale, queryStart, spec.SlidingWindow,
			)
		} else {
			attention = builder.AttentionWindowWithOffset(
				query, cacheKey, cacheValue, spec.AttentionScale, true, queryStart, spec.SlidingWindow,
			)
		}
	} else {
		attention = builder.AttentionWithOffset(
			query, cacheKey, cacheValue, spec.AttentionScale, true, queryStart,
		)
	}
	attention = builder.Reshape(attention, headCount*valueLength, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	attention = builder.WeightedRMSNorm(attention, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	attentionOutput := builder.Add(input, attention)
	var feedForward *tensor.Tensor
	if usesExperts {
		denseInput := builder.WeightedRMSNorm(attentionOutput, weights.FeedForwardNorm, spec.RMSNormEpsilon)
		denseGate := builder.MulMat(weights.FeedForwardGate, denseInput)
		denseUp := builder.MulMat(weights.FeedForwardUp, denseInput)
		dense := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(denseGate, denseUp))
		dense = builder.WeightedRMSNorm(dense, weights.FeedForwardPostNorm1, spec.RMSNormEpsilon)
		expertInput := builder.WeightedRMSNorm(attentionOutput, weights.FeedForwardPreNorm2, spec.RMSNormEpsilon)
		routerInput := builder.Scale(builder.RMSNorm(attentionOutput, spec.RMSNormEpsilon), 1/float32(math.Sqrt(float64(spec.EmbeddingLength))))
		routerInput = builder.Multiply(routerInput, weights.FeedForwardRouterScale)
		if weights.FeedForwardGateUpExperts != nil {
			feedForward = builder.MoEGELUFusedGateUpWithRouterInput(
				expertInput, routerInput, weights.FeedForwardRouter,
				weights.FeedForwardGateUpExperts, weights.FeedForwardDownExperts,
				weights.FeedForwardDownExpertsScale,
				spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		} else {
			feedForward = builder.MoEGELUWithRouterInput(
				expertInput, routerInput, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, weights.FeedForwardDownExpertsScale,
				spec.ExpertUsedCount, true,
				spec.ExpertWeightsScale,
			)
		}
		feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm2, spec.RMSNormEpsilon)
		feedForward = builder.Add(dense, feedForward)
	} else {
		feedForwardInput := builder.WeightedRMSNorm(attentionOutput, weights.FeedForwardNorm, spec.RMSNormEpsilon)
		gate := builder.MulMat(weights.FeedForwardGate, feedForwardInput)
		up := builder.MulMat(weights.FeedForwardUp, feedForwardInput)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
	}
	feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	output := builder.Add(attentionOutput, feedForward)
	if spec.EmbeddingPerLayer > 0 {
		perLayer := builder.GELU(builder.MulMat(weights.PerLayerInputGate, output))
		perLayer = builder.Multiply(perLayer, weights.PerLayerInput)
		perLayer = builder.MulMat(weights.PerLayerProjection, perLayer)
		perLayer = builder.WeightedRMSNorm(perLayer, weights.PerLayerPostNorm, spec.RMSNormEpsilon)
		output = builder.Add(output, perLayer)
	}
	if weights.LayerOutputScale != nil {
		output = builder.Multiply(output, weights.LayerOutputScale)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}

// Gemma3nAttentionResult: attention/Laurel stage plus FFN projections.
type Gemma3nAttentionResult struct {
	Residual *tensor.Tensor
	Gate     *tensor.Tensor
	Up       *tensor.Tensor
	Key      *tensor.Tensor
	Value    *tensor.Tensor
}

// BuildGemma3nAttentionStage: active AltUp attention, Laurel, FFN projections.
func BuildGemma3nAttentionStage(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (Gemma3nAttentionResult, error) {
	if builder == nil || input == nil || spec.Architecture != "gemma3n" ||
		input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return Gemma3nAttentionResult{}, errors.New("Gemma 3n attention input is invalid")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return Gemma3nAttentionResult{}, errors.New("Gemma 3n cache pair is incomplete")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":          weights.AttentionNorm,
		"attention query":         weights.AttentionQ,
		"attention query norm":    weights.AttentionQNorm,
		"attention output":        weights.AttentionOutput,
		"attention post norm":     weights.AttentionPostNorm,
		"feed-forward norm":       weights.FeedForwardNorm,
		"feed-forward gate":       weights.FeedForwardGate,
		"feed-forward up":         weights.FeedForwardUp,
		"Laurel left projection":  weights.LaurelLeft,
		"Laurel right projection": weights.LaurelRight,
		"Laurel post norm":        weights.LaurelPostNorm,
	}
	if spec.LayerHasKV(layerIndex) {
		required["attention key"] = weights.AttentionK
		required["attention value"] = weights.AttentionV
		required["attention key norm"] = weights.AttentionKNorm
	} else if pastKey == nil {
		return Gemma3nAttentionResult{}, errors.New("Gemma 3n shared-KV layer has no source cache")
	}
	for name, item := range required {
		if item == nil {
			return Gemma3nAttentionResult{}, fmt.Errorf("Gemma 3n %s weight is nil", name)
		}
	}
	tokens := input.Shape.Dims[1]
	headCount := uint64(spec.LayerHeadCount(layerIndex))
	kvHeadCount := uint64(spec.LayerKVHeadCount(layerIndex))
	keyLength := uint64(spec.LayerKeyLength(layerIndex))
	valueLength := uint64(spec.LayerValueLength(layerIndex))
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	laurel := builder.MulMat(weights.LaurelLeft, normalized)
	laurel = builder.MulMat(weights.LaurelRight, laurel)
	laurel = builder.WeightedRMSNorm(laurel, weights.LaurelPostNorm, spec.RMSNormEpsilon)
	laurel = builder.Add(laurel, normalized)
	query := builder.Reshape(builder.MulMat(weights.AttentionQ, normalized), keyLength, headCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	frequencyBase := spec.RopeFrequencyBase
	if spec.IsSlidingLayer(layerIndex) {
		frequencyBase = spec.RopeFrequencySWA
	}
	query = builder.RoPENeoXScaled(query, positions, spec.LayerRopeDimensionCount(layerIndex), frequencyBase, 1)
	cacheKey, cacheValue := pastKey, pastValue
	queryStart := uint32(0)
	if spec.LayerHasKV(layerIndex) {
		key := builder.Reshape(builder.MulMat(weights.AttentionK, normalized), keyLength, kvHeadCount, tokens)
		value := builder.Reshape(builder.MulMat(weights.AttentionV, normalized), valueLength, kvHeadCount, tokens)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
		value = builder.RMSNorm(value, spec.RMSNormEpsilon)
		key = builder.RoPENeoXScaled(key, positions, spec.LayerRopeDimensionCount(layerIndex), frequencyBase, 1)
		cacheKey, cacheValue = key, value
		if pastKey != nil {
			if pastKey.Shape.Rank != 3 || pastKey.Shape.Dims[2] > math.MaxUint32 {
				return Gemma3nAttentionResult{}, errors.New("Gemma 3n cache shape is invalid")
			}
			queryStart = uint32(pastKey.Shape.Dims[2])
			cacheKey = builder.Concat(pastKey, key, 2)
			cacheValue = builder.Concat(pastValue, value, 2)
		}
	} else {
		if pastKey.Shape.Rank != 3 || pastValue.Shape.Rank != 3 ||
			pastKey.Shape.Dims[0] != keyLength || pastValue.Shape.Dims[0] != valueLength ||
			pastKey.Shape.Dims[1] != kvHeadCount || pastValue.Shape.Dims[1] != kvHeadCount ||
			pastKey.Shape.Dims[2] != pastValue.Shape.Dims[2] || pastKey.Shape.Dims[2] < tokens {
			return Gemma3nAttentionResult{}, errors.New("Gemma 3n shared-KV source shape is invalid")
		}
		queryStart = uint32(pastKey.Shape.Dims[2] - tokens)
	}
	var attention *tensor.Tensor
	attentionScale := spec.AttentionScale
	if attentionScale == 0 {
		attentionScale = 1 / float32(math.Sqrt(float64(keyLength)))
	}
	query = builder.Scale(query, attentionScale)
	if spec.IsSlidingLayer(layerIndex) {
		attention = builder.AttentionWindowWithOffset(
			query, cacheKey, cacheValue, 1, true, queryStart, spec.SlidingWindow,
		)
	} else {
		attention = builder.AttentionWithOffset(query, cacheKey, cacheValue, 1, true, queryStart)
	}
	attention = builder.MulMat(weights.AttentionOutput, builder.Reshape(attention, headCount*valueLength, tokens))
	attention = builder.WeightedRMSNorm(attention, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	residual := builder.Scale(builder.Add(builder.Add(input, attention), laurel), 1/float32(math.Sqrt2))
	feedForwardInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	result := Gemma3nAttentionResult{
		Residual: residual,
		Gate:     builder.MulMat(weights.FeedForwardGate, feedForwardInput),
		Up:       builder.MulMat(weights.FeedForwardUp, feedForwardInput),
		Key:      cacheKey,
		Value:    cacheValue,
	}
	if err := builder.Err(); err != nil {
		return Gemma3nAttentionResult{}, err
	}
	return result, nil
}

// BuildGemma3nFeedForwardOutput: down projection, post norm, residual.
func BuildGemma3nFeedForwardOutput(
	builder *tensor.Builder,
	residual, activated *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if builder == nil || residual == nil || activated == nil || spec.Architecture != "gemma3n" ||
		weights.FeedForwardDown == nil || weights.FeedForwardPostNorm == nil {
		return nil, errors.New("Gemma 3n feed-forward output is incomplete")
	}
	output := builder.MulMat(weights.FeedForwardDown, activated)
	output = builder.WeightedRMSNorm(output, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	output = builder.Add(residual, output)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildGemma4PerLayerInputs: projects token and model embeddings per block.
func BuildGemma4PerLayerInputs(
	builder *tensor.Builder,
	input, tokenEmbedding, modelProjection, projectionNorm *tensor.Tensor,
	spec Spec,
) ([]*tensor.Tensor, error) {
	if builder == nil || input == nil || tokenEmbedding == nil || modelProjection == nil || projectionNorm == nil {
		return nil, errors.New("Gemma 4 per-layer input is incomplete")
	}
	if (spec.Architecture != "gemma4" && spec.Architecture != "gemma3n") ||
		spec.EmbeddingPerLayer == 0 || spec.BlockCount == 0 ||
		input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("Gemma 4 per-layer input configuration is invalid")
	}
	width := uint64(spec.EmbeddingPerLayer)
	layers := uint64(spec.BlockCount)
	tokens := input.Shape.Dims[1]
	combinedWidth := width * layers
	if tokenEmbedding.Shape.Rank != 2 || tokenEmbedding.Shape.Dims[0] != combinedWidth ||
		tokenEmbedding.Shape.Dims[1] != tokens ||
		modelProjection.Shape.Rank != 2 || modelProjection.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		modelProjection.Shape.Dims[1] != combinedWidth ||
		projectionNorm.Shape.Rank != 1 || projectionNorm.Shape.Dims[0] != width {
		return nil, errors.New("Gemma 4 per-layer input shape is invalid")
	}
	projected := builder.MulMat(modelProjection, input)
	projected = builder.Scale(projected, 1/float32(math.Sqrt(float64(spec.EmbeddingLength))))
	projected = builder.Reshape(projected, width, layers*tokens)
	projected = builder.WeightedRMSNorm(projected, projectionNorm, spec.RMSNormEpsilon)
	selected := builder.Scale(tokenEmbedding, float32(math.Sqrt(float64(width))))
	selected = builder.Reshape(selected, width, layers*tokens)
	combined := builder.Scale(builder.Add(projected, selected), 1/float32(math.Sqrt(2)))
	combined = builder.Reshape(combined, combinedWidth, tokens)
	result := make([]*tensor.Tensor, spec.BlockCount)
	for layer := uint64(0); layer < layers; layer++ {
		result[layer] = builder.Reshape(
			builder.GroupSlice(combined, layer*width, width, 1, combinedWidth),
			width, tokens,
		)
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func limitedSwiGLU(builder *tensor.Builder, gate, up *tensor.Tensor, limit float32) *tensor.Tensor {
	if limit <= 0 {
		return builder.SwiGLU(gate, up)
	}
	up = builder.Clamp(up, -limit, limit)
	gate = builder.Clamp(builder.SiLU(gate), -math.MaxFloat32, limit)
	return builder.Multiply(gate, up)
}

func deepSeek4LimitedSwiGLU(builder *tensor.Builder, gate, up *tensor.Tensor, limit float32) *tensor.Tensor {
	if limit <= 0 {
		return builder.SwiGLU(gate, up)
	}
	gate = builder.Clamp(gate, -math.MaxFloat32, limit)
	up = builder.Clamp(up, -limit, limit)
	return builder.SwiGLU(gate, up)
}

func buildDeciSparseBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey *tensor.Tensor,
	pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Deci position count is invalid")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("Deci cache must contain both sentinel tensors")
	}
	sentinel := builder.GroupSlice(input, 0, 1, 1, input.Shape.Dims[0])
	cacheKey, cacheValue := sentinel, sentinel
	if pastKey != nil {
		wantPrefix := tensor.MustShape(1, 1, pastKey.Shape.Dims[2])
		if !pastKey.Shape.Equal(wantPrefix) || !pastValue.Shape.Equal(wantPrefix) {
			return DenseBlockResult{}, errors.New("Deci sentinel cache shape is invalid")
		}
		cacheKey = builder.Concat(pastKey, sentinel, 2)
		cacheValue = builder.Concat(pastValue, sentinel, 2)
	}
	if spec.LayerFeedForwardLength(layerIndex) == 0 {
		return DenseBlockResult{Output: input, Key: cacheKey, Value: cacheValue}, builder.Err()
	}
	ffnInput := input
	if spec.LayerHeadCount(layerIndex) > 0 {
		if weights.AttentionNorm == nil || weights.AttentionOutput == nil {
			return DenseBlockResult{}, errors.New("Deci linear-attention weights are incomplete")
		}
		projected := builder.MulMat(
			weights.AttentionOutput,
			builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon),
		)
		if weights.AttentionOutputBias != nil {
			projected = builder.Add(projected, weights.AttentionOutputBias)
		}
		ffnInput = builder.Add(projected, input)
	}
	for name, weight := range map[string]*tensor.Tensor{
		"norm": weights.FeedForwardNorm,
		"gate": weights.FeedForwardGate,
		"up":   weights.FeedForwardUp,
		"down": weights.FeedForwardDown,
	} {
		if weight == nil {
			return DenseBlockResult{}, fmt.Errorf("Deci feed-forward %s weight is nil", name)
		}
	}
	normalized := builder.WeightedRMSNorm(ffnInput, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	if weights.FeedForwardGateBias != nil {
		gate = builder.Add(gate, weights.FeedForwardGateBias)
	}
	if weights.FeedForwardUpBias != nil {
		up = builder.Add(up, weights.FeedForwardUpBias)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	if weights.FeedForwardDownBias != nil {
		feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
	}
	output := builder.Add(feedForward, ffnInput)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}

// BuildQwen35BlockCached: gated attention/GDN hybrid.
func BuildQwen35BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	recurrent bool,
	pastKey, pastValue, convState, ssmState *tensor.Tensor,
) (Qwen35BlockResult, error) {
	return buildQwen35BlockCached(
		builder, input, spec, weights, positions, nil, recurrent,
		pastKey, pastValue, convState, ssmState,
	)
}

// BuildQwen35BlockCachedWithMultiPositions: distinct MRoPE axes.
func BuildQwen35BlockCachedWithMultiPositions(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	multiPositions [4][]uint32,
	recurrent bool,
	pastKey, pastValue, convState, ssmState *tensor.Tensor,
) (Qwen35BlockResult, error) {
	if spec.Architecture != "qwen35" && spec.Architecture != "qwen35moe" {
		return Qwen35BlockResult{}, errors.New("Qwen hybrid architecture does not support multi-axis positions")
	}
	return buildQwen35BlockCached(
		builder, input, spec, weights, multiPositions[0], &multiPositions, recurrent,
		pastKey, pastValue, convState, ssmState,
	)
}

func buildQwen35BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	multiPositions *[4][]uint32,
	recurrent bool,
	pastKey, pastValue, convState, ssmState *tensor.Tensor,
) (Qwen35BlockResult, error) {
	if spec.Architecture != "qwen3next" && spec.Architecture != "qwen35" && spec.Architecture != "qwen35moe" {
		return Qwen35BlockResult{}, errors.New("Qwen hybrid block architecture is invalid")
	}
	if recurrent {
		return buildQwen35RecurrentBlock(
			builder,
			input,
			spec,
			weights,
			positions,
			convState,
			ssmState,
		)
	}
	result, err := buildQwen35AttentionBlock(
		builder,
		input,
		spec,
		weights,
		positions,
		multiPositions,
		pastKey,
		pastValue,
	)
	if err != nil {
		return Qwen35BlockResult{}, err
	}
	return Qwen35BlockResult{
		Output: result.Output,
		Key:    result.Key,
		Value:  result.Value,
	}, nil
}

func buildQwen35AttentionBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	multiPositions *[4][]uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil {
		return DenseBlockResult{}, errors.New("Qwen3.5 attention block input is nil")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":      weights.AttentionNorm,
		"attention Q/gate":    weights.AttentionQ,
		"attention K":         weights.AttentionK,
		"attention V":         weights.AttentionV,
		"attention output":    weights.AttentionOutput,
		"attention Q norm":    weights.AttentionQNorm,
		"attention K norm":    weights.AttentionKNorm,
		"post-attention norm": weights.FeedForwardNorm,
	}
	addQwen35FeedForwardRequirements(required, spec, weights)
	for name, item := range required {
		if item == nil {
			return DenseBlockResult{}, fmt.Errorf("Qwen3.5 attention block %s weight is nil", name)
		}
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Qwen3.5 attention position count is invalid")
	}
	if (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("Qwen3.5 attention cache must contain both key and value")
	}

	tokens := uint64(len(positions))
	headWidth := uint64(spec.KeyLength)
	heads := uint64(spec.HeadCount)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	queryAndGate := builder.MulMat(weights.AttentionQ, normalized)
	query := builder.GroupSlice(queryAndGate, 0, headWidth, heads, 2*headWidth)
	gate := builder.GroupSlice(queryAndGate, headWidth, headWidth, heads, 2*headWidth)
	key := builder.Reshape(
		builder.MulMat(weights.AttentionK, normalized),
		headWidth,
		uint64(spec.HeadCountKV),
		tokens,
	)
	value := builder.Reshape(
		builder.MulMat(weights.AttentionV, normalized),
		uint64(spec.ValueLength),
		uint64(spec.HeadCountKV),
		tokens,
	)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	frequencyScale := float32(1)
	if spec.RopeScalingType == "linear" {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	if spec.Architecture == "qwen3next" {
		query = builder.RoPENeoXScaled(
			query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, frequencyScale,
		)
		key = builder.RoPENeoXScaled(
			key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, frequencyScale,
		)
	} else {
		resolved := [4][]uint32{}
		if multiPositions == nil {
			for axis := range resolved {
				resolved[axis] = positions
			}
		} else {
			resolved = *multiPositions
		}
		query = builder.RoPEMultiScaled(
			query, resolved, spec.RopeSections, spec.RopeDimensionCount,
			spec.RopeFrequencyBase, frequencyScale,
		)
		key = builder.RoPEMultiScaled(
			key, resolved, spec.RopeSections, spec.RopeDimensionCount,
			spec.RopeFrequencyBase, frequencyScale,
		)
	}

	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("Qwen3.5 attention cache exceeds uint32")
		}
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	if spec.AttentionScale > 0 {
		attentionScale = spec.AttentionScale
	}
	attention := builder.AttentionWithOffset(
		query,
		cacheKey,
		cacheValue,
		attentionScale,
		true,
		queryStart,
	)
	attention = builder.Reshape(
		attention,
		uint64(spec.HeadCount)*uint64(spec.ValueLength),
		tokens,
	)
	gate = builder.Reshape(gate, uint64(spec.HeadCount)*headWidth, tokens)
	attention = builder.Multiply(attention, builder.Sigmoid(gate))
	attention = builder.MulMat(weights.AttentionOutput, attention)
	residual := builder.Add(input, attention)
	output := buildQwen35FeedForward(builder, residual, spec, weights)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}

func buildQwen35RecurrentBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	convState, ssmState *tensor.Tensor,
) (Qwen35BlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return Qwen35BlockResult{}, errors.New("Qwen3.5 recurrent block input/state is nil")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":      weights.AttentionNorm,
		"QKV":                 weights.AttentionQKV,
		"SSM convolution":     weights.SSMConv1D,
		"SSM time-step bias":  weights.SSMTimeStep,
		"SSM A":               weights.SSMA,
		"SSM norm":            weights.SSMNorm,
		"SSM output":          weights.SSMOutput,
		"post-attention norm": weights.FeedForwardNorm,
	}
	if spec.Architecture == "qwen3next" {
		required["SSM beta/alpha"] = weights.SSMBetaAlpha
		if weights.AttentionQKV.Shape.Dims[1] == uint64(spec.SSMInnerSize)+
			2*uint64(spec.SSMStateSize)*uint64(spec.SSMGroupCount) {
			required["attention gate"] = weights.AttentionGate
		}
	} else {
		required["attention gate"] = weights.AttentionGate
		required["SSM beta"] = weights.SSMBeta
		required["SSM alpha"] = weights.SSMAlpha
	}
	addQwen35FeedForwardRequirements(required, spec, weights)
	for name, item := range required {
		if item == nil {
			return Qwen35BlockResult{}, fmt.Errorf("Qwen3.5 recurrent block %s weight is nil", name)
		}
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return Qwen35BlockResult{}, errors.New("Qwen3.5 recurrent position count is invalid")
	}
	tokens := uint64(len(positions))
	stateWidth := uint64(spec.SSMStateSize)
	keyHeads := uint64(spec.SSMGroupCount)
	valueHeads := uint64(spec.SSMTimeStepRank)
	keyDimension := stateWidth * keyHeads
	valueDimension := uint64(spec.SSMInnerSize)
	convChannels := 2*keyDimension + valueDimension
	if !convState.Shape.Equal(tensor.MustShape(uint64(spec.SSMConvKernel-1), convChannels)) ||
		!ssmState.Shape.Equal(tensor.MustShape(stateWidth, stateWidth, valueHeads, 1)) {
		return Qwen35BlockResult{}, errors.New("Qwen3.5 recurrent cache shape is invalid")
	}

	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	qkvProjection := builder.MulMat(weights.AttentionQKV, normalized)
	qkvMixed := qkvProjection
	var z *tensor.Tensor
	if spec.Architecture == "qwen3next" && weights.AttentionGate == nil {
		valueHeadsPerGroup := valueHeads / keyHeads
		valueWidthPerGroup := stateWidth * valueHeadsPerGroup
		groupStride := 2*stateWidth + 2*valueWidthPerGroup
		query := builder.GroupSlice(qkvProjection, 0, stateWidth, keyHeads, groupStride)
		key := builder.GroupSlice(qkvProjection, stateWidth, stateWidth, keyHeads, groupStride)
		value := builder.GroupSlice(
			qkvProjection, 2*stateWidth, valueWidthPerGroup, keyHeads, groupStride,
		)
		z = builder.GroupSlice(
			qkvProjection, 2*stateWidth+valueWidthPerGroup,
			valueWidthPerGroup, keyHeads, groupStride,
		)
		qkvMixed = builder.Concat(
			builder.Concat(
				builder.Reshape(query, keyDimension, tokens),
				builder.Reshape(key, keyDimension, tokens),
				0,
			),
			builder.Reshape(value, valueDimension, tokens),
			0,
		)
		z = builder.Reshape(z, stateWidth, valueHeads, tokens, 1)
	} else {
		z = builder.MulMat(weights.AttentionGate, normalized)
	}
	var beta, alpha *tensor.Tensor
	if spec.Architecture == "qwen3next" {
		valueHeadsPerGroup := valueHeads / keyHeads
		betaAlpha := builder.MulMat(weights.SSMBetaAlpha, normalized)
		beta = builder.GroupSlice(
			betaAlpha, 0, valueHeadsPerGroup, keyHeads, 2*valueHeadsPerGroup,
		)
		alpha = builder.GroupSlice(
			betaAlpha, valueHeadsPerGroup, valueHeadsPerGroup,
			keyHeads, 2*valueHeadsPerGroup,
		)
		beta = builder.Reshape(builder.Sigmoid(beta), 1, valueHeads, tokens, 1)
		alpha = builder.Reshape(alpha, valueHeads, tokens)
	} else {
		beta = builder.Reshape(
			builder.Sigmoid(builder.MulMat(weights.SSMBeta, normalized)),
			1, valueHeads, tokens, 1,
		)
		alpha = builder.MulMat(weights.SSMAlpha, normalized)
	}
	gate := builder.Multiply(
		builder.Softplus(builder.Add(alpha, weights.SSMTimeStep)),
		weights.SSMA,
	)
	gate = builder.Reshape(gate, 1, valueHeads, tokens, 1)

	convInput := builder.Concat(convState, builder.Transpose2D(qkvMixed), 0)
	nextConvState := builder.GroupSlice(
		convInput,
		tokens,
		uint64(spec.SSMConvKernel-1),
		1,
		uint64(spec.SSMConvKernel-1),
	)
	nextConvState = builder.Reshape(
		nextConvState,
		uint64(spec.SSMConvKernel-1),
		convChannels,
	)
	convolved := builder.SiLU(builder.SSMConv(convInput, weights.SSMConv1D))
	query := builder.GroupSlice(convolved, 0, stateWidth, keyHeads, stateWidth)
	key := builder.GroupSlice(convolved, keyDimension, stateWidth, keyHeads, stateWidth)
	value := builder.GroupSlice(convolved, 2*keyDimension, stateWidth, valueHeads, stateWidth)
	query = builder.Reshape(builder.L2Norm(query, spec.RMSNormEpsilon), stateWidth, keyHeads, tokens, 1)
	key = builder.Reshape(builder.L2Norm(key, spec.RMSNormEpsilon), stateWidth, keyHeads, tokens, 1)
	value = builder.Reshape(value, stateWidth, valueHeads, tokens, 1)
	var packed *tensor.Tensor
	if spec.Architecture == "qwen3next" {
		packed = builder.GatedDeltaNetRepeatInterleave(query, key, value, gate, beta, ssmState)
	} else {
		packed = builder.GatedDeltaNet(query, key, value, gate, beta, ssmState)
	}
	attentionElements := stateWidth * valueHeads * tokens
	attention := builder.FlatSlice(
		packed,
		0,
		stateWidth,
		valueHeads,
		tokens,
		1,
	)
	nextSSMState := builder.FlatSlice(
		packed,
		attentionElements,
		stateWidth,
		stateWidth,
		valueHeads,
		1,
	)
	z = builder.Reshape(z, stateWidth, valueHeads, tokens, 1)
	attention = builder.Multiply(
		builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon),
		builder.SiLU(z),
	)
	attention = builder.Reshape(attention, valueDimension, tokens)
	attention = builder.MulMat(weights.SSMOutput, attention)
	residual := builder.Add(input, attention)
	output := buildQwen35FeedForward(builder, residual, spec, weights)
	if err := builder.Err(); err != nil {
		return Qwen35BlockResult{}, err
	}
	return Qwen35BlockResult{
		Output:    output,
		ConvState: nextConvState,
		SSMState:  nextSSMState,
		Recurrent: true,
	}, nil
}

func buildQwen35FeedForward(
	builder *tensor.Builder,
	residual *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) *tensor.Tensor {
	normalized := builder.WeightedRMSNorm(
		residual,
		weights.FeedForwardNorm,
		spec.RMSNormEpsilon,
	)
	var feedForward *tensor.Tensor
	if spec.Architecture == "qwen3next" || spec.Architecture == "qwen35moe" {
		if weights.FeedForwardGateUpExperts != nil {
			feedForward = builder.MoESoftmaxFusedGateUp(
				normalized, weights.FeedForwardRouter, weights.FeedForwardGateUpExperts,
				weights.FeedForwardDownExperts, nil, spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		} else {
			feedForward = builder.MoE(
				normalized, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
				weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
				spec.ExpertUsedCount, true, spec.ExpertWeightsScale,
			)
		}
		sharedRouter := builder.Reshape(
			weights.FeedForwardSharedRouter, uint64(spec.EmbeddingLength), 1,
		)
		sharedGate := builder.Sigmoid(builder.MulMat(sharedRouter, normalized))
		shared := builder.MulMat(
			weights.FeedForwardSharedDown,
			builder.SwiGLU(
				builder.MulMat(weights.FeedForwardSharedGate, normalized),
				builder.MulMat(weights.FeedForwardSharedUp, normalized),
			),
		)
		feedForward = builder.Add(feedForward, builder.Multiply(shared, sharedGate))
	} else {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	}
	return builder.Add(residual, feedForward)
}

func addQwen35FeedForwardRequirements(
	required map[string]*tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) {
	if spec.Architecture == "qwen3next" || spec.Architecture == "qwen35moe" {
		required["feed-forward router"] = weights.FeedForwardRouter
		required["feed-forward expert down"] = weights.FeedForwardDownExperts
		if weights.FeedForwardGateUpExperts != nil {
			required["feed-forward fused expert gate/up"] = weights.FeedForwardGateUpExperts
		} else {
			required["feed-forward expert gate"] = weights.FeedForwardGateExperts
			required["feed-forward expert up"] = weights.FeedForwardUpExperts
		}
		required["feed-forward shared router"] = weights.FeedForwardSharedRouter
		required["feed-forward shared gate"] = weights.FeedForwardSharedGate
		required["feed-forward shared up"] = weights.FeedForwardSharedUp
		required["feed-forward shared down"] = weights.FeedForwardSharedDown
		return
	}
	required["feed-forward gate"] = weights.FeedForwardGate
	required["feed-forward up"] = weights.FeedForwardUp
	required["feed-forward down"] = weights.FeedForwardDown
}
