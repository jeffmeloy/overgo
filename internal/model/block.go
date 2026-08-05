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

// ApplyNormalization: learned normalization stage.
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
	if spec.Profile().DenseWeights.RMSNormBias && bias != nil {
		return builder.Add(normalized, bias)
	}
	return normalized
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

// DenseBlockOptions: dense layer graph inputs.
type DenseBlockOptions struct {
	Context CachedBlockContext
	Spec    Spec
	Weights LayerGraphWeights
	Plan    *LayerPlan
}

// BuildDenseBlockWithOptions: typed dense layer construction.
func BuildDenseBlockWithOptions(options DenseBlockOptions) (DenseBlockResult, error) {
	if options.Context.MultiPositions != nil {
		switch options.Spec.Architecture {
		case "glm4", "glm4moe":
			if options.Spec.RopeSections[0] <= 0 || options.Spec.RopeSections[1] <= 0 {
				return DenseBlockResult{}, errors.New("GLM4 multi-axis positions require multimodal RoPE sections")
			}
		case "hunyuan-dense", "hunyuan_vl":
			if options.Spec.RopeSections[0] <= 0 || options.Spec.RopeSections[1] <= 0 {
				return DenseBlockResult{}, errors.New("Hunyuan multi-axis positions require multimodal RoPE sections")
			}
		case "paddleocr", "qwen2vl", "qwen3vl", "qwen3vlmoe":
		default:
			return DenseBlockResult{}, errors.New("dense block architecture does not support multi-axis positions")
		}
		options.Context.Positions = options.Context.MultiPositions[0]
	}
	return buildDenseBlockCachedForLayer(options)
}

// BuildDenseBlock: pre-normalized grouped-query transformer block.
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
	return BuildDenseBlockWithOptions(DenseBlockOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, Positions: positions,
			PastKey: pastKey, PastValue: pastValue,
		},
		Spec: spec, Weights: weights,
	})
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
	return BuildDenseBlockWithOptions(DenseBlockOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, Positions: positions,
			PastKey: pastKey, PastValue: pastValue, Layer: layerIndex,
		},
		Spec: spec, Weights: weights,
	})
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
	return BuildDenseBlockWithOptions(DenseBlockOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, MultiPositions: &multiPositions,
			PastKey: pastKey, PastValue: pastValue, Layer: layerIndex,
		},
		Spec: spec, Weights: weights,
	})
}

func buildDenseBlockCachedForLayer(options DenseBlockOptions) (DenseBlockResult, error) {
	context := options.Context
	builder := context.Builder
	input := context.Input
	spec := options.Spec
	weights := options.Weights
	positions := context.Positions
	multiPositions := context.MultiPositions
	pastKey := context.PastKey
	pastValue := context.PastValue
	layerIndex := context.Layer
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
	layerPlan := spec.PlanLayer(layerIndex, false)
	if options.Plan != nil {
		layerPlan = *options.Plan
	}
	normPlan := layerPlan.Normalization
	switch layerPlan.DenseGraph {
	case DenseGraphBERT:
		return buildBERTEncoderBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	case DenseGraphModernBERT:
		return buildModernBERTBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	case DenseGraphGemmaEmbedding:
		return buildGemmaEmbeddingBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	case DenseGraphTalkie:
		return buildTalkieBlock(builder, input, spec, weights, positions, pastKey, pastValue)
	case DenseGraphGemma4:
		return buildGemma4BlockCached(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	case DenseGraphGemma3n:
		return DenseBlockResult{}, errors.New("Gemma 3n block requires AltUp execution")
	case DenseGraphRWKV6Qwen2:
		return BuildRWKV6Qwen2BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	case DenseGraphRWKV6:
		return BuildRWKV6BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	case DenseGraphRWKV7:
		return BuildRWKV7BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	}
	isPostOnlyNorm := !normPlan.PreAttention
	if err := layerPlan.ExpertComposition.Validate(spec); err != nil {
		return DenseBlockResult{}, err
	}
	headCount := spec.LayerHeadCount(layerIndex)
	kvHeadCount := spec.LayerKVHeadCount(layerIndex)
	if layerPlan.DeciSparse {
		return buildDeciSparseBlockCached(
			builder, input, spec, weights, positions, pastKey, pastValue, layerIndex,
		)
	}
	if profile.FeedForward == FeedForwardXIELU &&
		(int(layerIndex) >= len(spec.XIELUAlphaN) || int(layerIndex) >= len(spec.XIELUAlphaP) ||
			int(layerIndex) >= len(spec.XIELUBeta) || int(layerIndex) >= len(spec.XIELUEpsilon)) {
		return DenseBlockResult{}, errors.New("Apertus xIELU parameters are missing for layer")
	}
	usesExperts := weights.FeedForwardRouter != nil
	if err := layerPlan.DenseWeights.Validate(spec, profile, weights, usesExperts); err != nil {
		return DenseBlockResult{}, err
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, fmt.Errorf("dense block has %d positions for %d tokens", len(positions), input.Shape.Dims[1])
	}
	if err := requireTensorPair(pastKey, pastValue, "dense block past key/value cache must both be present"); err != nil {
		return DenseBlockResult{}, err
	}
	if spec.NonCausalAttention && profile.Forward != ForwardDFlash && pastKey != nil {
		return DenseBlockResult{}, errors.New("non-causal dense block does not support a KV cache")
	}

	tokens := uint64(len(positions))
	normalized := input
	if !isPostOnlyNorm && !spec.SandwichNorm {
		normalized = ApplyNormalization(
			builder, input, weights.AttentionNorm, weights.AttentionNormBias, spec,
		)
	}
	feedForwardNormalized := normalized
	attentionGate := layerPlan.AttentionOutput.PrepareGate(builder, normalized, weights)
	if layerPlan.DenseWeights.validateFalconNorm && weights.AttentionNorm2 != nil {
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
	query, key, err := layerPlan.QKPreprocess.Apply(builder, query, key, spec, weights, qkProjection)
	if err != nil {
		return DenseBlockResult{}, err
	}
	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(headCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(kvHeadCount), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(kvHeadCount), tokens)

	query, key, err = layerPlan.QKPreprocess.Apply(builder, query, key, spec, weights, qkHeads)
	if err != nil {
		return DenseBlockResult{}, err
	}
	query, key = layerPlan.Rotary.Apply(
		builder, query, key, positions, multiPositions, weights.RopeFactors,
	)
	query, key, err = layerPlan.QKPreprocess.Apply(builder, query, key, spec, weights, qkPostRotary)
	if err != nil {
		return DenseBlockResult{}, err
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
	query, attentionScale = layerPlan.QueryScale.Apply(builder, query, spec, weights, attentionScale)
	attention := layerPlan.AttentionGraph.Build(
		builder, query, cacheKey, cacheValue, weights.AttentionSinks, nil,
		attentionScale, queryStart,
	)
	attention = layerPlan.AttentionOutput.ApplyGate(
		builder, attention, attentionGate, weights, headCount, tokens, attentionGateHeads,
	)
	attention = builder.Reshape(attention, uint64(headCount)*uint64(spec.ValueLength), tokens)
	attention = layerPlan.AttentionOutput.ApplyGate(
		builder, attention, attentionGate, weights, headCount, tokens, attentionGateFlat,
	)
	attention, err = layerPlan.AttentionOutput.ApplyProjection(builder, attention, spec, weights)
	if err != nil {
		return DenseBlockResult{}, err
	}
	residual := builder.Add(input, attention)
	normalized = layerPlan.ResidualStages.AfterAttention(builder, residual, normalized, spec, weights)
	if usesExperts && layerPlan.ExpertComposition.kind == expertArctic {
		feedForward, err := layerPlan.ExpertComposition.Build(
			builder, input, residual, normalized, spec, weights, layerPlan,
		)
		if err != nil {
			return DenseBlockResult{}, err
		}
		return DenseBlockResult{
			Output: builder.Add(residual, feedForward), Key: cacheKey, Value: cacheValue,
		}, nil
	}

	normalized = layerPlan.ResidualStages.FeedForwardInput(
		builder, input, residual, normalized, feedForwardNormalized, spec, weights,
	)
	if usesExperts {
		feedForward, err := layerPlan.ExpertComposition.Build(
			builder, input, residual, normalized, spec, weights, layerPlan,
		)
		if err != nil {
			return DenseBlockResult{}, err
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
	} else if profile.FeedForward == FeedForwardXIELU {
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
		if !layerPlan.DenseWeights.allowActivationScale {
			return DenseBlockResult{}, errors.New("feed-forward activation scale requires MPT")
		}
		activation = builder.Divide(activation, weights.FeedForwardActivationScale)
	}
	if layerPlan.DenseWeights.requireSubNorm {
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
	feedForward = layerPlan.ResidualStages.ApplyFeedForwardOutput(
		builder, feedForward, spec, weights, normPlan,
	)
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
	layerPlan := spec.PlanLayer(layerIndex, false)
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Gemma 4 block input shape is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "Gemma 4 cache pair is incomplete"); err != nil {
		return DenseBlockResult{}, err
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
	if err := requireBlockWeights("Gemma 4 block", required); err != nil {
		return DenseBlockResult{}, err
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
	query = layerPlan.Rotary.ApplyOne(builder, query, positions, nil, weights.RopeFactors)
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
		key = layerPlan.Rotary.ApplyOne(builder, key, positions, nil, weights.RopeFactors)
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
	attention := layerPlan.AttentionGraph.Build(
		builder, query, cacheKey, cacheValue, nil, weights.AttentionBlockIDs,
		spec.AttentionScale, queryStart,
	)
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
		feedForward = layerPlan.Experts.BuildLayer(builder, expertInput, routerInput, weights)
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
	if builder == nil || input == nil || spec.Profile().DenseGraph != DenseGraphGemma3n ||
		input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return Gemma3nAttentionResult{}, errors.New("Gemma 3n attention input is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "Gemma 3n cache pair is incomplete"); err != nil {
		return Gemma3nAttentionResult{}, err
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
	if err := requireBlockWeights("Gemma 3n", required); err != nil {
		return Gemma3nAttentionResult{}, err
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
	if builder == nil || residual == nil || activated == nil || spec.Profile().DenseGraph != DenseGraphGemma3n ||
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
	if !spec.Profile().Has(ArchitecturePerLayerEmbeddings) ||
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

func buildSharedSwiGLU(
	builder *tensor.Builder,
	input *tensor.Tensor,
	weights LayerGraphWeights,
) *tensor.Tensor {
	gate := builder.MulMat(weights.FeedForwardSharedGate, input)
	up := builder.MulMat(weights.FeedForwardSharedUp, input)
	return builder.MulMat(weights.FeedForwardSharedDown, builder.SwiGLU(gate, up))
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
	if err := requireTensorPair(pastKey, pastValue, "Deci cache must contain both sentinel tensors"); err != nil {
		return DenseBlockResult{}, err
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
	if err := requireBlockWeights("Deci feed-forward", map[string]*tensor.Tensor{
		"norm": weights.FeedForwardNorm,
		"gate": weights.FeedForwardGate,
		"up":   weights.FeedForwardUp,
		"down": weights.FeedForwardDown,
	}); err != nil {
		return DenseBlockResult{}, err
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
	profile := spec.Profile()
	if profile.AttentionGraph.QwenGDN == qwenGDNNone || !profile.Has(ArchitectureMultiAxisPositions) {
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
	if spec.Profile().AttentionGraph.QwenGDN == qwenGDNNone {
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
	if err := requireBlockWeights("Qwen3.5 attention block", required); err != nil {
		return DenseBlockResult{}, err
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Qwen3.5 attention position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "Qwen3.5 attention cache must contain both key and value"); err != nil {
		return DenseBlockResult{}, err
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
	if spec.Profile().AttentionGraph.QwenGDN == qwenGDNRepeatInterleave {
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
	qwenPolicy := spec.Profile().AttentionGraph.QwenGDN
	if qwenPolicy == qwenGDNRepeatInterleave {
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
	if err := requireBlockWeights("Qwen3.5 recurrent block", required); err != nil {
		return Qwen35BlockResult{}, err
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
	if qwenPolicy == qwenGDNRepeatInterleave && weights.AttentionGate == nil {
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
	if qwenPolicy == qwenGDNRepeatInterleave {
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
	if qwenPolicy == qwenGDNRepeatInterleave {
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
	if spec.expertCompositionPlan().kind == expertSharedGated {
		plan := spec.moeGraphPlan(0)
		plan.NormalizeTopKProb = true
		feedForward = plan.BuildLayer(builder, normalized, nil, weights)
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
	if spec.expertCompositionPlan().kind == expertSharedGated {
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
