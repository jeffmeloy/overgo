package model

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor"
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
	Output    *tensor.Tensor
	Key       *tensor.Tensor
	Value     *tensor.Tensor
	Auxiliary *tensor.Tensor
	States    CacheStates[*tensor.Tensor]
}

type Qwen35BlockResult struct {
	Output    *tensor.Tensor
	Key       *tensor.Tensor
	Value     *tensor.Tensor
	ConvState *tensor.Tensor
	SSMState  *tensor.Tensor
	Recurrent bool
}

// Qwen35BlockOptions: hybrid block graph inputs.
type Qwen35BlockOptions struct {
	Builder        *tensor.Builder
	Input          *tensor.Tensor
	Spec           Spec
	Weights        LayerGraphWeights
	Positions      []uint32
	MultiPositions *[4][]uint32
	Sequences      uint64
	Recurrent      bool
	PastKey        *tensor.Tensor
	PastValue      *tensor.Tensor
	ConvState      *tensor.Tensor
	SSMState       *tensor.Tensor
	CacheWrite     tensor.CacheWriteMode
}

// buildDenseBlockWithOptions: typed dense layer construction.
func buildDenseBlockWithOptions(options BlockDispatchOptions) (DenseBlockResult, error) {
	if options.Context.MultiPositions != nil {
		if !options.Spec.SupportsMultiAxisPositions() {
			return DenseBlockResult{}, errors.New("dense block architecture does not support multi-axis positions")
		}
		options.Context.Positions = options.Context.MultiPositions[0]
	}
	return buildDenseBlock(options)
}

func buildDenseBlock(options BlockDispatchOptions) (DenseBlockResult, error) {
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
	if options.Plan == nil {
		return DenseBlockResult{}, errors.New("compiled dense layer plan is required")
	}
	profile := spec.Profile()
	layerPlan := *options.Plan
	if layerPlan.Layer != layerIndex {
		return DenseBlockResult{}, errors.New("compiled dense layer plan index differs")
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
		return buildGemma4BlockCached(builder, input, spec, weights, positions, pastKey, pastValue, layerPlan)
	case DenseGraphGemma3n:
		return DenseBlockResult{}, errors.New("Gemma 3n block requires AltUp execution")
	case DenseGraphRWKV6Qwen2:
		return buildRWKV6Qwen2BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	case DenseGraphRWKV6:
		return buildRWKV6BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
	case DenseGraphRWKV7:
		return buildRWKV7BlockCached(builder, input, spec, weights, pastKey, pastValue, layerIndex)
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
	runtime := denseBlockRuntime{
		builder: builder, spec: spec, weights: weights, plan: layerPlan,
		profile: profile, layer: layerIndex, tokens: tokens,
	}
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
	query, key, value := runtime.projectAttention(normalized)
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
		queryStart = builder.CacheTokenOffset(uint32(pastKey.Shape.Dims[2]))
		cacheKey = builder.WriteCache(pastKey, key, 2, context.CacheWrite)
		cacheValue = builder.WriteCache(pastValue, value, 2, context.CacheWrite)
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
	feedForward, err := runtime.buildFeedForward(normalized, normPlan)
	if err != nil {
		return DenseBlockResult{}, err
	}
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}

type denseBlockRuntime struct {
	builder *tensor.Builder
	spec    Spec
	weights LayerGraphWeights
	plan    LayerPlan
	profile ArchitectureProfile
	layer   uint32
	tokens  uint64
}

func (r denseBlockRuntime) projectAttention(normalized *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	if r.weights.AttentionQKV != nil {
		shapes := r.spec.TensorShapes(r.layer)
		queryLength := shapes.QueryProjectionWidth()
		keyLength := shapes.KeyProjectionWidth()
		valueLength := shapes.ValueProjectionWidth()
		mixed := r.builder.MulMat(r.weights.AttentionQKV, normalized)
		if r.weights.AttentionQKVBias != nil {
			mixed = r.builder.Add(mixed, r.weights.AttentionQKVBias)
		}
		if r.spec.AttentionClamp > 0 {
			mixed = r.builder.Clamp(mixed, -r.spec.AttentionClamp, r.spec.AttentionClamp)
		}
		stride := queryLength + keyLength + valueLength
		return r.builder.Reshape(
				r.builder.GroupSlice(mixed, 0, queryLength, 1, stride), queryLength, r.tokens,
			), r.builder.Reshape(
				r.builder.GroupSlice(mixed, queryLength, keyLength, 1, stride), keyLength, r.tokens,
			), r.builder.Reshape(
				r.builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride), valueLength, r.tokens,
			)
	}
	query := r.builder.MulMat(r.weights.AttentionQ, normalized)
	key := r.builder.MulMat(r.weights.AttentionK, normalized)
	value := r.builder.MulMat(r.weights.AttentionV, normalized)
	for _, item := range []struct {
		tensor **tensor.Tensor
		scale  *tensor.Tensor
		bias   *tensor.Tensor
	}{
		{&query, r.weights.AttentionQScale, r.weights.AttentionQBias},
		{&key, r.weights.AttentionKScale, r.weights.AttentionKBias},
		{&value, r.weights.AttentionVScale, r.weights.AttentionVBias},
	} {
		if item.scale != nil {
			*item.tensor = r.builder.Multiply(*item.tensor, item.scale)
		}
		if item.bias != nil {
			*item.tensor = r.builder.Add(*item.tensor, item.bias)
		}
		if r.spec.AttentionClamp > 0 {
			*item.tensor = r.builder.Clamp(*item.tensor, -r.spec.AttentionClamp, r.spec.AttentionClamp)
		}
	}
	return query, key, value
}

func (r denseBlockRuntime) buildFeedForward(
	normalized *tensor.Tensor,
	normPlan NormalizationPlan,
) (*tensor.Tensor, error) {
	up := r.builder.MulMat(r.weights.FeedForwardUp, normalized)
	if r.weights.FeedForwardUpScale != nil {
		up = r.builder.Multiply(up, r.weights.FeedForwardUpScale)
	}
	if r.weights.FeedForwardUpBias != nil {
		up = r.builder.Add(up, r.weights.FeedForwardUpBias)
	}
	var activation *tensor.Tensor
	switch r.profile.FeedForward {
	case FeedForwardFusedGateUp:
		width := uint64(r.spec.FeedForwardLength)
		stride := 2 * width
		gate := r.builder.Reshape(r.builder.GroupSlice(up, 0, width, 1, stride), width, r.tokens)
		up = r.builder.Reshape(r.builder.GroupSlice(up, width, width, 1, stride), width, r.tokens)
		activation = r.builder.SwiGLU(gate, up)
	case FeedForwardXIELU:
		activation = r.builder.XIELU(
			up, r.spec.XIELUAlphaN[r.layer], r.spec.XIELUAlphaP[r.layer],
			r.spec.XIELUBeta[r.layer], r.spec.XIELUEpsilon[r.layer],
		)
	case FeedForwardGELU, FeedForwardSequentialGELU:
		activation = r.builder.GELU(up)
	case FeedForwardSquaredReLU:
		activation = r.builder.ReLUSquared(up)
	default:
		gate := r.builder.MulMat(r.weights.FeedForwardGate, normalized)
		if r.weights.FeedForwardGateScale != nil {
			gate = r.builder.Multiply(gate, r.weights.FeedForwardGateScale)
		}
		if r.weights.FeedForwardGateBias != nil {
			gate = r.builder.Add(gate, r.weights.FeedForwardGateBias)
		}
		activation = r.builder.SwiGLU(gate, up)
		if r.profile.Has(ArchitectureGemma) {
			activation = r.builder.GEGLU(gate, up)
		}
	}
	if r.weights.FeedForwardActivationScale != nil {
		if !r.plan.DenseWeights.allowActivationScale {
			return nil, errors.New("feed-forward activation scale requires MPT")
		}
		activation = r.builder.Divide(activation, r.weights.FeedForwardActivationScale)
	}
	if r.plan.DenseWeights.requireSubNorm {
		activation = r.builder.WeightedRMSNorm(
			activation, r.weights.FeedForwardSubNorm, r.spec.RMSNormEpsilon,
		)
	}
	feedForward := r.builder.MulMat(r.weights.FeedForwardDown, activation)
	if r.weights.FeedForwardDownScale != nil {
		feedForward = r.builder.Multiply(feedForward, r.weights.FeedForwardDownScale)
	}
	if r.weights.FeedForwardDownBias != nil {
		feedForward = r.builder.Add(feedForward, r.weights.FeedForwardDownBias)
	}
	return r.plan.ResidualStages.ApplyFeedForwardOutput(
		r.builder, feedForward, r.spec, r.weights, normPlan,
	), r.builder.Err()
}

func buildGemma4BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerPlan LayerPlan,
) (DenseBlockResult, error) {
	layerIndex := layerPlan.Layer
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Gemma 4 block input shape is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "Gemma 4 cache pair is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention query", weights.AttentionQ),
		requireGraphWeight("attention query norm", weights.AttentionQNorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention post norm", weights.AttentionPostNorm),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
		requireGraphWeight("feed-forward post norm", weights.FeedForwardPostNorm),
	}
	if spec.LayerHasKV(layerIndex) {
		required.add("attention key", weights.AttentionK)
		required.add("attention key norm", weights.AttentionKNorm)
	} else if pastKey == nil {
		return DenseBlockResult{}, errors.New("Gemma 4 shared-KV layer has no source cache")
	}
	usesExperts := weights.FeedForwardRouter != nil
	if usesExperts {
		required.add("expert router", weights.FeedForwardRouter)
		required.add("expert router scale", weights.FeedForwardRouterScale)
		required.add("expert down", weights.FeedForwardDownExperts)
		required.add("expert pre norm", weights.FeedForwardPreNorm2)
		required.add("dense expert post norm", weights.FeedForwardPostNorm1)
		required.add("routed expert post norm", weights.FeedForwardPostNorm2)
		if weights.FeedForwardGateUpExperts == nil {
			required.add("expert gate", weights.FeedForwardGateExperts)
			required.add("expert up", weights.FeedForwardUpExperts)
		}
	}
	if spec.EmbeddingPerLayer > 0 {
		required.add("per-layer input", weights.PerLayerInput)
		required.add("per-layer input gate", weights.PerLayerInputGate)
		required.add("per-layer projection", weights.PerLayerProjection)
		required.add("per-layer post norm", weights.PerLayerPostNorm)
	}
	if err := required.validate("Gemma 4 block"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := input.Shape.Dims[1]
	shapes := spec.TensorShapes(layerIndex)
	headCount, kvHeadCount := shapes.QueryHeads, shapes.KVHeads
	keyLength, valueLength := shapes.Key, shapes.Value
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
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention query", weights.AttentionQ),
		requireGraphWeight("attention query norm", weights.AttentionQNorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention post norm", weights.AttentionPostNorm),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("Laurel left projection", weights.LaurelLeft),
		requireGraphWeight("Laurel right projection", weights.LaurelRight),
		requireGraphWeight("Laurel post norm", weights.LaurelPostNorm),
	}
	if spec.LayerHasKV(layerIndex) {
		required.add("attention key", weights.AttentionK)
		required.add("attention value", weights.AttentionV)
		required.add("attention key norm", weights.AttentionKNorm)
	} else if pastKey == nil {
		return Gemma3nAttentionResult{}, errors.New("Gemma 3n shared-KV layer has no source cache")
	}
	if err := required.validate("Gemma 3n"); err != nil {
		return Gemma3nAttentionResult{}, err
	}
	tokens := input.Shape.Dims[1]
	shapes := spec.TensorShapes(layerIndex)
	headCount, kvHeadCount := shapes.QueryHeads, shapes.KVHeads
	keyLength, valueLength := shapes.Key, shapes.Value
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
	cacheKey, cacheValue, err := buildSentinelCache(builder, input, pastKey, pastValue)
	if err != nil {
		return DenseBlockResult{}, fmt.Errorf("Deci: %w", err)
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
	if err := (graphWeights{
		requireGraphWeight("norm", weights.FeedForwardNorm),
		requireGraphWeight("gate", weights.FeedForwardGate),
		requireGraphWeight("up", weights.FeedForwardUp),
		requireGraphWeight("down", weights.FeedForwardDown),
	}).validate("Deci feed-forward"); err != nil {
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

// BuildQwen35BlockWithOptions: typed hybrid block construction.
func BuildQwen35BlockWithOptions(options Qwen35BlockOptions) (Qwen35BlockResult, error) {
	if options.Sequences == 0 {
		return Qwen35BlockResult{}, errors.New("Qwen hybrid sequence count is zero")
	}
	plan := options.Spec.PlanLayer(0, options.Recurrent)
	pastKey, pastValue := options.PastKey, options.PastValue
	if options.Recurrent {
		pastKey, pastValue = options.ConvState, options.SSMState
	}
	result, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: options.Spec, Weights: options.Weights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: options.Builder, Input: options.Input, Positions: options.Positions,
			MultiPositions: options.MultiPositions, PastKey: pastKey, PastValue: pastValue,
			Recurrent: options.Recurrent, CacheWrite: options.CacheWrite,
			Sequences: options.Sequences,
		},
	})
	if err != nil {
		return Qwen35BlockResult{}, err
	}
	qwen := Qwen35BlockResult{Output: result.Output, Key: result.Key, Value: result.Value}
	if options.Recurrent {
		qwen.Key, qwen.Value = nil, nil
		qwen.ConvState, qwen.SSMState, qwen.Recurrent = result.Key, result.Value, true
	}
	return qwen, nil
}

func buildQwen35AttentionMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	multiPositions *[4][]uint32,
	sequences uint64,
	pastKey, pastValue *tensor.Tensor,
	cacheWrite tensor.CacheWriteMode,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil {
		return DenseBlockResult{}, errors.New("Qwen3.5 attention mix input is nil")
	}
	required := graphWeights{
		requireGraphWeight("attention Q/gate", weights.AttentionQ),
		requireGraphWeight("attention K", weights.AttentionK),
		requireGraphWeight("attention V", weights.AttentionV),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention Q norm", weights.AttentionQNorm),
		requireGraphWeight("attention K norm", weights.AttentionKNorm),
	}
	if err := required.validate("Qwen3.5 attention mix"); err != nil {
		return DenseBlockResult{}, err
	}
	if len(positions) == 0 || sequences == 0 ||
		uint64(len(positions)) > math.MaxUint64/sequences ||
		uint64(len(positions))*sequences != normalized.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Qwen3.5 attention position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "Qwen3.5 attention cache must contain both key and value"); err != nil {
		return DenseBlockResult{}, err
	}

	tokens := uint64(len(positions))
	headWidth := uint64(spec.KeyLength)
	heads := uint64(spec.HeadCount)
	queryAndGate := builder.MulMat(weights.AttentionQ, normalized)
	query := builder.GroupSlice(queryAndGate, 0, headWidth, heads, 2*headWidth)
	gate := builder.GroupSlice(queryAndGate, headWidth, headWidth, heads, 2*headWidth)
	if sequences > 1 {
		query = builder.Reshape(query, headWidth, heads, tokens, sequences)
		gate = builder.Reshape(gate, headWidth, heads, tokens, sequences)
	}
	keyProjection := builder.MulMat(weights.AttentionK, normalized)
	valueProjection := builder.MulMat(weights.AttentionV, normalized)
	var key, value *tensor.Tensor
	if sequences == 1 {
		key = builder.Reshape(keyProjection, headWidth, uint64(spec.HeadCountKV), tokens)
		value = builder.Reshape(valueProjection, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	} else {
		key = builder.Reshape(keyProjection, headWidth, uint64(spec.HeadCountKV), tokens, sequences)
		value = builder.Reshape(
			valueProjection, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens, sequences,
		)
	}
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
		queryStart = builder.CacheTokenOffset(uint32(pastKey.Shape.Dims[2]))
		cacheKey = builder.WriteCache(pastKey, key, 2, cacheWrite)
		cacheValue = builder.WriteCache(pastValue, value, 2, cacheWrite)
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
		tokens*sequences,
	)
	gate = builder.Reshape(gate, uint64(spec.HeadCount)*headWidth, tokens*sequences)
	attention = builder.Multiply(attention, builder.Sigmoid(gate))
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: cacheKey, Value: cacheValue}, nil
}

func buildQwen35RecurrentMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	sequences uint64,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("Qwen3.5 recurrent mix input/state is nil")
	}
	required := graphWeights{
		requireGraphWeight("QKV", weights.AttentionQKV),
		requireGraphWeight("SSM convolution", weights.SSMConv1D),
		requireGraphWeight("SSM time-step bias", weights.SSMTimeStep),
		requireGraphWeight("SSM A", weights.SSMA),
		requireGraphWeight("SSM norm", weights.SSMNorm),
		requireGraphWeight("SSM output", weights.SSMOutput),
	}
	qwenPolicy := spec.Profile().AttentionGraph.QwenGDN
	if qwenPolicy == qwenGDNRepeatInterleave {
		required.add("SSM beta/alpha", weights.SSMBetaAlpha)
		if weights.AttentionQKV.Shape.Dims[1] == uint64(spec.SSMInnerSize)+
			2*uint64(spec.SSMStateSize)*uint64(spec.SSMGroupCount) {
			required.add("attention gate", weights.AttentionGate)
		}
	} else {
		required.add("attention gate", weights.AttentionGate)
		required.add("SSM beta", weights.SSMBeta)
		required.add("SSM alpha", weights.SSMAlpha)
	}
	if err := required.validate("Qwen3.5 recurrent mix"); err != nil {
		return DenseBlockResult{}, err
	}
	if len(positions) == 0 || sequences == 0 ||
		uint64(len(positions)) > math.MaxUint64/sequences ||
		uint64(len(positions))*sequences != normalized.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Qwen3.5 recurrent position count is invalid")
	}
	tokens := uint64(len(positions))
	stateWidth := uint64(spec.SSMStateSize)
	keyHeads := uint64(spec.SSMGroupCount)
	valueHeads := uint64(spec.SSMTimeStepRank)
	keyDimension := stateWidth * keyHeads
	valueDimension := uint64(spec.SSMInnerSize)
	convChannels := 2*keyDimension + valueDimension
	wantConvState := tensor.MustShape(uint64(spec.SSMConvKernel-1), convChannels)
	if sequences > 1 {
		wantConvState = tensor.MustShape(uint64(spec.SSMConvKernel-1), convChannels, sequences)
	}
	if !convState.Shape.Equal(wantConvState) ||
		!ssmState.Shape.Equal(tensor.MustShape(stateWidth, stateWidth, valueHeads, sequences)) {
		return DenseBlockResult{}, errors.New("Qwen3.5 recurrent cache shape is invalid")
	}

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
		z = builder.Reshape(z, stateWidth, valueHeads, tokens, sequences)
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
		beta = builder.Reshape(builder.Sigmoid(beta), 1, valueHeads, tokens, sequences)
		alpha = builder.Reshape(alpha, valueHeads, tokens, sequences)
	} else {
		beta = builder.Reshape(
			builder.Sigmoid(builder.MulMat(weights.SSMBeta, normalized)),
			1, valueHeads, tokens, sequences,
		)
		alpha = builder.MulMat(weights.SSMAlpha, normalized)
		if sequences > 1 {
			alpha = builder.Reshape(alpha, valueHeads, tokens, sequences)
		}
	}
	gate := builder.Multiply(
		builder.Softplus(builder.Add(alpha, weights.SSMTimeStep)),
		weights.SSMA,
	)
	gate = builder.Reshape(gate, 1, valueHeads, tokens, sequences)

	var qkvTime *tensor.Tensor
	if sequences > 1 && tokens == 1 {
		qkvTime = builder.Reshape(qkvMixed, tokens, convChannels, sequences)
	} else {
		qkvTime = builder.Transpose2D(qkvMixed)
		if sequences > 1 {
			qkvTime = builder.Reshape(qkvTime, tokens, convChannels, sequences)
		}
	}
	convInput := builder.Concat(convState, qkvTime, 0)
	nextConvState := builder.GroupSlice(
		convInput,
		tokens,
		uint64(spec.SSMConvKernel-1),
		1,
		uint64(spec.SSMConvKernel-1),
	)
	if sequences == 1 {
		nextConvState = builder.Reshape(
			nextConvState, uint64(spec.SSMConvKernel-1), convChannels,
		)
	} else {
		nextConvState = builder.Reshape(
			nextConvState, uint64(spec.SSMConvKernel-1), convChannels, sequences,
		)
	}
	convolved := builder.SiLU(builder.SSMConv(convInput, weights.SSMConv1D))
	query := builder.GroupSlice(convolved, 0, stateWidth, keyHeads, stateWidth)
	key := builder.GroupSlice(convolved, keyDimension, stateWidth, keyHeads, stateWidth)
	value := builder.GroupSlice(convolved, 2*keyDimension, stateWidth, valueHeads, stateWidth)
	query = builder.Reshape(builder.L2Norm(query, spec.RMSNormEpsilon), stateWidth, keyHeads, tokens, sequences)
	key = builder.Reshape(builder.L2Norm(key, spec.RMSNormEpsilon), stateWidth, keyHeads, tokens, sequences)
	value = builder.Reshape(value, stateWidth, valueHeads, tokens, sequences)
	var packed *tensor.Tensor
	if qwenPolicy == qwenGDNRepeatInterleave {
		packed = builder.GatedDeltaNetRepeatInterleave(query, key, value, gate, beta, ssmState)
	} else {
		packed = builder.GatedDeltaNet(query, key, value, gate, beta, ssmState)
	}
	attentionElements := stateWidth * valueHeads * tokens * sequences
	attention := builder.FlatSlice(
		packed,
		0,
		stateWidth,
		valueHeads,
		tokens,
		sequences,
	)
	nextSSMState := builder.FlatSlice(
		packed,
		attentionElements,
		stateWidth,
		stateWidth,
		valueHeads,
		sequences,
	)
	z = builder.Reshape(z, stateWidth, valueHeads, tokens, sequences)
	attention = builder.Multiply(
		builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon),
		builder.SiLU(z),
	)
	attention = builder.Reshape(attention, valueDimension, tokens*sequences)
	attention = builder.MulMat(weights.SSMOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextConvState, Value: nextSSMState}, nil
}

func buildQwen35FeedForwardMix(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	weights LayerGraphWeights,
	experts MoEGraphPlan,
	composition ExpertCompositionPlan,
) (*tensor.Tensor, error) {
	required := graphWeights{}
	addQwen35FeedForwardRequirements(&required, composition, weights)
	if err := required.validate("Qwen3.5 feed-forward mix"); err != nil {
		return nil, err
	}
	var feedForward *tensor.Tensor
	if composition.kind == expertSharedGated {
		feedForward = experts.BuildLayer(builder, normalized, nil, weights)
		sharedRouter := builder.Reshape(
			weights.FeedForwardSharedRouter, normalized.Shape.Dims[0], 1,
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
	return feedForward, builder.Err()
}

func addQwen35FeedForwardRequirements(
	required *graphWeights,
	composition ExpertCompositionPlan,
	weights LayerGraphWeights,
) {
	if composition.kind == expertSharedGated {
		required.add("feed-forward router", weights.FeedForwardRouter)
		required.add("feed-forward expert down", weights.FeedForwardDownExperts)
		if weights.FeedForwardGateUpExperts != nil {
			required.add("feed-forward fused expert gate/up", weights.FeedForwardGateUpExperts)
		} else {
			required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		}
		required.add("feed-forward shared router", weights.FeedForwardSharedRouter)
		required.add("feed-forward shared gate", weights.FeedForwardSharedGate)
		required.add("feed-forward shared up", weights.FeedForwardSharedUp)
		required.add("feed-forward shared down", weights.FeedForwardSharedDown)
		return
	}
	required.add("feed-forward gate", weights.FeedForwardGate)
	required.add("feed-forward up", weights.FeedForwardUp)
	required.add("feed-forward down", weights.FeedForwardDown)
}
