package model

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
)

// LayerGraphWeights are the graph inputs for one dense Llama/Qwen3 block.
type LayerGraphWeights struct {
	AttentionNorm         *tensor.Tensor
	AttentionNormBias     *tensor.Tensor
	AttentionNorm2        *tensor.Tensor
	AttentionNorm2Bias    *tensor.Tensor
	AttentionQ            *tensor.Tensor
	AttentionK            *tensor.Tensor
	AttentionV            *tensor.Tensor
	AttentionOutput       *tensor.Tensor
	AttentionQScale       *tensor.Tensor
	AttentionKScale       *tensor.Tensor
	AttentionVScale       *tensor.Tensor
	AttentionOutputScale  *tensor.Tensor
	AttentionSubNorm      *tensor.Tensor
	AttentionQBias        *tensor.Tensor
	AttentionKBias        *tensor.Tensor
	AttentionVBias        *tensor.Tensor
	AttentionOutputBias   *tensor.Tensor
	AttentionQNorm        *tensor.Tensor
	AttentionKNorm        *tensor.Tensor
	AttentionPostNorm     *tensor.Tensor
	AttentionRelativeBias *tensor.Tensor
	RopeFactors           *tensor.Tensor
	FeedForwardNorm       *tensor.Tensor
	FeedForwardNormBias   *tensor.Tensor
	FeedForwardGate       *tensor.Tensor
	FeedForwardUp         *tensor.Tensor
	FeedForwardDown       *tensor.Tensor
	FeedForwardGateScale  *tensor.Tensor
	FeedForwardUpScale    *tensor.Tensor
	FeedForwardDownScale  *tensor.Tensor
	FeedForwardSubNorm    *tensor.Tensor
	FeedForwardGateBias   *tensor.Tensor
	FeedForwardUpBias     *tensor.Tensor
	FeedForwardDownBias   *tensor.Tensor
	FeedForwardPostNorm   *tensor.Tensor

	AttentionQKV     *tensor.Tensor
	AttentionQKVBias *tensor.Tensor
	AttentionGate    *tensor.Tensor
	SSMConv1D        *tensor.Tensor
	SSMTimeStep      *tensor.Tensor
	SSMA             *tensor.Tensor
	SSMBeta          *tensor.Tensor
	SSMAlpha         *tensor.Tensor
	SSMNorm          *tensor.Tensor
	SSMOutput        *tensor.Tensor
}

// ApplyNormalization applies the architecture's learned pre/post
// normalization. Dense LayerNorm architectures require an affine bias while
// RMSNorm architectures intentionally ignore it.
func ApplyNormalization(
	builder *tensor.Builder,
	input, weight, bias *tensor.Tensor,
	spec Spec,
) *tensor.Tensor {
	if spec.UsesUnweightedLayerNorm() {
		return builder.LayerNorm(input, spec.LayerNormEpsilon)
	}
	if spec.UsesWeightOnlyLayerNorm() {
		return builder.Multiply(builder.LayerNorm(input, spec.LayerNormEpsilon), weight)
	}
	if spec.UsesLayerNorm() {
		return builder.AffineLayerNorm(input, weight, bias, spec.LayerNormEpsilon)
	}
	return builder.WeightedRMSNorm(input, weight, spec.RMSNormEpsilon)
}

// BuildT5EncoderBlock constructs one full, bidirectional T5 encoder block.
func BuildT5EncoderBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if builder == nil || input == nil {
		return nil, errors.New("T5 encoder block input is nil")
	}
	if spec.Architecture != "t5encoder" {
		return nil, errors.New("T5 encoder block requires t5encoder architecture")
	}
	required := map[string]*tensor.Tensor{
		"attention norm":          weights.AttentionNorm,
		"attention Q":             weights.AttentionQ,
		"attention K":             weights.AttentionK,
		"attention V":             weights.AttentionV,
		"attention output":        weights.AttentionOutput,
		"attention relative bias": weights.AttentionRelativeBias,
		"feed-forward norm":       weights.FeedForwardNorm,
		"feed-forward gate":       weights.FeedForwardGate,
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
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

type DenseBlockResult struct {
	Output *tensor.Tensor
	Key    *tensor.Tensor
	Value  *tensor.Tensor
}

type Qwen35BlockResult struct {
	Output    *tensor.Tensor
	Key       *tensor.Tensor
	Value     *tensor.Tensor
	ConvState *tensor.Tensor
	SSMState  *tensor.Tensor
	Recurrent bool
}

// BuildDenseBlock constructs one pre-normalized grouped-query transformer
// block. It covers the initial Llama layout and Qwen3's per-head Q/K norms.
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

// BuildDenseBlockCached additionally accepts and returns the layer's rank-3
// RoPE-key/value cache. Past cache tensors must either both be nil or both be
// present.
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
	if builder == nil {
		return DenseBlockResult{}, errors.New("dense block builder is nil")
	}
	if input == nil {
		return DenseBlockResult{}, errors.New("dense block input is nil")
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	isOLMo2 := spec.Architecture == "olmo2"
	isCommandRQKNorm := spec.Architecture == "command-r" && spec.BlockCount >= 64
	if spec.Architecture == "apertus" &&
		(int(layerIndex) >= len(spec.XIELUAlphaN) || int(layerIndex) >= len(spec.XIELUAlphaP) ||
			int(layerIndex) >= len(spec.XIELUBeta) || int(layerIndex) >= len(spec.XIELUEpsilon)) {
		return DenseBlockResult{}, errors.New("Apertus xIELU parameters are missing for layer")
	}
	required := map[string]*tensor.Tensor{
		"attention output":  weights.AttentionOutput,
		"feed-forward up":   weights.FeedForwardUp,
		"feed-forward down": weights.FeedForwardDown,
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
	if !usesGateFreeFFN(spec.Architecture) {
		required["feed-forward gate"] = weights.FeedForwardGate
	} else if usesSequentialGELU(spec.Architecture) {
		required["attention output bias"] = weights.AttentionOutputBias
		required["feed-forward up bias"] = weights.FeedForwardUpBias
		required["feed-forward down bias"] = weights.FeedForwardDownBias
	}
	if isOLMo2 {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
		required["attention post norm"] = weights.AttentionPostNorm
		required["feed-forward post norm"] = weights.FeedForwardPostNorm
	} else if !spec.UsesUnweightedLayerNorm() {
		required["attention norm"] = weights.AttentionNorm
		if !usesParallelResidual(spec.Architecture) {
			if spec.Architecture != "stablelm" {
				required["feed-forward norm"] = weights.FeedForwardNorm
			}
		}
		if spec.UsesLayerNorm() {
			required["attention norm bias"] = weights.AttentionNormBias
			if !usesParallelResidual(spec.Architecture) && spec.Architecture != "stablelm" {
				required["feed-forward norm bias"] = weights.FeedForwardNormBias
			}
		}
	}
	if isCommandRQKNorm {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if spec.Architecture == "stablelm" &&
		(weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
		return DenseBlockResult{}, errors.New("StableLM Q/K norm weights must both be present or absent")
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

	tokens := uint64(len(positions))
	normalized := input
	if !isOLMo2 {
		normalized = ApplyNormalization(
			builder, input, weights.AttentionNorm, weights.AttentionNormBias, spec,
		)
	}
	feedForwardNormalized := normalized
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
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
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
	}
	if isOLMo2 {
		query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	}

	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)

	if spec.Architecture == "apertus" || spec.Architecture == "qwen3" || spec.Architecture == "gemma3" {
		if weights.AttentionQNorm == nil || weights.AttentionKNorm == nil {
			return DenseBlockResult{}, errors.New("dense block architecture requires Q/K norm weights")
		}
		query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	}
	if isCommandRQKNorm {
		query = ApplyNormalization(builder, query, weights.AttentionQNorm, nil, spec)
		key = ApplyNormalization(builder, key, weights.AttentionKNorm, nil, spec)
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
		rotaryDimensions = spec.RopeDimensionCount
	}
	if !spec.UsesRoPE(layerIndex) {
		// Some dense architectures intentionally leave periodic layers
		// position-independent.
	} else if usesNormalRoPE(spec.Architecture) {
		frequencyBase := spec.RopeFrequencyBase
		frequencyScale := float32(1)
		if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		if spec.Architecture == "cohere2" && spec.IsSlidingLayer(layerIndex) {
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
	} else if isGemmaArchitecture(spec.Architecture) {
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
	}
	if supportsLongRoPE(spec.Architecture) && spec.RopeAttentionFactor != 1 {
		query = builder.Scale(query, spec.RopeAttentionFactor)
		key = builder.Scale(key, spec.RopeAttentionFactor)
	}
	if spec.Architecture == "maincoder" {
		if weights.AttentionQNorm == nil || weights.AttentionKNorm == nil {
			return DenseBlockResult{}, errors.New("dense block architecture requires Q/K norm weights")
		}
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
	if spec.Architecture == "phi2" || spec.Architecture == "phi3" {
		// Phi decoders scale the rotated query before the dot product to
		// preserve upstream precision behavior.
		query = builder.Scale(query, attentionScale)
		attentionScale = 1
	}
	if isGemmaArchitecture(spec.Architecture) {
		// Gemma scales Q before the attention dot product, rather than scaling
		// the accumulated score. Preserve that ordering for quantized parity.
		if spec.Architecture == "gemma2" && spec.BlockCount == 46 {
			attentionScale = float32(1 / math.Sqrt(
				float64(spec.EmbeddingLength)/float64(spec.HeadCount),
			))
		}
		query = builder.Scale(query, attentionScale)
		attentionScale = 1
	}
	var attention *tensor.Tensor
	if spec.IsSlidingLayer(layerIndex) {
		if spec.AttentionSoftcap > 0 {
			attention = builder.AttentionWindowSoftcappedWithOffset(
				query, cacheKey, cacheValue, attentionScale, spec.AttentionSoftcap,
				true, queryStart, spec.SlidingWindow,
			)
		} else {
			attention = builder.AttentionWindowWithOffset(
				query, cacheKey, cacheValue, attentionScale, true, queryStart, spec.SlidingWindow,
			)
		}
	} else {
		if spec.AttentionSoftcap > 0 {
			attention = builder.AttentionSoftcappedWithOffset(
				query, cacheKey, cacheValue, attentionScale, spec.AttentionSoftcap,
				true, queryStart,
			)
		} else {
			attention = builder.AttentionWithOffset(
				query, cacheKey, cacheValue, attentionScale, true, queryStart,
			)
		}
	}
	attention = builder.Reshape(
		attention,
		uint64(spec.HeadCount)*uint64(spec.ValueLength),
		tokens,
	)
	if spec.Architecture == "bitnet" {
		attention = builder.WeightedRMSNorm(
			attention, weights.AttentionSubNorm, spec.RMSNormEpsilon,
		)
	}
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputScale != nil {
		attention = builder.Multiply(attention, weights.AttentionOutputScale)
	}
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	if hasPostNorm(spec.Architecture) || isOLMo2 {
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

	parallelResidual := usesParallelResidual(spec.Architecture) ||
		(spec.Architecture == "gptneox" && spec.ParallelResidual) ||
		(spec.Architecture == "stablelm" && weights.FeedForwardNorm == nil)
	if spec.Architecture == "falcon" {
		normalized = feedForwardNormalized
	} else if spec.Architecture == "gptneox" && spec.ParallelResidual {
		// GPT-NeoX parallel blocks use a distinct FFN LayerNorm over the
		// original residual input rather than sharing the attention norm.
		normalized = ApplyNormalization(
			builder, input, weights.FeedForwardNorm, weights.FeedForwardNormBias, spec,
		)
	} else if !parallelResidual {
		normalized = residual
		if !isOLMo2 {
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
	// Cohere decoders leave normalized pointing at the block input so attention
	// and FFN run in parallel before both branches are added to the residual.
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	if weights.FeedForwardUpScale != nil {
		up = builder.Multiply(up, weights.FeedForwardUpScale)
	}
	if weights.FeedForwardUpBias != nil {
		up = builder.Add(up, weights.FeedForwardUpBias)
	}
	var activation *tensor.Tensor
	if usesFusedGateUp(spec.Architecture) {
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
	} else if usesGELU(spec.Architecture) {
		activation = builder.GELU(up)
	} else if usesSquaredReLU(spec.Architecture) {
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
		if isGemmaArchitecture(spec.Architecture) {
			activation = builder.GEGLU(gate, up)
		}
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
	if hasPostNorm(spec.Architecture) || isOLMo2 {
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

// BuildQwen35BlockCached constructs either a gated full-attention block or a
// fused gated-delta-net recurrent block, following the layer cadence recorded
// in the weight catalog.
func BuildQwen35BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	recurrent bool,
	pastKey, pastValue, convState, ssmState *tensor.Tensor,
) (Qwen35BlockResult, error) {
	if spec.Architecture != "qwen35" {
		return Qwen35BlockResult{}, errors.New("Qwen3.5 block requires qwen35 architecture")
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
		"feed-forward gate":   weights.FeedForwardGate,
		"feed-forward up":     weights.FeedForwardUp,
		"feed-forward down":   weights.FeedForwardDown,
	}
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
	var multiPositions [4][]uint32
	for axis := range multiPositions {
		multiPositions[axis] = positions
	}
	query = builder.RoPEMulti(
		query,
		multiPositions,
		spec.RopeSections,
		spec.RopeDimensionCount,
		spec.RopeFrequencyBase,
	)
	key = builder.RoPEMulti(
		key,
		multiPositions,
		spec.RopeSections,
		spec.RopeDimensionCount,
		spec.RopeFrequencyBase,
	)

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
	attention := builder.AttentionWithOffset(
		query,
		cacheKey,
		cacheValue,
		float32(1/math.Sqrt(float64(spec.KeyLength))),
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
		"attention gate":      weights.AttentionGate,
		"SSM convolution":     weights.SSMConv1D,
		"SSM time-step bias":  weights.SSMTimeStep,
		"SSM A":               weights.SSMA,
		"SSM beta":            weights.SSMBeta,
		"SSM alpha":           weights.SSMAlpha,
		"SSM norm":            weights.SSMNorm,
		"SSM output":          weights.SSMOutput,
		"post-attention norm": weights.FeedForwardNorm,
		"feed-forward gate":   weights.FeedForwardGate,
		"feed-forward up":     weights.FeedForwardUp,
		"feed-forward down":   weights.FeedForwardDown,
	}
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
	qkvMixed := builder.MulMat(weights.AttentionQKV, normalized)
	z := builder.MulMat(weights.AttentionGate, normalized)
	beta := builder.Reshape(
		builder.Sigmoid(builder.MulMat(weights.SSMBeta, normalized)),
		1,
		valueHeads,
		tokens,
		1,
	)
	alpha := builder.MulMat(weights.SSMAlpha, normalized)
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
	packed := builder.GatedDeltaNet(query, key, value, gate, beta, ssmState)
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
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	return builder.Add(residual, feedForward)
}
