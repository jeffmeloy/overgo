package model

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
)

// LayerGraphWeights: graph inputs for one dense Llama/Qwen3 block
type LayerGraphWeights struct {
	AttentionNorm            *tensor.Tensor
	AttentionNormBias        *tensor.Tensor
	AttentionNorm2           *tensor.Tensor
	AttentionNorm2Bias       *tensor.Tensor
	AttentionQ               *tensor.Tensor
	AttentionQB              *tensor.Tensor
	AttentionK               *tensor.Tensor
	AttentionV               *tensor.Tensor
	AttentionOutput          *tensor.Tensor
	AttentionQScale          *tensor.Tensor
	AttentionKScale          *tensor.Tensor
	AttentionVScale          *tensor.Tensor
	AttentionOutputScale     *tensor.Tensor
	AttentionSubNorm         *tensor.Tensor
	AttentionQBias           *tensor.Tensor
	AttentionKBias           *tensor.Tensor
	AttentionVBias           *tensor.Tensor
	AttentionOutputBias      *tensor.Tensor
	AttentionQNorm           *tensor.Tensor
	AttentionKNorm           *tensor.Tensor
	AttentionQNormBias       *tensor.Tensor
	AttentionKNormBias       *tensor.Tensor
	AttentionPostNorm        *tensor.Tensor
	AttentionPostNormBias    *tensor.Tensor
	AttentionRelativeBias    *tensor.Tensor
	AttentionOutputGate      *tensor.Tensor
	RopeFactors              *tensor.Tensor
	FeedForwardNorm          *tensor.Tensor
	FeedForwardNormBias      *tensor.Tensor
	FeedForwardExpertNorm    *tensor.Tensor
	FeedForwardGate          *tensor.Tensor
	FeedForwardUp            *tensor.Tensor
	FeedForwardDown          *tensor.Tensor
	FeedForwardGateScale     *tensor.Tensor
	FeedForwardUpScale       *tensor.Tensor
	FeedForwardDownScale     *tensor.Tensor
	FeedForwardSubNorm       *tensor.Tensor
	FeedForwardGateBias      *tensor.Tensor
	FeedForwardUpBias        *tensor.Tensor
	FeedForwardDownBias      *tensor.Tensor
	FeedForwardPostNorm      *tensor.Tensor
	FeedForwardPostNormBias  *tensor.Tensor
	FeedForwardRouter        *tensor.Tensor
	FeedForwardGateUpExperts *tensor.Tensor
	FeedForwardGateExperts   *tensor.Tensor
	FeedForwardUpExperts     *tensor.Tensor
	FeedForwardDownExperts   *tensor.Tensor
	FeedForwardExpertBias    *tensor.Tensor
	FeedForwardSharedGate    *tensor.Tensor
	FeedForwardSharedUp      *tensor.Tensor
	FeedForwardSharedDown    *tensor.Tensor
	FeedForwardSharedRouter  *tensor.Tensor
	ShortConvKernel          *tensor.Tensor
	ShortConvInput           *tensor.Tensor
	ShortConvOutput          *tensor.Tensor
	AttentionKVAMQA          *tensor.Tensor
	AttentionKVANorm         *tensor.Tensor
	AttentionKVB             *tensor.Tensor

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

// ApplyNormalization: applies architecture's learned pre/post
// normalization; Affine LayerNorm architectures and PhiMoE's affine RMSNorm
// require learned bias
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
		if bias == nil {
			return builder.Multiply(builder.LayerNorm(input, spec.LayerNormEpsilon), weight)
		}
		return builder.AffineLayerNorm(input, weight, bias, spec.LayerNormEpsilon)
	}
	normalized := builder.WeightedRMSNorm(input, weight, spec.RMSNormEpsilon)
	if spec.Architecture == "phimoe" && bias != nil {
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
	usesExperts := spec.Architecture == "nomic-bert-moe" && spec.IsInterleavedMoELayer(layerIndex)
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
	if spec.HiddenActivation == "silu" {
		fused = builder.SwiGLU(gate, up)
	} else {
		fused = builder.GEGLU(gate, up)
	}
	output := builder.Add(residual, builder.MulMat(weights.FeedForwardDown, fused))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: key, Value: value}, nil
}

// BuildT5EncoderBlock: constructs one full, bidirectional T5 encoder block
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

type LFM2BlockResult struct {
	Output    *tensor.Tensor
	Key       *tensor.Tensor
	Value     *tensor.Tensor
	Recurrent bool
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

// BuildMLABlockCached: PLM/MiniCPM3 MLA block
func BuildMLABlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Architecture != "plm" && spec.Architecture != "minicpm3" {
		return DenseBlockResult{}, errors.New("MLA block requires plm or minicpm3 architecture")
	}
	isMiniCPM3 := spec.Architecture == "minicpm3"
	required := map[string]*tensor.Tensor{
		"attention norm": weights.AttentionNorm, "attention Q": weights.AttentionQ,
		"attention KV-A": weights.AttentionKVAMQA, "attention KV-A norm": weights.AttentionKVANorm,
		"attention KV-B": weights.AttentionKVB, "attention output": weights.AttentionOutput,
		"feed-forward norm": weights.FeedForwardNorm, "feed-forward up": weights.FeedForwardUp,
		"feed-forward down": weights.FeedForwardDown,
	}
	if isMiniCPM3 {
		required["attention Q-B"] = weights.AttentionQB
		required["attention Q-A norm"] = weights.AttentionQNorm
		required["feed-forward gate"] = weights.FeedForwardGate
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
	if isMiniCPM3 {
		queryMixed = builder.WeightedRMSNorm(queryMixed, weights.AttentionQNorm, spec.RMSNormEpsilon)
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
	kv := builder.MulMat(weights.AttentionKVB, kvCompressed)
	stride := nopeWidth + valueWidth
	kNoPE := builder.GroupSlice(kv, 0, nopeWidth, heads, stride)
	value := builder.GroupSlice(kv, nopeWidth, valueWidth, heads, stride)
	frequencyScale := float32(1)
	if spec.RopeScalingType == "linear" && spec.RopeScalingFactor > 0 {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	if isMiniCPM3 {
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
	kPEHeads := builder.RepeatHeads(kPE, spec.HeadCount)
	query := builder.Concat(qNoPE, qPE, 0)
	key := builder.Concat(kNoPE, kPEHeads, 0)
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attention := builder.AttentionWithOffset(
		query, cacheKey, cacheValue, float32(1/math.Sqrt(float64(spec.KeyLength))), true, queryStart,
	)
	attention = builder.Reshape(attention, heads*valueWidth, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if isMiniCPM3 {
		attention = builder.Scale(attention, spec.ResidualScale)
	}
	residual := builder.Add(input, attention)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	activated := builder.ReLUSquared(up)
	if isMiniCPM3 {
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
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
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
	convInput := builder.Concat(pastKey, builder.Transpose2D(builder.Multiply(b, x)), 0)
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
	if builder == nil {
		return DenseBlockResult{}, errors.New("dense block builder is nil")
	}
	if input == nil {
		return DenseBlockResult{}, errors.New("dense block input is nil")
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	if spec.Architecture == "bert" || spec.Architecture == "jina-bert-v2" || spec.Architecture == "jina-bert-v3" || spec.Architecture == "nomic-bert" || spec.Architecture == "nomic-bert-moe" {
		return buildBERTEncoderBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	if spec.Architecture == "modern-bert" {
		return buildModernBERTBlock(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	isOLMo2 := spec.Architecture == "olmo2"
	isOLMoE := spec.Architecture == "olmoe"
	isPhiMoE := spec.Architecture == "phimoe"
	isEXAOneMoE := spec.Architecture == "exaone-moe"
	isPostOnlyNorm := usesPostOnlyNorm(spec.Architecture)
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
	isGraniteMoE := spec.Architecture == "granitemoe"
	isGrok := spec.Architecture == "grok"
	isHunyuanMoE := spec.Architecture == "hunyuan-moe"
	isHunyuan := isHunyuanMoE || spec.Architecture == "hunyuan-dense"
	isHYV3 := spec.Architecture == "hy_v3"
	isMellum := spec.Architecture == "mellum"
	isSmallThinker := spec.Architecture == "smallthinker"
	isMiniMaxM2 := spec.Architecture == "minimax-m2"
	isLFM2MoE := spec.Architecture == "lfm2moe"
	isArctic := spec.Architecture == "arctic"
	isLLaDAMoE := spec.Architecture == "llada-moe"
	isQwen2MoE := spec.Architecture == "qwen2moe"
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
		if !((spec.Architecture == "llama" || spec.Architecture == "llama-embed") && spec.ExpertCount > 0) && spec.Architecture != "qwen3moe" && spec.Architecture != "rnd1" && !isArctic && !isLLaDAMoE && !isBailingMoE && !isBailingMoE2 && !isCohere2MoE && !isDeepSeek && !isDeepSeek2OCR && !isDBRX && !isDOTS1 && !isErnieMoE && !isGraniteMoE && !isGrok && !isHunyuanMoE && !isHYV3 && !isMellum && !isSmallThinker && !isMiniMaxM2 && !isLFM2MoE && !isLaguna && !isAFMoE && !isQwen2MoE && !isOLMoE && !isPhiMoE && !isEXAOneMoE {
			return DenseBlockResult{}, errors.New("dense block expert weights require a supported MoE architecture")
		}
		required["feed-forward router"] = weights.FeedForwardRouter
		if (isCohere2MoE || isDeepSeek2OCR || isHYV3) && weights.FeedForwardGateUpExperts != nil {
			required["feed-forward fused expert gate/up"] = weights.FeedForwardGateUpExperts
		} else {
			if (!isGraniteMoE && !isGrok && !isErnieMoE) || weights.FeedForwardGateExperts != nil {
				required["feed-forward expert gate"] = weights.FeedForwardGateExperts
			}
			required["feed-forward expert up"] = weights.FeedForwardUpExperts
		}
		required["feed-forward expert down"] = weights.FeedForwardDownExperts
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
	if !usesExperts && !usesGateFreeFFN(spec.Architecture) {
		required["feed-forward gate"] = weights.FeedForwardGate
	} else if usesSequentialGELU(spec.Architecture) {
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
		if !usesParallelResidual(spec.Architecture) {
			if spec.Architecture != "stablelm" {
				required["feed-forward norm"] = weights.FeedForwardNorm
			}
		}
		if spec.RequiresLayerNormBias() {
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
	if spec.Architecture == "plamo3" {
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
	if spec.NonCausalAttention && pastKey != nil {
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
		if isDBRX && spec.AttentionClamp > 0 {
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
	}
	if isOLMo2 || isOLMoE || isMiniMaxM2 {
		query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	}

	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(headCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(kvHeadCount), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(kvHeadCount), tokens)

	if spec.Architecture == "apertus" || isAFMoE || isBailingMoE2 || isDOTS1 || spec.Architecture == "exaone4" || isEXAOneMoE || isHYV3 || isLLaDAMoE || isMellum || spec.Architecture == "openelm" || spec.Architecture == "plamo3" || spec.Architecture == "qwen3" || spec.Architecture == "qwen3moe" || spec.Architecture == "rnd1" || isLaguna || spec.Architecture == "lfm2" || isLFM2MoE || spec.Architecture == "gemma3" {
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
		rotaryDimensions = spec.RopeDimensionCount
	}
	if !spec.UsesRoPE(layerIndex) {
		// Some dense architectures leave periodic layers
		// position-independent
	} else if spec.Architecture == "paddleocr" {
		var multiPositions [4][]uint32
		for axis := range multiPositions {
			multiPositions[axis] = positions
		}
		query = builder.RoPEMulti(
			query, multiPositions, spec.RopeSections,
			rotaryDimensions, spec.RopeFrequencyBase,
		)
		key = builder.RoPEMulti(
			key, multiPositions, spec.RopeSections,
			rotaryDimensions, spec.RopeFrequencyBase,
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
	} else if usesNormalRoPE(spec.Architecture) {
		frequencyBase := spec.RopeFrequencyBase
		frequencyScale := float32(1)
		if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		if (spec.Architecture == "cohere2" || isCohere2MoE) && spec.IsSlidingLayer(layerIndex) {
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
		if (isAFMoE || isEXAOneMoE || isSmallThinker || spec.Architecture == "plamo3") && spec.IsSlidingLayer(layerIndex) {
			frequencyBase = spec.RopeFrequencySWA
		}
		if isMellum && spec.IsSlidingLayer(layerIndex) {
			frequencyBase = spec.RopeFrequencySWA
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
	if supportsLongRoPE(spec.Architecture) && spec.RopeAttentionFactor > 0 && spec.RopeAttentionFactor != 1 {
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
	if isGemmaArchitecture(spec.Architecture) {
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
		causal := !spec.NonCausalAttention
		if spec.MaxALiBiBias > 0 {
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
	if isLaguna && weights.AttentionOutputGate.Shape.Dims[1] == uint64(headCount) {
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
	if weights.AttentionOutputScale != nil {
		attention = builder.Multiply(attention, weights.AttentionOutputScale)
	}
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	if spec.SandwichNorm {
		attention = builder.WeightedRMSNorm(attention, weights.AttentionNorm, spec.RMSNormEpsilon)
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

	parallelResidual := usesParallelResidual(spec.Architecture) ||
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
	} else if !parallelResidual {
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
		if isErnieMoE {
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
			feedForward = builder.MoE(
				normalized,
				weights.FeedForwardRouter,
				weights.FeedForwardGateExperts,
				weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts,
				spec.ExpertUsedCount,
				true,
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
	if spec.SandwichNorm {
		feedForward = builder.WeightedRMSNorm(
			feedForward, weights.FeedForwardNorm, spec.RMSNormEpsilon,
		)
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

// BuildQwen35BlockCached: constructs either gated full-attention block or
// fused gated-delta-net recurrent block, following layer cadence recorded
// in weight catalog
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
