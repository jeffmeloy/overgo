package model

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

// LayerWeights: initial dense transformer weight set
type LayerWeights struct {
	Recurrent                   bool
	AttentionNorm               gguf.TensorInfo
	AttentionNormBias           *gguf.TensorInfo
	AttentionNorm2              *gguf.TensorInfo
	AttentionNorm2Bias          *gguf.TensorInfo
	AttentionQ                  gguf.TensorInfo
	AttentionQB                 *gguf.TensorInfo
	AttentionK                  gguf.TensorInfo
	AttentionV                  gguf.TensorInfo
	AttentionOutput             gguf.TensorInfo
	AttentionQScale             *gguf.TensorInfo
	AttentionKScale             *gguf.TensorInfo
	AttentionVScale             *gguf.TensorInfo
	AttentionOutputScale        *gguf.TensorInfo
	AttentionSubNorm            *gguf.TensorInfo
	AttentionQBias              *gguf.TensorInfo
	AttentionKBias              *gguf.TensorInfo
	AttentionVBias              *gguf.TensorInfo
	AttentionOutputBias         *gguf.TensorInfo
	AttentionQNorm              *gguf.TensorInfo
	AttentionKNorm              *gguf.TensorInfo
	AttentionQNormBias          *gguf.TensorInfo
	AttentionKNormBias          *gguf.TensorInfo
	AttentionPostNorm           *gguf.TensorInfo
	AttentionPostNormBias       *gguf.TensorInfo
	AttentionRelativeBias       *gguf.TensorInfo
	CrossAttentionNorm          *gguf.TensorInfo
	CrossAttentionQ             *gguf.TensorInfo
	CrossAttentionK             *gguf.TensorInfo
	CrossAttentionV             *gguf.TensorInfo
	CrossAttentionOutput        *gguf.TensorInfo
	AttentionOutputGate         *gguf.TensorInfo
	AttentionSinks              *gguf.TensorInfo
	RopeFactors                 *gguf.TensorInfo
	FeedForwardNorm             gguf.TensorInfo
	FeedForwardNormBias         *gguf.TensorInfo
	FeedForwardExpertNorm       *gguf.TensorInfo
	FeedForwardGate             gguf.TensorInfo
	FeedForwardUp               gguf.TensorInfo
	FeedForwardDown             gguf.TensorInfo
	FeedForwardGateScale        *gguf.TensorInfo
	FeedForwardUpScale          *gguf.TensorInfo
	FeedForwardDownScale        *gguf.TensorInfo
	FeedForwardSubNorm          *gguf.TensorInfo
	FeedForwardGateBias         *gguf.TensorInfo
	FeedForwardUpBias           *gguf.TensorInfo
	FeedForwardDownBias         *gguf.TensorInfo
	FeedForwardPostNorm         *gguf.TensorInfo
	FeedForwardPostNormBias     *gguf.TensorInfo
	FeedForwardPreNorm2         *gguf.TensorInfo
	FeedForwardPostNorm1        *gguf.TensorInfo
	FeedForwardPostNorm2        *gguf.TensorInfo
	FeedForwardRouter           *gguf.TensorInfo
	FeedForwardRouterBias       *gguf.TensorInfo
	FeedForwardRouterScale      *gguf.TensorInfo
	FeedForwardGateUpExperts    *gguf.TensorInfo
	FeedForwardGateExperts      *gguf.TensorInfo
	FeedForwardUpExperts        *gguf.TensorInfo
	FeedForwardDownExperts      *gguf.TensorInfo
	FeedForwardDownExpertsScale *gguf.TensorInfo
	FeedForwardGateChunkExperts *gguf.TensorInfo
	FeedForwardUpChunkExperts   *gguf.TensorInfo
	FeedForwardDownChunkExperts *gguf.TensorInfo
	FeedForwardExpertBias       *gguf.TensorInfo
	FeedForwardLatentDown       *gguf.TensorInfo
	FeedForwardLatentUp         *gguf.TensorInfo
	FeedForwardSharedGate       *gguf.TensorInfo
	FeedForwardSharedUp         *gguf.TensorInfo
	FeedForwardSharedDown       *gguf.TensorInfo
	FeedForwardSharedRouter     *gguf.TensorInfo
	LayerOutputScale            *gguf.TensorInfo
	PerLayerInputGate           *gguf.TensorInfo
	PerLayerProjection          *gguf.TensorInfo
	PerLayerPostNorm            *gguf.TensorInfo
	ShortConvKernel             *gguf.TensorInfo
	ShortConvInput              *gguf.TensorInfo
	ShortConvOutput             *gguf.TensorInfo
	AttentionKVAMQA             *gguf.TensorInfo
	AttentionKVANorm            *gguf.TensorInfo
	AttentionKVB                *gguf.TensorInfo
	AttentionKB                 *gguf.TensorInfo
	AttentionVB                 *gguf.TensorInfo
	IndexerKNorm                *gguf.TensorInfo
	IndexerKNormBias            *gguf.TensorInfo
	IndexerProjection           *gguf.TensorInfo
	IndexerAttentionK           *gguf.TensorInfo
	IndexerAttentionQB          *gguf.TensorInfo
	AttentionOutputA            *gguf.TensorInfo
	AttentionCompressorKV       *gguf.TensorInfo
	AttentionCompressorGate     *gguf.TensorInfo
	AttentionCompressorAPE      *gguf.TensorInfo
	AttentionCompressorNorm     *gguf.TensorInfo
	IndexerCompressorKV         *gguf.TensorInfo
	IndexerCompressorGate       *gguf.TensorInfo
	IndexerCompressorAPE        *gguf.TensorInfo
	IndexerCompressorNorm       *gguf.TensorInfo
	HyperAttentionFN            *gguf.TensorInfo
	HyperAttentionBase          *gguf.TensorInfo
	HyperAttentionScale         *gguf.TensorInfo
	HyperFeedForwardFN          *gguf.TensorInfo
	HyperFeedForwardBase        *gguf.TensorInfo
	HyperFeedForwardScale       *gguf.TensorInfo
	HyperHeadFN                 *gguf.TensorInfo
	HyperHeadBase               *gguf.TensorInfo
	HyperHeadScale              *gguf.TensorInfo
	FeedForwardHashExperts      *gguf.TensorInfo
	VisualAttentionQKV          *gguf.TensorInfo
	VisualAttentionOutput       *gguf.TensorInfo
	VisualFeedForwardGate       *gguf.TensorInfo
	VisualFeedForwardUp         *gguf.TensorInfo
	VisualFeedForwardDown       *gguf.TensorInfo
	AltUpCorrectCoefficient     *gguf.TensorInfo
	AltUpCorrectScale           *gguf.TensorInfo
	AltUpPredictCoefficient     *gguf.TensorInfo
	AltUpRouter                 *gguf.TensorInfo
	AltUpRouterNorm             *gguf.TensorInfo
	LaurelLeft                  *gguf.TensorInfo
	LaurelRight                 *gguf.TensorInfo
	LaurelPostNorm              *gguf.TensorInfo

	AttentionQKV         *gguf.TensorInfo
	AttentionQKVBias     *gguf.TensorInfo
	AttentionGate        *gguf.TensorInfo
	SSMConv1D            *gguf.TensorInfo
	SSMConv1DBias        *gguf.TensorInfo
	SSMInput             *gguf.TensorInfo
	SSMX                 *gguf.TensorInfo
	SSMTimeStepWeight    *gguf.TensorInfo
	SSMTimeStep          *gguf.TensorInfo
	SSMTimeStepNorm      *gguf.TensorInfo
	SSMA                 *gguf.TensorInfo
	SSMD                 *gguf.TensorInfo
	SSMBNorm             *gguf.TensorInfo
	SSMCNorm             *gguf.TensorInfo
	SSMBeta              *gguf.TensorInfo
	SSMAlpha             *gguf.TensorInfo
	SSMBetaAlpha         *gguf.TensorInfo
	SSMNorm              *gguf.TensorInfo
	SSMOutput            *gguf.TensorInfo
	SSMQueryConv         *gguf.TensorInfo
	SSMKeyConv           *gguf.TensorInfo
	SSMValueConv         *gguf.TensorInfo
	SSMForgetA           *gguf.TensorInfo
	SSMForgetB           *gguf.TensorInfo
	SSMOutputGateA       *gguf.TensorInfo
	SSMOutputGateB       *gguf.TensorInfo
	TimeMixW1            *gguf.TensorInfo
	TimeMixW2            *gguf.TensorInfo
	TimeMixW0            *gguf.TensorInfo
	TimeMixA0            *gguf.TensorInfo
	TimeMixA1            *gguf.TensorInfo
	TimeMixA2            *gguf.TensorInfo
	TimeMixV0            *gguf.TensorInfo
	TimeMixV1            *gguf.TensorInfo
	TimeMixV2            *gguf.TensorInfo
	TimeMixG1            *gguf.TensorInfo
	TimeMixG2            *gguf.TensorInfo
	TimeMixKK            *gguf.TensorInfo
	TimeMixKA            *gguf.TensorInfo
	TimeMixRK            *gguf.TensorInfo
	TimeMixLerpX         *gguf.TensorInfo
	TimeMixLerpFused     *gguf.TensorInfo
	TimeMixLerpW         *gguf.TensorInfo
	TimeMixLerpK         *gguf.TensorInfo
	TimeMixLerpV         *gguf.TensorInfo
	TimeMixLerpR         *gguf.TensorInfo
	TimeMixLerpG         *gguf.TensorInfo
	TimeMixFirst         *gguf.TensorInfo
	TimeMixDecay         *gguf.TensorInfo
	TimeMixDecayW1       *gguf.TensorInfo
	TimeMixDecayW2       *gguf.TensorInfo
	TimeMixKey           *gguf.TensorInfo
	TimeMixValue         *gguf.TensorInfo
	TimeMixReceptance    *gguf.TensorInfo
	TimeMixGate          *gguf.TensorInfo
	TimeMixLN            *gguf.TensorInfo
	TimeMixLNBias        *gguf.TensorInfo
	TimeMixOutput        *gguf.TensorInfo
	ChannelMixLerpK      *gguf.TensorInfo
	ChannelMixLerpR      *gguf.TensorInfo
	ChannelMixKey        *gguf.TensorInfo
	ChannelMixValue      *gguf.TensorInfo
	ChannelMixReceptance *gguf.TensorInfo
}

// WavPosNetWeights: layer-specific PosNet tensors
type WavPosNetWeights struct {
	Norm1, Norm1Bias, Conv1, Conv1Bias gguf.TensorInfo
	Norm2, Norm2Bias, Conv2, Conv2Bias gguf.TensorInfo
	AttentionNorm, AttentionNormBias   gguf.TensorInfo
	AttentionQ, AttentionQBias         gguf.TensorInfo
	AttentionK, AttentionKBias         gguf.TensorInfo
	AttentionV, AttentionVBias         gguf.TensorInfo
	AttentionOutput, AttentionOutBias  gguf.TensorInfo
}

// WavConvNextWeights: ConvNeXt block tensors
type WavConvNextWeights struct {
	Depthwise, DepthwiseBias   gguf.TensorInfo
	Norm, NormBias             gguf.TensorInfo
	Pointwise1, Pointwise1Bias gguf.TensorInfo
	Pointwise2, Pointwise2Bias gguf.TensorInfo
	Gamma                      gguf.TensorInfo
}

// WavTokenizerWeights: decoder-only audio tensors
type WavTokenizerWeights struct {
	InputConv, InputConvBias   gguf.TensorInfo
	PosNet                     []WavPosNetWeights
	TokenNorm, TokenNormBias   gguf.TensorInfo
	ConvNext                   []WavConvNextWeights
	OutputNorm, OutputNormBias gguf.TensorInfo
	Output, OutputBias         gguf.TensorInfo
}

// Qwen35MTPWeights: one pinned dense NextN block.
type Qwen35MTPWeights struct {
	Layer          LayerWeights
	EHProjection   gguf.TensorInfo
	EmbeddingNorm  gguf.TensorInfo
	HiddenNorm     gguf.TensorInfo
	TokenEmbedding *gguf.TensorInfo
	OutputNorm     *gguf.TensorInfo
	Output         *gguf.TensorInfo
}

// Weights: validated initial Llama/Qwen3 tensor catalog
type Weights struct {
	TokenEmbedding          gguf.TensorInfo
	TokenTypeEmbedding      *gguf.TensorInfo
	PositionEmbedding       *gguf.TensorInfo
	TokenEmbeddingNorm      *gguf.TensorInfo
	TokenEmbeddingNormBias  *gguf.TensorInfo
	OutputNorm              gguf.TensorInfo
	EncoderOutputNorm       *gguf.TensorInfo
	OutputNormBias          *gguf.TensorInfo
	Output                  *gguf.TensorInfo
	OutputBias              *gguf.TensorInfo
	Dense2Output            *gguf.TensorInfo
	Dense3Output            *gguf.TensorInfo
	PerLayerTokenEmbedding  *gguf.TensorInfo
	PerLayerModelProjection *gguf.TensorInfo
	PerLayerProjectionNorm  *gguf.TensorInfo
	FeatureProjection       *gguf.TensorInfo
	FeatureProjectionPost   *gguf.TensorInfo
	DraftToTarget           *gguf.TensorInfo
	AltUpProjection         *gguf.TensorInfo
	AltUpUnembedding        *gguf.TensorInfo
	Layers                  []LayerWeights
	EncoderLayers           []LayerWeights
	WavTokenizer            *WavTokenizerWeights
	Qwen35MTP               *Qwen35MTPWeights
}

// ReadWeights: validates names and shapes without loading tensor bytes
func ReadWeights(file *gguf.File, spec Spec) (Weights, error) {
	if file == nil {
		return Weights{}, errors.New("model file is nil")
	}
	tensors := make(map[string]gguf.TensorInfo, len(file.Tensors))
	for _, item := range file.Tensors {
		if _, exists := tensors[item.Name]; exists {
			return Weights{}, fmt.Errorf("duplicate tensor %q", item.Name)
		}
		tensors[item.Name] = item
	}
	required := func(name string, shape ...uint64) (gguf.TensorInfo, error) {
		item, ok := tensors[name]
		if !ok {
			return gguf.TensorInfo{}, fmt.Errorf("required tensor %q is missing", name)
		}
		if item.Dimensions != uint32(len(shape)) {
			return gguf.TensorInfo{}, fmt.Errorf(
				"tensor %q has rank %d, need %d",
				name,
				item.Dimensions,
				len(shape),
			)
		}
		for index, dimension := range shape {
			if item.Shape[index] != dimension {
				return gguf.TensorInfo{}, fmt.Errorf(
					"tensor %q dimension %d is %d, need %d",
					name,
					index,
					item.Shape[index],
					dimension,
				)
			}
		}
		return item, nil
	}

	var result Weights
	var err error
	if spec.Architecture == "dflash" {
		featureWidth := uint64(len(spec.TargetLayers)) * uint64(spec.EmbeddingLength)
		projection, loadErr := required("fc.weight", featureWidth, uint64(spec.EmbeddingLength))
		if loadErr != nil {
			return Weights{}, loadErr
		}
		encoderNorm, loadErr := required("enc.output_norm.weight", uint64(spec.EmbeddingLength))
		if loadErr != nil {
			return Weights{}, loadErr
		}
		if result.OutputNorm, loadErr = required("output_norm.weight", uint64(spec.EmbeddingLength)); loadErr != nil {
			return Weights{}, loadErr
		}
		result.FeatureProjection = &projection
		result.EncoderOutputNorm = &encoderNorm
		result.Layers = make([]LayerWeights, spec.BlockCount)
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		for block := uint32(0); block < spec.BlockCount; block++ {
			prefix := fmt.Sprintf("blk.%d.", block)
			layer := &result.Layers[block]
			for name, item := range map[string]struct {
				destination *gguf.TensorInfo
				shape       []uint64
			}{
				"attn_norm.weight":   {&layer.AttentionNorm, []uint64{uint64(spec.EmbeddingLength)}},
				"attn_q.weight":      {&layer.AttentionQ, []uint64{uint64(spec.EmbeddingLength), queryLength}},
				"attn_k.weight":      {&layer.AttentionK, []uint64{uint64(spec.EmbeddingLength), keyLength}},
				"attn_v.weight":      {&layer.AttentionV, []uint64{uint64(spec.EmbeddingLength), valueLength}},
				"attn_output.weight": {&layer.AttentionOutput, []uint64{queryLength, uint64(spec.EmbeddingLength)}},
				"ffn_norm.weight":    {&layer.FeedForwardNorm, []uint64{uint64(spec.EmbeddingLength)}},
				"ffn_gate.weight":    {&layer.FeedForwardGate, []uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}},
				"ffn_up.weight":      {&layer.FeedForwardUp, []uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}},
				"ffn_down.weight":    {&layer.FeedForwardDown, []uint64{uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)}},
			} {
				loaded, itemErr := required(prefix+name, item.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*item.destination = loaded
			}
			qNorm, itemErr := required(prefix+"attn_q_norm.weight", uint64(spec.KeyLength))
			if itemErr != nil {
				return Weights{}, itemErr
			}
			kNorm, itemErr := required(prefix+"attn_k_norm.weight", uint64(spec.KeyLength))
			if itemErr != nil {
				return Weights{}, itemErr
			}
			layer.AttentionQNorm, layer.AttentionKNorm = &qNorm, &kNorm
		}
		return result, nil
	}
	if spec.Architecture == "eagle3" {
		draftVocabulary := uint64(spec.VocabularySize)
		if item, ok := tensors["d2t"]; ok {
			if item.Type != dtype.I64 || item.Dimensions != 1 || item.Shape[0] == 0 {
				return Weights{}, fmt.Errorf("tensor %q has incompatible shape/type", item.Name)
			}
			draftVocabulary = item.Shape[0]
			result.DraftToTarget = &item
		}
		projection, loadErr := required(
			"fc.weight", 3*uint64(spec.TargetHiddenSize), uint64(spec.EmbeddingLength),
		)
		if loadErr != nil {
			return Weights{}, loadErr
		}
		if result.OutputNorm, loadErr = required("output_norm.weight", uint64(spec.EmbeddingLength)); loadErr != nil {
			return Weights{}, loadErr
		}
		result.FeatureProjection = &projection
		if item, ok := tensors["token_embd.weight"]; ok {
			validated, itemErr := required(item.Name, uint64(spec.EmbeddingLength), uint64(spec.VocabularySize))
			if itemErr != nil {
				return Weights{}, itemErr
			}
			result.TokenEmbedding = validated
		}
		if item, ok := tensors["output.weight"]; ok {
			validated, itemErr := required(item.Name, uint64(spec.EmbeddingLength), draftVocabulary)
			if itemErr != nil {
				return Weights{}, itemErr
			}
			result.Output = &validated
		}
		if result.DraftToTarget != nil && result.Output == nil {
			return Weights{}, errors.New(`required tensor "output.weight" is missing for Eagle3 vocabulary mapping`)
		}
		result.Layers = make([]LayerWeights, 1)
		layer := &result.Layers[0]
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		for name, item := range map[string]struct {
			destination *gguf.TensorInfo
			shape       []uint64
		}{
			"attn_norm.weight":   {&layer.AttentionNorm, []uint64{uint64(spec.EmbeddingLength)}},
			"attn_q.weight":      {&layer.AttentionQ, []uint64{2 * uint64(spec.EmbeddingLength), queryLength}},
			"attn_k.weight":      {&layer.AttentionK, []uint64{2 * uint64(spec.EmbeddingLength), keyLength}},
			"attn_v.weight":      {&layer.AttentionV, []uint64{2 * uint64(spec.EmbeddingLength), valueLength}},
			"attn_output.weight": {&layer.AttentionOutput, []uint64{queryLength, uint64(spec.EmbeddingLength)}},
			"ffn_norm.weight":    {&layer.FeedForwardNorm, []uint64{uint64(spec.EmbeddingLength)}},
			"ffn_gate.weight":    {&layer.FeedForwardGate, []uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}},
			"ffn_up.weight":      {&layer.FeedForwardUp, []uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}},
			"ffn_down.weight":    {&layer.FeedForwardDown, []uint64{uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)}},
		} {
			loaded, itemErr := required("blk.0."+name, item.shape...)
			if itemErr != nil {
				return Weights{}, itemErr
			}
			*item.destination = loaded
		}
		hiddenNorm, itemErr := required("blk.0.attn_norm_2.weight", uint64(spec.EmbeddingLength))
		if itemErr != nil {
			return Weights{}, itemErr
		}
		layer.AttentionNorm2 = &hiddenNorm
		if item, ok := tensors["blk.0.rope_freqs.weight"]; ok {
			validated, itemErr := required(item.Name, uint64(spec.RopeDimensionCount/2))
			if itemErr != nil {
				return Weights{}, itemErr
			}
			layer.RopeFactors = &validated
		}
		return result, nil
	}
	if spec.Architecture == "gemma4-assistant" {
		width := uint64(spec.EmbeddingLength)
		targetWidth := uint64(spec.TargetHiddenSize)
		if result.TokenEmbedding, err = required("token_embd.weight", width, uint64(spec.VocabularySize)); err != nil {
			return Weights{}, err
		}
		if result.OutputNorm, err = required("output_norm.weight", width); err != nil {
			return Weights{}, err
		}
		pre, loadErr := required("blk.0.nextn.pre_projection.weight", 2*targetWidth, width)
		if loadErr != nil {
			return Weights{}, loadErr
		}
		post, loadErr := required("nextn.post_projection.weight", width, targetWidth)
		if loadErr != nil {
			return Weights{}, loadErr
		}
		result.FeatureProjection = &pre
		result.FeatureProjectionPost = &post
		result.Layers = make([]LayerWeights, spec.BlockCount)
		var sharedRope *gguf.TensorInfo
		for block := uint32(0); block < spec.BlockCount; block++ {
			prefix := fmt.Sprintf("blk.%d.", block)
			layer := &result.Layers[block]
			queryLength := uint64(spec.HeadCount) * uint64(spec.LayerKeyLength(block))
			attentionOutputLength := uint64(spec.HeadCount) * uint64(spec.LayerValueLength(block))
			for name, item := range map[string]struct {
				destination *gguf.TensorInfo
				shape       []uint64
			}{
				"attn_norm.weight":   {&layer.AttentionNorm, []uint64{width}},
				"attn_q.weight":      {&layer.AttentionQ, []uint64{width, queryLength}},
				"attn_output.weight": {&layer.AttentionOutput, []uint64{attentionOutputLength, width}},
				"ffn_norm.weight":    {&layer.FeedForwardNorm, []uint64{width}},
				"ffn_gate.weight":    {&layer.FeedForwardGate, []uint64{width, uint64(spec.FeedForwardLength)}},
				"ffn_up.weight":      {&layer.FeedForwardUp, []uint64{width, uint64(spec.FeedForwardLength)}},
				"ffn_down.weight":    {&layer.FeedForwardDown, []uint64{uint64(spec.FeedForwardLength), width}},
			} {
				loaded, itemErr := required(prefix+name, item.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*item.destination = loaded
			}
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"attn_q_norm.weight":         {[]uint64{uint64(spec.LayerKeyLength(block))}, &layer.AttentionQNorm},
				"post_attention_norm.weight": {[]uint64{width}, &layer.AttentionPostNorm},
				"post_ffw_norm.weight":       {[]uint64{width}, &layer.FeedForwardPostNorm},
				"layer_output_scale.weight":  {[]uint64{1}, &layer.LayerOutputScale},
			} {
				loaded, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &loaded
			}
			if !spec.IsSlidingLayer(block) {
				rope, ok := tensors[prefix+"rope_freqs.weight"]
				if !ok {
					rope, ok = tensors["rope_freqs.weight"]
				}
				if !ok && sharedRope != nil {
					rope, ok = *sharedRope, true
				}
				if !ok {
					return Weights{}, fmt.Errorf("required Gemma 4 assistant RoPE factors for layer %d are missing", block)
				}
				if rope.Type != dtype.F32 || rope.Dimensions != 1 || rope.Shape[0] != uint64(spec.RopeDimensionCount/2) {
					return Weights{}, fmt.Errorf("tensor %q has incompatible Gemma 4 assistant RoPE factors", rope.Name)
				}
				layer.RopeFactors = &rope
				sharedRope = layer.RopeFactors
			}
		}
		return result, nil
	}
	tokenEmbeddingName := "token_embd.weight"
	if spec.Architecture == "codeshell" {
		if _, ok := tensors[tokenEmbeddingName]; !ok {
			tokenEmbeddingName = "output.weight"
		}
	}
	if result.TokenEmbedding, err = required(
		tokenEmbeddingName,
		uint64(spec.EmbeddingLength),
		uint64(spec.VocabularySize),
	); err != nil {
		return Weights{}, err
	}
	if spec.Architecture == "deepseek4" {
		width := uint64(spec.EmbeddingLength)
		headWidth := uint64(spec.KeyLength)
		hyper := uint64(spec.HyperConnectionCount)
		hyperWidth := hyper * width
		mixWidth := (2 + hyper) * hyper
		if result.OutputNorm, err = required("output_norm.weight", width); err != nil {
			return Weights{}, err
		}
		output, loadErr := required("output.weight", width, uint64(spec.VocabularySize))
		if loadErr != nil {
			return Weights{}, loadErr
		}
		result.Output = &output
		result.Layers = make([]LayerWeights, spec.BlockCount)
		for block := uint32(0); block < spec.BlockCount; block++ {
			prefix := fmt.Sprintf("blk.%d.", block)
			layer := &result.Layers[block]
			for name, item := range map[string]struct {
				destination *gguf.TensorInfo
				shape       []uint64
			}{
				"attn_norm.weight":   {&layer.AttentionNorm, []uint64{width}},
				"attn_q_a.weight":    {&layer.AttentionQ, []uint64{width, uint64(spec.QLoRARank)}},
				"attn_kv.weight":     {&layer.AttentionK, []uint64{width, headWidth}},
				"attn_output.weight": {&layer.AttentionOutput, []uint64{uint64(spec.AttentionOutputGroups * spec.AttentionOutputRank), width}},
				"ffn_norm.weight":    {&layer.FeedForwardNorm, []uint64{width}},
			} {
				loaded, itemErr := required(prefix+name, item.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*item.destination = loaded
			}
			for name, item := range map[string]struct {
				destination **gguf.TensorInfo
				shape       []uint64
			}{
				"attn_sinks.weight":     {&layer.AttentionSinks, []uint64{uint64(spec.HeadCount)}},
				"attn_q_a_norm.weight":  {&layer.AttentionQNorm, []uint64{uint64(spec.QLoRARank)}},
				"attn_q_b.weight":       {&layer.AttentionQB, []uint64{uint64(spec.QLoRARank), uint64(spec.HeadCount) * headWidth}},
				"attn_kv_a_norm.weight": {&layer.AttentionKNorm, []uint64{headWidth}},
				"attn_output_a.weight":  {&layer.AttentionOutputA, []uint64{uint64(spec.HeadCount) * headWidth / uint64(spec.AttentionOutputGroups), uint64(spec.AttentionOutputRank * spec.AttentionOutputGroups)}},
				"hc_attn_fn.weight":     {&layer.HyperAttentionFN, []uint64{hyperWidth, mixWidth}},
				"hc_attn_base.weight":   {&layer.HyperAttentionBase, []uint64{mixWidth}},
				"hc_attn_scale.weight":  {&layer.HyperAttentionScale, []uint64{3}},
				"hc_ffn_fn.weight":      {&layer.HyperFeedForwardFN, []uint64{hyperWidth, mixWidth}},
				"hc_ffn_base.weight":    {&layer.HyperFeedForwardBase, []uint64{mixWidth}},
				"hc_ffn_scale.weight":   {&layer.HyperFeedForwardScale, []uint64{3}},
				"ffn_gate_inp.weight":   {&layer.FeedForwardRouter, []uint64{width, uint64(spec.ExpertCount)}},
				"ffn_gate_exps.weight":  {&layer.FeedForwardGateExperts, []uint64{width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)}},
				"ffn_up_exps.weight":    {&layer.FeedForwardUpExperts, []uint64{width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)}},
				"ffn_down_exps.weight":  {&layer.FeedForwardDownExperts, []uint64{uint64(spec.ExpertFeedForward), width, uint64(spec.ExpertCount)}},
				"ffn_gate_shexp.weight": {&layer.FeedForwardSharedGate, []uint64{width, uint64(spec.SharedExpertFF)}},
				"ffn_up_shexp.weight":   {&layer.FeedForwardSharedUp, []uint64{width, uint64(spec.SharedExpertFF)}},
				"ffn_down_shexp.weight": {&layer.FeedForwardSharedDown, []uint64{uint64(spec.SharedExpertFF), width}},
			} {
				loaded, itemErr := required(prefix+name, item.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*item.destination = &loaded
			}
			if block < spec.HashLayerCount {
				loaded, itemErr := required(prefix+"ffn_gate_tid2eid.weight", uint64(spec.ExpertUsedCount), uint64(spec.VocabularySize))
				if itemErr != nil {
					return Weights{}, itemErr
				}
				if loaded.Type != dtype.I32 {
					return Weights{}, fmt.Errorf("tensor %q must use I32 storage", loaded.Name)
				}
				layer.FeedForwardHashExperts = &loaded
			} else {
				loaded, itemErr := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.FeedForwardRouterBias = &loaded
			}
			ratio := spec.CompressRatios[block]
			if ratio != 0 {
				coefficient := uint64(1)
				if ratio == 4 {
					coefficient = 2
				}
				for name, item := range map[string]struct {
					destination **gguf.TensorInfo
					shape       []uint64
				}{
					"attn_compressor_kv.weight":   {&layer.AttentionCompressorKV, []uint64{width, coefficient * headWidth}},
					"attn_compressor_gate.weight": {&layer.AttentionCompressorGate, []uint64{width, coefficient * headWidth}},
					"attn_compressor_ape.weight":  {&layer.AttentionCompressorAPE, []uint64{coefficient * headWidth, uint64(ratio)}},
					"attn_compressor_norm.weight": {&layer.AttentionCompressorNorm, []uint64{headWidth}},
				} {
					loaded, itemErr := required(prefix+name, item.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*item.destination = &loaded
				}
			}
			if ratio == 4 {
				indexerWidth := uint64(spec.IndexerKeyLength)
				for name, item := range map[string]struct {
					destination **gguf.TensorInfo
					shape       []uint64
				}{
					"indexer.proj.weight":            {&layer.IndexerProjection, []uint64{width, uint64(spec.IndexerHeadCount)}},
					"indexer.attn_q_b.weight":        {&layer.IndexerAttentionQB, []uint64{uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount) * indexerWidth}},
					"indexer_compressor_kv.weight":   {&layer.IndexerCompressorKV, []uint64{width, 2 * indexerWidth}},
					"indexer_compressor_gate.weight": {&layer.IndexerCompressorGate, []uint64{width, 2 * indexerWidth}},
					"indexer_compressor_ape.weight":  {&layer.IndexerCompressorAPE, []uint64{2 * indexerWidth, 4}},
					"indexer_compressor_norm.weight": {&layer.IndexerCompressorNorm, []uint64{indexerWidth}},
				} {
					loaded, itemErr := required(prefix+name, item.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*item.destination = &loaded
				}
			}
		}
		last := &result.Layers[len(result.Layers)-1]
		for name, item := range map[string]struct {
			destination **gguf.TensorInfo
			shape       []uint64
		}{
			"output_hc_fn.weight":    {&last.HyperHeadFN, []uint64{hyperWidth, hyper}},
			"output_hc_base.weight":  {&last.HyperHeadBase, []uint64{hyper}},
			"output_hc_scale.weight": {&last.HyperHeadScale, []uint64{1}},
		} {
			loaded, itemErr := required(name, item.shape...)
			if itemErr != nil {
				return Weights{}, itemErr
			}
			*item.destination = &loaded
		}
		return result, nil
	}
	if spec.Architecture == "wavtokenizer-dec" {
		width := uint64(spec.PosNetEmbeddingLength)
		ffn := uint64(spec.FeedForwardLength)
		wav := &WavTokenizerWeights{
			PosNet:   make([]WavPosNetWeights, spec.PosNetBlockCount),
			ConvNext: make([]WavConvNextWeights, spec.ConvNextBlockCount),
		}
		load := func(destination *gguf.TensorInfo, name string, shape ...uint64) error {
			item, loadErr := required(name, shape...)
			if loadErr == nil {
				*destination = item
			}
			return loadErr
		}
		if err = load(&wav.InputConv, "conv1d.weight", 7, uint64(spec.EmbeddingLength), width); err != nil {
			return Weights{}, err
		}
		if err = load(&wav.InputConvBias, "conv1d.bias", 1, width); err != nil {
			return Weights{}, err
		}
		for block := uint32(0); block < spec.PosNetBlockCount; block++ {
			prefix := fmt.Sprintf("posnet.%d.", block)
			layer := &wav.PosNet[block]
			switch block {
			case 0, 1, 3, 4:
				for name, item := range map[string]struct {
					destination *gguf.TensorInfo
					shape       []uint64
				}{
					"norm1.weight": {&layer.Norm1, []uint64{1, width}},
					"norm1.bias":   {&layer.Norm1Bias, []uint64{1, width}},
					"conv1.weight": {&layer.Conv1, []uint64{3, width, width}},
					"conv1.bias":   {&layer.Conv1Bias, []uint64{1, width}},
					"norm2.weight": {&layer.Norm2, []uint64{1, width}},
					"norm2.bias":   {&layer.Norm2Bias, []uint64{1, width}},
					"conv2.weight": {&layer.Conv2, []uint64{3, width, width}},
					"conv2.bias":   {&layer.Conv2Bias, []uint64{1, width}},
				} {
					if err = load(item.destination, prefix+name, item.shape...); err != nil {
						return Weights{}, err
					}
				}
			case 2:
				for name, destination := range map[string]*gguf.TensorInfo{
					"attn_norm.weight": &layer.AttentionNorm,
					"attn_norm.bias":   &layer.AttentionNormBias,
					"attn_q.bias":      &layer.AttentionQBias,
					"attn_k.bias":      &layer.AttentionKBias,
					"attn_v.bias":      &layer.AttentionVBias,
					"attn_output.bias": &layer.AttentionOutBias,
				} {
					if err = load(destination, prefix+name, 1, width); err != nil {
						return Weights{}, err
					}
				}
				for name, destination := range map[string]*gguf.TensorInfo{
					"attn_q.weight":      &layer.AttentionQ,
					"attn_k.weight":      &layer.AttentionK,
					"attn_v.weight":      &layer.AttentionV,
					"attn_output.weight": &layer.AttentionOutput,
				} {
					if err = load(destination, prefix+name, 1, width, width); err != nil {
						return Weights{}, err
					}
				}
			case 5:
				if err = load(&layer.AttentionNorm, prefix+"attn_norm.weight", 1, width); err != nil {
					return Weights{}, err
				}
				if err = load(&layer.AttentionNormBias, prefix+"attn_norm.bias", 1, width); err != nil {
					return Weights{}, err
				}
			}
		}
		if err = load(&wav.TokenNorm, "token_embd_norm.weight", width); err != nil {
			return Weights{}, err
		}
		if err = load(&wav.TokenNormBias, "token_embd_norm.bias", width); err != nil {
			return Weights{}, err
		}
		for block := uint32(0); block < spec.ConvNextBlockCount; block++ {
			prefix := fmt.Sprintf("convnext.%d.", block)
			layer := &wav.ConvNext[block]
			for name, item := range map[string]struct {
				destination *gguf.TensorInfo
				shape       []uint64
			}{
				"dw.weight":    {&layer.Depthwise, []uint64{7, 1, width}},
				"dw.bias":      {&layer.DepthwiseBias, []uint64{1, width}},
				"norm.weight":  {&layer.Norm, []uint64{width}},
				"norm.bias":    {&layer.NormBias, []uint64{width}},
				"pw1.weight":   {&layer.Pointwise1, []uint64{width, ffn}},
				"pw1.bias":     {&layer.Pointwise1Bias, []uint64{ffn}},
				"pw2.weight":   {&layer.Pointwise2, []uint64{ffn, width}},
				"pw2.bias":     {&layer.Pointwise2Bias, []uint64{width}},
				"gamma.weight": {&layer.Gamma, []uint64{width}},
			} {
				if err = load(item.destination, prefix+name, item.shape...); err != nil {
					return Weights{}, err
				}
			}
		}
		for name, item := range map[string]struct {
			destination *gguf.TensorInfo
			shape       []uint64
		}{
			"output_norm.weight": {&wav.OutputNorm, []uint64{width}},
			"output_norm.bias":   {&wav.OutputNormBias, []uint64{width}},
			"output.weight":      {&wav.Output, []uint64{width, uint64(spec.OutputEmbeddingLength)}},
			"output.bias":        {&wav.OutputBias, []uint64{uint64(spec.OutputEmbeddingLength)}},
		} {
			if err = load(item.destination, name, item.shape...); err != nil {
				return Weights{}, err
			}
		}
		result.WavTokenizer = wav
		result.Output = &wav.Output
		result.OutputBias = &wav.OutputBias
		return result, nil
	}
	if spec.Architecture == "bert" || spec.Architecture == "gpt2" || spec.Architecture == "starcoder" {
		positionEmbedding, positionErr := required(
			"position_embd.weight",
			uint64(spec.EmbeddingLength),
			uint64(spec.ContextLength),
		)
		if positionErr != nil {
			return Weights{}, positionErr
		}
		result.PositionEmbedding = &positionEmbedding
	}
	if spec.Architecture == "bert" || spec.Architecture == "jina-bert-v2" || spec.Architecture == "jina-bert-v3" || spec.Architecture == "nomic-bert" || spec.Architecture == "nomic-bert-moe" {
		if typeEmbedding, ok := tensors["token_types.weight"]; ok {
			if typeEmbedding.Dimensions != 2 ||
				typeEmbedding.Shape[0] != uint64(spec.EmbeddingLength) ||
				typeEmbedding.Shape[1] != uint64(spec.TokenTypeCount) {
				return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", typeEmbedding.Name, typeEmbedding.Shape)
			}
			result.TokenTypeEmbedding = &typeEmbedding
		}
		if spec.Architecture == "jina-bert-v2" && result.TokenTypeEmbedding == nil {
			return Weights{}, errors.New(`required tensor "token_types.weight" is missing`)
		}
		tokenNorm, normErr := required("token_embd_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		tokenNormBias, normErr := required("token_embd_norm.bias", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.TokenEmbeddingNorm = &tokenNorm
		result.TokenEmbeddingNormBias = &tokenNormBias
	}
	if spec.Architecture == "modern-bert" {
		tokenNorm, normErr := required("token_embd_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.TokenEmbeddingNorm = &tokenNorm
	}
	if spec.Architecture == "mpt" {
		if _, ok := tensors["position_embd.weight"]; ok {
			positionEmbedding, positionErr := required(
				"position_embd.weight",
				uint64(spec.EmbeddingLength),
				uint64(spec.ContextLength),
			)
			if positionErr != nil {
				return Weights{}, positionErr
			}
			result.PositionEmbedding = &positionEmbedding
		}
	}
	if spec.Architecture == "bloom" {
		tokenNorm, normErr := required("token_embd_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		tokenNormBias, normErr := required("token_embd_norm.bias", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.TokenEmbeddingNorm = &tokenNorm
		result.TokenEmbeddingNormBias = &tokenNormBias
	}
	if spec.Architecture == "rwkv6" || spec.Architecture == "rwkv7" {
		tokenNorm, normErr := required("token_embd_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		tokenNormBias, normErr := required("token_embd_norm.bias", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.TokenEmbeddingNorm = &tokenNorm
		result.TokenEmbeddingNormBias = &tokenNormBias
	}
	outputNormName := "output_norm.weight"
	if spec.Architecture == "t5encoder" {
		outputNormName = "enc.output_norm.weight"
	} else if spec.Architecture == "t5" {
		outputNormName = "dec.output_norm.weight"
	} else if spec.Architecture == "neo-bert" {
		outputNormName = "enc.output_norm.weight"
	} else if spec.Architecture == "lfm2" || spec.Architecture == "lfm2moe" {
		outputNormName = "token_embd_norm.weight"
	}
	if !spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() && spec.Architecture != "bert" && spec.Architecture != "jina-bert-v2" && spec.Architecture != "jina-bert-v3" && spec.Architecture != "nomic-bert" && spec.Architecture != "nomic-bert-moe" {
		if result.OutputNorm, err = required(outputNormName, uint64(spec.EmbeddingLength)); err != nil {
			return Weights{}, err
		}
	}
	if spec.RequiresLayerNormBias() && spec.Architecture != "bert" && spec.Architecture != "jina-bert-v2" && spec.Architecture != "jina-bert-v3" && spec.Architecture != "nomic-bert" && spec.Architecture != "nomic-bert-moe" {
		outputNormBias, biasErr := required("output_norm.bias", uint64(spec.EmbeddingLength))
		if biasErr != nil {
			return Weights{}, biasErr
		}
		result.OutputNormBias = &outputNormBias
	}
	if spec.Architecture == "rwkv6qwen2" {
		if item, ok := tensors["output_norm.bias"]; ok {
			validated, biasErr := required(item.Name, uint64(spec.EmbeddingLength))
			if biasErr != nil {
				return Weights{}, biasErr
			}
			result.OutputNormBias = &validated
		}
	}
	if spec.Architecture == "t5encoder" {
		result.Layers = make([]LayerWeights, spec.BlockCount)
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		attentionOutputLength := uint64(spec.HeadCount) * uint64(spec.ValueLength)
		for block := uint32(0); block < spec.BlockCount; block++ {
			prefix := fmt.Sprintf("enc.blk.%d.", block)
			layer := &result.Layers[block]
			if layer.AttentionNorm, err = required(prefix+"attn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if layer.AttentionQ, err = required(prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength); err != nil {
				return Weights{}, err
			}
			if layer.AttentionK, err = required(prefix+"attn_k.weight", uint64(spec.EmbeddingLength), keyLength); err != nil {
				return Weights{}, err
			}
			if layer.AttentionV, err = required(prefix+"attn_v.weight", uint64(spec.EmbeddingLength), valueLength); err != nil {
				return Weights{}, err
			}
			if layer.AttentionOutput, err = required(prefix+"attn_o.weight", attentionOutputLength, uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			relativeBias, biasErr := required(
				prefix+"attn_rel_b.weight",
				uint64(spec.HeadCount),
				uint64(spec.RelativeBuckets),
			)
			if biasErr != nil {
				return Weights{}, biasErr
			}
			layer.AttentionRelativeBias = &relativeBias
			if layer.FeedForwardNorm, err = required(prefix+"ffn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if layer.FeedForwardGate, err = required(prefix+"ffn_gate.weight", uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
				return Weights{}, err
			}
			if layer.FeedForwardUp, err = required(prefix+"ffn_up.weight", uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
				return Weights{}, err
			}
			if layer.FeedForwardDown, err = required(prefix+"ffn_down.weight", uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
		}
		return result, nil
	}
	if spec.Architecture == "t5" {
		encoderNorm, normErr := required("enc.output_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.EncoderOutputNorm = &encoderNorm
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		attentionOutputLength := uint64(spec.HeadCount) * uint64(spec.ValueLength)
		loadFFN := func(prefix string, layer *LayerWeights) error {
			var loadErr error
			if layer.FeedForwardNorm, loadErr = required(prefix+"ffn_norm.weight", uint64(spec.EmbeddingLength)); loadErr != nil {
				return loadErr
			}
			if item, ok := tensors[prefix+"ffn_gate.weight"]; ok {
				gate, gateErr := required(item.Name, uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength))
				if gateErr != nil {
					return gateErr
				}
				layer.FeedForwardGate = gate
			}
			if layer.FeedForwardUp, loadErr = required(prefix+"ffn_up.weight", uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); loadErr != nil {
				return loadErr
			}
			if layer.FeedForwardDown, loadErr = required(prefix+"ffn_down.weight", uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)); loadErr != nil {
				return loadErr
			}
			return nil
		}
		loadAttention := func(prefix string, layer *LayerWeights) error {
			var loadErr error
			if layer.AttentionNorm, loadErr = required(prefix+"attn_norm.weight", uint64(spec.EmbeddingLength)); loadErr != nil {
				return loadErr
			}
			if layer.AttentionQ, loadErr = required(prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength); loadErr != nil {
				return loadErr
			}
			if layer.AttentionK, loadErr = required(prefix+"attn_k.weight", uint64(spec.EmbeddingLength), keyLength); loadErr != nil {
				return loadErr
			}
			if layer.AttentionV, loadErr = required(prefix+"attn_v.weight", uint64(spec.EmbeddingLength), valueLength); loadErr != nil {
				return loadErr
			}
			if layer.AttentionOutput, loadErr = required(prefix+"attn_o.weight", attentionOutputLength, uint64(spec.EmbeddingLength)); loadErr != nil {
				return loadErr
			}
			return nil
		}
		result.EncoderLayers = make([]LayerWeights, spec.BlockCount)
		var encoderRelativeBias *gguf.TensorInfo
		for block := uint32(0); block < spec.BlockCount; block++ {
			prefix := fmt.Sprintf("enc.blk.%d.", block)
			layer := &result.EncoderLayers[block]
			if loadErr := loadAttention(prefix, layer); loadErr != nil {
				return Weights{}, loadErr
			}
			if loadErr := loadFFN(prefix, layer); loadErr != nil {
				return Weights{}, loadErr
			}
			if item, ok := tensors[prefix+"attn_rel_b.weight"]; ok {
				bias, biasErr := required(item.Name, uint64(spec.HeadCount), uint64(spec.RelativeBuckets))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionRelativeBias = &bias
				encoderRelativeBias = &bias
			} else if block == 0 {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_rel_b.weight")
			} else {
				layer.AttentionRelativeBias = encoderRelativeBias
			}
		}
		result.Layers = make([]LayerWeights, spec.DecoderBlockCount)
		var decoderRelativeBias *gguf.TensorInfo
		for block := uint32(0); block < spec.DecoderBlockCount; block++ {
			prefix := fmt.Sprintf("dec.blk.%d.", block)
			layer := &result.Layers[block]
			if loadErr := loadAttention(prefix, layer); loadErr != nil {
				return Weights{}, loadErr
			}
			if loadErr := loadFFN(prefix, layer); loadErr != nil {
				return Weights{}, loadErr
			}
			if item, ok := tensors[prefix+"attn_rel_b.weight"]; ok {
				bias, biasErr := required(item.Name, uint64(spec.HeadCount), uint64(spec.RelativeBuckets))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionRelativeBias = &bias
				decoderRelativeBias = &bias
			} else if block == 0 {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_rel_b.weight")
			} else {
				layer.AttentionRelativeBias = decoderRelativeBias
			}
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"cross_attn_norm.weight": {[]uint64{uint64(spec.EmbeddingLength)}, &layer.CrossAttentionNorm},
				"cross_attn_q.weight":    {[]uint64{uint64(spec.EmbeddingLength), queryLength}, &layer.CrossAttentionQ},
				"cross_attn_k.weight":    {[]uint64{uint64(spec.EmbeddingLength), keyLength}, &layer.CrossAttentionK},
				"cross_attn_v.weight":    {[]uint64{uint64(spec.EmbeddingLength), valueLength}, &layer.CrossAttentionV},
				"cross_attn_o.weight":    {[]uint64{attentionOutputLength, uint64(spec.EmbeddingLength)}, &layer.CrossAttentionOutput},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		}
		if output, ok := tensors["output.weight"]; ok {
			validated, outputErr := required(output.Name, uint64(spec.EmbeddingLength), uint64(spec.VocabularySize))
			if outputErr != nil {
				return Weights{}, outputErr
			}
			result.Output = &validated
		}
		return result, nil
	}
	if output, ok := tensors["output.weight"]; ok &&
		spec.Architecture != "cohere2" && spec.Architecture != "command-r" {
		if output.Dimensions != 2 ||
			output.Shape[0] != uint64(spec.EmbeddingLength) ||
			output.Shape[1] != uint64(spec.VocabularySize) {
			return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", output.Name, output.Shape)
		}
		result.Output = &output
	}
	if (spec.Architecture == "internlm2" ||
		spec.Architecture == "apertus" ||
		spec.Architecture == "baichuan" ||
		spec.Architecture == "bailingmoe" ||
		spec.Architecture == "bailingmoe2" ||
		spec.Architecture == "dbrx" ||
		spec.Architecture == "dots1" ||
		spec.Architecture == "minimax-m2" ||
		spec.Architecture == "mimo2" ||
		spec.Architecture == "step35" ||
		spec.Architecture == "kimi-linear" ||
		spec.Architecture == "rwkv6" ||
		spec.Architecture == "rwkv6qwen2" ||
		spec.Architecture == "rwkv7" ||
		spec.Architecture == "arwkv7" ||
		spec.Architecture == "mellum" ||
		spec.Architecture == "jais" ||
		spec.Architecture == "llada-moe" ||
		spec.Architecture == "xverse" ||
		spec.Architecture == "olmo2" ||
		spec.Architecture == "nemotron" ||
		spec.Architecture == "orion" ||
		spec.Architecture == "gptneox" ||
		spec.Architecture == "gpt-oss" ||
		spec.Architecture == "phi2" ||
		spec.Architecture == "phimoe" ||
		spec.Architecture == "plamo" ||
		spec.Architecture == "qwen" ||
		spec.Architecture == "stablelm" ||
		spec.Architecture == "talkie" ||
		spec.Architecture == "codeshell") &&
		result.Output == nil {
		return Weights{}, errors.New(`required tensor "output.weight" is missing`)
	}
	if outputBias, ok := tensors["output.bias"]; ok {
		if outputBias.Type != dtype.F32 ||
			outputBias.Dimensions != 1 ||
			outputBias.Shape[0] != uint64(spec.VocabularySize) {
			return Weights{}, fmt.Errorf(
				"tensor %q has incompatible shape %v",
				outputBias.Name,
				outputBias.Shape,
			)
		}
		result.OutputBias = &outputBias
	}
	if (spec.Architecture == "phi2" || spec.Architecture == "phimoe") && result.OutputBias == nil {
		return Weights{}, errors.New(`required tensor "output.bias" is missing`)
	}
	if spec.Architecture == "gemma-embedding" {
		if item, ok := tensors["dense_2.weight"]; ok {
			if spec.Dense2FeatureIn == 0 || spec.Dense2FeatureOut == 0 {
				return Weights{}, errors.New("Gemma embedding dense-2 tensor has no shape metadata")
			}
			validated, denseErr := required(
				item.Name, uint64(spec.Dense2FeatureIn), uint64(spec.Dense2FeatureOut),
			)
			if denseErr != nil {
				return Weights{}, denseErr
			}
			result.Dense2Output = &validated
		}
		if item, ok := tensors["dense_3.weight"]; ok {
			if spec.Dense3FeatureIn == 0 || spec.Dense3FeatureOut == 0 {
				return Weights{}, errors.New("Gemma embedding dense-3 tensor has no shape metadata")
			}
			validated, denseErr := required(
				item.Name, uint64(spec.Dense3FeatureIn), uint64(spec.Dense3FeatureOut),
			)
			if denseErr != nil {
				return Weights{}, denseErr
			}
			result.Dense3Output = &validated
		}
		if result.Dense2Output != nil && result.Dense3Output != nil &&
			result.Dense2Output.Shape[1] != result.Dense3Output.Shape[0] {
			return Weights{}, errors.New("Gemma embedding dense projection widths do not compose")
		}
	}
	if (spec.Architecture == "gemma4" || spec.Architecture == "gemma3n") && spec.EmbeddingPerLayer > 0 {
		perLayerTokenEmbedding, itemErr := required(
			"per_layer_token_embd.weight",
			uint64(spec.EmbeddingPerLayer)*uint64(spec.BlockCount),
			uint64(spec.VocabularySize),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		perLayerModelProjection, itemErr := required(
			"per_layer_model_proj.weight",
			uint64(spec.EmbeddingLength),
			uint64(spec.EmbeddingPerLayer)*uint64(spec.BlockCount),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		perLayerProjectionNorm, itemErr := required(
			"per_layer_proj_norm.weight", uint64(spec.EmbeddingPerLayer),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		result.PerLayerTokenEmbedding = &perLayerTokenEmbedding
		result.PerLayerModelProjection = &perLayerModelProjection
		result.PerLayerProjectionNorm = &perLayerProjectionNorm
	}
	if spec.Architecture == "gemma3n" {
		projection, itemErr := required(
			"altup_proj.weight", uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength), uint64(spec.AltUpCount-1),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		unembedding, itemErr := required(
			"altup_unembd_proj.weight", uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength), uint64(spec.AltUpCount-1),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		result.AltUpProjection, result.AltUpUnembedding = &projection, &unembedding
	}

	result.Layers = make([]LayerWeights, spec.BlockCount)
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		queryLength := uint64(spec.LayerHeadCount(block)) * uint64(spec.LayerKeyLength(block))
		keyLength := uint64(spec.LayerKVHeadCount(block)) * uint64(spec.LayerKeyLength(block))
		valueLength := uint64(spec.LayerKVHeadCount(block)) * uint64(spec.LayerValueLength(block))
		attentionOutputLength := uint64(spec.LayerHeadCount(block)) * uint64(spec.LayerValueLength(block))
		biasNames := []string{
			"attn_q.bias",
			"attn_k.bias",
			"attn_v.bias",
			"attn_output.bias",
			"ffn_gate.bias",
			"ffn_up.bias",
			"ffn_down.bias",
		}
		if spec.Architecture == "qwen3next" || spec.Architecture == "qwen35" || spec.Architecture == "qwen35moe" || spec.Architecture == "cogvlm" {
			for _, name := range biasNames {
				if _, ok := tensors[prefix+name]; ok {
					return Weights{}, fmt.Errorf(
						"tensor %q requires unsupported %s projection biases",
						prefix+name, spec.Architecture,
					)
				}
			}
		}
		for _, unsupported := range []string{
			"rope_factors_long.weight",
			"rope_factors_short.weight",
		} {
			if _, ok := tensors[prefix+unsupported]; ok {
				if supportsLongRoPE(spec.Architecture) {
					continue
				}
				return Weights{}, fmt.Errorf(
					"tensor %q requires an unsupported dense-decoder feature",
					prefix+unsupported,
				)
			}
		}
		layer := &result.Layers[block]
		if spec.Architecture == "mpt" {
			for _, unsupported := range []string{
				"attn_q_norm.weight", "attn_q_norm.bias",
				"attn_k_norm.weight", "attn_k_norm.bias", "ffn_act.scales",
			} {
				if _, ok := tensors[prefix+unsupported]; ok {
					return Weights{}, fmt.Errorf(
						"tensor %q requires an unsupported MPT variant", prefix+unsupported,
					)
				}
			}
		}
		ropeFactors, hasRopeFactors := tensors[prefix+"rope_freqs.weight"]
		if (spec.Architecture == "step35" || spec.Architecture == "gemma4" && !spec.IsSlidingLayer(block)) && !hasRopeFactors {
			ropeFactors, hasRopeFactors = tensors["rope_freqs.weight"]
		}
		if hasRopeFactors {
			if spec.Architecture == "qwen3next" || spec.Architecture == "qwen35" || spec.Architecture == "qwen35moe" {
				return Weights{}, fmt.Errorf(
					"tensor %q requires unsupported hybrid RoPE factors",
					ropeFactors.Name,
				)
			}
			rotaryDimensions := spec.KeyLength
			if spec.RopeDimensionCount > 0 {
				rotaryDimensions = spec.RopeDimensionCount
			}
			if rotaryDimensions%2 != 0 ||
				ropeFactors.Type != dtype.F32 ||
				ropeFactors.Dimensions != 1 ||
				ropeFactors.Shape[0] != uint64(rotaryDimensions/2) {
				return Weights{}, fmt.Errorf(
					"tensor %q has incompatible shape %v",
					ropeFactors.Name,
					ropeFactors.Shape,
				)
			}
			layer.RopeFactors = &ropeFactors
		}
		if supportsLongRoPE(spec.Architecture) {
			longName := prefix + "rope_factors_long.weight"
			shortName := prefix + "rope_factors_short.weight"
			if spec.Architecture == "step35" {
				longName = "rope_factors_long.weight"
				shortName = "rope_factors_short.weight"
			}
			longFactors, hasLong := tensors[longName]
			shortFactors, hasShort := tensors[shortName]
			if block > 0 {
				if !hasLong {
					longFactors, hasLong = tensors["blk.0.rope_factors_long.weight"]
				}
				if !hasShort {
					shortFactors, hasShort = tensors["blk.0.rope_factors_short.weight"]
				}
			}
			if hasLong != hasShort {
				return Weights{}, fmt.Errorf("%s LongRoPE factor tensors must both be present or absent", spec.Architecture)
			}
			if spec.RopeScalingType == "longrope" && !hasLong {
				return Weights{}, fmt.Errorf("%s LongRoPE factor tensors are missing", spec.Architecture)
			}
			if hasLong {
				for _, item := range []gguf.TensorInfo{longFactors, shortFactors} {
					if item.Type != dtype.F32 || item.Dimensions != 1 ||
						item.Shape[0] != uint64(spec.RopeDimensionCount/2) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
				}
				if spec.ContextLength > spec.OriginalContextLength {
					layer.RopeFactors = &longFactors
				} else {
					layer.RopeFactors = &shortFactors
				}
			}
		}
		if spec.Architecture == "modern-bert" {
			if item, ok := tensors[prefix+"attn_norm.weight"]; ok {
				if item.Dimensions != 1 || item.Shape[0] != uint64(spec.EmbeddingLength) {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
				}
				layer.AttentionNorm = item
			} else if block > 0 {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_norm.weight")
			}
		} else if spec.Architecture != "olmo2" && !usesPostOnlyNorm(spec.Architecture) &&
			(spec.Architecture != "deci" || spec.LayerHeadCount(block) > 0) &&
			!spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() {
			if layer.AttentionNorm, err = required(prefix+"attn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if spec.RequiresLayerNormBias() {
				attentionNormBias, biasErr := required(
					prefix+"attn_norm.bias",
					uint64(spec.EmbeddingLength),
				)
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionNormBias = &attentionNormBias
			}
		}
		if spec.Architecture == "kimi-linear" {
			layer.Recurrent = spec.IsRecurrentLayer(block)
			if layer.Recurrent {
				inner := uint64(spec.SSMInnerSize)
				for name, destination := range map[string]*gguf.TensorInfo{
					"attn_q.weight": &layer.AttentionQ, "attn_k.weight": &layer.AttentionK,
					"attn_v.weight": &layer.AttentionV, "attn_output.weight": &layer.AttentionOutput,
				} {
					shape := []uint64{uint64(spec.EmbeddingLength), inner}
					if name == "attn_output.weight" {
						shape = []uint64{inner, uint64(spec.EmbeddingLength)}
					}
					item, itemErr := required(prefix+name, shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*destination = item
				}
				conv := func(name string) (*gguf.TensorInfo, error) {
					item, ok := tensors[prefix+name]
					if !ok {
						return nil, fmt.Errorf("required tensor %q is missing", prefix+name)
					}
					valid3 := item.Dimensions == 3 && item.Shape[0] == uint64(spec.SSMConvKernel) && item.Shape[1] == 1 && item.Shape[2] == inner
					valid4 := item.Dimensions == 4 && item.Shape[0] == uint64(spec.SSMConvKernel) && item.Shape[1] == 1 && item.Shape[2] == inner && item.Shape[3] == 1
					if !valid3 && !valid4 {
						return nil, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
					return &item, nil
				}
				for name, destination := range map[string]**gguf.TensorInfo{
					"ssm_conv1d_q.weight": &layer.SSMQueryConv,
					"ssm_conv1d_k.weight": &layer.SSMKeyConv,
					"ssm_conv1d_v.weight": &layer.SSMValueConv,
				} {
					item, itemErr := conv(name)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*destination = item
				}
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ssm_f_a.weight":  {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.KDAHeadDim)}, &layer.SSMForgetA},
					"ssm_f_b.weight":  {[]uint64{uint64(spec.KDAHeadDim), inner}, &layer.SSMForgetB},
					"ssm_beta.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.HeadCount)}, &layer.SSMBeta},
					"ssm_dt.bias":     {[]uint64{inner}, &layer.SSMTimeStep},
					"ssm_g_a.weight":  {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.KDAHeadDim)}, &layer.SSMOutputGateA},
					"ssm_g_b.weight":  {[]uint64{uint64(spec.KDAHeadDim), inner}, &layer.SSMOutputGateB},
					"ssm_norm.weight": {[]uint64{uint64(spec.KDAHeadDim)}, &layer.SSMNorm},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
				ssmA, ok := tensors[prefix+"ssm_a"]
				if !ok {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"ssm_a")
				}
				validA2 := ssmA.Dimensions == 2 && ssmA.Shape[0] == 1 && ssmA.Shape[1] == uint64(spec.HeadCount)
				validA4 := ssmA.Dimensions == 4 && ssmA.Shape[0] == 1 && ssmA.Shape[1] == uint64(spec.HeadCount) && ssmA.Shape[2] == 1 && ssmA.Shape[3] == 1
				if !validA2 && !validA4 {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", ssmA.Name, ssmA.Shape)
				}
				layer.SSMA = &ssmA
			} else {
				if spec.QLoRARank > 0 {
					item, itemErr := required(prefix+"attn_q_a.weight", uint64(spec.EmbeddingLength), uint64(spec.QLoRARank))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionQ = item
					item, itemErr = required(prefix+"attn_q_a_norm.weight", uint64(spec.QLoRARank))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionQNorm = &item
					item, itemErr = required(prefix+"attn_q_b.weight", uint64(spec.QLoRARank), queryLength)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionQB = &item
				} else if layer.AttentionQ, err = required(prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength); err != nil {
					return Weights{}, err
				}
				if layer.AttentionOutput, err = required(prefix+"attn_output.weight", attentionOutputLength, uint64(spec.EmbeddingLength)); err != nil {
					return Weights{}, err
				}
				kvA, itemErr := required(prefix+"attn_kv_a_mqa.weight", uint64(spec.EmbeddingLength), uint64(spec.KVLoRARank+spec.RopeDimensionCount))
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.AttentionKVAMQA = &kvA
				kvNorm, itemErr := required(prefix+"attn_kv_a_norm.weight", uint64(spec.KVLoRARank))
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.AttentionKVANorm = &kvNorm
				nope := uint64(spec.KeyLength - spec.RopeDimensionCount)
				if item, ok := tensors[prefix+"attn_k_b.weight"]; ok {
					validated, itemErr := required(item.Name, nope, uint64(spec.KVLoRARank), uint64(spec.HeadCount))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionKB = &validated
					validated, itemErr = required(prefix+"attn_v_b.weight", uint64(spec.KVLoRARank), uint64(spec.ValueLength), uint64(spec.HeadCount))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionVB = &validated
				} else {
					validated, itemErr := required(prefix+"attn_kv_b.weight", uint64(spec.KVLoRARank), uint64(spec.HeadCount)*(nope+uint64(spec.ValueLength)))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionKVB = &validated
				}
			}
			if layer.FeedForwardNorm, err = required(prefix+"ffn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if block < spec.LeadingDenseBlocks {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination *gguf.TensorInfo
				}{
					"ffn_gate.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}, &layer.FeedForwardGate},
					"ffn_up.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}, &layer.FeedForwardUp},
					"ffn_down.weight": {[]uint64{uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)}, &layer.FeedForwardDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = item
				}
			} else {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_inp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)}, &layer.FeedForwardRouter},
					"ffn_gate_exps.weight":  {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)}, &layer.FeedForwardGateExperts},
					"ffn_up_exps.weight":    {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)}, &layer.FeedForwardUpExperts},
					"ffn_down_exps.weight":  {[]uint64{uint64(spec.ExpertFeedForward), uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)}, &layer.FeedForwardDownExperts},
					"exp_probs_b.bias":      {[]uint64{uint64(spec.ExpertCount)}, &layer.FeedForwardExpertBias},
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			continue
		}
		if spec.Architecture == "rwkv6" {
			layer.Recurrent = true
			embedding := uint64(spec.EmbeddingLength)
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"attn_norm_2.weight":            {[]uint64{embedding}, &layer.AttentionNorm2},
				"attn_norm_2.bias":              {[]uint64{embedding}, &layer.AttentionNorm2Bias},
				"time_mix_w1.weight":            {[]uint64{embedding, uint64(spec.TimeMixExtraDim) * 5}, &layer.TimeMixW1},
				"time_mix_w2.weight":            {[]uint64{uint64(spec.TimeMixExtraDim), embedding, 5}, &layer.TimeMixW2},
				"time_mix_lerp_x.weight":        {[]uint64{embedding, 1, 1}, &layer.TimeMixLerpX},
				"time_mix_first.weight":         {[]uint64{uint64(spec.WKVHeadSize), uint64(spec.HeadCount)}, &layer.TimeMixFirst},
				"time_mix_decay.weight":         {[]uint64{embedding}, &layer.TimeMixDecay},
				"time_mix_decay_w1.weight":      {[]uint64{embedding, uint64(spec.TimeDecayExtraDim)}, &layer.TimeMixDecayW1},
				"time_mix_decay_w2.weight":      {[]uint64{uint64(spec.TimeDecayExtraDim), embedding}, &layer.TimeMixDecayW2},
				"time_mix_key.weight":           {[]uint64{embedding, embedding}, &layer.TimeMixKey},
				"time_mix_value.weight":         {[]uint64{embedding, embedding}, &layer.TimeMixValue},
				"time_mix_receptance.weight":    {[]uint64{embedding, embedding}, &layer.TimeMixReceptance},
				"time_mix_gate.weight":          {[]uint64{embedding, embedding}, &layer.TimeMixGate},
				"time_mix_ln.weight":            {[]uint64{embedding}, &layer.TimeMixLN},
				"time_mix_ln.bias":              {[]uint64{embedding}, &layer.TimeMixLNBias},
				"time_mix_output.weight":        {[]uint64{embedding, embedding}, &layer.TimeMixOutput},
				"channel_mix_lerp_k.weight":     {[]uint64{embedding, 1, 1}, &layer.ChannelMixLerpK},
				"channel_mix_lerp_r.weight":     {[]uint64{embedding, 1, 1}, &layer.ChannelMixLerpR},
				"channel_mix_key.weight":        {[]uint64{embedding, uint64(spec.FeedForwardLength)}, &layer.ChannelMixKey},
				"channel_mix_value.weight":      {[]uint64{uint64(spec.FeedForwardLength), embedding}, &layer.ChannelMixValue},
				"channel_mix_receptance.weight": {[]uint64{embedding, embedding}, &layer.ChannelMixReceptance},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if item, ok := tensors[prefix+"time_mix_lerp_fused.weight"]; ok {
				validated, itemErr := required(item.Name, embedding, 1, 1, 5)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.TimeMixLerpFused = &validated
			} else {
				for suffix, destination := range map[string]**gguf.TensorInfo{
					"w": &layer.TimeMixLerpW, "k": &layer.TimeMixLerpK,
					"v": &layer.TimeMixLerpV, "r": &layer.TimeMixLerpR,
					"g": &layer.TimeMixLerpG,
				} {
					item, itemErr := required(prefix+"time_mix_lerp_"+suffix+".weight", embedding, 1, 1)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*destination = &item
				}
			}
			continue
		}
		if spec.Architecture == "rwkv7" || spec.Architecture == "arwkv7" {
			layer.Recurrent = true
			embedding := uint64(spec.EmbeddingLength)
			valueRank := uint64(spec.ValueMixLoRARank)
			if block == 0 {
				valueRank = uint64(spec.ICLRLoRARank)
			}
			lerpCount := uint64(6)
			if spec.Architecture == "arwkv7" && spec.GateLoRARank == 0 {
				lerpCount = 5
			}
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"time_mix_w0.weight":         {[]uint64{embedding}, &layer.TimeMixW0},
				"time_mix_w1.weight":         {[]uint64{embedding, uint64(spec.DecayLoRARank)}, &layer.TimeMixW1},
				"time_mix_w2.weight":         {[]uint64{uint64(spec.DecayLoRARank), embedding}, &layer.TimeMixW2},
				"time_mix_a0.weight":         {[]uint64{embedding}, &layer.TimeMixA0},
				"time_mix_a1.weight":         {[]uint64{embedding, uint64(spec.ICLRLoRARank)}, &layer.TimeMixA1},
				"time_mix_a2.weight":         {[]uint64{uint64(spec.ICLRLoRARank), embedding}, &layer.TimeMixA2},
				"time_mix_v0.weight":         {[]uint64{embedding}, &layer.TimeMixV0},
				"time_mix_v1.weight":         {[]uint64{embedding, valueRank}, &layer.TimeMixV1},
				"time_mix_v2.weight":         {[]uint64{valueRank, embedding}, &layer.TimeMixV2},
				"time_mix_lerp_fused.weight": {[]uint64{embedding, 1, 1, lerpCount}, &layer.TimeMixLerpFused},
				"time_mix_k_k.weight":        {[]uint64{embedding}, &layer.TimeMixKK},
				"time_mix_k_a.weight":        {[]uint64{embedding}, &layer.TimeMixKA},
				"time_mix_r_k.weight":        {[]uint64{embedding}, &layer.TimeMixRK},
				"time_mix_key.weight":        {[]uint64{embedding, embedding}, &layer.TimeMixKey},
				"time_mix_value.weight":      {[]uint64{embedding, embedding}, &layer.TimeMixValue},
				"time_mix_receptance.weight": {[]uint64{embedding, embedding}, &layer.TimeMixReceptance},
				"time_mix_output.weight":     {[]uint64{embedding, embedding}, &layer.TimeMixOutput},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if spec.GateLoRARank > 0 {
				g1, itemErr := required(prefix+"time_mix_g1.weight", embedding, uint64(spec.GateLoRARank))
				if itemErr != nil {
					return Weights{}, itemErr
				}
				g2, itemErr := required(prefix+"time_mix_g2.weight", uint64(spec.GateLoRARank), embedding)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.TimeMixG1, layer.TimeMixG2 = &g1, &g2
			}
			if spec.Architecture == "rwkv7" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"attn_norm_2.weight":        {[]uint64{embedding}, &layer.AttentionNorm2},
					"attn_norm_2.bias":          {[]uint64{embedding}, &layer.AttentionNorm2Bias},
					"time_mix_ln.weight":        {[]uint64{embedding}, &layer.TimeMixLN},
					"time_mix_ln.bias":          {[]uint64{embedding}, &layer.TimeMixLNBias},
					"channel_mix_lerp_k.weight": {[]uint64{embedding, 1, 1}, &layer.ChannelMixLerpK},
					"channel_mix_key.weight":    {[]uint64{embedding, uint64(spec.FeedForwardLength)}, &layer.ChannelMixKey},
					"channel_mix_value.weight":  {[]uint64{uint64(spec.FeedForwardLength), embedding}, &layer.ChannelMixValue},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			} else {
				if item, ok := tensors[prefix+"time_mix_ln.weight"]; ok {
					norm, itemErr := required(item.Name, embedding)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					bias, itemErr := required(prefix+"time_mix_ln.bias", embedding)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.TimeMixLN, layer.TimeMixLNBias = &norm, &bias
				}
				if layer.FeedForwardNorm, err = required(prefix+"ffn_norm.weight", embedding); err != nil {
					return Weights{}, err
				}
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination *gguf.TensorInfo
				}{
					"ffn_gate.weight": {[]uint64{embedding, uint64(spec.FeedForwardLength)}, &layer.FeedForwardGate},
					"ffn_up.weight":   {[]uint64{embedding, uint64(spec.FeedForwardLength)}, &layer.FeedForwardUp},
					"ffn_down.weight": {[]uint64{uint64(spec.FeedForwardLength), embedding}, &layer.FeedForwardDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = item
				}
			}
			continue
		}
		if spec.Architecture == "rwkv6qwen2" {
			layer.Recurrent = true
			embedding := uint64(spec.EmbeddingLength)
			keyValue := uint64(spec.HeadCountKV) * uint64(spec.WKVHeadSize)
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"time_mix_w1.weight":         {[]uint64{embedding, uint64(spec.TimeMixExtraDim) * 5}, &layer.TimeMixW1},
				"time_mix_w2.weight":         {[]uint64{uint64(spec.TimeMixExtraDim), embedding, 5}, &layer.TimeMixW2},
				"time_mix_lerp_x.weight":     {[]uint64{embedding, 1, 1}, &layer.TimeMixLerpX},
				"time_mix_lerp_fused.weight": {[]uint64{embedding, 1, 1, 5}, &layer.TimeMixLerpFused},
				"time_mix_decay.weight":      {[]uint64{embedding}, &layer.TimeMixDecay},
				"time_mix_decay_w1.weight":   {[]uint64{embedding, uint64(spec.TimeDecayExtraDim)}, &layer.TimeMixDecayW1},
				"time_mix_decay_w2.weight":   {[]uint64{uint64(spec.TimeDecayExtraDim), embedding}, &layer.TimeMixDecayW2},
				"time_mix_key.weight":        {[]uint64{embedding, keyValue}, &layer.TimeMixKey},
				"time_mix_value.weight":      {[]uint64{embedding, keyValue}, &layer.TimeMixValue},
				"time_mix_receptance.weight": {[]uint64{embedding, embedding}, &layer.TimeMixReceptance},
				"time_mix_gate.weight":       {[]uint64{embedding, embedding}, &layer.TimeMixGate},
				"time_mix_output.weight":     {[]uint64{embedding, embedding}, &layer.TimeMixOutput},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			for name, widthAndDestination := range map[string]struct {
				width       uint64
				destination **gguf.TensorInfo
			}{
				"time_mix_key.bias":        {keyValue, &layer.AttentionKBias},
				"time_mix_value.bias":      {keyValue, &layer.AttentionVBias},
				"time_mix_receptance.bias": {embedding, &layer.AttentionQBias},
			} {
				if item, ok := tensors[prefix+name]; ok {
					validated, itemErr := required(item.Name, widthAndDestination.width)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*widthAndDestination.destination = &validated
				}
			}
			if layer.FeedForwardNorm, err = required(prefix+"ffn_norm.weight", embedding); err != nil {
				return Weights{}, err
			}
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination *gguf.TensorInfo
			}{
				"ffn_gate.weight": {[]uint64{embedding, uint64(spec.FeedForwardLength)}, &layer.FeedForwardGate},
				"ffn_up.weight":   {[]uint64{embedding, uint64(spec.FeedForwardLength)}, &layer.FeedForwardUp},
				"ffn_down.weight": {[]uint64{uint64(spec.FeedForwardLength), embedding}, &layer.FeedForwardDown},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = item
			}
			continue
		}
		if spec.Architecture == "nemotron_h" || spec.Architecture == "nemotron_h_moe" {
			if spec.IsRecurrentLayer(block) {
				layer.Recurrent = true
				convDimension := uint64(spec.SSMInnerSize) +
					2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
				inputDimension := uint64(spec.SSMInnerSize) + convDimension + uint64(spec.SSMTimeStepRank)
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ssm_in.weight":     {[]uint64{uint64(spec.EmbeddingLength), inputDimension}, &layer.SSMInput},
					"ssm_conv1d.weight": {[]uint64{uint64(spec.SSMConvKernel), convDimension}, &layer.SSMConv1D},
					"ssm_dt.bias":       {[]uint64{uint64(spec.SSMTimeStepRank)}, &layer.SSMTimeStep},
					"ssm_a":             {[]uint64{1, uint64(spec.SSMTimeStepRank)}, &layer.SSMA},
					"ssm_d":             {[]uint64{1, uint64(spec.SSMTimeStepRank)}, &layer.SSMD},
					"ssm_norm.weight":   {[]uint64{uint64(spec.SSMInnerSize / spec.SSMGroupCount), uint64(spec.SSMGroupCount)}, &layer.SSMNorm},
					"ssm_out.weight":    {[]uint64{uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)}, &layer.SSMOutput},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
				if item, ok := tensors[prefix+"ssm_conv1d.bias"]; ok {
					if item.Dimensions != 1 || item.Shape[0] != convDimension {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
					layer.SSMConv1DBias = &item
				}
				continue
			}
			if spec.LayerFeedForwardLength(block) == 0 {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination *gguf.TensorInfo
				}{
					"attn_q.weight":      {[]uint64{uint64(spec.EmbeddingLength), queryLength}, &layer.AttentionQ},
					"attn_k.weight":      {[]uint64{uint64(spec.EmbeddingLength), keyLength}, &layer.AttentionK},
					"attn_v.weight":      {[]uint64{uint64(spec.EmbeddingLength), valueLength}, &layer.AttentionV},
					"attn_output.weight": {[]uint64{attentionOutputLength, uint64(spec.EmbeddingLength)}, &layer.AttentionOutput},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = item
				}
				for name, widthAndDestination := range map[string]struct {
					width       uint64
					destination **gguf.TensorInfo
				}{
					"attn_q.bias":      {queryLength, &layer.AttentionQBias},
					"attn_k.bias":      {keyLength, &layer.AttentionKBias},
					"attn_v.bias":      {valueLength, &layer.AttentionVBias},
					"attn_output.bias": {uint64(spec.EmbeddingLength), &layer.AttentionOutputBias},
				} {
					if item, ok := tensors[prefix+name]; ok {
						if item.Type != dtype.F32 || item.Dimensions != 1 || item.Shape[0] != widthAndDestination.width {
							return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
						}
						*widthAndDestination.destination = &item
					}
				}
				continue
			}
			if spec.Architecture == "nemotron_h" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination *gguf.TensorInfo
				}{
					"ffn_up.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.LayerFeedForwardLength(block))}, &layer.FeedForwardUp},
					"ffn_down.weight": {[]uint64{uint64(spec.LayerFeedForwardLength(block)), uint64(spec.EmbeddingLength)}, &layer.FeedForwardDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = item
				}
				for name, widthAndDestination := range map[string]struct {
					width       uint64
					destination **gguf.TensorInfo
				}{
					"ffn_up.bias":   {uint64(spec.LayerFeedForwardLength(block)), &layer.FeedForwardUpBias},
					"ffn_down.bias": {uint64(spec.EmbeddingLength), &layer.FeedForwardDownBias},
				} {
					if item, ok := tensors[prefix+name]; ok {
						if item.Type != dtype.F32 || item.Dimensions != 1 || item.Shape[0] != widthAndDestination.width {
							return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
						}
						*widthAndDestination.destination = &item
					}
				}
				continue
			}
			moeWidth := uint64(spec.EmbeddingLength)
			if spec.MoELatentSize > 0 {
				moeWidth = uint64(spec.MoELatentSize)
				latentDown, latentErr := required(prefix+"ffn_latent_down.weight", uint64(spec.EmbeddingLength), moeWidth)
				if latentErr != nil {
					return Weights{}, latentErr
				}
				latentUp, latentErr := required(prefix+"ffn_latent_up.weight", moeWidth, uint64(spec.EmbeddingLength))
				if latentErr != nil {
					return Weights{}, latentErr
				}
				layer.FeedForwardLatentDown = &latentDown
				layer.FeedForwardLatentUp = &latentUp
			}
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ffn_gate_inp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)}, &layer.FeedForwardRouter},
				"exp_probs_b.bias":      {[]uint64{uint64(spec.ExpertCount)}, &layer.FeedForwardExpertBias},
				"ffn_up_exps.weight":    {[]uint64{moeWidth, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)}, &layer.FeedForwardUpExperts},
				"ffn_down_exps.weight":  {[]uint64{uint64(spec.ExpertFeedForward), moeWidth, uint64(spec.ExpertCount)}, &layer.FeedForwardDownExperts},
				"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
				"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				if name == "exp_probs_b.bias" && item.Type != dtype.F32 {
					return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", item.Name)
				}
				*shapeAndDestination.destination = &item
			}
			continue
		}
		if spec.Architecture == "falcon" {
			if item, ok := tensors[prefix+"attn_norm_2.weight"]; ok {
				if item.Dimensions != 1 || item.Shape[0] != uint64(spec.EmbeddingLength) {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
				}
				layer.AttentionNorm2 = &item
				if bias, hasBias := tensors[prefix+"attn_norm_2.bias"]; hasBias {
					if bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.EmbeddingLength) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.AttentionNorm2Bias = &bias
				}
			} else if _, ok := tensors[prefix+"attn_norm_2.bias"]; ok {
				return Weights{}, errors.New("Falcon secondary attention norm bias has no weight")
			}
		}
		if spec.Architecture == "bitnet" {
			attentionSubNorm, subNormErr := required(
				prefix+"attn_sub_norm.weight", uint64(spec.EmbeddingLength),
			)
			if subNormErr != nil {
				return Weights{}, subNormErr
			}
			feedForwardSubNorm, subNormErr := required(
				prefix+"ffn_sub_norm.weight", uint64(spec.FeedForwardLength),
			)
			if subNormErr != nil {
				return Weights{}, subNormErr
			}
			layer.AttentionSubNorm = &attentionSubNorm
			layer.FeedForwardSubNorm = &feedForwardSubNorm
			for name, destination := range map[string]**gguf.TensorInfo{
				"attn_q.scale":      &layer.AttentionQScale,
				"attn_k.scale":      &layer.AttentionKScale,
				"attn_v.scale":      &layer.AttentionVScale,
				"attn_output.scale": &layer.AttentionOutputScale,
				"ffn_gate.scale":    &layer.FeedForwardGateScale,
				"ffn_up.scale":      &layer.FeedForwardUpScale,
				"ffn_down.scale":    &layer.FeedForwardDownScale,
			} {
				if item, ok := tensors[prefix+name]; ok {
					if item.Type != dtype.F32 || item.Dimensions != 1 || item.Shape[0] != 1 {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
					*destination = &item
				}
			}
		}
		if spec.Architecture == "deci" && spec.LayerHeadCount(block) == 0 {
			// Attention-free layer.
		} else if spec.Architecture == "deci" && spec.LayerKVHeadCount(block) == 0 {
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight",
				uint64(spec.EmbeddingLength),
				uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else if spec.Architecture == "jamba" && spec.IsRecurrentLayer(block) {
			layer.Recurrent = true
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ssm_in.weight":      {[]uint64{uint64(spec.EmbeddingLength), 2 * uint64(spec.SSMInnerSize)}, &layer.SSMInput},
				"ssm_conv1d.weight":  {[]uint64{uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)}, &layer.SSMConv1D},
				"ssm_conv1d.bias":    {[]uint64{uint64(spec.SSMInnerSize)}, &layer.SSMConv1DBias},
				"ssm_x.weight":       {[]uint64{uint64(spec.SSMInnerSize), uint64(spec.SSMTimeStepRank + 2*spec.SSMStateSize)}, &layer.SSMX},
				"ssm_dt_norm.weight": {[]uint64{uint64(spec.SSMTimeStepRank)}, &layer.SSMTimeStepNorm},
				"ssm_dt.weight":      {[]uint64{uint64(spec.SSMTimeStepRank), uint64(spec.SSMInnerSize)}, &layer.SSMTimeStepWeight},
				"ssm_dt.bias":        {[]uint64{uint64(spec.SSMInnerSize)}, &layer.SSMTimeStep},
				"ssm_b_norm.weight":  {[]uint64{uint64(spec.SSMStateSize)}, &layer.SSMBNorm},
				"ssm_c_norm.weight":  {[]uint64{uint64(spec.SSMStateSize)}, &layer.SSMCNorm},
				"ssm_a":              {[]uint64{uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)}, &layer.SSMA},
				"ssm_d":              {[]uint64{uint64(spec.SSMInnerSize)}, &layer.SSMD},
				"ssm_out.weight":     {[]uint64{uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)}, &layer.SSMOutput},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		} else if spec.Architecture == "plamo2" && spec.IsRecurrentLayer(block) {
			layer.Recurrent = true
			dtDimension := uint64(64)
			if candidate := uint64(spec.EmbeddingLength / 16); candidate > dtDimension {
				dtDimension = candidate
			}
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ssm_in.weight":      {[]uint64{uint64(spec.EmbeddingLength), 2 * uint64(spec.SSMInnerSize)}, &layer.SSMInput},
				"ssm_conv1d.weight":  {[]uint64{uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)}, &layer.SSMConv1D},
				"ssm_x.weight":       {[]uint64{uint64(spec.SSMInnerSize), dtDimension + 2*uint64(spec.SSMStateSize)}, &layer.SSMX},
				"ssm_dt.weight":      {[]uint64{dtDimension, uint64(spec.SSMTimeStepRank)}, &layer.SSMTimeStepWeight},
				"ssm_dt.bias":        {[]uint64{uint64(spec.SSMTimeStepRank)}, &layer.SSMTimeStep},
				"ssm_a":              {[]uint64{uint64(spec.SSMTimeStepRank)}, &layer.SSMA},
				"ssm_d":              {[]uint64{uint64(spec.SSMTimeStepRank)}, &layer.SSMD},
				"ssm_out.weight":     {[]uint64{uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)}, &layer.SSMOutput},
				"ssm_dt_norm.weight": {[]uint64{dtDimension}, &layer.SSMTimeStepNorm},
				"ssm_b_norm.weight":  {[]uint64{uint64(spec.SSMStateSize)}, &layer.SSMBNorm},
				"ssm_c_norm.weight":  {[]uint64{uint64(spec.SSMStateSize)}, &layer.SSMCNorm},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		} else if spec.Architecture == "mamba" {
			layer.Recurrent = true
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ssm_in.weight": {
					[]uint64{uint64(spec.EmbeddingLength), 2 * uint64(spec.SSMInnerSize)}, &layer.SSMInput,
				},
				"ssm_conv1d.weight": {
					[]uint64{uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)}, &layer.SSMConv1D,
				},
				"ssm_conv1d.bias": {
					[]uint64{uint64(spec.SSMInnerSize)}, &layer.SSMConv1DBias,
				},
				"ssm_x.weight": {
					[]uint64{uint64(spec.SSMInnerSize), uint64(spec.SSMTimeStepRank + 2*spec.SSMStateSize)}, &layer.SSMX,
				},
				"ssm_dt.weight": {
					[]uint64{uint64(spec.SSMTimeStepRank), uint64(spec.SSMInnerSize)}, &layer.SSMTimeStepWeight,
				},
				"ssm_dt.bias": {
					[]uint64{uint64(spec.SSMInnerSize)}, &layer.SSMTimeStep,
				},
				"ssm_a": {
					[]uint64{uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)}, &layer.SSMA,
				},
				"ssm_d": {
					[]uint64{uint64(spec.SSMInnerSize)}, &layer.SSMD,
				},
				"ssm_out.weight": {
					[]uint64{uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)}, &layer.SSMOutput,
				},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		} else if spec.Architecture == "falcon-h1" {
			convDimension := uint64(spec.SSMInnerSize) +
				2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
			inputDimension := uint64(spec.SSMInnerSize) + convDimension + uint64(spec.SSMTimeStepRank)
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ssm_in.weight":     {[]uint64{uint64(spec.EmbeddingLength), inputDimension}, &layer.SSMInput},
				"ssm_conv1d.weight": {[]uint64{uint64(spec.SSMConvKernel), convDimension}, &layer.SSMConv1D},
				"ssm_dt.bias":       {[]uint64{uint64(spec.SSMTimeStepRank)}, &layer.SSMTimeStep},
				"ssm_a":             {[]uint64{1, uint64(spec.SSMTimeStepRank)}, &layer.SSMA},
				"ssm_d":             {[]uint64{1, uint64(spec.SSMTimeStepRank)}, &layer.SSMD},
				"ssm_out.weight":    {[]uint64{uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)}, &layer.SSMOutput},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if item, ok := tensors[prefix+"ssm_conv1d.bias"]; ok {
				if item.Type != dtype.F32 || item.Dimensions != 1 || item.Shape[0] != convDimension {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
				}
				layer.SSMConv1DBias = &item
			}
			if item, ok := tensors[prefix+"ssm_norm.weight"]; ok {
				norm, itemErr := required(
					item.Name, uint64(spec.SSMInnerSize/spec.SSMGroupCount), uint64(spec.SSMGroupCount),
				)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.SSMNorm = &norm
			}
			if item, ok := tensors[prefix+"attn_qkv.weight"]; ok {
				qkv, itemErr := required(item.Name, uint64(spec.EmbeddingLength), queryLength+keyLength+valueLength)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.AttentionQKV = &qkv
				if bias, ok := tensors[prefix+"attn_qkv.bias"]; ok {
					validated, biasErr := required(bias.Name, queryLength+keyLength+valueLength)
					if biasErr != nil {
						return Weights{}, biasErr
					}
					if validated.Type != dtype.F32 {
						return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", validated.Name)
					}
					layer.AttentionQKVBias = &validated
				}
			} else {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination *gguf.TensorInfo
				}{
					"attn_q.weight": {[]uint64{uint64(spec.EmbeddingLength), queryLength}, &layer.AttentionQ},
					"attn_k.weight": {[]uint64{uint64(spec.EmbeddingLength), keyLength}, &layer.AttentionK},
					"attn_v.weight": {[]uint64{uint64(spec.EmbeddingLength), valueLength}, &layer.AttentionV},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = item
				}
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight", attentionOutputLength, uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else if spec.Architecture == "mamba2" ||
			(spec.Architecture == "granitehybrid" && spec.IsRecurrentLayer(block)) {
			layer.Recurrent = true
			convDimension := uint64(spec.SSMInnerSize) +
				2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
			inputDimension := uint64(spec.SSMInnerSize) + convDimension + uint64(spec.SSMTimeStepRank)
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ssm_in.weight": {
					[]uint64{uint64(spec.EmbeddingLength), inputDimension}, &layer.SSMInput,
				},
				"ssm_conv1d.weight": {
					[]uint64{uint64(spec.SSMConvKernel), convDimension}, &layer.SSMConv1D,
				},
				"ssm_dt.bias": {
					[]uint64{uint64(spec.SSMTimeStepRank)}, &layer.SSMTimeStep,
				},
				"ssm_a": {
					[]uint64{1, uint64(spec.SSMTimeStepRank)}, &layer.SSMA,
				},
				"ssm_d": {
					[]uint64{1, uint64(spec.SSMTimeStepRank)}, &layer.SSMD,
				},
				"ssm_norm.weight": {
					[]uint64{uint64(spec.SSMInnerSize / spec.SSMGroupCount), uint64(spec.SSMGroupCount)}, &layer.SSMNorm,
				},
				"ssm_out.weight": {
					[]uint64{uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)}, &layer.SSMOutput,
				},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if item, ok := tensors[prefix+"ssm_conv1d.bias"]; ok {
				if item.Dimensions != 1 || item.Shape[0] != convDimension {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
				}
				layer.SSMConv1DBias = &item
			} else if spec.Architecture == "mamba2" {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"ssm_conv1d.bias")
			}
		} else if spec.Architecture == "qwen3next" || spec.Architecture == "qwen35" || spec.Architecture == "qwen35moe" {
			layer.Recurrent = spec.IsRecurrentLayer(block)
			if layer.Recurrent {
				keyDimension := uint64(spec.SSMStateSize) * uint64(spec.SSMGroupCount)
				valueDimension := uint64(spec.SSMInnerSize)
				if spec.Architecture == "qwen3next" {
					if _, ok := tensors[prefix+"attn_qkv.weight"]; ok {
						qkv, qkvErr := required(
							prefix+"attn_qkv.weight", uint64(spec.EmbeddingLength),
							keyDimension*2+valueDimension,
						)
						if qkvErr != nil {
							return Weights{}, qkvErr
						}
						layer.AttentionQKV = &qkv
						attentionGate, gateErr := required(
							prefix+"attn_gate.weight", uint64(spec.EmbeddingLength), valueDimension,
						)
						if gateErr != nil {
							return Weights{}, gateErr
						}
						layer.AttentionGate = &attentionGate
					} else {
						qkvz, qkvzErr := required(
							prefix+"ssm_in.weight", uint64(spec.EmbeddingLength),
							keyDimension*2+valueDimension*2,
						)
						if qkvzErr != nil {
							return Weights{}, qkvzErr
						}
						layer.AttentionQKV = &qkvz
					}
					betaAlpha, betaAlphaErr := required(
						prefix+"ssm_ba.weight", uint64(spec.EmbeddingLength),
						2*uint64(spec.SSMTimeStepRank),
					)
					if betaAlphaErr != nil {
						return Weights{}, betaAlphaErr
					}
					layer.SSMBetaAlpha = &betaAlpha
				} else {
					qkv, qkvErr := required(
						prefix+"attn_qkv.weight", uint64(spec.EmbeddingLength),
						keyDimension*2+valueDimension,
					)
					if qkvErr != nil {
						return Weights{}, qkvErr
					}
					layer.AttentionQKV = &qkv
					attentionGate, gateErr := required(
						prefix+"attn_gate.weight", uint64(spec.EmbeddingLength), valueDimension,
					)
					if gateErr != nil {
						return Weights{}, gateErr
					}
					layer.AttentionGate = &attentionGate
				}
				for name, shape := range map[string][]uint64{
					"ssm_conv1d.weight": {
						uint64(spec.SSMConvKernel),
						keyDimension*2 + valueDimension,
					},
					"ssm_dt.bias":     {uint64(spec.SSMTimeStepRank)},
					"ssm_a":           {uint64(spec.SSMTimeStepRank)},
					"ssm_norm.weight": {uint64(spec.SSMStateSize)},
					"ssm_out.weight":  {valueDimension, uint64(spec.EmbeddingLength)},
				} {
					item, itemErr := required(prefix+name, shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					switch name {
					case "ssm_conv1d.weight":
						layer.SSMConv1D = &item
					case "ssm_dt.bias":
						layer.SSMTimeStep = &item
					case "ssm_a":
						layer.SSMA = &item
					case "ssm_norm.weight":
						layer.SSMNorm = &item
					case "ssm_out.weight":
						layer.SSMOutput = &item
					}
				}
				if spec.Architecture != "qwen3next" {
					for name, destination := range map[string]**gguf.TensorInfo{
						"ssm_beta.weight":  &layer.SSMBeta,
						"ssm_alpha.weight": &layer.SSMAlpha,
					} {
						item, itemErr := required(
							prefix+name, uint64(spec.EmbeddingLength), uint64(spec.SSMTimeStepRank),
						)
						if itemErr != nil {
							return Weights{}, itemErr
						}
						*destination = &item
					}
				}
			} else {
				if layer.AttentionQ, err = required(
					prefix+"attn_q.weight",
					uint64(spec.EmbeddingLength),
					queryLength*2,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionK, err = required(
					prefix+"attn_k.weight",
					uint64(spec.EmbeddingLength),
					keyLength,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionV, err = required(
					prefix+"attn_v.weight",
					uint64(spec.EmbeddingLength),
					valueLength,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionOutput, err = required(
					prefix+"attn_output.weight",
					attentionOutputLength,
					uint64(spec.EmbeddingLength),
				); err != nil {
					return Weights{}, err
				}
			}
		} else if (spec.Architecture == "lfm2" || spec.Architecture == "lfm2moe") && spec.IsRecurrentLayer(block) {
			layer.Recurrent = true
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"shortconv.conv.weight": {
					[]uint64{uint64(spec.ShortConvCacheLength), uint64(spec.EmbeddingLength)},
					&layer.ShortConvKernel,
				},
				"shortconv.in_proj.weight": {
					[]uint64{uint64(spec.EmbeddingLength), 3 * uint64(spec.EmbeddingLength)},
					&layer.ShortConvInput,
				},
				"shortconv.out_proj.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)},
					&layer.ShortConvOutput,
				},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		} else if spec.Architecture == "gemma4" || spec.Architecture == "gemma3n" {
			if layer.AttentionQ, err = required(
				prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength,
			); err != nil {
				return Weights{}, err
			}
			if spec.LayerHasKV(block) {
				if layer.AttentionK, err = required(
					prefix+"attn_k.weight", uint64(spec.EmbeddingLength), keyLength,
				); err != nil {
					return Weights{}, err
				}
				if item, ok := tensors[prefix+"attn_v.weight"]; ok {
					value, valueErr := required(item.Name, uint64(spec.EmbeddingLength), valueLength)
					if valueErr != nil {
						return Weights{}, valueErr
					}
					layer.AttentionV = value
				} else if spec.Architecture == "gemma3n" {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_v.weight")
				}
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight", attentionOutputLength, uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else if isMLAArchitecture(spec.Architecture) {
			nope := uint64(spec.KeyLength - spec.RopeDimensionCount)
			if spec.Architecture == "minicpm3" || ((isDeepSeek2Family(spec.Architecture) || spec.Architecture == "glm-dsa") && spec.QLoRARank > 0) {
				if layer.AttentionQ, err = required(
					prefix+"attn_q_a.weight", uint64(spec.EmbeddingLength), uint64(spec.QLoRARank),
				); err != nil {
					return Weights{}, err
				}
				qB, qBErr := required(prefix+"attn_q_b.weight", uint64(spec.QLoRARank), queryLength)
				if qBErr != nil {
					return Weights{}, qBErr
				}
				layer.AttentionQB = &qB
				qNorm, qNormErr := required(prefix+"attn_q_a_norm.weight", uint64(spec.QLoRARank))
				if qNormErr != nil {
					return Weights{}, qNormErr
				}
				layer.AttentionQNorm = &qNorm
			} else if layer.AttentionQ, err = required(
				prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength,
			); err != nil {
				return Weights{}, err
			}
			mlaTensors := map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"attn_kv_a_mqa.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.KVLoRARank + spec.RopeDimensionCount)},
					&layer.AttentionKVAMQA,
				},
				"attn_kv_a_norm.weight": {
					[]uint64{uint64(spec.KVLoRARank)}, &layer.AttentionKVANorm,
				},
			}
			if _, modern := tensors[prefix+"attn_k_b.weight"]; modern {
				mlaTensors["attn_k_b.weight"] = struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{[]uint64{nope, uint64(spec.KVLoRARank), uint64(spec.HeadCount)}, &layer.AttentionKB}
				mlaTensors["attn_v_b.weight"] = struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{[]uint64{uint64(spec.KVLoRARank), uint64(spec.ValueLength), uint64(spec.HeadCount)}, &layer.AttentionVB}
			} else {
				mlaTensors["attn_kv_b.weight"] = struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{[]uint64{uint64(spec.KVLoRARank), uint64(spec.HeadCount) * (nope + uint64(spec.ValueLength))}, &layer.AttentionKVB}
			}
			for name, shapeAndDestination := range mlaTensors {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if spec.LayerHasFullIndexer(block) {
				indexerTensors := map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"indexer.k_norm.weight": {[]uint64{uint64(spec.IndexerKeyLength)}, &layer.IndexerKNorm},
					"indexer.k_norm.bias":   {[]uint64{uint64(spec.IndexerKeyLength)}, &layer.IndexerKNormBias},
					"indexer.proj.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.IndexerHeadCount)},
						&layer.IndexerProjection},
					"indexer.attn_k.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.IndexerKeyLength)},
						&layer.IndexerAttentionK},
					"indexer.attn_q_b.weight": {[]uint64{uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount) * uint64(spec.IndexerKeyLength)},
						&layer.IndexerAttentionQB},
				}
				for name, shapeAndDestination := range indexerTensors {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight", uint64(spec.HeadCount)*uint64(spec.ValueLength), uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else {
			if spec.Architecture == "apertus" || spec.Architecture == "bailingmoe2" || spec.Architecture == "bert" || spec.Architecture == "bloom" || spec.Architecture == "chatglm" || spec.Architecture == "cogvlm" || spec.Architecture == "cohere2moe" || spec.Architecture == "deci" || spec.Architecture == "dbrx" || spec.Architecture == "dots1" || spec.Architecture == "ernie4_5" || spec.Architecture == "ernie4_5-moe" || spec.Architecture == "eurobert" || spec.Architecture == "exaone4" || spec.Architecture == "gemma-embedding" || spec.Architecture == "glm4" || spec.Architecture == "glm4moe" || spec.Architecture == "grok" || spec.Architecture == "hunyuan-dense" || spec.Architecture == "hunyuan_vl" || spec.Architecture == "hy_v3" || spec.Architecture == "jina-bert-v2" || spec.Architecture == "jina-bert-v3" || spec.Architecture == "mimo2" || spec.Architecture == "step35" || spec.Architecture == "minimax-m2" || spec.Architecture == "modern-bert" || spec.Architecture == "neo-bert" || spec.Architecture == "nomic-bert" || spec.Architecture == "nomic-bert-moe" || spec.Architecture == "openelm" || spec.Architecture == "paddleocr" || spec.Architecture == "pangu-embedded" || spec.Architecture == "phi2" || spec.Architecture == "phi3" || spec.Architecture == "phimoe" || spec.Architecture == "plamo2" || spec.Architecture == "plamo3" || spec.Architecture == "gpt2" || spec.Architecture == "gptneox" || spec.Architecture == "jais" || spec.Architecture == "mpt" || spec.Architecture == "qwen" || spec.Architecture == "qwen2vl" || spec.Architecture == "qwen3vl" || spec.Architecture == "qwen3vlmoe" || spec.Architecture == "refact" || spec.Architecture == "smallthinker" || spec.Architecture == "starcoder" || spec.Architecture == "talkie" ||
				spec.Architecture == "falcon" {
				_, hasQKV := tensors[prefix+"attn_qkv.weight"]
				if hasQKV || spec.Architecture == "bailingmoe2" || spec.Architecture == "bloom" || spec.Architecture == "cogvlm" || spec.Architecture == "dbrx" || spec.Architecture == "gpt2" || spec.Architecture == "gptneox" || spec.Architecture == "jais" || spec.Architecture == "modern-bert" || spec.Architecture == "mpt" || spec.Architecture == "neo-bert" || spec.Architecture == "qwen" || spec.Architecture == "starcoder" || spec.Architecture == "falcon" {
					qkv, qkvErr := required(
						prefix+"attn_qkv.weight",
						uint64(spec.EmbeddingLength),
						queryLength+keyLength+valueLength,
					)
					if qkvErr != nil {
						return Weights{}, qkvErr
					}
					layer.AttentionQKV = &qkv
					if _, ok := tensors[prefix+"attn_qkv.bias"]; ok {
						qkvBias, biasErr := required(
							prefix+"attn_qkv.bias",
							queryLength+keyLength+valueLength,
						)
						if biasErr != nil {
							return Weights{}, biasErr
						}
						if qkvBias.Type != dtype.F32 {
							return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", qkvBias.Name)
						}
						layer.AttentionQKVBias = &qkvBias
					}
					if (spec.Architecture == "bloom" || spec.Architecture == "gpt2" || spec.Architecture == "gptneox" || spec.Architecture == "jais" || spec.Architecture == "qwen" || spec.Architecture == "starcoder") && layer.AttentionQKVBias == nil {
						return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_qkv.bias")
					}
				}
			}
			if layer.AttentionQKV == nil {
				if spec.Architecture == "apertus" || spec.Architecture == "exaone4" || spec.Architecture == "glm4" || spec.Architecture == "phi2" || spec.Architecture == "phi3" || spec.Architecture == "phimoe" || spec.Architecture == "smallthinker" {
					if _, ok := tensors[prefix+"attn_qkv.bias"]; ok {
						return Weights{}, fmt.Errorf("%s fused QKV bias has no fused weight", spec.Architecture)
					}
				}
				if layer.AttentionQ, err = required(
					prefix+"attn_q.weight",
					uint64(spec.EmbeddingLength),
					queryLength,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionK, err = required(
					prefix+"attn_k.weight",
					uint64(spec.EmbeddingLength),
					keyLength,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionV, err = required(
					prefix+"attn_v.weight",
					uint64(spec.EmbeddingLength),
					valueLength,
				); err != nil {
					return Weights{}, err
				}
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight",
				attentionOutputLength,
				uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		}
		if spec.Architecture == "apertus" || spec.Architecture == "afmoe" || spec.Architecture == "bailingmoe2" || spec.Architecture == "dots1" || spec.Architecture == "exaone4" || spec.Architecture == "exaone-moe" || spec.Architecture == "gemma-embedding" || spec.Architecture == "grovemoe" || spec.Architecture == "hunyuan-dense" || spec.Architecture == "hunyuan_vl" || spec.Architecture == "hy_v3" || spec.Architecture == "llada-moe" || spec.Architecture == "mellum" || spec.Architecture == "openelm" || spec.Architecture == "plamo3" || spec.Architecture == "qwen3" || spec.Architecture == "qwen3moe" || spec.Architecture == "qwen3vl" || spec.Architecture == "qwen3vlmoe" || spec.Architecture == "rnd1" || spec.Architecture == "laguna" || spec.Architecture == "gemma3" || spec.Architecture == "hunyuan-moe" ||
			spec.Architecture == "maincoder" ||
			((spec.Architecture == "qwen3next" || spec.Architecture == "qwen35" || spec.Architecture == "qwen35moe") && !layer.Recurrent) ||
			((spec.Architecture == "lfm2" || spec.Architecture == "lfm2moe") && !layer.Recurrent) {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", uint64(spec.KeyLength))
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", uint64(spec.KeyLength))
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "gemma4" || spec.Architecture == "gemma3n" {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", uint64(spec.LayerKeyLength(block)))
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			if spec.LayerHasKV(block) {
				kNorm, kNormErr := required(prefix+"attn_k_norm.weight", uint64(spec.LayerKeyLength(block)))
				if kNormErr != nil {
					return Weights{}, kNormErr
				}
				layer.AttentionKNorm = &kNorm
			}
		}
		if spec.Architecture == "glm4moe" {
			qNorm, hasQNorm := tensors[prefix+"attn_q_norm.weight"]
			kNorm, hasKNorm := tensors[prefix+"attn_k_norm.weight"]
			if hasQNorm != hasKNorm {
				return Weights{}, errors.New("GLM4-MoE Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				if qNorm.Dimensions != 1 || qNorm.Shape[0] != uint64(spec.KeyLength) ||
					kNorm.Dimensions != 1 || kNorm.Shape[0] != uint64(spec.KeyLength) {
					return Weights{}, errors.New("GLM4-MoE Q/K norm tensor shape is invalid")
				}
				layer.AttentionQNorm = &qNorm
				layer.AttentionKNorm = &kNorm
			}
		}
		if spec.Architecture == "step35" {
			qNorm, hasQNorm := tensors[prefix+"attn_q_norm.weight"]
			kNorm, hasKNorm := tensors[prefix+"attn_k_norm.weight"]
			if hasQNorm != hasKNorm {
				return Weights{}, errors.New("Step3.5 Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				if qNorm.Dimensions != 1 || qNorm.Shape[0] != uint64(spec.KeyLength) ||
					kNorm.Dimensions != 1 || kNorm.Shape[0] != uint64(spec.KeyLength) {
					return Weights{}, errors.New("Step3.5 Q/K norm tensor shape is invalid")
				}
				layer.AttentionQNorm = &qNorm
				layer.AttentionKNorm = &kNorm
			}
			if gate, ok := tensors[prefix+"attn_gate.weight"]; ok {
				if gate.Dimensions != 2 || gate.Shape[0] != uint64(spec.EmbeddingLength) ||
					gate.Shape[1] != uint64(spec.LayerHeadCount(block)) {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", gate.Name, gate.Shape)
				}
				layer.AttentionOutputGate = &gate
			}
		}
		if spec.Architecture == "mimo2" || spec.Architecture == "gpt-oss" {
			if sinks, ok := tensors[prefix+"attn_sinks.weight"]; ok {
				if sinks.Type != dtype.F32 || sinks.Dimensions != 1 ||
					sinks.Shape[0] != uint64(spec.LayerHeadCount(block)) {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", sinks.Name, sinks.Shape)
				}
				layer.AttentionSinks = &sinks
			}
			if spec.Architecture == "gpt-oss" && layer.AttentionSinks == nil {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_sinks.weight")
			}
		}
		if spec.Architecture == "gpt-oss" {
			postNorm, normErr := required(prefix+"post_attention_norm.weight", uint64(spec.EmbeddingLength))
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionPostNorm = &postNorm
		}
		if spec.Architecture == "laguna" || spec.Architecture == "afmoe" {
			gate, ok := tensors[prefix+"attn_gate.weight"]
			if !ok {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_gate.weight")
			}
			if gate.Dimensions != 2 || gate.Shape[0] != uint64(spec.EmbeddingLength) {
				return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", gate.Name, gate.Shape)
			}
			heads := uint64(spec.LayerHeadCount(block))
			if spec.Architecture == "afmoe" && gate.Shape[1] != heads*uint64(spec.ValueLength) {
				return Weights{}, fmt.Errorf("tensor %q has gate width %d, need %d", gate.Name, gate.Shape[1], heads*uint64(spec.ValueLength))
			}
			if spec.Architecture == "laguna" && gate.Shape[1] != heads && gate.Shape[1] != heads*uint64(spec.ValueLength) {
				return Weights{}, fmt.Errorf("tensor %q has gate width %d, need %d or %d", gate.Name, gate.Shape[1], heads, heads*uint64(spec.ValueLength))
			}
			layer.AttentionOutputGate = &gate
		}
		if spec.Architecture == "chameleon" {
			qNorm, normErr := required(
				prefix+"attn_q_norm.weight", uint64(spec.KeyLength), uint64(spec.HeadCount),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(
				prefix+"attn_k_norm.weight", uint64(spec.KeyLength), uint64(spec.HeadCountKV),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
			if _, ok := tensors[prefix+"attn_q_norm.bias"]; ok {
				bias, biasErr := required(prefix+"attn_q_norm.bias", uint64(spec.KeyLength), uint64(spec.HeadCount))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionQNormBias = &bias
			}
			if _, ok := tensors[prefix+"attn_k_norm.bias"]; ok {
				bias, biasErr := required(prefix+"attn_k_norm.bias", uint64(spec.KeyLength), uint64(spec.HeadCountKV))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionKNormBias = &bias
			}
		}
		if spec.Architecture == "talkie" {
			qNorm, normErr := required(
				prefix+"attn_q_norm.weight", 1, uint64(spec.HeadCount),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layerScale, scaleErr := required(prefix+"layer_out_scale.weight", 1)
			if scaleErr != nil {
				return Weights{}, scaleErr
			}
			layer.LayerOutputScale = &layerScale
		}
		if spec.Architecture == "olmo2" {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", queryLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", keyLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "olmoe" {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", queryLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", keyLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "minimax-m2" {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", queryLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", keyLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "command-r" && spec.BlockCount >= 64 {
			qNorm, normErr := required(
				prefix+"attn_q_norm.weight",
				uint64(spec.KeyLength),
				uint64(spec.HeadCount),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(
				prefix+"attn_k_norm.weight",
				uint64(spec.KeyLength),
				uint64(spec.HeadCountKV),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "plamo2" && !layer.Recurrent {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", uint64(spec.KeyLength), uint64(spec.HeadCount))
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", uint64(spec.KeyLength), uint64(spec.HeadCountKV))
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "stablelm" {
			_, hasQNorm := tensors[prefix+"attn_q_norm.weight"]
			_, hasKNorm := tensors[prefix+"attn_k_norm.weight"]
			if hasQNorm != hasKNorm {
				return Weights{}, errors.New("StableLM Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				qNorm, normErr := required(
					prefix+"attn_q_norm.weight",
					uint64(spec.KeyLength),
					uint64(spec.HeadCount),
				)
				if normErr != nil {
					return Weights{}, normErr
				}
				kNorm, normErr := required(
					prefix+"attn_k_norm.weight",
					uint64(spec.KeyLength),
					uint64(spec.HeadCountKV),
				)
				if normErr != nil {
					return Weights{}, normErr
				}
				layer.AttentionQNorm = &qNorm
				layer.AttentionKNorm = &kNorm
			}
		}
		if spec.Architecture == "jina-bert-v2" {
			for name, destination := range map[string]**gguf.TensorInfo{
				"attn_q_norm.weight": &layer.AttentionQNorm,
				"attn_k_norm.weight": &layer.AttentionKNorm,
			} {
				if _, ok := tensors[prefix+name]; ok {
					norm, normErr := required(prefix+name, uint64(spec.EmbeddingLength))
					if normErr != nil {
						return Weights{}, normErr
					}
					*destination = &norm
					biasName := prefix + name[:len(name)-len("weight")] + "bias"
					if _, hasBias := tensors[biasName]; hasBias {
						bias, biasErr := required(biasName, uint64(spec.EmbeddingLength))
						if biasErr != nil {
							return Weights{}, biasErr
						}
						if name == "attn_q_norm.weight" {
							layer.AttentionQNormBias = &bias
						} else {
							layer.AttentionKNormBias = &bias
						}
					}
				} else if _, hasBias := tensors[prefix+name[:len(name)-len("weight")]+"bias"]; hasBias {
					return Weights{}, fmt.Errorf("JinaBERT v2 %s bias has no weight", name[:len(name)-len("weight")])
				}
			}
			if layer.AttentionKNorm != nil && keyLength != uint64(spec.EmbeddingLength) {
				return Weights{}, errors.New("JinaBERT v2 K norm requires full-width KV projection")
			}
			if item, ok := tensors[prefix+"attn_norm_2.weight"]; ok {
				norm, normErr := required(item.Name, uint64(spec.EmbeddingLength))
				if normErr != nil {
					return Weights{}, normErr
				}
				layer.AttentionNorm2 = &norm
				if _, hasBias := tensors[prefix+"attn_norm_2.bias"]; hasBias {
					bias, biasErr := required(prefix+"attn_norm_2.bias", uint64(spec.EmbeddingLength))
					if biasErr != nil {
						return Weights{}, biasErr
					}
					layer.AttentionNorm2Bias = &bias
				}
			} else if _, hasBias := tensors[prefix+"attn_norm_2.bias"]; hasBias {
				return Weights{}, errors.New("JinaBERT v2 secondary norm bias has no weight")
			}
		}
		if !layer.Recurrent && spec.Architecture != "ernie4_5" && spec.Architecture != "ernie4_5-moe" {
			if item, ok := tensors[prefix+"attn_output.bias"]; ok {
				if item.Type != dtype.F32 ||
					item.Dimensions != 1 ||
					item.Shape[0] != uint64(spec.EmbeddingLength) {
					return Weights{}, fmt.Errorf(
						"tensor %q has incompatible shape %v",
						item.Name,
						item.Shape,
					)
				}
				layer.AttentionOutputBias = &item
			}
		}
		if spec.Architecture == "pangu-embedded" && layer.AttentionOutputBias == nil {
			return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
		}
		if !layer.Recurrent && layer.AttentionQKV == nil &&
			(spec.Architecture != "deci" || spec.LayerKVHeadCount(block) > 0) {
			for name, shapeAndDestination := range map[string]struct {
				shape       uint64
				destination **gguf.TensorInfo
			}{
				"attn_q.bias": {layer.AttentionQ.Shape[1], &layer.AttentionQBias},
				"attn_k.bias": {layer.AttentionK.Shape[1], &layer.AttentionKBias},
				"attn_v.bias": {layer.AttentionV.Shape[1], &layer.AttentionVBias},
			} {
				if item, ok := tensors[prefix+name]; ok {
					if item.Type != dtype.F32 ||
						item.Dimensions != 1 ||
						item.Shape[0] != shapeAndDestination.shape {
						return Weights{}, fmt.Errorf(
							"tensor %q has incompatible shape %v",
							item.Name,
							item.Shape,
						)
					}
					*shapeAndDestination.destination = &item
				}
			}
		}
		if hasPostNorm(spec.Architecture) || spec.Architecture == "olmo2" {
			attentionPostNormName := "post_attention_norm.weight"
			feedForwardPostNormName := "post_ffw_norm.weight"
			if spec.Architecture == "bert" || spec.Architecture == "jina-bert-v2" || spec.Architecture == "jina-bert-v3" || spec.Architecture == "nomic-bert" || spec.Architecture == "nomic-bert-moe" || spec.Architecture == "grok" {
				attentionPostNormName = "attn_output_norm.weight"
				feedForwardPostNormName = "layer_output_norm.weight"
				if spec.Architecture == "grok" {
					if _, ok := tensors[prefix+feedForwardPostNormName]; !ok {
						feedForwardPostNormName = "ffn_post_norm.weight"
					}
				}
			}
			attentionPostNorm, normErr := required(
				prefix+attentionPostNormName,
				uint64(spec.EmbeddingLength),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			feedForwardPostNorm, normErr := required(
				prefix+feedForwardPostNormName,
				uint64(spec.EmbeddingLength),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionPostNorm = &attentionPostNorm
			layer.FeedForwardPostNorm = &feedForwardPostNorm
			if spec.Architecture == "bert" || spec.Architecture == "jina-bert-v2" || spec.Architecture == "jina-bert-v3" || spec.Architecture == "nomic-bert" || spec.Architecture == "nomic-bert-moe" {
				attentionBias, biasErr := required(prefix+"attn_output_norm.bias", uint64(spec.EmbeddingLength))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				feedForwardBias, biasErr := required(prefix+"layer_output_norm.bias", uint64(spec.EmbeddingLength))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionPostNormBias = &attentionBias
				layer.FeedForwardPostNormBias = &feedForwardBias
			}
		}
		if spec.Architecture == "gemma4" || spec.Architecture == "gemma3n" {
			if scale, ok := tensors[prefix+"layer_output_scale.weight"]; ok && spec.Architecture == "gemma4" {
				if scale.Type != dtype.F32 || scale.Dimensions != 1 || scale.Shape[0] != 1 {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", scale.Name, scale.Shape)
				}
				layer.LayerOutputScale = &scale
			}
			if spec.EmbeddingPerLayer > 0 {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"per_layer_inp_gate.weight":  {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.EmbeddingPerLayer)}, &layer.PerLayerInputGate},
					"per_layer_proj.weight":      {[]uint64{uint64(spec.EmbeddingPerLayer), uint64(spec.EmbeddingLength)}, &layer.PerLayerProjection},
					"per_layer_post_norm.weight": {[]uint64{uint64(spec.EmbeddingLength)}, &layer.PerLayerPostNorm},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
		}
		if spec.Architecture == "gemma3n" {
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"altup_correct_coef.weight":  {[]uint64{uint64(spec.AltUpCount), uint64(spec.AltUpCount)}, &layer.AltUpCorrectCoefficient},
				"altup_correct_scale.weight": {[]uint64{uint64(spec.EmbeddingLength)}, &layer.AltUpCorrectScale},
				"altup_predict_coef.weight":  {[]uint64{uint64(spec.AltUpCount), uint64(spec.AltUpCount * spec.AltUpCount)}, &layer.AltUpPredictCoefficient},
				"altup_router.weight":        {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.AltUpCount)}, &layer.AltUpRouter},
				"altup_router_norm.weight":   {[]uint64{uint64(spec.EmbeddingLength)}, &layer.AltUpRouterNorm},
				"laurel_l.weight":            {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.LaurelRank)}, &layer.LaurelLeft},
				"laurel_r.weight":            {[]uint64{uint64(spec.LaurelRank), uint64(spec.EmbeddingLength)}, &layer.LaurelRight},
				"laurel_post_norm.weight":    {[]uint64{uint64(spec.EmbeddingLength)}, &layer.LaurelPostNorm},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		}
		if spec.Architecture == "mamba" || spec.Architecture == "mamba2" {
			continue
		}
		feedForwardNormName := "ffn_norm.weight"
		if spec.Architecture == "falcon-h1" {
			feedForwardNormName = "ffn_norm"
		}
		if spec.Architecture == "dbrx" {
			feedForwardNormName = "attn_output_norm.weight"
		}
		if spec.Architecture == "glm4moe" {
			feedForwardNormName = "attn_post_norm.weight"
		}
		if spec.Architecture == "qwen3next" || spec.Architecture == "qwen35" || spec.Architecture == "qwen35moe" || spec.Architecture == "seed_oss" {
			feedForwardNormName = "post_attention_norm.weight"
		}
		if spec.Architecture == "stablelm" {
			if _, ok := tensors[prefix+feedForwardNormName]; ok {
				if layer.FeedForwardNorm, err = required(prefix+feedForwardNormName, uint64(spec.EmbeddingLength)); err != nil {
					return Weights{}, err
				}
				if _, ok := tensors[prefix+"ffn_norm.bias"]; ok {
					feedForwardNormBias, biasErr := required(
						prefix+"ffn_norm.bias",
						uint64(spec.EmbeddingLength),
					)
					if biasErr != nil {
						return Weights{}, biasErr
					}
					layer.FeedForwardNormBias = &feedForwardNormBias
				}
			} else if _, ok := tensors[prefix+"ffn_norm.bias"]; ok {
				return Weights{}, errors.New("StableLM FFN norm bias has no weight")
			}
		} else if spec.Architecture != "olmo2" && spec.Architecture != "gpt-oss" && !usesPostOnlyNorm(spec.Architecture) &&
			(spec.Architecture != "deci" || spec.LayerFeedForwardLength(block) > 0) &&
			!usesParallelResidual(spec.Architecture) &&
			!spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() {
			if layer.FeedForwardNorm, err = required(prefix+feedForwardNormName, uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if spec.RequiresLayerNormBias() {
				feedForwardNormBias, biasErr := required(
					prefix+"ffn_norm.bias",
					uint64(spec.EmbeddingLength),
				)
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.FeedForwardNormBias = &feedForwardNormBias
			}
		}
		_, tensorSelectedMoE := tensors[prefix+"ffn_gate_inp.weight"]
		if ((spec.Architecture == "llama" || spec.Architecture == "llama-embed") && spec.ExpertCount > 0) || spec.Architecture == "arctic" || spec.Architecture == "bailingmoe" || spec.Architecture == "dbrx" || spec.Architecture == "grovemoe" || spec.Architecture == "grok" || spec.Architecture == "hunyuan-moe" || spec.Architecture == "llada-moe" || spec.Architecture == "mellum" || spec.Architecture == "minimax-m2" || spec.Architecture == "qwen3moe" || spec.Architecture == "qwen3vlmoe" || spec.Architecture == "qwen3next" || spec.Architecture == "qwen35moe" || spec.Architecture == "qwen2moe" || spec.Architecture == "olmoe" || spec.Architecture == "phimoe" || spec.Architecture == "rnd1" || spec.Architecture == "smallthinker" ||
			(spec.Architecture == "jamba" && tensorSelectedMoE) ||
			(spec.Architecture == "granitehybrid" && spec.ExpertCount > 0) ||
			((spec.Architecture == "mimo2" || spec.Architecture == "step35" || spec.Architecture == "gemma4") && tensorSelectedMoE) ||
			spec.Architecture == "granitemoe" ||
			(spec.Architecture == "glm4moe" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "nomic-bert-moe" && spec.IsInterleavedMoELayer(block)) ||
			spec.Architecture == "gpt-oss" ||
			(spec.Architecture == "hy_v3" && tensorSelectedMoE) ||
			((isDeepSeek2Family(spec.Architecture) || spec.Architecture == "glm-dsa" || spec.Architecture == "deepseek2-ocr") && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "cohere2moe" && block >= spec.LeadingDenseBlocks) ||
			spec.IsInterleavedMoELayer(block) ||
			(spec.Architecture == "dots1" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "deepseek" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "bailingmoe2" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "lfm2moe" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "exaone-moe" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "afmoe" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "laguna" && block >= spec.LeadingDenseBlocks) {
			expertTensors := map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ffn_gate_inp.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)},
					&layer.FeedForwardRouter,
				},
				"ffn_up_exps.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)},
					&layer.FeedForwardUpExperts,
				},
				"ffn_down_exps.weight": {
					[]uint64{uint64(spec.ExpertFeedForward), uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)},
					&layer.FeedForwardDownExperts,
				},
			}
			fusedGateUp := false
			if spec.Architecture == "cohere2moe" || isDeepSeek2Family(spec.Architecture) || spec.Architecture == "glm-dsa" || spec.Architecture == "deepseek2-ocr" || spec.Architecture == "gemma4" || spec.Architecture == "hy_v3" || spec.Architecture == "qwen3next" || spec.Architecture == "qwen35moe" {
				if item, ok := tensors[prefix+"ffn_gate_up_exps.weight"]; ok {
					if item.Dimensions != 3 || item.Shape[0] != uint64(spec.EmbeddingLength) ||
						item.Shape[1] != 2*uint64(spec.ExpertFeedForward) || item.Shape[2] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
					layer.FeedForwardGateUpExperts = &item
					delete(expertTensors, "ffn_up_exps.weight")
					fusedGateUp = true
				}
			}
			if spec.Architecture != "granitemoe" && spec.Architecture != "granitehybrid" && spec.Architecture != "grok" && spec.Architecture != "ernie4_5-moe" && spec.Architecture != "nomic-bert-moe" && !fusedGateUp {
				expertTensors["ffn_gate_exps.weight"] = struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)},
					&layer.FeedForwardGateExperts,
				}
			}
			for name, shapeAndDestination := range expertTensors {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if spec.Architecture == "gemma4" {
				routerScale, scaleErr := required(prefix+"ffn_gate_inp.scale", uint64(spec.EmbeddingLength))
				if scaleErr != nil {
					return Weights{}, scaleErr
				}
				layer.FeedForwardRouterScale = &routerScale
				if _, ok := tensors[prefix+"ffn_down_exps.scale"]; ok {
					downScale, scaleErr := required(prefix+"ffn_down_exps.scale", uint64(spec.ExpertCount))
					if scaleErr != nil {
						return Weights{}, scaleErr
					}
					layer.FeedForwardDownExpertsScale = &downScale
				}
				for name, destination := range map[string]**gguf.TensorInfo{
					"pre_ffw_norm_2.weight":  &layer.FeedForwardPreNorm2,
					"post_ffw_norm_1.weight": &layer.FeedForwardPostNorm1,
					"post_ffw_norm_2.weight": &layer.FeedForwardPostNorm2,
				} {
					item, itemErr := required(prefix+name, uint64(spec.EmbeddingLength))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*destination = &item
				}
			}
			if spec.Architecture == "llama4" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "gpt-oss" {
				if layer.AttentionOutputBias == nil {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
				}
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_inp.bias":  {[]uint64{uint64(spec.ExpertCount)}, &layer.FeedForwardRouterBias},
					"ffn_gate_exps.bias": {[]uint64{uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)}, &layer.FeedForwardGateBias},
					"ffn_up_exps.bias":   {[]uint64{uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)}, &layer.FeedForwardUpBias},
					"ffn_down_exps.bias": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)}, &layer.FeedForwardDownBias},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					if item.Type != dtype.F32 {
						return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", item.Name)
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "grovemoe" {
				chunkExperts := uint64(spec.ExpertCount / spec.ExpertsPerGroup)
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_chexps.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts}, &layer.FeedForwardGateChunkExperts},
					"ffn_up_chexps.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts}, &layer.FeedForwardUpChunkExperts},
					"ffn_down_chexps.weight": {[]uint64{uint64(spec.ExpertChunkFeedForward), uint64(spec.EmbeddingLength), chunkExperts}, &layer.FeedForwardDownChunkExperts},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "granitemoe" || spec.Architecture == "granitehybrid" || spec.Architecture == "grok" || spec.Architecture == "ernie4_5-moe" {
				if item, ok := tensors[prefix+"ffn_gate_exps.weight"]; ok {
					if item.Dimensions != 3 || item.Shape[0] != uint64(spec.EmbeddingLength) ||
						item.Shape[1] != uint64(spec.ExpertFeedForward) || item.Shape[2] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
					layer.FeedForwardGateExperts = &item
				}
				if (spec.Architecture == "granitemoe" || spec.Architecture == "granitehybrid") && spec.SharedExpertFF > 0 {
					for name, shapeAndDestination := range map[string]struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{
						"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
						"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
						"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
					} {
						item, itemErr := required(prefix+name, shapeAndDestination.shape...)
						if itemErr != nil {
							return Weights{}, itemErr
						}
						*shapeAndDestination.destination = &item
					}
				}
			}
			if spec.Architecture == "ernie4_5-moe" {
				if bias, ok := tensors[prefix+"exp_probs_b.bias"]; ok {
					if bias.Type != dtype.F32 || bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.FeedForwardExpertBias = &bias
				}
				if spec.SharedExpertFF > 0 {
					for name, shapeAndDestination := range map[string]struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{
						"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
						"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
						"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
					} {
						item, itemErr := required(prefix+name, shapeAndDestination.shape...)
						if itemErr != nil {
							return Weights{}, itemErr
						}
						*shapeAndDestination.destination = &item
					}
				}
			}
			if spec.Architecture == "laguna" || spec.Architecture == "afmoe" {
				shared := map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"exp_probs_b.bias": {
						[]uint64{uint64(spec.ExpertCount)}, &layer.FeedForwardExpertBias,
					},
				}
				if spec.SharedExpertFF > 0 {
					shared["ffn_gate_shexp.weight"] = struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate}
					shared["ffn_up_shexp.weight"] = struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp}
					shared["ffn_down_shexp.weight"] = struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown}
				}
				for name, shapeAndDestination := range shared {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "hunyuan-moe" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "glm4moe" {
				bias, biasErr := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				if bias.Type != dtype.F32 {
					return Weights{}, fmt.Errorf("tensor %q must use F32 selection-bias storage", bias.Name)
				}
				layer.FeedForwardExpertBias = &bias
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "cohere2moe" && spec.SharedExpertFF > 0 {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "hy_v3" {
				bias, ok := tensors[prefix+"exp_probs_b"]
				if !ok {
					bias, ok = tensors[prefix+"exp_probs_b.bias"]
				}
				if ok {
					if bias.Type != dtype.F32 || bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.FeedForwardExpertBias = &bias
				}
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if isDeepSeek2Family(spec.Architecture) || spec.Architecture == "glm-dsa" || spec.Architecture == "deepseek2-ocr" {
				if bias, ok := tensors[prefix+"exp_probs_b.bias"]; ok {
					if bias.Type != dtype.F32 || bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.FeedForwardExpertBias = &bias
				}
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "exaone-moe" || spec.Architecture == "bailingmoe2" || spec.Architecture == "dots1" {
				if bias, ok := tensors[prefix+"exp_probs_b.bias"]; ok {
					if bias.Type != dtype.F32 || bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.FeedForwardExpertBias = &bias
				}
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "bailingmoe" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "deepseek" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "qwen2moe" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_inp_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedRouter},
					"ffn_gate_shexp.weight":     {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":       {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight":     {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "qwen3next" || spec.Architecture == "qwen35moe" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_inp_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedRouter},
					"ffn_gate_shexp.weight":     {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":       {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight":     {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "lfm2moe" {
				bias, biasErr := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				if bias.Type != dtype.F32 {
					return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
				}
				layer.FeedForwardExpertBias = &bias
			}
			if spec.Architecture == "minimax-m2" {
				bias, biasErr := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				if bias.Type != dtype.F32 {
					return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
				}
				layer.FeedForwardExpertBias = &bias
			}
			if spec.Architecture == "mimo2" {
				if bias, ok := tensors[prefix+"exp_probs_b.bias"]; ok {
					if bias.Type != dtype.F32 || bias.Dimensions != 1 ||
						bias.Shape[0] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.FeedForwardExpertBias = &bias
				}
			}
			if spec.Architecture == "step35" {
				if bias, ok := tensors[prefix+"exp_probs_b.bias"]; ok {
					if bias.Type != dtype.F32 || bias.Dimensions != 1 ||
						bias.Shape[0] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.FeedForwardExpertBias = &bias
				}
				if spec.SharedExpertFF > 0 {
					for name, shapeAndDestination := range map[string]struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{
						"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
						"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
						"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
					} {
						item, itemErr := required(prefix+name, shapeAndDestination.shape...)
						if itemErr != nil {
							return Weights{}, itemErr
						}
						*shapeAndDestination.destination = &item
					}
				}
			}
			if spec.Architecture == "phimoe" && layer.AttentionOutputBias == nil {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
			}
			if spec.Architecture == "grok" {
				denseNames := []string{"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
				var present int
				for _, name := range denseNames {
					if _, ok := tensors[prefix+name]; ok {
						present++
					}
				}
				if present != 0 && present != len(denseNames) {
					return Weights{}, errors.New("Grok dense FFN tensors must be all present or all absent")
				}
				if present == len(denseNames) {
					if layer.FeedForwardGate, err = required(prefix+denseNames[0], uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
						return Weights{}, err
					}
					if layer.FeedForwardUp, err = required(prefix+denseNames[1], uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
						return Weights{}, err
					}
					if layer.FeedForwardDown, err = required(prefix+denseNames[2], uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)); err != nil {
						return Weights{}, err
					}
				}
				continue
			}
			if spec.Architecture == "arctic" {
				expertNorm, normErr := required(prefix+"ffn_norm_exps.weight", uint64(spec.EmbeddingLength))
				if normErr != nil {
					return Weights{}, normErr
				}
				layer.FeedForwardExpertNorm = &expertNorm
			} else if spec.Architecture != "gemma4" {
				continue
			}
		}
		feedForwardLength := spec.LayerFeedForwardLength(block)
		if spec.Architecture == "deci" && feedForwardLength == 0 {
			continue
		}
		if spec.Architecture == "arctic" {
			feedForwardLength = spec.EmbeddingLength
		}
		if spec.Architecture == "jina-bert-v2" {
			if gate, ok := tensors[prefix+"ffn_gate.weight"]; ok {
				if gate.Dimensions != 2 || gate.Shape[0] != uint64(spec.EmbeddingLength) || gate.Shape[1] != uint64(feedForwardLength) {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", gate.Name, gate.Shape)
				}
				layer.FeedForwardGate = gate
			}
			up, ok := tensors[prefix+"ffn_up.weight"]
			if !ok {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"ffn_up.weight")
			}
			upWidth := uint64(feedForwardLength)
			if up.Dimensions != 2 || up.Shape[0] != uint64(spec.EmbeddingLength) ||
				(up.Shape[1] != upWidth && up.Shape[1] != 2*upWidth) {
				return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", up.Name, up.Shape)
			}
			if layer.FeedForwardGate.Name != "" && up.Shape[1] != upWidth {
				return Weights{}, errors.New("JinaBERT v2 separate and fused FFN gates cannot be combined")
			}
			layer.FeedForwardUp = up
			if _, ok := tensors[prefix+"ffn_up.bias"]; ok {
				bias, biasErr := required(prefix+"ffn_up.bias", up.Shape[1])
				if biasErr != nil {
					return Weights{}, biasErr
				}
				if bias.Type != dtype.F32 {
					return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
				}
				layer.FeedForwardUpBias = &bias
			}
			if layer.FeedForwardDown, err = required(prefix+"ffn_down.weight", upWidth, uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			downBias, biasErr := required(prefix+"ffn_down.bias", uint64(spec.EmbeddingLength))
			if biasErr != nil {
				return Weights{}, biasErr
			}
			if downBias.Type != dtype.F32 {
				return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", downBias.Name)
			}
			layer.FeedForwardDownBias = &downBias
			if layer.AttentionOutputBias == nil {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
			}
			continue
		}
		if !usesGateFreeFFN(spec.Architecture) {
			if layer.FeedForwardGate, err = required(
				prefix+"ffn_gate.weight",
				uint64(spec.EmbeddingLength),
				uint64(feedForwardLength),
			); err != nil {
				return Weights{}, err
			}
		}
		feedForwardUpLength := uint64(feedForwardLength)
		if usesFusedGateUp(spec.Architecture) {
			feedForwardUpLength *= 2
		}
		if layer.FeedForwardUp, err = required(
			prefix+"ffn_up.weight",
			uint64(spec.EmbeddingLength),
			feedForwardUpLength,
		); err != nil {
			return Weights{}, err
		}
		if layer.FeedForwardDown, err = required(
			prefix+"ffn_down.weight",
			uint64(feedForwardLength),
			uint64(spec.EmbeddingLength),
		); err != nil {
			return Weights{}, err
		}
		for name, shapeAndDestination := range map[string]struct {
			shape       uint64
			destination **gguf.TensorInfo
		}{
			"ffn_gate.bias": {uint64(feedForwardLength), &layer.FeedForwardGateBias},
			"ffn_up.bias":   {uint64(feedForwardLength), &layer.FeedForwardUpBias},
			"ffn_down.bias": {uint64(spec.EmbeddingLength), &layer.FeedForwardDownBias},
		} {
			if item, ok := tensors[prefix+name]; ok {
				if item.Type != dtype.F32 ||
					item.Dimensions != 1 ||
					item.Shape[0] != shapeAndDestination.shape {
					return Weights{}, fmt.Errorf(
						"tensor %q has incompatible shape %v",
						item.Name,
						item.Shape,
					)
				}
				*shapeAndDestination.destination = &item
			}
		}
		if usesSequentialGELU(spec.Architecture) {
			for name, item := range map[string]*gguf.TensorInfo{
				"attn_output.bias": layer.AttentionOutputBias,
				"ffn_up.bias":      layer.FeedForwardUpBias,
				"ffn_down.bias":    layer.FeedForwardDownBias,
			} {
				if item == nil {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+name)
				}
			}
		}
		if spec.Architecture == "jais2" {
			for name, item := range map[string]*gguf.TensorInfo{
				"attn_q.bias":      layer.AttentionQBias,
				"attn_k.bias":      layer.AttentionKBias,
				"attn_v.bias":      layer.AttentionVBias,
				"attn_output.bias": layer.AttentionOutputBias,
				"ffn_up.bias":      layer.FeedForwardUpBias,
				"ffn_down.bias":    layer.FeedForwardDownBias,
			} {
				if item == nil {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+name)
				}
			}
		}
		if spec.Architecture == "jais" {
			for name, item := range map[string]*gguf.TensorInfo{
				"attn_output.bias": layer.AttentionOutputBias,
				"ffn_gate.bias":    layer.FeedForwardGateBias,
				"ffn_up.bias":      layer.FeedForwardUpBias,
				"ffn_down.bias":    layer.FeedForwardDownBias,
			} {
				if item == nil {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+name)
				}
			}
		}
		if spec.Architecture == "cogvlm" {
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"vis_attn_qkv.weight": {
					[]uint64{uint64(spec.EmbeddingLength), 3 * uint64(spec.EmbeddingLength)},
					&layer.VisualAttentionQKV,
				},
				"vis_attn_output.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)},
					&layer.VisualAttentionOutput,
				},
				"vis_gate.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)},
					&layer.VisualFeedForwardGate,
				},
				"vis_up.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)},
					&layer.VisualFeedForwardUp,
				},
				"vis_down.weight": {
					[]uint64{uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)},
					&layer.VisualFeedForwardDown,
				},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		}
	}
	if spec.NextNPredictLayers == 1 && (spec.Architecture == "qwen35" || spec.Architecture == "qwen35moe") {
		prefix := fmt.Sprintf("blk.%d.", spec.BlockCount)
		mtp := &Qwen35MTPWeights{}
		mtp.Layer.Recurrent = false
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		for name, item := range map[string]struct {
			destination *gguf.TensorInfo
			shape       []uint64
		}{
			"attn_norm.weight":           {&mtp.Layer.AttentionNorm, []uint64{uint64(spec.EmbeddingLength)}},
			"post_attention_norm.weight": {&mtp.Layer.FeedForwardNorm, []uint64{uint64(spec.EmbeddingLength)}},
			"attn_q.weight":              {&mtp.Layer.AttentionQ, []uint64{uint64(spec.EmbeddingLength), 2 * queryLength}},
			"attn_k.weight":              {&mtp.Layer.AttentionK, []uint64{uint64(spec.EmbeddingLength), keyLength}},
			"attn_v.weight":              {&mtp.Layer.AttentionV, []uint64{uint64(spec.EmbeddingLength), valueLength}},
			"attn_output.weight":         {&mtp.Layer.AttentionOutput, []uint64{queryLength, uint64(spec.EmbeddingLength)}},
			"ffn_gate.weight":            {&mtp.Layer.FeedForwardGate, []uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}},
			"ffn_up.weight":              {&mtp.Layer.FeedForwardUp, []uint64{uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)}},
			"ffn_down.weight":            {&mtp.Layer.FeedForwardDown, []uint64{uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)}},
			"nextn.eh_proj.weight":       {&mtp.EHProjection, []uint64{2 * uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)}},
			"nextn.enorm.weight":         {&mtp.EmbeddingNorm, []uint64{uint64(spec.EmbeddingLength)}},
			"nextn.hnorm.weight":         {&mtp.HiddenNorm, []uint64{uint64(spec.EmbeddingLength)}},
		} {
			loaded, loadErr := required(prefix+name, item.shape...)
			if loadErr != nil {
				return Weights{}, loadErr
			}
			*item.destination = loaded
		}
		for name, destination := range map[string]**gguf.TensorInfo{
			"nextn.embed_tokens.weight":     &mtp.TokenEmbedding,
			"nextn.shared_head_norm.weight": &mtp.OutputNorm,
			"nextn.shared_head_head.weight": &mtp.Output,
		} {
			item, ok := tensors[prefix+name]
			if !ok {
				continue
			}
			shape := []uint64{uint64(spec.EmbeddingLength)}
			if name != "nextn.shared_head_norm.weight" {
				shape = append(shape, uint64(spec.VocabularySize))
			}
			validated, loadErr := required(item.Name, shape...)
			if loadErr != nil {
				return Weights{}, loadErr
			}
			*destination = &validated
		}
		qNorm, loadErr := required(prefix+"attn_q_norm.weight", uint64(spec.KeyLength))
		if loadErr != nil {
			return Weights{}, loadErr
		}
		kNorm, loadErr := required(prefix+"attn_k_norm.weight", uint64(spec.KeyLength))
		if loadErr != nil {
			return Weights{}, loadErr
		}
		mtp.Layer.AttentionQNorm, mtp.Layer.AttentionKNorm = &qNorm, &kNorm
		result.Qwen35MTP = mtp
	}
	return result, nil
}
