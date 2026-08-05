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
	FeedForwardActivationScale  *gguf.TensorInfo
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

type singleMTPWeights struct {
	MTPOnly        bool
	Layer          LayerWeights
	EHProjection   gguf.TensorInfo
	EmbeddingNorm  gguf.TensorInfo
	HiddenNorm     gguf.TensorInfo
	TokenEmbedding *gguf.TensorInfo
	OutputNorm     *gguf.TensorInfo
	Output         *gguf.TensorInfo
}

// Qwen35MTPWeights: one pinned dense NextN block.
type Qwen35MTPWeights singleMTPWeights

// Step35MTPWeights: one full Step3.5 draft head.
type Step35MTPWeights struct {
	Layer           LayerWeights
	EHProjection    gguf.TensorInfo
	EmbeddingNorm   gguf.TensorInfo
	HiddenNorm      gguf.TensorInfo
	TokenEmbedding  *gguf.TensorInfo
	LayerOutputNorm *gguf.TensorInfo
	OutputNorm      *gguf.TensorInfo
	Output          *gguf.TensorInfo
}

// Cohere2MTPWeights: one full Cohere2-MoE draft block.
type Cohere2MTPWeights singleMTPWeights

type weightRequirementLoader func(string, ...uint64) (gguf.TensorInfo, error)

func loadQKNormPair(
	load weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	layer *LayerWeights,
	queryShape, keyShape []uint64,
	optionalLabel string,
) error {
	if optionalLabel != "" {
		_, hasQuery := tensors[prefix+"attn_q_norm.weight"]
		_, hasKey := tensors[prefix+"attn_k_norm.weight"]
		if hasQuery != hasKey {
			return fmt.Errorf("%s Q/K norm tensors must both be present or absent", optionalLabel)
		}
		if !hasQuery {
			return nil
		}
	}
	return loadTensorRequirements(load, tensors, prefix, []tensorRequirement{
		requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, queryShape...),
		requiredTensorPointer("attn_k_norm.weight", &layer.AttentionKNorm, keyShape...),
	})
}

func loadOptionalWeightBias(
	load weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	weight, bias tensorRequirement,
	orphanError string,
) error {
	_, hasWeight := tensors[prefix+weight.name]
	_, hasBias := tensors[prefix+bias.name]
	if !hasWeight {
		if hasBias {
			return errors.New(orphanError)
		}
		return nil
	}
	requirements := []tensorRequirement{weight}
	if hasBias {
		requirements = append(requirements, bias)
	}
	return loadTensorRequirements(load, tensors, prefix, requirements)
}

type mtpCommonDestinations struct {
	ehProjection, embeddingNorm, hiddenNorm *gguf.TensorInfo
	tokenEmbedding, outputNorm, output      **gguf.TensorInfo
}

func loadMTPCommonWeights(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	destination mtpCommonDestinations,
) error {
	width := uint64(spec.EmbeddingLength)
	return loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
		requiredTensor("nextn.eh_proj.weight", destination.ehProjection, 2*width, width),
		requiredTensor("nextn.enorm.weight", destination.embeddingNorm, width),
		requiredTensor("nextn.hnorm.weight", destination.hiddenNorm, width),
		optionalTensorPointer("nextn.embed_tokens.weight", destination.tokenEmbedding, width, uint64(spec.VocabularySize)),
		optionalTensorPointer("nextn.shared_head_norm.weight", destination.outputNorm, width),
		optionalTensorPointer("nextn.shared_head_head.weight", destination.output, width, uint64(spec.VocabularySize)),
	})
}

func loadSharedExpertWeights(
	required weightRequirementLoader,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	withRouter bool,
) error {
	return loadSharedExpertWeightsForWidth(required, prefix, uint64(spec.EmbeddingLength), spec, layer, withRouter)
}

func loadSharedExpertWeightsForWidth(
	required weightRequirementLoader,
	prefix string,
	width uint64,
	spec Spec,
	layer *LayerWeights,
	withRouter bool,
) error {
	requirements := []tensorRequirement{
		requiredTensorPointer("ffn_gate_shexp.weight", &layer.FeedForwardSharedGate, width, uint64(spec.SharedExpertFF)),
		requiredTensorPointer("ffn_up_shexp.weight", &layer.FeedForwardSharedUp, width, uint64(spec.SharedExpertFF)),
		requiredTensorPointer("ffn_down_shexp.weight", &layer.FeedForwardSharedDown, uint64(spec.SharedExpertFF), width),
	}
	if withRouter {
		requirements = append(requirements,
			requiredTensorPointer("ffn_gate_inp_shexp.weight", &layer.FeedForwardSharedRouter, width),
		)
	}
	return loadTensorRequirements(required, nil, prefix, requirements)
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
	ClassifierOutput        *gguf.TensorInfo
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
	Step35MTP               []Step35MTPWeights
	HYV3MTP                 []Step35MTPWeights
	Cohere2MTP              *Cohere2MTPWeights
	NextNMTP                []Step35MTPWeights
}

func readWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
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
	profile := spec.Profile()
	draftPlan := profile.DraftPlan(spec.NextNPredictLayers)
	normPlan := spec.NormPlan()
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
			width := uint64(spec.EmbeddingLength)
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
				requiredTensor("attn_q.weight", &layer.AttentionQ, width, queryLength),
				requiredTensor("attn_k.weight", &layer.AttentionK, width, keyLength),
				requiredTensor("attn_v.weight", &layer.AttentionV, width, valueLength),
				requiredTensor("attn_output.weight", &layer.AttentionOutput, queryLength, width),
				requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
				requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
				requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
				requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
			}); itemErr != nil {
				return Weights{}, itemErr
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
		width := uint64(spec.EmbeddingLength)
		if itemErr := loadTensorRequirements(required, tensors, "blk.0.", []tensorRequirement{
			requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
			requiredTensor("attn_q.weight", &layer.AttentionQ, 2*width, queryLength),
			requiredTensor("attn_k.weight", &layer.AttentionK, 2*width, keyLength),
			requiredTensor("attn_v.weight", &layer.AttentionV, 2*width, valueLength),
			requiredTensor("attn_output.weight", &layer.AttentionOutput, queryLength, width),
			requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
			requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
		}); itemErr != nil {
			return Weights{}, itemErr
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
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
				requiredTensor("attn_q.weight", &layer.AttentionQ, width, queryLength),
				requiredTensor("attn_output.weight", &layer.AttentionOutput, attentionOutputLength, width),
				requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
				requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
				requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
				requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
				requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, uint64(spec.LayerKeyLength(block))),
				requiredTensorPointer("post_attention_norm.weight", &layer.AttentionPostNorm, width),
				requiredTensorPointer("post_ffw_norm.weight", &layer.FeedForwardPostNorm, width),
				requiredTensorPointer("layer_output_scale.weight", &layer.LayerOutputScale, 1),
			}); itemErr != nil {
				return Weights{}, itemErr
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
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
				requiredTensor("attn_q_a.weight", &layer.AttentionQ, width, uint64(spec.QLoRARank)),
				requiredTensor("attn_kv.weight", &layer.AttentionK, width, headWidth),
				requiredTensor("attn_output.weight", &layer.AttentionOutput, uint64(spec.AttentionOutputGroups*spec.AttentionOutputRank), width),
				requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
				requiredTensorPointer("attn_sinks.weight", &layer.AttentionSinks, uint64(spec.HeadCount)),
				requiredTensorPointer("attn_q_a_norm.weight", &layer.AttentionQNorm, uint64(spec.QLoRARank)),
				requiredTensorPointer("attn_q_b.weight", &layer.AttentionQB, uint64(spec.QLoRARank), uint64(spec.HeadCount)*headWidth),
				requiredTensorPointer("attn_kv_a_norm.weight", &layer.AttentionKNorm, headWidth),
				requiredTensorPointer("attn_output_a.weight", &layer.AttentionOutputA, uint64(spec.HeadCount)*headWidth/uint64(spec.AttentionOutputGroups), uint64(spec.AttentionOutputRank*spec.AttentionOutputGroups)),
				requiredTensorPointer("hc_attn_fn.weight", &layer.HyperAttentionFN, hyperWidth, mixWidth),
				requiredTensorPointer("hc_attn_base.weight", &layer.HyperAttentionBase, mixWidth),
				requiredTensorPointer("hc_attn_scale.weight", &layer.HyperAttentionScale, 3),
				requiredTensorPointer("hc_ffn_fn.weight", &layer.HyperFeedForwardFN, hyperWidth, mixWidth),
				requiredTensorPointer("hc_ffn_base.weight", &layer.HyperFeedForwardBase, mixWidth),
				requiredTensorPointer("hc_ffn_scale.weight", &layer.HyperFeedForwardScale, 3),
				requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, width, uint64(spec.ExpertCount)),
				requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
				requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
				requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, uint64(spec.ExpertFeedForward), width, uint64(spec.ExpertCount)),
			}); itemErr != nil {
				return Weights{}, itemErr
			}
			if itemErr := loadSharedExpertWeightsForWidth(required, prefix, width, spec, layer, false); itemErr != nil {
				return Weights{}, itemErr
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
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensorPointer("attn_compressor_kv.weight", &layer.AttentionCompressorKV, width, coefficient*headWidth),
					requiredTensorPointer("attn_compressor_gate.weight", &layer.AttentionCompressorGate, width, coefficient*headWidth),
					requiredTensorPointer("attn_compressor_ape.weight", &layer.AttentionCompressorAPE, coefficient*headWidth, uint64(ratio)),
					requiredTensorPointer("attn_compressor_norm.weight", &layer.AttentionCompressorNorm, headWidth),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			}
			if ratio == 4 {
				indexerWidth := uint64(spec.IndexerKeyLength)
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensorPointer("indexer.proj.weight", &layer.IndexerProjection, width, uint64(spec.IndexerHeadCount)),
					requiredTensorPointer("indexer.attn_q_b.weight", &layer.IndexerAttentionQB, uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount)*indexerWidth),
					requiredTensorPointer("indexer_compressor_kv.weight", &layer.IndexerCompressorKV, width, 2*indexerWidth),
					requiredTensorPointer("indexer_compressor_gate.weight", &layer.IndexerCompressorGate, width, 2*indexerWidth),
					requiredTensorPointer("indexer_compressor_ape.weight", &layer.IndexerCompressorAPE, 2*indexerWidth, 4),
					requiredTensorPointer("indexer_compressor_norm.weight", &layer.IndexerCompressorNorm, indexerWidth),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			}
		}
		last := &result.Layers[len(result.Layers)-1]
		if itemErr := loadTensorRequirements(required, tensors, "", []tensorRequirement{
			requiredTensorPointer("output_hc_fn.weight", &last.HyperHeadFN, hyperWidth, hyper),
			requiredTensorPointer("output_hc_base.weight", &last.HyperHeadBase, hyper),
			requiredTensorPointer("output_hc_scale.weight", &last.HyperHeadScale, 1),
		}); itemErr != nil {
			return Weights{}, itemErr
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
		if err = loadTensorRequirements(required, tensors, "", []tensorRequirement{
			requiredTensor("conv1d.weight", &wav.InputConv, 7, uint64(spec.EmbeddingLength), width),
			requiredTensor("conv1d.bias", &wav.InputConvBias, 1, width),
		}); err != nil {
			return Weights{}, err
		}
		for block := uint32(0); block < spec.PosNetBlockCount; block++ {
			prefix := fmt.Sprintf("posnet.%d.", block)
			layer := &wav.PosNet[block]
			switch block {
			case 0, 1, 3, 4:
				if err = loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensor("norm1.weight", &layer.Norm1, 1, width),
					requiredTensor("norm1.bias", &layer.Norm1Bias, 1, width),
					requiredTensor("conv1.weight", &layer.Conv1, 3, width, width),
					requiredTensor("conv1.bias", &layer.Conv1Bias, 1, width),
					requiredTensor("norm2.weight", &layer.Norm2, 1, width),
					requiredTensor("norm2.bias", &layer.Norm2Bias, 1, width),
					requiredTensor("conv2.weight", &layer.Conv2, 3, width, width),
					requiredTensor("conv2.bias", &layer.Conv2Bias, 1, width),
				}); err != nil {
					return Weights{}, err
				}
			case 2:
				if err = loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensor("attn_norm.weight", &layer.AttentionNorm, 1, width),
					requiredTensor("attn_norm.bias", &layer.AttentionNormBias, 1, width),
					requiredTensor("attn_q.bias", &layer.AttentionQBias, 1, width),
					requiredTensor("attn_k.bias", &layer.AttentionKBias, 1, width),
					requiredTensor("attn_v.bias", &layer.AttentionVBias, 1, width),
					requiredTensor("attn_output.bias", &layer.AttentionOutBias, 1, width),
					requiredTensor("attn_q.weight", &layer.AttentionQ, 1, width, width),
					requiredTensor("attn_k.weight", &layer.AttentionK, 1, width, width),
					requiredTensor("attn_v.weight", &layer.AttentionV, 1, width, width),
					requiredTensor("attn_output.weight", &layer.AttentionOutput, 1, width, width),
				}); err != nil {
					return Weights{}, err
				}
			case 5:
				if err = loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensor("attn_norm.weight", &layer.AttentionNorm, 1, width),
					requiredTensor("attn_norm.bias", &layer.AttentionNormBias, 1, width),
				}); err != nil {
					return Weights{}, err
				}
			}
		}
		if err = loadTensorRequirements(required, tensors, "", []tensorRequirement{
			requiredTensor("token_embd_norm.weight", &wav.TokenNorm, width),
			requiredTensor("token_embd_norm.bias", &wav.TokenNormBias, width),
		}); err != nil {
			return Weights{}, err
		}
		for block := uint32(0); block < spec.ConvNextBlockCount; block++ {
			prefix := fmt.Sprintf("convnext.%d.", block)
			layer := &wav.ConvNext[block]
			if err = loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensor("dw.weight", &layer.Depthwise, 7, 1, width),
				requiredTensor("dw.bias", &layer.DepthwiseBias, 1, width),
				requiredTensor("norm.weight", &layer.Norm, width),
				requiredTensor("norm.bias", &layer.NormBias, width),
				requiredTensor("pw1.weight", &layer.Pointwise1, width, ffn),
				requiredTensor("pw1.bias", &layer.Pointwise1Bias, ffn),
				requiredTensor("pw2.weight", &layer.Pointwise2, ffn, width),
				requiredTensor("pw2.bias", &layer.Pointwise2Bias, width),
				requiredTensor("gamma.weight", &layer.Gamma, width),
			}); err != nil {
				return Weights{}, err
			}
		}
		if err = loadTensorRequirements(required, tensors, "", []tensorRequirement{
			requiredTensor("output_norm.weight", &wav.OutputNorm, width),
			requiredTensor("output_norm.bias", &wav.OutputNormBias, width),
			requiredTensor("output.weight", &wav.Output, width, uint64(spec.OutputEmbeddingLength)),
			requiredTensor("output.bias", &wav.OutputBias, uint64(spec.OutputEmbeddingLength)),
		}); err != nil {
			return Weights{}, err
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
	if normPlan.PostNormLayout == PostNormLayoutBERT {
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
		if normErr := loadTensorRequirements(required, tensors, "", []tensorRequirement{
			requiredTensorPointer("token_embd_norm.weight", &result.TokenEmbeddingNorm,
				uint64(spec.EmbeddingLength)),
		}); normErr != nil {
			return Weights{}, normErr
		}
	}
	if spec.Architecture == "mpt" {
		if positionErr := loadTensorRequirements(required, tensors, "", []tensorRequirement{
			optionalTensorPointer("position_embd.weight", &result.PositionEmbedding,
				uint64(spec.EmbeddingLength), uint64(spec.ContextLength)),
		}); positionErr != nil {
			return Weights{}, positionErr
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
	if !spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() &&
		profile.OutputNorm != OutputNormAbsent {
		if result.OutputNorm, err = required(profile.OutputNormTensor(), uint64(spec.EmbeddingLength)); err != nil {
			return Weights{}, err
		}
	}
	if spec.RequiresLayerNormBias() && profile.OutputNorm != OutputNormAbsent {
		outputNormBias, biasErr := required("output_norm.bias", uint64(spec.EmbeddingLength))
		if biasErr != nil {
			return Weights{}, biasErr
		}
		result.OutputNormBias = &outputNormBias
	}
	if spec.Architecture == "rwkv6qwen2" {
		if biasErr := loadTensorRequirements(required, tensors, "", []tensorRequirement{
			optionalTensorPointer("output_norm.bias", &result.OutputNormBias,
				uint64(spec.EmbeddingLength)),
		}); biasErr != nil {
			return Weights{}, biasErr
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
			width := uint64(spec.EmbeddingLength)
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("cross_attn_norm.weight", &layer.CrossAttentionNorm, width),
				requiredTensorPointer("cross_attn_q.weight", &layer.CrossAttentionQ, width, queryLength),
				requiredTensorPointer("cross_attn_k.weight", &layer.CrossAttentionK, width, keyLength),
				requiredTensorPointer("cross_attn_v.weight", &layer.CrossAttentionV, width, valueLength),
				requiredTensorPointer("cross_attn_o.weight", &layer.CrossAttentionOutput, attentionOutputLength, width),
			}); itemErr != nil {
				return Weights{}, itemErr
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
	if spec.Architecture != "cohere2" && spec.Architecture != "command-r" {
		if outputErr := loadTensorRequirements(required, tensors, "", []tensorRequirement{
			optionalTensorPointer("output.weight", &result.Output,
				uint64(spec.EmbeddingLength), uint64(spec.VocabularySize)),
		}); outputErr != nil {
			return Weights{}, outputErr
		}
	}
	if profile.Has(ArchitectureRequiresOutput) && result.Output == nil {
		return Weights{}, errors.New(`required tensor "output.weight" is missing`)
	}
	if outputErr := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		optionalF32TensorPointer("output.bias", &result.OutputBias, uint64(spec.VocabularySize)),
	}); outputErr != nil {
		return Weights{}, outputErr
	}
	if profile.Has(ArchitectureClassifierHead) {
		outputCount := uint64(1)
		if len(spec.ClassifierLabels) > 0 {
			outputCount = uint64(len(spec.ClassifierLabels))
		}
		if classifierErr := loadTensorRequirements(required, tensors, "", []tensorRequirement{
			optionalTensorPointer("cls.output.weight", &result.ClassifierOutput,
				uint64(spec.EmbeddingLength), outputCount),
		}); classifierErr != nil {
			return Weights{}, classifierErr
		}
	}
	if (spec.Architecture == "gptj" || spec.Architecture == "phi2" || spec.Architecture == "phimoe") && result.OutputBias == nil {
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
	if profile.Has(ArchitecturePerLayerEmbeddings) && spec.EmbeddingPerLayer > 0 {
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

	trunkBlockCount := spec.BlockCount
	if draftPlan.AppendedBlocks {
		trunkBlockCount += draftPlan.Heads
	}
	cohere2HasMTP := false
	cohere2MTPOnly := false
	if draftPlan.Kind == DraftCohere2MTP && draftPlan.SessionEligible() {
		mtpPrefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, 0))
		_, cohere2HasMTP = tensors[mtpPrefix+"nextn.eh_proj.weight"]
		_, hasTrunk := tensors["blk.0.attn_norm.weight"]
		cohere2MTPOnly = cohere2HasMTP && !hasTrunk
		if cohere2HasMTP {
			trunkBlockCount++
		}
	}
	mtpOnly := draftPlan.Kind == DraftQwen35MTP && draftPlan.SessionEligible()
	if mtpOnly {
		_, hasTrunk := tensors["blk.0.attn_norm.weight"]
		mtpOnly = !hasTrunk
	}
	if mtpOnly {
		trunkBlockCount = 0
	}
	result.Layers = make([]LayerWeights, trunkBlockCount)
	for block := uint32(0); block < trunkBlockCount; block++ {
		if cohere2MTPOnly && block < spec.BlockCount {
			continue
		}
		prefix := fmt.Sprintf("blk.%d.", block)
		isNextNBlock := block >= spec.BlockCount
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
		if profile.Has(ArchitectureBiasFreeProjections) {
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
				if profile.Has(ArchitectureLongRoPE) {
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
			_, hasQNorm := tensors[prefix+"attn_q_norm.weight"]
			_, hasKNorm := tensors[prefix+"attn_k_norm.weight"]
			if hasQNorm != hasKNorm {
				return Weights{}, errors.New("MPT Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				if queryLength != uint64(spec.EmbeddingLength) || keyLength != uint64(spec.EmbeddingLength) {
					return Weights{}, errors.New("MPT Q/K norm requires full-width Q/K projections")
				}
				qNorm, normErr := required(prefix+"attn_q_norm.weight", uint64(spec.EmbeddingLength))
				if normErr != nil {
					return Weights{}, normErr
				}
				kNorm, normErr := required(prefix+"attn_k_norm.weight", uint64(spec.EmbeddingLength))
				if normErr != nil {
					return Weights{}, normErr
				}
				if qNorm.Type != dtype.F32 || kNorm.Type != dtype.F32 {
					return Weights{}, errors.New("MPT Q/K norm tensors must use F32 storage")
				}
				layer.AttentionQNorm = &qNorm
				layer.AttentionKNorm = &kNorm
				if normErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					optionalF32TensorPointer("attn_q_norm.bias", &layer.AttentionQNormBias, uint64(spec.EmbeddingLength)),
					optionalF32TensorPointer("attn_k_norm.bias", &layer.AttentionKNormBias, uint64(spec.EmbeddingLength)),
				}); normErr != nil {
					return Weights{}, normErr
				}
			} else if _, hasQBias := tensors[prefix+"attn_q_norm.bias"]; hasQBias {
				return Weights{}, errors.New("MPT Q norm bias has no weight")
			} else if _, hasKBias := tensors[prefix+"attn_k_norm.bias"]; hasKBias {
				return Weights{}, errors.New("MPT K norm bias has no weight")
			}
			if _, ok := tensors[prefix+"ffn_act.scales"]; ok {
				activationScale, scaleErr := required(
					prefix+"ffn_act.scales", uint64(spec.FeedForwardLength),
				)
				if scaleErr != nil {
					return Weights{}, scaleErr
				}
				if activationScale.Type != dtype.F32 {
					return Weights{}, fmt.Errorf(
						"tensor %q must use F32 activation-scale storage", activationScale.Name,
					)
				}
				layer.FeedForwardActivationScale = &activationScale
			}
		}
		ropeFactors, hasRopeFactors := tensors[prefix+"rope_freqs.weight"]
		if (spec.Architecture == "step35" || spec.Architecture == "gemma4" && !spec.IsSlidingLayer(block)) && !hasRopeFactors {
			ropeFactors, hasRopeFactors = tensors["rope_freqs.weight"]
		}
		if hasRopeFactors {
			if profile.Has(ArchitectureQwenGDN) {
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
		if profile.Has(ArchitectureLongRoPE) {
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
		} else if normPlan.PreAttention &&
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
				width := uint64(spec.EmbeddingLength)
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensor("attn_q.weight", &layer.AttentionQ, width, inner),
					requiredTensor("attn_k.weight", &layer.AttentionK, width, inner),
					requiredTensor("attn_v.weight", &layer.AttentionV, width, inner),
					requiredTensor("attn_output.weight", &layer.AttentionOutput, inner, width),
				}); itemErr != nil {
					return Weights{}, itemErr
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
				for _, binding := range []struct {
					name        string
					destination **gguf.TensorInfo
				}{
					{name: "ssm_conv1d_q.weight", destination: &layer.SSMQueryConv},
					{name: "ssm_conv1d_k.weight", destination: &layer.SSMKeyConv},
					{name: "ssm_conv1d_v.weight", destination: &layer.SSMValueConv},
				} {
					item, itemErr := conv(binding.name)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*binding.destination = item
				}
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensorPointer("ssm_f_a.weight", &layer.SSMForgetA, width, uint64(spec.KDAHeadDim)),
					requiredTensorPointer("ssm_f_b.weight", &layer.SSMForgetB, uint64(spec.KDAHeadDim), inner),
					requiredTensorPointer("ssm_beta.weight", &layer.SSMBeta, width, uint64(spec.HeadCount)),
					requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, inner),
					requiredTensorPointer("ssm_g_a.weight", &layer.SSMOutputGateA, width, uint64(spec.KDAHeadDim)),
					requiredTensorPointer("ssm_g_b.weight", &layer.SSMOutputGateB, uint64(spec.KDAHeadDim), inner),
					requiredTensorPointer("ssm_norm.weight", &layer.SSMNorm, uint64(spec.KDAHeadDim)),
				}); itemErr != nil {
					return Weights{}, itemErr
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
				width := uint64(spec.EmbeddingLength)
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
					requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
					requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			} else {
				width := uint64(spec.EmbeddingLength)
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, width, uint64(spec.ExpertCount)),
					requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
					requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
					requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, uint64(spec.ExpertFeedForward), width, uint64(spec.ExpertCount)),
					requiredTensorPointer("exp_probs_b.bias", &layer.FeedForwardExpertBias, uint64(spec.ExpertCount)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
				if itemErr := loadSharedExpertWeights(required, prefix, spec, layer, false); itemErr != nil {
					return Weights{}, itemErr
				}
			}
			continue
		}
		if handled, familyErr := loadRWKVLayer(required, tensors, prefix, spec, layer, block); familyErr != nil {
			return Weights{}, familyErr
		} else if handled {
			continue
		}
		if spec.Architecture == "nemotron_h" || spec.Architecture == "nemotron_h_moe" {
			if spec.IsRecurrentLayer(block) {
				layer.Recurrent = true
				convDimension := uint64(spec.SSMInnerSize) +
					2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
				requirements := append(
					mamba2TensorRequirements(spec, layer, true),
					optionalTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, convDimension),
				)
				if itemErr := loadTensorRequirements(required, tensors, prefix, requirements); itemErr != nil {
					return Weights{}, itemErr
				}
				continue
			}
			if spec.LayerFeedForwardLength(block) == 0 {
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensor("attn_q.weight", &layer.AttentionQ, uint64(spec.EmbeddingLength), queryLength),
					requiredTensor("attn_k.weight", &layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
					requiredTensor("attn_v.weight", &layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
					requiredTensor("attn_output.weight", &layer.AttentionOutput, attentionOutputLength, uint64(spec.EmbeddingLength)),
					optionalF32TensorPointer("attn_q.bias", &layer.AttentionQBias, queryLength),
					optionalF32TensorPointer("attn_k.bias", &layer.AttentionKBias, keyLength),
					optionalF32TensorPointer("attn_v.bias", &layer.AttentionVBias, valueLength),
					optionalF32TensorPointer("attn_output.bias", &layer.AttentionOutputBias, uint64(spec.EmbeddingLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
				continue
			}
			if spec.Architecture == "nemotron_h" {
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensor("ffn_up.weight", &layer.FeedForwardUp, uint64(spec.EmbeddingLength), uint64(spec.LayerFeedForwardLength(block))),
					requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.LayerFeedForwardLength(block)), uint64(spec.EmbeddingLength)),
					optionalF32TensorPointer("ffn_up.bias", &layer.FeedForwardUpBias, uint64(spec.LayerFeedForwardLength(block))),
					optionalF32TensorPointer("ffn_down.bias", &layer.FeedForwardDownBias, uint64(spec.EmbeddingLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
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
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)),
				requiredF32TensorPointer("exp_probs_b.bias", &layer.FeedForwardExpertBias, uint64(spec.ExpertCount)),
				requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, moeWidth, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
				requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, uint64(spec.ExpertFeedForward), moeWidth, uint64(spec.ExpertCount)),
				requiredTensorPointer("ffn_up_shexp.weight", &layer.FeedForwardSharedUp, uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)),
				requiredTensorPointer("ffn_down_shexp.weight", &layer.FeedForwardSharedDown, uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)),
			}); itemErr != nil {
				return Weights{}, itemErr
			}
			continue
		}
		if profile.DenseWeights.ValidateFalconNorm {
			if normErr := loadOptionalWeightBias(
				required, tensors, prefix,
				requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, uint64(spec.EmbeddingLength)),
				"Falcon secondary attention norm bias has no weight",
			); normErr != nil {
				return Weights{}, normErr
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
			if scaleErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				optionalF32TensorPointer("attn_q.scale", &layer.AttentionQScale, 1),
				optionalF32TensorPointer("attn_k.scale", &layer.AttentionKScale, 1),
				optionalF32TensorPointer("attn_v.scale", &layer.AttentionVScale, 1),
				optionalF32TensorPointer("attn_output.scale", &layer.AttentionOutputScale, 1),
				optionalF32TensorPointer("ffn_gate.scale", &layer.FeedForwardGateScale, 1),
				optionalF32TensorPointer("ffn_up.scale", &layer.FeedForwardUpScale, 1),
				optionalF32TensorPointer("ffn_down.scale", &layer.FeedForwardDownScale, 1),
			}); scaleErr != nil {
				return Weights{}, scaleErr
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
		} else if handled, familyErr := loadStateSpaceLayer(
			required, tensors, prefix, spec, layer, block,
			queryLength, keyLength, valueLength, attentionOutputLength,
		); familyErr != nil {
			return Weights{}, familyErr
		} else if handled {
		} else if profile.Has(ArchitectureSharedKV) {
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
		} else if profile.Attention == AttentionMLA || profile.Attention == AttentionDSA {
			nope := uint64(spec.KeyLength - spec.RopeDimensionCount)
			if spec.Architecture == "minicpm3" || (profile.Has(ArchitectureDeepSeek2Layout) && spec.QLoRARank > 0) {
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
			mlaTensors := []tensorRequirement{
				requiredTensorPointer("attn_kv_a_mqa.weight", &layer.AttentionKVAMQA,
					uint64(spec.EmbeddingLength), uint64(spec.KVLoRARank+spec.RopeDimensionCount)),
				requiredTensorPointer("attn_kv_a_norm.weight", &layer.AttentionKVANorm, uint64(spec.KVLoRARank)),
			}
			if _, modern := tensors[prefix+"attn_k_b.weight"]; modern {
				mlaTensors = append(mlaTensors,
					requiredTensorPointer("attn_k_b.weight", &layer.AttentionKB, nope, uint64(spec.KVLoRARank), uint64(spec.HeadCount)),
					requiredTensorPointer("attn_v_b.weight", &layer.AttentionVB, uint64(spec.KVLoRARank), uint64(spec.ValueLength), uint64(spec.HeadCount)),
				)
			} else {
				mlaTensors = append(mlaTensors, requiredTensorPointer(
					"attn_kv_b.weight", &layer.AttentionKVB,
					uint64(spec.KVLoRARank), uint64(spec.HeadCount)*(nope+uint64(spec.ValueLength)),
				))
			}
			if itemErr := loadTensorRequirements(required, tensors, prefix, mlaTensors); itemErr != nil {
				return Weights{}, itemErr
			}
			if spec.LayerHasFullIndexer(block) {
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensorPointer("indexer.k_norm.weight", &layer.IndexerKNorm, uint64(spec.IndexerKeyLength)),
					requiredTensorPointer("indexer.k_norm.bias", &layer.IndexerKNormBias, uint64(spec.IndexerKeyLength)),
					requiredTensorPointer("indexer.proj.weight", &layer.IndexerProjection, uint64(spec.EmbeddingLength), uint64(spec.IndexerHeadCount)),
					requiredTensorPointer("indexer.attn_k.weight", &layer.IndexerAttentionK, uint64(spec.EmbeddingLength), uint64(spec.IndexerKeyLength)),
					requiredTensorPointer("indexer.attn_q_b.weight", &layer.IndexerAttentionQB, uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount)*uint64(spec.IndexerKeyLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight", uint64(spec.HeadCount)*uint64(spec.ValueLength), uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else if attentionErr := loadStandardAttentionCatalog(
			required, tensors, prefix, spec, layer,
			queryLength, keyLength, valueLength, attentionOutputLength,
		); attentionErr != nil {
			return Weights{}, attentionErr
		}
		qkPlan := spec.qkPreprocessPlan(block)
		if !layer.Recurrent && profile.RecurrentBlock != BlockPLaMo2 &&
			(qkPlan.Heads == qkNormWeighted || qkPlan.PostRotary == qkNormWeighted) {
			shape := []uint64{uint64(spec.KeyLength)}
			if normErr := loadQKNormPair(required, tensors, prefix, layer, shape, shape, ""); normErr != nil {
				return Weights{}, normErr
			}
		}
		if profile.Has(ArchitectureSharedKV) {
			shape := []uint64{uint64(spec.LayerKeyLength(block))}
			requirements := []tensorRequirement{
				requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, shape...),
			}
			if spec.LayerHasKV(block) {
				requirements = append(requirements,
					requiredTensorPointer("attn_k_norm.weight", &layer.AttentionKNorm, shape...))
			}
			if normErr := loadTensorRequirements(required, tensors, prefix, requirements); normErr != nil {
				return Weights{}, normErr
			}
		}
		if qkPlan.Heads == qkNormOptionalWeighted {
			shape := []uint64{uint64(spec.KeyLength)}
			if normErr := loadQKNormPair(required, tensors, prefix, layer, shape, shape, spec.Architecture); normErr != nil {
				return Weights{}, normErr
			}
		}
		if qkPlan.Heads == qkNormOptionalWeighted && profile.DenseStages.AttentionGate != attentionGateNone {
			if gateErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				optionalTensorPointer("attn_gate.weight", &layer.AttentionOutputGate,
					uint64(spec.EmbeddingLength), uint64(spec.LayerHeadCount(block))),
			}); gateErr != nil {
				return Weights{}, gateErr
			}
		}
		if profile.AttentionGraph.UseSinks {
			sinkRequirement := optionalF32TensorPointer(
				"attn_sinks.weight", &layer.AttentionSinks, uint64(spec.LayerHeadCount(block)))
			requirements := []tensorRequirement{sinkRequirement}
			if profile.DenseWeights.RequireAttentionSinks {
				requirements[0] = requiredF32TensorPointer(
					"attn_sinks.weight", &layer.AttentionSinks, uint64(spec.LayerHeadCount(block)))
				requirements = append(requirements, requiredTensorPointer(
					"post_attention_norm.weight", &layer.AttentionPostNorm, uint64(spec.EmbeddingLength)))
			}
			if sinkErr := loadTensorRequirements(required, tensors, prefix, requirements); sinkErr != nil {
				return Weights{}, sinkErr
			}
		}
		if profile.DenseWeights.RequireAttentionGate {
			gate, ok := tensors[prefix+"attn_gate.weight"]
			if !ok {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_gate.weight")
			}
			if gate.Dimensions != 2 || gate.Shape[0] != uint64(spec.EmbeddingLength) {
				return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", gate.Name, gate.Shape)
			}
			heads := uint64(spec.LayerHeadCount(block))
			if profile.DenseStages.AttentionFlatGate && gate.Shape[1] != heads*uint64(spec.ValueLength) {
				return Weights{}, fmt.Errorf("tensor %q has gate width %d, need %d", gate.Name, gate.Shape[1], heads*uint64(spec.ValueLength))
			}
			if profile.DenseStages.AttentionFlatGateElse && gate.Shape[1] != heads && gate.Shape[1] != heads*uint64(spec.ValueLength) {
				return Weights{}, fmt.Errorf("tensor %q has gate width %d, need %d or %d", gate.Name, gate.Shape[1], heads, heads*uint64(spec.ValueLength))
			}
			layer.AttentionOutputGate = &gate
		}
		headQKNorm := qkPlan.Heads == qkNormAffine || qkPlan.Heads == qkNormConfiguredNoBias ||
			qkPlan.Heads == qkNormLayer || profile.RecurrentBlock == BlockPLaMo2 && !layer.Recurrent
		if headQKNorm {
			optionalLabel := ""
			if qkPlan.Heads == qkNormLayer {
				optionalLabel = spec.Architecture
			}
			if normErr := loadQKNormPair(
				required, tensors, prefix, layer,
				[]uint64{uint64(spec.KeyLength), uint64(spec.HeadCount)},
				[]uint64{uint64(spec.KeyLength), uint64(spec.HeadCountKV)}, optionalLabel,
			); normErr != nil {
				return Weights{}, normErr
			}
		}
		if qkPlan.Heads == qkNormAffine {
			if biasErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				optionalTensorPointer("attn_q_norm.bias", &layer.AttentionQNormBias,
					uint64(spec.KeyLength), uint64(spec.HeadCount)),
				optionalTensorPointer("attn_k_norm.bias", &layer.AttentionKNormBias,
					uint64(spec.KeyLength), uint64(spec.HeadCountKV)),
			}); biasErr != nil {
				return Weights{}, biasErr
			}
		}
		if profile.DenseGraph == DenseGraphTalkie {
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
		if qkPlan.Projection == qkNormWeighted {
			if normErr := loadQKNormPair(
				required, tensors, prefix, layer, []uint64{queryLength}, []uint64{keyLength}, "",
			); normErr != nil {
				return Weights{}, normErr
			}
		}
		if spec.Architecture == "jina-bert-v2" {
			for _, binding := range []struct {
				name   string
				weight **gguf.TensorInfo
				bias   **gguf.TensorInfo
			}{
				{name: "attn_q_norm", weight: &layer.AttentionQNorm, bias: &layer.AttentionQNormBias},
				{name: "attn_k_norm", weight: &layer.AttentionKNorm, bias: &layer.AttentionKNormBias},
			} {
				shape := []uint64{uint64(spec.EmbeddingLength)}
				if normErr := loadOptionalWeightBias(
					required, tensors, prefix,
					requiredTensorPointer(binding.name+".weight", binding.weight, shape...),
					requiredTensorPointer(binding.name+".bias", binding.bias, shape...),
					"JinaBERT v2 "+binding.name+" bias has no weight",
				); normErr != nil {
					return Weights{}, normErr
				}
			}
			if layer.AttentionKNorm != nil && keyLength != uint64(spec.EmbeddingLength) {
				return Weights{}, errors.New("JinaBERT v2 K norm requires full-width KV projection")
			}
			if normErr := loadOptionalWeightBias(
				required, tensors, prefix,
				requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, uint64(spec.EmbeddingLength)),
				"JinaBERT v2 secondary norm bias has no weight",
			); normErr != nil {
				return Weights{}, normErr
			}
		}
		if !layer.Recurrent && spec.Architecture != "ernie4_5" && spec.Architecture != "ernie4_5-moe" {
			if biasErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				optionalF32TensorPointer("attn_output.bias", &layer.AttentionOutputBias,
					uint64(spec.EmbeddingLength)),
			}); biasErr != nil {
				return Weights{}, biasErr
			}
		}
		if profile.DenseWeights.RequireAttentionOutputBias && layer.AttentionOutputBias == nil {
			return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
		}
		if !layer.Recurrent && layer.AttentionQKV == nil &&
			(spec.Architecture != "deci" || spec.LayerKVHeadCount(block) > 0) {
			if biasErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				optionalF32TensorPointer("attn_q.bias", &layer.AttentionQBias, layer.AttentionQ.Shape[1]),
				optionalF32TensorPointer("attn_k.bias", &layer.AttentionKBias, layer.AttentionK.Shape[1]),
				optionalF32TensorPointer("attn_v.bias", &layer.AttentionVBias, layer.AttentionV.Shape[1]),
			}); biasErr != nil {
				return Weights{}, biasErr
			}
		}
		if normPlan.PostAttention {
			normNames := normPlan.PostNormTensors()
			if normNames.FeedForwardFallback != "" {
				if _, ok := tensors[prefix+normNames.FeedForwardWeight]; !ok {
					normNames.FeedForwardWeight = normNames.FeedForwardFallback
				}
			}
			width := uint64(spec.EmbeddingLength)
			requirements := []tensorRequirement{
				requiredTensorPointer(normNames.AttentionWeight, &layer.AttentionPostNorm, width),
				requiredTensorPointer(normNames.FeedForwardWeight, &layer.FeedForwardPostNorm, width),
			}
			if normNames.AttentionBias != "" {
				requirements = append(requirements,
					requiredTensorPointer(normNames.AttentionBias, &layer.AttentionPostNormBias, width),
					requiredTensorPointer(normNames.FeedForwardBias, &layer.FeedForwardPostNormBias, width),
				)
			}
			if normErr := loadTensorRequirements(required, tensors, prefix, requirements); normErr != nil {
				return Weights{}, normErr
			}
		}
		if profile.Has(ArchitecturePerLayerEmbeddings) {
			if spec.Architecture == "gemma4" {
				if scaleErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					optionalF32TensorPointer("layer_output_scale.weight", &layer.LayerOutputScale, 1),
				}); scaleErr != nil {
					return Weights{}, scaleErr
				}
			}
			if spec.EmbeddingPerLayer > 0 {
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensorPointer("per_layer_inp_gate.weight", &layer.PerLayerInputGate, uint64(spec.EmbeddingLength), uint64(spec.EmbeddingPerLayer)),
					requiredTensorPointer("per_layer_proj.weight", &layer.PerLayerProjection, uint64(spec.EmbeddingPerLayer), uint64(spec.EmbeddingLength)),
					requiredTensorPointer("per_layer_post_norm.weight", &layer.PerLayerPostNorm, uint64(spec.EmbeddingLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			}
		}
		if spec.Architecture == "gemma3n" {
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("altup_correct_coef.weight", &layer.AltUpCorrectCoefficient, uint64(spec.AltUpCount), uint64(spec.AltUpCount)),
				requiredTensorPointer("altup_correct_scale.weight", &layer.AltUpCorrectScale, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("altup_predict_coef.weight", &layer.AltUpPredictCoefficient, uint64(spec.AltUpCount), uint64(spec.AltUpCount*spec.AltUpCount)),
				requiredTensorPointer("altup_router.weight", &layer.AltUpRouter, uint64(spec.EmbeddingLength), uint64(spec.AltUpCount)),
				requiredTensorPointer("altup_router_norm.weight", &layer.AltUpRouterNorm, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("laurel_l.weight", &layer.LaurelLeft, uint64(spec.EmbeddingLength), uint64(spec.LaurelRank)),
				requiredTensorPointer("laurel_r.weight", &layer.LaurelRight, uint64(spec.LaurelRank), uint64(spec.EmbeddingLength)),
				requiredTensorPointer("laurel_post_norm.weight", &layer.LaurelPostNorm, uint64(spec.EmbeddingLength)),
			}); itemErr != nil {
				return Weights{}, itemErr
			}
		}
		if spec.Architecture == "mamba" || spec.Architecture == "mamba2" {
			continue
		}
		feedForwardNormName := normPlan.FeedForwardNormTensor()
		if spec.Architecture == "stablelm" {
			if normErr := loadOptionalWeightBias(
				required, tensors, prefix,
				requiredTensor(feedForwardNormName, &layer.FeedForwardNorm, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("ffn_norm.bias", &layer.FeedForwardNormBias, uint64(spec.EmbeddingLength)),
				"StableLM FFN norm bias has no weight",
			); normErr != nil {
				return Weights{}, normErr
			}
		} else if spec.Architecture != "gpt-oss" && normPlan.PreFeedForward &&
			(spec.Architecture != "deci" || spec.LayerFeedForwardLength(block) > 0) &&
			profile.Residual != ResidualParallel &&
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
		if layerUsesMoECatalog(tensors, prefix, spec, block, isNextNBlock) {
			loadDense, moeErr := loadMoECatalog(required, tensors, prefix, spec, layer)
			if moeErr != nil {
				return Weights{}, moeErr
			}
			if !loadDense {
				continue
			}
		}
		if ffnErr := loadDenseFFNCatalog(required, tensors, prefix, spec, layer, block); ffnErr != nil {
			return Weights{}, ffnErr
		}
	}
	if draftPlan.Kind == DraftQwen35MTP && draftPlan.SessionEligible() {
		prefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, 0))
		mtp := &Qwen35MTPWeights{}
		mtp.MTPOnly = mtpOnly
		mtp.Layer.Recurrent = false
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		if loadErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensor("attn_norm.weight", &mtp.Layer.AttentionNorm, uint64(spec.EmbeddingLength)),
			requiredTensor("post_attention_norm.weight", &mtp.Layer.FeedForwardNorm, uint64(spec.EmbeddingLength)),
			requiredTensor("attn_q.weight", &mtp.Layer.AttentionQ, uint64(spec.EmbeddingLength), 2*queryLength),
			requiredTensor("attn_k.weight", &mtp.Layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
			requiredTensor("attn_v.weight", &mtp.Layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
			requiredTensor("attn_output.weight", &mtp.Layer.AttentionOutput, queryLength, uint64(spec.EmbeddingLength)),
			requiredTensor("ffn_gate.weight", &mtp.Layer.FeedForwardGate, uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_up.weight", &mtp.Layer.FeedForwardUp, uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_down.weight", &mtp.Layer.FeedForwardDown, uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)),
		}); loadErr != nil {
			return Weights{}, loadErr
		}
		if loadErr := loadMTPCommonWeights(required, tensors, prefix, spec, mtpCommonDestinations{
			ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
			tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
		}); loadErr != nil {
			return Weights{}, loadErr
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
	if (draftPlan.Kind == DraftStep35MTP || draftPlan.Kind == DraftHYV3MTP) && draftPlan.HasHead(0) {
		heads := make([]Step35MTPWeights, draftPlan.Heads)
		for offset := range draftPlan.Heads {
			block := draftPlan.Block(spec.BlockCount, offset)
			prefix := fmt.Sprintf("blk.%d.", block)
			mtp := &heads[offset]
			mtp.Layer = result.Layers[block]
			if loadErr := loadMTPCommonWeights(required, tensors, prefix, spec, mtpCommonDestinations{
				ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
				tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
			}); loadErr != nil {
				return Weights{}, loadErr
			}
		}
		if draftPlan.Kind == DraftHYV3MTP {
			result.HYV3MTP = heads
		} else {
			result.Step35MTP = heads
		}
		result.Layers = result.Layers[:spec.BlockCount]
	}
	if cohere2HasMTP {
		block := draftPlan.Block(spec.BlockCount, 0)
		prefix := fmt.Sprintf("blk.%d.", block)
		mtp := &Cohere2MTPWeights{MTPOnly: cohere2MTPOnly, Layer: result.Layers[block]}
		if loadErr := loadMTPCommonWeights(required, tensors, prefix, spec, mtpCommonDestinations{
			ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
			tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
		}); loadErr != nil {
			return Weights{}, loadErr
		}
		result.Cohere2MTP = mtp
		if cohere2MTPOnly {
			result.Layers = result.Layers[:0]
		} else {
			result.Layers = result.Layers[:spec.BlockCount]
		}
	}
	if draftPlan.Kind == DraftNextNMTP && draftPlan.HasHead(0) {
		result.NextNMTP = make([]Step35MTPWeights, spec.NextNPredictLayers)
		for offset := range draftPlan.Heads {
			block := draftPlan.Block(spec.BlockCount, offset)
			prefix := fmt.Sprintf("blk.%d.", block)
			mtp := &result.NextNMTP[offset]
			mtp.Layer = result.Layers[block]
			if loadErr := loadMTPCommonWeights(required, tensors, prefix, spec, mtpCommonDestinations{
				ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
				tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
			}); loadErr != nil {
				return Weights{}, loadErr
			}
			if spec.Architecture == "mimo2" || spec.Architecture == "bailingmoe2" {
				loaded, loadErr := required(prefix+"layer_output_norm.weight", uint64(spec.EmbeddingLength))
				if loadErr != nil {
					return Weights{}, loadErr
				}
				mtp.LayerOutputNorm = &loaded
			}
		}
		result.Layers = result.Layers[:spec.BlockCount]
	}
	return result, nil
}
