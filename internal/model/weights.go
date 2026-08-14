package model

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
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

// AudioDecoderWeights: decoder-only audio tensors
type AudioDecoderWeights struct {
	InputConv, InputConvBias   gguf.TensorInfo
	PosNet                     []WavPosNetWeights
	TokenNorm, TokenNormBias   gguf.TensorInfo
	ConvNext                   []WavConvNextWeights
	OutputNorm, OutputNormBias gguf.TensorInfo
	Output, OutputBias         gguf.TensorInfo
}

// SingleDraftWeights: one retained draft block.
type SingleDraftWeights struct {
	MTPOnly        bool
	Layer          LayerWeights
	EHProjection   gguf.TensorInfo
	EmbeddingNorm  gguf.TensorInfo
	HiddenNorm     gguf.TensorInfo
	TokenEmbedding *gguf.TensorInfo
	OutputNorm     *gguf.TensorInfo
	Output         *gguf.TensorInfo
}

// AppendedDraftWeights: one appended draft head.
type AppendedDraftWeights struct {
	Layer           LayerWeights
	EHProjection    gguf.TensorInfo
	EmbeddingNorm   gguf.TensorInfo
	HiddenNorm      gguf.TensorInfo
	TokenEmbedding  *gguf.TensorInfo
	LayerOutputNorm *gguf.TensorInfo
	OutputNorm      *gguf.TensorInfo
	Output          *gguf.TensorInfo
}

func loadQKNormPair(
	catalog weightCatalog,
	prefix string,
	layer *LayerWeights,
	queryShape, keyShape []uint64,
	optionalLabel string,
) error {
	if optionalLabel != "" {
		_, hasQuery := catalog.tensor(prefix + "attn_q_norm.weight")
		_, hasKey := catalog.tensor(prefix + "attn_k_norm.weight")
		if hasQuery != hasKey {
			return fmt.Errorf("%s Q/K norm tensors must both be present or absent", optionalLabel)
		}
		if !hasQuery {
			return nil
		}
	}
	return loadTensorRequirements(catalog, prefix, []tensorRequirement{
		requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, queryShape...),
		requiredTensorPointer("attn_k_norm.weight", &layer.AttentionKNorm, keyShape...),
	})
}

func loadOptionalWeightBias(
	catalog weightCatalog,
	prefix string,
	weight, bias tensorRequirement,
	orphanError string,
) error {
	_, hasWeight := catalog.tensor(prefix + weight.name)
	_, hasBias := catalog.tensor(prefix + bias.name)
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
	return loadTensorRequirements(catalog, prefix, requirements)
}

type mtpCommonDestinations struct {
	ehProjection, embeddingNorm, hiddenNorm *gguf.TensorInfo
	tokenEmbedding, outputNorm, output      **gguf.TensorInfo
}

func loadMTPCommonWeights(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	destination mtpCommonDestinations,
) error {
	width := uint64(spec.EmbeddingLength)
	return loadTensorRequirements(catalog, prefix, []tensorRequirement{
		requiredTensor("nextn.eh_proj.weight", destination.ehProjection, 2*width, width),
		requiredTensor("nextn.enorm.weight", destination.embeddingNorm, width),
		requiredTensor("nextn.hnorm.weight", destination.hiddenNorm, width),
		optionalTensorPointer("nextn.embed_tokens.weight", destination.tokenEmbedding, width, uint64(spec.VocabularySize)),
		optionalTensorPointer("nextn.shared_head_norm.weight", destination.outputNorm, width),
		optionalTensorPointer("nextn.shared_head_head.weight", destination.output, width, uint64(spec.VocabularySize)),
	})
}

func loadSharedExpertWeights(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	withRouter bool,
) error {
	return loadSharedExpertWeightsForWidth(catalog, prefix, uint64(spec.EmbeddingLength), spec, layer, withRouter)
}

func loadSharedExpertWeightsForWidth(
	catalog weightCatalog,
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
	return loadTensorRequirements(catalog, prefix, requirements)
}

type encoderDecoderCatalogPlan struct {
	width, query, key, value, output, feedForward, heads, relativeBuckets uint64
}

func newEncoderDecoderCatalogPlan(spec Spec) encoderDecoderCatalogPlan {
	return encoderDecoderCatalogPlan{
		width:           uint64(spec.EmbeddingLength),
		query:           uint64(spec.HeadCount) * uint64(spec.KeyLength),
		key:             uint64(spec.HeadCountKV) * uint64(spec.KeyLength),
		value:           uint64(spec.HeadCountKV) * uint64(spec.ValueLength),
		output:          uint64(spec.HeadCount) * uint64(spec.ValueLength),
		feedForward:     uint64(spec.FeedForwardLength),
		heads:           uint64(spec.HeadCount),
		relativeBuckets: uint64(spec.RelativeBuckets),
	}
}

func (p encoderDecoderCatalogPlan) loadLayer(
	catalog weightCatalog,
	prefix string,
	layer *LayerWeights,
	requireGate bool,
	inheritedBias **gguf.TensorInfo,
) error {
	gate := optionalTensor("ffn_gate.weight", &layer.FeedForwardGate, p.width, p.feedForward)
	gate.optional = !requireGate
	if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
		requiredTensor("attn_norm.weight", &layer.AttentionNorm, p.width),
		requiredTensor("attn_q.weight", &layer.AttentionQ, p.width, p.query),
		requiredTensor("attn_k.weight", &layer.AttentionK, p.width, p.key),
		requiredTensor("attn_v.weight", &layer.AttentionV, p.width, p.value),
		requiredTensor("attn_o.weight", &layer.AttentionOutput, p.output, p.width),
		requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, p.width),
		gate,
		requiredTensor("ffn_up.weight", &layer.FeedForwardUp, p.width, p.feedForward),
		requiredTensor("ffn_down.weight", &layer.FeedForwardDown, p.feedForward, p.width),
	}); err != nil {
		return err
	}
	bias := requiredTensorPointer(
		"attn_rel_b.weight", &layer.AttentionRelativeBias, p.heads, p.relativeBuckets)
	bias.optional = inheritedBias != nil
	if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{bias}); err != nil {
		return err
	}
	if inheritedBias == nil {
		return nil
	}
	if layer.AttentionRelativeBias != nil {
		*inheritedBias = layer.AttentionRelativeBias
	} else if *inheritedBias == nil {
		return fmt.Errorf("required tensor %q is missing", prefix+bias.name)
	} else {
		layer.AttentionRelativeBias = *inheritedBias
	}
	return nil
}

// Weights: validated model tensor catalog
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
	AudioDecoder            *AudioDecoderWeights
	SingleCatalogDraft      *SingleDraftWeights
	AppendedMultiCarryDraft []AppendedDraftWeights
	AppendedMultiDraft      []AppendedDraftWeights
	OptionalCatalogDraft    *SingleDraftWeights
	AppendedSingleDraft     []AppendedDraftWeights
}

type weightCatalog struct {
	tensors map[string]int
	items   []gguf.TensorInfo
}

func newWeightCatalog(file *gguf.File) (weightCatalog, error) {
	if file == nil {
		return weightCatalog{}, errors.New("model file is nil")
	}
	tensors := make(map[string]int, len(file.Tensors))
	for index, item := range file.Tensors {
		if _, exists := tensors[item.Name]; exists {
			return weightCatalog{}, fmt.Errorf("duplicate tensor %q", item.Name)
		}
		tensors[item.Name] = index
	}
	return weightCatalog{tensors: tensors, items: file.Tensors}, nil
}

func (c weightCatalog) tensor(name string) (gguf.TensorInfo, bool) {
	index, ok := c.tensors[name]
	if !ok {
		return gguf.TensorInfo{}, false
	}
	return c.items[index], true
}

func (c weightCatalog) required(name string, shape ...uint64) (gguf.TensorInfo, error) {
	item, ok := c.tensor(name)
	if !ok {
		return gguf.TensorInfo{}, fmt.Errorf("required tensor %q is missing", name)
	}
	if len(shape) == 0 {
		return item, nil
	}
	if err := validateTensorInfo(item, nil, shape); err != nil {
		return gguf.TensorInfo{}, err
	}
	return item, nil
}

func readWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	catalog, err := newWeightCatalog(file)
	if err != nil {
		return Weights{}, err
	}
	policy := spec.Profile().ModelCatalog.Weights
	if int(policy) >= len(weightCatalogReaders) || weightCatalogReaders[policy] == nil {
		return Weights{}, fmt.Errorf("weight catalog policy %d for %q is unsupported", policy, spec.Architecture)
	}
	return weightCatalogReaders[policy](catalog, spec)
}

type weightCatalogReader func(weightCatalog, Spec) (Weights, error)

var weightCatalogReaders = [...]weightCatalogReader{
	WeightCatalogLayered:          readLayeredWeightCatalog,
	WeightCatalogCompressedHyper:  readCompressedHyperWeightCatalog,
	WeightCatalogTargetFeatures:   readTargetFeatureWeightCatalog,
	WeightCatalogHiddenFusion:     readHiddenFusionWeightCatalog,
	WeightCatalogPairedProjection: readPairedProjectionWeightCatalog,
	WeightCatalogEncoder:          readRelativeEncoderWeightCatalog,
	WeightCatalogAudioDecoder:     readAudioDecoderWeightCatalog,
	WeightCatalogEncoderDecoder:   readEncoderDecoderWeightCatalog,
}

func readLayeredWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	loader := &layerCatalogLoader{
		catalog: catalog,
		spec:    spec,
		profile: spec.Profile(),
	}
	loader.draftPlan = loader.profile.DraftPlan(spec.NextNPredictLayers)
	loader.normPlan = loader.profile.Runtime.normalizationPlan(spec, loader.profile)
	var result Weights
	var err error
	for _, stage := range []func(Weights) (Weights, error){
		loader.loadModelCatalog,
		loader.loadLayerCatalogs,
		loader.loadDraftCatalogs,
	} {
		result, err = stage(result)
		if err != nil {
			return Weights{}, err
		}
	}
	return result, nil
}

type layerCatalogLoader struct {
	catalog         weightCatalog
	spec            Spec
	profile         ArchitectureProfile
	draftPlan       DraftPlan
	normPlan        NormalizationPlan
	trunkBlockCount uint32
	cohere2HasMTP   bool
	cohere2MTPOnly  bool
	mtpOnly         bool
}

func (l *layerCatalogLoader) loadModelCatalog(result Weights) (Weights, error) {
	catalog := l.catalog
	spec, profile, draftPlan, normPlan := l.spec, l.profile, l.draftPlan, l.normPlan
	var err error
	tokenEmbeddingName := "token_embd.weight"
	if profile.ModelCatalog.TokenEmbeddingFallback {
		if _, ok := catalog.tensors[tokenEmbeddingName]; !ok {
			tokenEmbeddingName = "output.weight"
		}
	}
	if result.TokenEmbedding, err = catalog.required(
		tokenEmbeddingName,
		uint64(spec.EmbeddingLength),
		uint64(spec.VocabularySize),
	); err != nil {
		return Weights{}, err
	}
	if profile.ModelCatalog.PositionEmbedding == positionEmbeddingRequired {
		positionEmbedding, positionErr := catalog.required(
			"position_embd.weight",
			uint64(spec.EmbeddingLength),
			uint64(spec.ContextLength),
		)
		if positionErr != nil {
			return Weights{}, positionErr
		}
		result.PositionEmbedding = &positionEmbedding
	}
	if normPlan.PostNormLayout == PostNormLayoutOutputLayer {
		tokenTypes := optionalTensorPointer(
			"token_types.weight", &result.TokenTypeEmbedding,
			uint64(spec.EmbeddingLength), uint64(spec.TokenTypeCount),
		)
		tokenTypes.optional = !profile.ModelCatalog.RequireTokenTypes
		if typeErr := loadTensorRequirements(catalog, "", []tensorRequirement{tokenTypes}); typeErr != nil {
			return Weights{}, typeErr
		}
		tokenNorm, normErr := catalog.required("token_embd_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		tokenNormBias, normErr := catalog.required("token_embd_norm.bias", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.TokenEmbeddingNorm = &tokenNorm
		result.TokenEmbeddingNormBias = &tokenNormBias
	}
	if profile.ModelCatalog.TokenNorm == tokenNormWeight {
		if normErr := loadTensorRequirements(catalog, "", []tensorRequirement{
			requiredTensorPointer("token_embd_norm.weight", &result.TokenEmbeddingNorm,
				uint64(spec.EmbeddingLength)),
		}); normErr != nil {
			return Weights{}, normErr
		}
	}
	if profile.ModelCatalog.PositionEmbedding == positionEmbeddingOptional {
		if positionErr := loadTensorRequirements(catalog, "", []tensorRequirement{
			optionalTensorPointer("position_embd.weight", &result.PositionEmbedding,
				uint64(spec.EmbeddingLength), uint64(spec.ContextLength)),
		}); positionErr != nil {
			return Weights{}, positionErr
		}
	}
	if profile.ModelCatalog.TokenNorm == tokenNormAffine {
		tokenNorm, normErr := catalog.required("token_embd_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		tokenNormBias, normErr := catalog.required("token_embd_norm.bias", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.TokenEmbeddingNorm = &tokenNorm
		result.TokenEmbeddingNormBias = &tokenNormBias
	}
	if !spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() &&
		profile.OutputNorm != OutputNormAbsent {
		if result.OutputNorm, err = catalog.required(profile.OutputNormTensor(), uint64(spec.EmbeddingLength)); err != nil {
			return Weights{}, err
		}
	}
	if spec.RequiresLayerNormBias() && profile.OutputNorm != OutputNormAbsent {
		outputNormBias, biasErr := catalog.required("output_norm.bias", uint64(spec.EmbeddingLength))
		if biasErr != nil {
			return Weights{}, biasErr
		}
		result.OutputNormBias = &outputNormBias
	}
	if profile.ModelCatalog.OptionalOutputNormBias {
		if biasErr := loadTensorRequirements(catalog, "", []tensorRequirement{
			optionalTensorPointer("output_norm.bias", &result.OutputNormBias,
				uint64(spec.EmbeddingLength)),
		}); biasErr != nil {
			return Weights{}, biasErr
		}
	}
	if !profile.ModelCatalog.SkipOutput {
		if outputErr := loadTensorRequirements(catalog, "", []tensorRequirement{
			optionalTensorPointer("output.weight", &result.Output,
				uint64(spec.EmbeddingLength), uint64(spec.VocabularySize)),
		}); outputErr != nil {
			return Weights{}, outputErr
		}
	}
	if profile.Has(ArchitectureRequiresOutput) && result.Output == nil {
		return Weights{}, errors.New(`required tensor "output.weight" is missing`)
	}
	if outputErr := loadTensorRequirements(catalog, "", []tensorRequirement{
		optionalF32TensorPointer("output.bias", &result.OutputBias, uint64(spec.VocabularySize)),
	}); outputErr != nil {
		return Weights{}, outputErr
	}
	if profile.Has(ArchitectureClassifierHead) {
		outputCount := uint64(1)
		if len(spec.ClassifierLabels) > 0 {
			outputCount = uint64(len(spec.ClassifierLabels))
		}
		if classifierErr := loadTensorRequirements(catalog, "", []tensorRequirement{
			optionalTensorPointer("cls.output.weight", &result.ClassifierOutput,
				uint64(spec.EmbeddingLength), outputCount),
		}); classifierErr != nil {
			return Weights{}, classifierErr
		}
	}
	if profile.ModelCatalog.RequireOutputBias && result.OutputBias == nil {
		return Weights{}, errors.New(`required tensor "output.bias" is missing`)
	}
	if profile.LayerTopology == LayerTopologyBidirectionalQKNorm {
		if item, ok := l.catalog.tensor("dense_2.weight"); ok {
			if spec.Dense2FeatureIn == 0 || spec.Dense2FeatureOut == 0 {
				return Weights{}, errors.New("Gemma embedding dense-2 tensor has no shape metadata")
			}
			validated, denseErr := catalog.required(
				item.Name, uint64(spec.Dense2FeatureIn), uint64(spec.Dense2FeatureOut),
			)
			if denseErr != nil {
				return Weights{}, denseErr
			}
			result.Dense2Output = &validated
		}
		if item, ok := l.catalog.tensor("dense_3.weight"); ok {
			if spec.Dense3FeatureIn == 0 || spec.Dense3FeatureOut == 0 {
				return Weights{}, errors.New("Gemma embedding dense-3 tensor has no shape metadata")
			}
			validated, denseErr := catalog.required(
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
		perLayerTokenEmbedding, itemErr := catalog.required(
			"per_layer_token_embd.weight",
			uint64(spec.EmbeddingPerLayer)*uint64(spec.BlockCount),
			uint64(spec.VocabularySize),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		perLayerModelProjection, itemErr := catalog.required(
			"per_layer_model_proj.weight",
			uint64(spec.EmbeddingLength),
			uint64(spec.EmbeddingPerLayer)*uint64(spec.BlockCount),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		perLayerProjectionNorm, itemErr := catalog.required(
			"per_layer_proj_norm.weight", uint64(spec.EmbeddingPerLayer),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		result.PerLayerTokenEmbedding = &perLayerTokenEmbedding
		result.PerLayerModelProjection = &perLayerModelProjection
		result.PerLayerProjectionNorm = &perLayerProjectionNorm
	}
	if profile.LayerTopology == LayerTopologySplitProjection {
		projection, itemErr := catalog.required(
			"altup_proj.weight", uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength), uint64(spec.AltUpCount-1),
		)
		if itemErr != nil {
			return Weights{}, itemErr
		}
		unembedding, itemErr := catalog.required(
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
	if draftPlan.Kind == DraftOptionalSingleCatalog && draftPlan.SessionEligible() {
		mtpPrefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, 0))
		_, cohere2HasMTP = catalog.tensors[mtpPrefix+"nextn.eh_proj.weight"]
		_, hasTrunk := catalog.tensors["blk.0.attn_norm.weight"]
		cohere2MTPOnly = cohere2HasMTP && !hasTrunk
		if cohere2HasMTP {
			trunkBlockCount++
		}
	}
	mtpOnly := draftPlan.Kind == DraftSingleCatalog && draftPlan.SessionEligible()
	if mtpOnly {
		_, hasTrunk := catalog.tensors["blk.0.attn_norm.weight"]
		mtpOnly = !hasTrunk
	}
	if mtpOnly {
		trunkBlockCount = 0
	}
	result.Layers = make([]LayerWeights, trunkBlockCount)
	l.trunkBlockCount = trunkBlockCount
	l.cohere2HasMTP = cohere2HasMTP
	l.cohere2MTPOnly = cohere2MTPOnly
	l.mtpOnly = mtpOnly
	return result, nil
}

func (l *layerCatalogLoader) loadLayerCatalogs(result Weights) (Weights, error) {
	catalog := l.catalog
	spec, profile, normPlan := l.spec, l.profile, l.normPlan
	trunkBlockCount, cohere2MTPOnly := l.trunkBlockCount, l.cohere2MTPOnly
	var err error
	for block := uint32(0); block < trunkBlockCount; block++ {
		if cohere2MTPOnly && block < spec.BlockCount {
			continue
		}
		prefix := fmt.Sprintf("blk.%d.", block)
		isDraftBlock := block >= spec.BlockCount
		layerPlan, planErr := spec.planLayer(profile, block, false)
		if planErr != nil {
			return Weights{}, planErr
		}
		shapes := spec.TensorShapes(block)
		queryLength := shapes.QueryProjectionWidth()
		keyLength := shapes.KeyProjectionWidth()
		valueLength := shapes.ValueProjectionWidth()
		attentionOutputLength := shapes.AttentionOutputWidth()
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
				if _, ok := catalog.tensors[prefix+name]; ok {
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
			if _, ok := catalog.tensors[prefix+unsupported]; ok {
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
		if layerPlan.DenseWeights.validateOptionalQKNorm && layerPlan.DenseWeights.allowActivationScale {
			_, hasQNorm := catalog.tensors[prefix+"attn_q_norm.weight"]
			_, hasKNorm := catalog.tensors[prefix+"attn_k_norm.weight"]
			if hasQNorm != hasKNorm {
				return Weights{}, errors.New("MPT Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				if queryLength != uint64(spec.EmbeddingLength) || keyLength != uint64(spec.EmbeddingLength) {
					return Weights{}, errors.New("MPT Q/K norm requires full-width Q/K projections")
				}
				if normErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredF32TensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, uint64(spec.EmbeddingLength)),
					requiredF32TensorPointer("attn_k_norm.weight", &layer.AttentionKNorm, uint64(spec.EmbeddingLength)),
					optionalF32TensorPointer("attn_q_norm.bias", &layer.AttentionQNormBias, uint64(spec.EmbeddingLength)),
					optionalF32TensorPointer("attn_k_norm.bias", &layer.AttentionKNormBias, uint64(spec.EmbeddingLength)),
				}); normErr != nil {
					return Weights{}, normErr
				}
			} else if _, hasQBias := catalog.tensors[prefix+"attn_q_norm.bias"]; hasQBias {
				return Weights{}, errors.New("MPT Q norm bias has no weight")
			} else if _, hasKBias := catalog.tensors[prefix+"attn_k_norm.bias"]; hasKBias {
				return Weights{}, errors.New("MPT K norm bias has no weight")
			}
			if scaleErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
				optionalF32TensorPointer("ffn_act.scales", &layer.FeedForwardActivationScale,
					uint64(spec.FeedForwardLength)),
			}); scaleErr != nil {
				return Weights{}, scaleErr
			}
		}
		ropeFactors, hasRopeFactors := l.catalog.tensor(prefix + "rope_freqs.weight")
		if (profile.Rotary.FactorPairs || profile.LayerTopology == LayerTopologySharedKVAdapter && !layerPlan.Sliding) && !hasRopeFactors {
			ropeFactors, hasRopeFactors = l.catalog.tensor("rope_freqs.weight")
		}
		if hasRopeFactors {
			if layerPlan.Attention == AttentionGatedDelta {
				return Weights{}, fmt.Errorf(
					"tensor %q requires unsupported hybrid RoPE factors",
					ropeFactors.Name,
				)
			}
			rotaryDimensions := spec.KeyLength
			if spec.RopeDimensionCount > 0 {
				rotaryDimensions = spec.RopeDimensionCount
			}
			if rotaryDimensions%2 != 0 {
				return Weights{}, fmt.Errorf("tensor %q has odd rotary width %d", ropeFactors.Name, rotaryDimensions)
			}
			if validateErr := validateTensorInfo(
				ropeFactors, []dtype.Type{dtype.F32}, []uint64{uint64(rotaryDimensions / 2)},
			); validateErr != nil {
				return Weights{}, validateErr
			}
			layer.RopeFactors = &ropeFactors
		}
		if profile.Has(ArchitectureLongRoPE) {
			longName := prefix + "rope_factors_long.weight"
			shortName := prefix + "rope_factors_short.weight"
			if profile.Rotary.FactorPairs {
				longName = "rope_factors_long.weight"
				shortName = "rope_factors_short.weight"
			}
			longFactors, hasLong := l.catalog.tensor(longName)
			shortFactors, hasShort := l.catalog.tensor(shortName)
			if block > 0 {
				if !hasLong {
					longFactors, hasLong = l.catalog.tensor("blk.0.rope_factors_long.weight")
				}
				if !hasShort {
					shortFactors, hasShort = l.catalog.tensor("blk.0.rope_factors_short.weight")
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
					if itemErr := validateTensorInfo(
						item, []dtype.Type{dtype.F32}, []uint64{uint64(spec.RopeDimensionCount / 2)},
					); itemErr != nil {
						return Weights{}, itemErr
					}
				}
				if spec.ContextLength > spec.OriginalContextLength {
					layer.RopeFactors = &longFactors
				} else {
					layer.RopeFactors = &shortFactors
				}
			}
		}
		if profile.LayerTopology == LayerTopologyBidirectionalFusedQKV {
			norm := optionalTensor("attn_norm.weight", &layer.AttentionNorm, uint64(spec.EmbeddingLength))
			norm.optional = block == 0
			if normErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{norm}); normErr != nil {
				return Weights{}, normErr
			}
		} else if normPlan.PreAttention &&
			(!layerPlan.DeciSparse || spec.LayerHeadCount(block) > 0) &&
			!spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() {
			if layer.AttentionNorm, err = catalog.required(prefix+"attn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if spec.RequiresLayerNormBias() {
				attentionNormBias, biasErr := catalog.required(
					prefix+"attn_norm.bias",
					uint64(spec.EmbeddingLength),
				)
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionNormBias = &attentionNormBias
			}
		}
		if layerPlan.Mixer == recurrentMixerKeyedDelta {
			layer.Recurrent = spec.IsRecurrentLayer(block)
			if layer.Recurrent {
				inner := uint64(spec.SSMInnerSize)
				width := uint64(spec.EmbeddingLength)
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredTensor("attn_q.weight", &layer.AttentionQ, width, inner),
					requiredTensor("attn_k.weight", &layer.AttentionK, width, inner),
					requiredTensor("attn_v.weight", &layer.AttentionV, width, inner),
					requiredTensor("attn_output.weight", &layer.AttentionOutput, inner, width),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
				convShape := []uint64{uint64(spec.SSMConvKernel), 1, inner}
				convShape4D := append(slices.Clone(convShape), 1)
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredTensorPointerShapes("ssm_conv1d_q.weight", &layer.SSMQueryConv, convShape, convShape4D),
					requiredTensorPointerShapes("ssm_conv1d_k.weight", &layer.SSMKeyConv, convShape, convShape4D),
					requiredTensorPointerShapes("ssm_conv1d_v.weight", &layer.SSMValueConv, convShape, convShape4D),
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
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredTensorPointerShapes("ssm_a", &layer.SSMA,
						[]uint64{1, uint64(spec.HeadCount)},
						[]uint64{1, uint64(spec.HeadCount), 1, 1}),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			} else {
				if spec.QLoRARank > 0 {
					item, itemErr := catalog.required(prefix+"attn_q_a.weight", uint64(spec.EmbeddingLength), uint64(spec.QLoRARank))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionQ = item
					item, itemErr = catalog.required(prefix+"attn_q_a_norm.weight", uint64(spec.QLoRARank))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionQNorm = &item
					item, itemErr = catalog.required(prefix+"attn_q_b.weight", uint64(spec.QLoRARank), queryLength)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionQB = &item
				} else if layer.AttentionQ, err = catalog.required(prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength); err != nil {
					return Weights{}, err
				}
				if layer.AttentionOutput, err = catalog.required(prefix+"attn_output.weight", attentionOutputLength, uint64(spec.EmbeddingLength)); err != nil {
					return Weights{}, err
				}
				kvA, itemErr := catalog.required(prefix+"attn_kv_a_mqa.weight", uint64(spec.EmbeddingLength), uint64(spec.KVLoRARank+spec.RopeDimensionCount))
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.AttentionKVAMQA = &kvA
				kvNorm, itemErr := catalog.required(prefix+"attn_kv_a_norm.weight", uint64(spec.KVLoRARank))
				if itemErr != nil {
					return Weights{}, itemErr
				}
				layer.AttentionKVANorm = &kvNorm
				nope := uint64(spec.KeyLength - spec.RopeDimensionCount)
				if _, ok := catalog.tensors[prefix+"attn_k_b.weight"]; ok {
					validated, itemErr := catalog.required(prefix+"attn_k_b.weight", nope, uint64(spec.KVLoRARank), uint64(spec.HeadCount))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionKB = &validated
					validated, itemErr = catalog.required(prefix+"attn_v_b.weight", uint64(spec.KVLoRARank), uint64(spec.ValueLength), uint64(spec.HeadCount))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionVB = &validated
				} else {
					validated, itemErr := catalog.required(prefix+"attn_kv_b.weight", uint64(spec.KVLoRARank), uint64(spec.HeadCount)*(nope+uint64(spec.ValueLength)))
					if itemErr != nil {
						return Weights{}, itemErr
					}
					layer.AttentionKVB = &validated
				}
			}
			if layer.FeedForwardNorm, err = catalog.required(prefix+"ffn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if block < spec.LeadingDenseBlocks {
				width := uint64(spec.EmbeddingLength)
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
					requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
					requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			} else {
				width := uint64(spec.EmbeddingLength)
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, width, uint64(spec.ExpertCount)),
					requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
					requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
					requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, uint64(spec.ExpertFeedForward), width, uint64(spec.ExpertCount)),
					requiredTensorPointer("exp_probs_b.bias", &layer.FeedForwardExpertBias, uint64(spec.ExpertCount)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
				if itemErr := loadSharedExpertWeights(catalog, prefix, spec, layer, false); itemErr != nil {
					return Weights{}, itemErr
				}
			}
			continue
		}
		if mixer := layerPlan.Mixer; mixer >= recurrentMixerDynamicWKV6 && mixer <= recurrentMixerDynamicWKV7 {
			if mixerErr := loadTokenShiftRecurrentLayer(catalog, prefix, spec, layer, block, layerPlan.Mixer); mixerErr != nil {
				return Weights{}, mixerErr
			}
			continue
		}
		if layerPlan.Mixer == recurrentMixerSparseGroupedSelectiveScan {
			if spec.IsRecurrentLayer(block) {
				layer.Recurrent = true
				convDimension := uint64(spec.SSMInnerSize) +
					2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
				requirements := append(
					groupedSelectiveScanTensorRequirements(spec, layer, true),
					optionalTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, convDimension),
				)
				if itemErr := loadTensorRequirements(catalog, prefix, requirements); itemErr != nil {
					return Weights{}, itemErr
				}
				continue
			}
			if spec.LayerFeedForwardLength(block) == 0 {
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
			if !profile.Has(ArchitectureMoE) {
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
				latentDown, latentErr := catalog.required(prefix+"ffn_latent_down.weight", uint64(spec.EmbeddingLength), moeWidth)
				if latentErr != nil {
					return Weights{}, latentErr
				}
				latentUp, latentErr := catalog.required(prefix+"ffn_latent_up.weight", moeWidth, uint64(spec.EmbeddingLength))
				if latentErr != nil {
					return Weights{}, latentErr
				}
				layer.FeedForwardLatentDown = &latentDown
				layer.FeedForwardLatentUp = &latentUp
			}
			if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
		if layerPlan.DenseWeights.useSecondaryAttentionNorm {
			if normErr := loadOptionalWeightBias(
				catalog, prefix,
				requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, uint64(spec.EmbeddingLength)),
				"secondary attention norm bias has no weight",
			); normErr != nil {
				return Weights{}, normErr
			}
		}
		if layerPlan.DenseWeights.requireSubNorm {
			attentionSubNorm, subNormErr := catalog.required(
				prefix+"attn_sub_norm.weight", uint64(spec.EmbeddingLength),
			)
			if subNormErr != nil {
				return Weights{}, subNormErr
			}
			feedForwardSubNorm, subNormErr := catalog.required(
				prefix+"ffn_sub_norm.weight", uint64(spec.FeedForwardLength),
			)
			if subNormErr != nil {
				return Weights{}, subNormErr
			}
			layer.AttentionSubNorm = &attentionSubNorm
			layer.FeedForwardSubNorm = &feedForwardSubNorm
			if scaleErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
		if layerPlan.DeciSparse && spec.LayerHeadCount(block) == 0 {
			// Attention-free layer.
		} else if layerPlan.DeciSparse && spec.LayerKVHeadCount(block) == 0 {
			if layer.AttentionOutput, err = catalog.required(
				prefix+"attn_output.weight",
				uint64(spec.EmbeddingLength),
				uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else if handled, mixerErr := loadRecurrentMixerLayer(
			catalog, prefix, spec, layer, block, layerPlan.Mixer,
			layerPlan.AttentionGraph.deltaProjection, layerPlan.Recurrent,
			queryLength, keyLength, valueLength, attentionOutputLength,
		); mixerErr != nil {
			return Weights{}, mixerErr
		} else if handled {
		} else if profile.Has(ArchitectureSharedKV) {
			if layer.AttentionQ, err = catalog.required(
				prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength,
			); err != nil {
				return Weights{}, err
			}
			if spec.LayerHasKV(block) {
				if layer.AttentionK, err = catalog.required(
					prefix+"attn_k.weight", uint64(spec.EmbeddingLength), keyLength,
				); err != nil {
					return Weights{}, err
				}
				if _, ok := catalog.tensors[prefix+"attn_v.weight"]; ok {
					value, valueErr := catalog.required(prefix+"attn_v.weight", uint64(spec.EmbeddingLength), valueLength)
					if valueErr != nil {
						return Weights{}, valueErr
					}
					layer.AttentionV = value
				} else if layerPlan.splitProjection() {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_v.weight")
				}
			}
			if layer.AttentionOutput, err = catalog.required(
				prefix+"attn_output.weight", attentionOutputLength, uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else if layerPlan.Attention == AttentionLatent || layerPlan.Attention == AttentionSparseLatent {
			nope := uint64(spec.KeyLength - spec.RopeDimensionCount)
			if profile.LatentAttention == latentAttentionNeoXResidualScale || (profile.Has(ArchitectureLatentKVLayout) && spec.QLoRARank > 0) {
				if layer.AttentionQ, err = catalog.required(
					prefix+"attn_q_a.weight", uint64(spec.EmbeddingLength), uint64(spec.QLoRARank),
				); err != nil {
					return Weights{}, err
				}
				qB, qBErr := catalog.required(prefix+"attn_q_b.weight", uint64(spec.QLoRARank), queryLength)
				if qBErr != nil {
					return Weights{}, qBErr
				}
				layer.AttentionQB = &qB
				qNorm, qNormErr := catalog.required(prefix+"attn_q_a_norm.weight", uint64(spec.QLoRARank))
				if qNormErr != nil {
					return Weights{}, qNormErr
				}
				layer.AttentionQNorm = &qNorm
			} else if layer.AttentionQ, err = catalog.required(
				prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength,
			); err != nil {
				return Weights{}, err
			}
			mlaTensors := []tensorRequirement{
				requiredTensorPointer("attn_kv_a_mqa.weight", &layer.AttentionKVAMQA,
					uint64(spec.EmbeddingLength), uint64(spec.KVLoRARank+spec.RopeDimensionCount)),
				requiredTensorPointer("attn_kv_a_norm.weight", &layer.AttentionKVANorm, uint64(spec.KVLoRARank)),
			}
			if _, modern := catalog.tensors[prefix+"attn_k_b.weight"]; modern {
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
			if itemErr := loadTensorRequirements(catalog, prefix, mlaTensors); itemErr != nil {
				return Weights{}, itemErr
			}
			if spec.LayerHasFullIndexer(block) {
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredTensorPointer("indexer.k_norm.weight", &layer.IndexerKNorm, uint64(spec.IndexerKeyLength)),
					requiredTensorPointer("indexer.k_norm.bias", &layer.IndexerKNormBias, uint64(spec.IndexerKeyLength)),
					requiredTensorPointer("indexer.proj.weight", &layer.IndexerProjection, uint64(spec.EmbeddingLength), uint64(spec.IndexerHeadCount)),
					requiredTensorPointer("indexer.attn_k.weight", &layer.IndexerAttentionK, uint64(spec.EmbeddingLength), uint64(spec.IndexerKeyLength)),
					requiredTensorPointer("indexer.attn_q_b.weight", &layer.IndexerAttentionQB, uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount)*uint64(spec.IndexerKeyLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			}
			if layer.AttentionOutput, err = catalog.required(
				prefix+"attn_output.weight", uint64(spec.HeadCount)*uint64(spec.ValueLength), uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else if attentionErr := loadStandardAttentionCatalog(
			catalog, prefix, spec, layer,
			queryLength, keyLength, valueLength, attentionOutputLength,
		); attentionErr != nil {
			return Weights{}, attentionErr
		}
		qkPlan := layerPlan.QKPreprocess
		if !layer.Recurrent && (qkPlan.Heads == qkNormWeighted || qkPlan.PostRotary == qkNormWeighted) {
			shape := []uint64{uint64(spec.KeyLength)}
			if normErr := loadQKNormPair(catalog, prefix, layer, shape, shape, ""); normErr != nil {
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
			if normErr := loadTensorRequirements(catalog, prefix, requirements); normErr != nil {
				return Weights{}, normErr
			}
		}
		if qkPlan.Heads == qkNormOptionalWeighted {
			shape := []uint64{uint64(spec.KeyLength)}
			if normErr := loadQKNormPair(catalog, prefix, layer, shape, shape, spec.Architecture); normErr != nil {
				return Weights{}, normErr
			}
		}
		if qkPlan.Heads == qkNormOptionalWeighted && layerPlan.AttentionOutput.gate != attentionGateNone {
			if gateErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
				optionalTensorPointer("attn_gate.weight", &layer.AttentionOutputGate,
					uint64(spec.EmbeddingLength), uint64(spec.LayerHeadCount(block))),
			}); gateErr != nil {
				return Weights{}, gateErr
			}
		}
		if layerPlan.AttentionGraph.UseSinks {
			sinkRequirement := optionalF32TensorPointer(
				"attn_sinks.weight", &layer.AttentionSinks, uint64(spec.LayerHeadCount(block)))
			requirements := []tensorRequirement{sinkRequirement}
			if layerPlan.DenseWeights.requireAttentionSinks {
				requirements[0] = requiredF32TensorPointer(
					"attn_sinks.weight", &layer.AttentionSinks, uint64(spec.LayerHeadCount(block)))
				requirements = append(requirements, requiredTensorPointer(
					"post_attention_norm.weight", &layer.AttentionPostNorm, uint64(spec.EmbeddingLength)))
			}
			if sinkErr := loadTensorRequirements(catalog, prefix, requirements); sinkErr != nil {
				return Weights{}, sinkErr
			}
		}
		if layerPlan.DenseWeights.requireAttentionGate {
			heads := uint64(spec.LayerHeadCount(block))
			wide := heads * uint64(spec.ValueLength)
			shapes := [][]uint64{{uint64(spec.EmbeddingLength), wide}}
			if layerPlan.AttentionOutput.flatGateElse {
				shapes = append(shapes, []uint64{uint64(spec.EmbeddingLength), heads})
			}
			if gateErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
				requiredTensorPointerShapes("attn_gate.weight", &layer.AttentionOutputGate, shapes...),
			}); gateErr != nil {
				return Weights{}, gateErr
			}
		}
		headQKNorm := qkPlan.Heads == qkNormAffine || qkPlan.Heads == qkNormConfiguredNoBias ||
			qkPlan.Heads == qkNormLayer
		if headQKNorm {
			optionalLabel := ""
			if qkPlan.Heads == qkNormLayer {
				optionalLabel = spec.Architecture
			}
			if normErr := loadQKNormPair(
				catalog, prefix, layer,
				[]uint64{uint64(spec.KeyLength), uint64(spec.HeadCount)},
				[]uint64{uint64(spec.KeyLength), uint64(spec.HeadCountKV)}, optionalLabel,
			); normErr != nil {
				return Weights{}, normErr
			}
		}
		if qkPlan.Heads == qkNormAffine {
			if biasErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
				optionalTensorPointer("attn_q_norm.bias", &layer.AttentionQNormBias,
					uint64(spec.KeyLength), uint64(spec.HeadCount)),
				optionalTensorPointer("attn_k_norm.bias", &layer.AttentionKNormBias,
					uint64(spec.KeyLength), uint64(spec.HeadCountKV)),
			}); biasErr != nil {
				return Weights{}, biasErr
			}
		}
		if profile.LayerTopology == LayerTopologyCausalPostQKNormSkip {
			qNorm, normErr := catalog.required(
				prefix+"attn_q_norm.weight", 1, uint64(spec.HeadCount),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layerScale, scaleErr := catalog.required(prefix+"layer_out_scale.weight", 1)
			if scaleErr != nil {
				return Weights{}, scaleErr
			}
			layer.LayerOutputScale = &layerScale
		}
		if qkPlan.Projection == qkNormWeighted {
			if normErr := loadQKNormPair(
				catalog, prefix, layer, []uint64{queryLength}, []uint64{keyLength}, "",
			); normErr != nil {
				return Weights{}, normErr
			}
		}
		if profile.EncoderOperator.usesALiBiQKNorm() {
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
					catalog, prefix,
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
				catalog, prefix,
				requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, uint64(spec.EmbeddingLength)),
				"JinaBERT v2 secondary norm bias has no weight",
			); normErr != nil {
				return Weights{}, normErr
			}
		}
		if !layer.Recurrent && !profile.ModelCatalog.SkipAttentionOutputBias {
			if biasErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
				optionalF32TensorPointer("attn_output.bias", &layer.AttentionOutputBias,
					uint64(spec.EmbeddingLength)),
			}); biasErr != nil {
				return Weights{}, biasErr
			}
		}
		if layerPlan.DenseWeights.requireAttentionOutputBias && layer.AttentionOutputBias == nil {
			return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
		}
		if !layer.Recurrent && layer.AttentionQKV == nil &&
			(!layerPlan.DeciSparse || spec.LayerKVHeadCount(block) > 0) {
			if biasErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
				if _, ok := catalog.tensors[prefix+normNames.FeedForwardWeight]; !ok {
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
			if normErr := loadTensorRequirements(catalog, prefix, requirements); normErr != nil {
				return Weights{}, normErr
			}
		}
		if profile.Has(ArchitecturePerLayerEmbeddings) {
			if profile.LayerTopology == LayerTopologySharedKVAdapter {
				if scaleErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					optionalF32TensorPointer("layer_output_scale.weight", &layer.LayerOutputScale, 1),
				}); scaleErr != nil {
					return Weights{}, scaleErr
				}
			}
			if spec.EmbeddingPerLayer > 0 {
				if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
					requiredTensorPointer("per_layer_inp_gate.weight", &layer.PerLayerInputGate, uint64(spec.EmbeddingLength), uint64(spec.EmbeddingPerLayer)),
					requiredTensorPointer("per_layer_proj.weight", &layer.PerLayerProjection, uint64(spec.EmbeddingPerLayer), uint64(spec.EmbeddingLength)),
					requiredTensorPointer("per_layer_post_norm.weight", &layer.PerLayerPostNorm, uint64(spec.EmbeddingLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			}
		}
		if layerPlan.splitProjection() {
			if itemErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
		if layerPlan.Mixer == recurrentMixerSelectiveScan || layerPlan.Mixer == recurrentMixerGroupedSelectiveScan {
			continue
		}
		feedForwardNormName := normPlan.FeedForwardNormTensor()
		if layerPlan.ResidualStages.kind == residualStable {
			if normErr := loadOptionalWeightBias(
				catalog, prefix,
				requiredTensor(feedForwardNormName, &layer.FeedForwardNorm, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("ffn_norm.bias", &layer.FeedForwardNormBias, uint64(spec.EmbeddingLength)),
				"StableLM FFN norm bias has no weight",
			); normErr != nil {
				return Weights{}, normErr
			}
		} else if !layerPlan.DenseWeights.requireExpertProjectionBiases && normPlan.PreFeedForward &&
			(!layerPlan.DeciSparse || spec.LayerFeedForwardLength(block) > 0) &&
			profile.Residual != ResidualParallel &&
			!spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() {
			if layer.FeedForwardNorm, err = catalog.required(prefix+feedForwardNormName, uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if spec.RequiresLayerNormBias() {
				feedForwardNormBias, biasErr := catalog.required(
					prefix+"ffn_norm.bias",
					uint64(spec.EmbeddingLength),
				)
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.FeedForwardNormBias = &feedForwardNormBias
			}
		}
		if layerUsesMoECatalog(catalog, prefix, spec, block, isDraftBlock) {
			loadDense, moeErr := loadMoECatalog(catalog, prefix, spec, layer)
			if moeErr != nil {
				return Weights{}, moeErr
			}
			if !loadDense {
				continue
			}
		}
		if ffnErr := loadDenseFFNCatalog(catalog, prefix, spec, layer, block); ffnErr != nil {
			return Weights{}, ffnErr
		}
	}
	return result, nil
}

func (l *layerCatalogLoader) loadDraftCatalogs(result Weights) (Weights, error) {
	catalog := l.catalog
	spec, profile, draftPlan, mtpOnly := l.spec, l.profile, l.draftPlan, l.mtpOnly
	cohere2HasMTP, cohere2MTPOnly := l.cohere2HasMTP, l.cohere2MTPOnly
	if draftPlan.Kind == DraftSingleCatalog && draftPlan.SessionEligible() {
		prefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, 0))
		mtp := &SingleDraftWeights{}
		mtp.MTPOnly = mtpOnly
		mtp.Layer.Recurrent = false
		shapes := spec.TensorShapes(0)
		queryLength := shapes.QueryProjectionWidth()
		keyLength := shapes.KeyProjectionWidth()
		valueLength := shapes.ValueProjectionWidth()
		if loadErr := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
		if loadErr := loadMTPCommonWeights(catalog, prefix, spec, mtpCommonDestinations{
			ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
			tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
		}); loadErr != nil {
			return Weights{}, loadErr
		}
		qNorm, loadErr := catalog.required(prefix+"attn_q_norm.weight", uint64(spec.KeyLength))
		if loadErr != nil {
			return Weights{}, loadErr
		}
		kNorm, loadErr := catalog.required(prefix+"attn_k_norm.weight", uint64(spec.KeyLength))
		if loadErr != nil {
			return Weights{}, loadErr
		}
		mtp.Layer.AttentionQNorm, mtp.Layer.AttentionKNorm = &qNorm, &kNorm
		result.SingleCatalogDraft = mtp
	}
	if (draftPlan.Kind == DraftAppendedMultiCarry || draftPlan.Kind == DraftAppendedMulti) && draftPlan.HasHead(0) {
		heads := make([]AppendedDraftWeights, draftPlan.Heads)
		for offset := range draftPlan.Heads {
			block := draftPlan.Block(spec.BlockCount, offset)
			prefix := fmt.Sprintf("blk.%d.", block)
			mtp := &heads[offset]
			mtp.Layer = result.Layers[block]
			if loadErr := loadMTPCommonWeights(catalog, prefix, spec, mtpCommonDestinations{
				ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
				tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
			}); loadErr != nil {
				return Weights{}, loadErr
			}
		}
		if draftPlan.Kind == DraftAppendedMulti {
			result.AppendedMultiDraft = heads
		} else {
			result.AppendedMultiCarryDraft = heads
		}
		result.Layers = result.Layers[:spec.BlockCount]
	}
	if cohere2HasMTP {
		block := draftPlan.Block(spec.BlockCount, 0)
		prefix := fmt.Sprintf("blk.%d.", block)
		mtp := &SingleDraftWeights{MTPOnly: cohere2MTPOnly, Layer: result.Layers[block]}
		if loadErr := loadMTPCommonWeights(catalog, prefix, spec, mtpCommonDestinations{
			ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
			tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
		}); loadErr != nil {
			return Weights{}, loadErr
		}
		result.OptionalCatalogDraft = mtp
		if cohere2MTPOnly {
			result.Layers = result.Layers[:0]
		} else {
			result.Layers = result.Layers[:spec.BlockCount]
		}
	}
	if draftPlan.Kind == DraftAppendedSingle && draftPlan.HasHead(0) {
		result.AppendedSingleDraft = make([]AppendedDraftWeights, spec.NextNPredictLayers)
		for offset := range draftPlan.Heads {
			block := draftPlan.Block(spec.BlockCount, offset)
			prefix := fmt.Sprintf("blk.%d.", block)
			mtp := &result.AppendedSingleDraft[offset]
			mtp.Layer = result.Layers[block]
			if loadErr := loadMTPCommonWeights(catalog, prefix, spec, mtpCommonDestinations{
				ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
				tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
			}); loadErr != nil {
				return Weights{}, loadErr
			}
			if profile.ModelCatalog.DraftLayerOutputNorm {
				loaded, loadErr := catalog.required(prefix+"layer_output_norm.weight", uint64(spec.EmbeddingLength))
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
