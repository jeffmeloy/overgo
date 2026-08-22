package model

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

var (
	projectionBiasNames = [...]string{
		"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias",
		"ffn_gate.bias", "ffn_up.bias", "ffn_down.bias",
	}
	longRoPEFactorNames = [...]string{"rope_factors_long.weight", "rope_factors_short.weight"}
)

const firstBlockTensorPrefix = "blk.0."

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
		_, hasQuery := catalog.tensor(prefix + attentionQueryNormTensor)
		_, hasKey := catalog.tensor(prefix + attentionKeyNormTensor)
		if hasQuery != hasKey {
			return fmt.Errorf("%s Q/K norm tensors must both be present or absent", optionalLabel)
		}
		if !hasQuery {
			return nil
		}
	}
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm, queryShape...),
		requiredTensorPointer(attentionKeyNormTensor, &layer.AttentionKNorm, keyShape...),
	})
}

func loadOptionalWeightBias(
	catalog weightCatalog,
	prefix string,
	weight, bias tensorBinding,
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
	requirements := []tensorBinding{weight}
	if hasBias {
		requirements = append(requirements, bias)
	}
	return bindTensorProgram(catalog, prefix, requirements)
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
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensor("nextn.eh_proj.weight", destination.ehProjection, tensor.PairedExtent*width, width),
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
	width uint64,
	spec Spec,
	layer *LayerWeights,
	withRouter bool,
) error {
	requirements := []tensorBinding{
		requiredTensorPointer("ffn_gate_shexp.weight", &layer.FeedForwardSharedGate, width, uint64(spec.SharedExpertFF)),
		requiredTensorPointer("ffn_up_shexp.weight", &layer.FeedForwardSharedUp, width, uint64(spec.SharedExpertFF)),
		requiredTensorPointer("ffn_down_shexp.weight", &layer.FeedForwardSharedDown, uint64(spec.SharedExpertFF), width),
	}
	if withRouter {
		requirements = append(requirements,
			requiredTensorPointer("ffn_gate_inp_shexp.weight", &layer.FeedForwardSharedRouter, width),
		)
	}
	return bindTensorProgram(catalog, prefix, requirements)
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
	gate := optionalTensorPointer(feedForwardGateWeightTensor, &layer.FeedForwardGate, p.width, p.feedForward)
	gate.optional = !requireGate
	if err := bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, p.width),
		requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, p.width, p.query),
		requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, p.width, p.key),
		requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, p.width, p.value),
		requiredTensorPointer("attn_o.weight", &layer.AttentionOutput, p.output, p.width),
		requiredTensorPointer(feedForwardNormWeightTensor, &layer.FeedForwardNorm, p.width),
		gate,
		requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp, p.width, p.feedForward),
		requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown, p.feedForward, p.width),
	}); err != nil {
		return err
	}
	bias := requiredTensorPointer(
		"attn_rel_b.weight", &layer.AttentionRelativeBias, p.heads, p.relativeBuckets)
	bias.optional = inheritedBias != nil
	if err := bindTensorProgram(catalog, prefix, []tensorBinding{bias}); err != nil {
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

func (c weightCatalog) selectName(prefix, primary, alternate string) string {
	if alternate == "" {
		return primary
	}
	if _, ok := c.tensors[prefix+primary]; ok {
		return primary
	}
	return alternate
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
	tokenEmbeddingAlternate := ""
	if profile.ModelCatalog.TiedTokenEmbedding {
		tokenEmbeddingAlternate = outputWeightTensor
	}
	if embeddingErr := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensor(catalog.selectName("", tokenEmbeddingWeightTensor, tokenEmbeddingAlternate),
			&result.TokenEmbedding, uint64(spec.EmbeddingLength), uint64(spec.VocabularySize)),
	}); embeddingErr != nil {
		return Weights{}, embeddingErr
	}
	if profile.ModelCatalog.PositionEmbedding == positionEmbeddingRequired {
		if positionErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("position_embd.weight", &result.PositionEmbedding,
				uint64(spec.EmbeddingLength), uint64(spec.ContextLength)),
		}); positionErr != nil {
			return Weights{}, positionErr
		}
	}
	if normPlan.PostNormLayout == PostNormLayoutOutputLayer {
		tokenTypes := optionalTensorPointer(
			"token_types.weight", &result.TokenTypeEmbedding,
			uint64(spec.EmbeddingLength), uint64(spec.TokenTypeCount),
		)
		tokenTypes.optional = !profile.ModelCatalog.RequireTokenTypes
		if typeErr := bindTensorProgram(catalog, "", []tensorBinding{tokenTypes}); typeErr != nil {
			return Weights{}, typeErr
		}
		if normErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("token_embd_norm.weight", &result.TokenEmbeddingNorm,
				uint64(spec.EmbeddingLength)),
			requiredTensorPointer("token_embd_norm.bias", &result.TokenEmbeddingNormBias,
				uint64(spec.EmbeddingLength)),
		}); normErr != nil {
			return Weights{}, normErr
		}
	}
	if profile.ModelCatalog.TokenNorm == tokenNormWeight {
		if normErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("token_embd_norm.weight", &result.TokenEmbeddingNorm,
				uint64(spec.EmbeddingLength)),
		}); normErr != nil {
			return Weights{}, normErr
		}
	}
	if profile.ModelCatalog.PositionEmbedding == positionEmbeddingOptional {
		if positionErr := bindTensorProgram(catalog, "", []tensorBinding{
			optionalTensorPointer("position_embd.weight", &result.PositionEmbedding,
				uint64(spec.EmbeddingLength), uint64(spec.ContextLength)),
		}); positionErr != nil {
			return Weights{}, positionErr
		}
	}
	if profile.ModelCatalog.TokenNorm == tokenNormAffine {
		if normErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("token_embd_norm.weight", &result.TokenEmbeddingNorm,
				uint64(spec.EmbeddingLength)),
			requiredTensorPointer("token_embd_norm.bias", &result.TokenEmbeddingNormBias,
				uint64(spec.EmbeddingLength)),
		}); normErr != nil {
			return Weights{}, normErr
		}
	}
	if !spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() &&
		profile.OutputNorm != OutputNormAbsent {
		if normErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensor(profile.OutputNormTensor(), &result.OutputNorm, uint64(spec.EmbeddingLength)),
		}); normErr != nil {
			return Weights{}, normErr
		}
	}
	if spec.RequiresLayerNormBias() && profile.OutputNorm != OutputNormAbsent {
		if biasErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("output_norm.bias", &result.OutputNormBias,
				uint64(spec.EmbeddingLength)),
		}); biasErr != nil {
			return Weights{}, biasErr
		}
	}
	if profile.ModelCatalog.OptionalOutputNormBias {
		if biasErr := bindTensorProgram(catalog, "", []tensorBinding{
			optionalTensorPointer("output_norm.bias", &result.OutputNormBias,
				uint64(spec.EmbeddingLength)),
		}); biasErr != nil {
			return Weights{}, biasErr
		}
	}
	if !profile.ModelCatalog.SkipOutput {
		if outputErr := bindTensorProgram(catalog, "", []tensorBinding{
			optionalTensorPointer(outputWeightTensor, &result.Output,
				uint64(spec.EmbeddingLength), uint64(spec.VocabularySize)),
		}); outputErr != nil {
			return Weights{}, outputErr
		}
	}
	if profile.Has(ArchitectureRequiresOutput) && result.Output == nil {
		return Weights{}, fmt.Errorf("required tensor %q is missing", outputWeightTensor)
	}
	if outputErr := bindTensorProgram(catalog, "", []tensorBinding{
		optionalF32TensorPointer("output.bias", &result.OutputBias, uint64(spec.VocabularySize)),
	}); outputErr != nil {
		return Weights{}, outputErr
	}
	if profile.Has(ArchitectureClassifierHead) {
		outputCount := uint64(tensor.SingletonExtent)
		if len(spec.ClassifierLabels) > tensor.FirstOffset {
			outputCount = uint64(len(spec.ClassifierLabels))
		}
		if classifierErr := bindTensorProgram(catalog, "", []tensorBinding{
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
			if spec.Dense2FeatureIn == tensor.FirstOffset || spec.Dense2FeatureOut == tensor.FirstOffset {
				return Weights{}, errors.New("Gemma embedding dense-2 tensor has no shape metadata")
			}
			if denseErr := bindTensorProgram(catalog, "", []tensorBinding{
				requiredTensorPointer(item.Name, &result.Dense2Output,
					uint64(spec.Dense2FeatureIn), uint64(spec.Dense2FeatureOut)),
			}); denseErr != nil {
				return Weights{}, denseErr
			}
		}
		if item, ok := l.catalog.tensor("dense_3.weight"); ok {
			if spec.Dense3FeatureIn == tensor.FirstOffset || spec.Dense3FeatureOut == tensor.FirstOffset {
				return Weights{}, errors.New("Gemma embedding dense-3 tensor has no shape metadata")
			}
			if denseErr := bindTensorProgram(catalog, "", []tensorBinding{
				requiredTensorPointer(item.Name, &result.Dense3Output,
					uint64(spec.Dense3FeatureIn), uint64(spec.Dense3FeatureOut)),
			}); denseErr != nil {
				return Weights{}, denseErr
			}
		}
	}
	if profile.Has(ArchitecturePerLayerEmbeddings) && spec.EmbeddingPerLayer > tensor.FirstOffset {
		perLayerWidth := uint64(spec.EmbeddingPerLayer) * uint64(spec.BlockCount)
		if itemErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("per_layer_token_embd.weight", &result.PerLayerTokenEmbedding,
				perLayerWidth, uint64(spec.VocabularySize)),
			requiredTensorPointer("per_layer_model_proj.weight", &result.PerLayerModelProjection,
				uint64(spec.EmbeddingLength), perLayerWidth),
			requiredTensorPointer("per_layer_proj_norm.weight", &result.PerLayerProjectionNorm,
				uint64(spec.EmbeddingPerLayer)),
		}); itemErr != nil {
			return Weights{}, itemErr
		}
	}
	if profile.LayerTopology == LayerTopologySplitProjection {
		shape := []uint64{
			uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength),
			uint64(spec.AltUpCount - tensor.SingletonExtent),
		}
		if itemErr := bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("altup_proj.weight", &result.AltUpProjection, shape...),
			requiredTensorPointer("altup_unembd_proj.weight", &result.AltUpUnembedding, shape...),
		}); itemErr != nil {
			return Weights{}, itemErr
		}
	}

	trunkBlockCount := spec.BlockCount
	if draftPlan.AppendedBlocks {
		trunkBlockCount += draftPlan.Heads
	}
	cohere2HasMTP := false
	cohere2MTPOnly := false
	if draftPlan.Kind == DraftOptionalSingleCatalog && draftPlan.SessionEligible() {
		mtpPrefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, tensor.FirstOffset))
		_, cohere2HasMTP = catalog.tensors[mtpPrefix+"nextn.eh_proj.weight"]
		_, hasTrunk := catalog.tensors[firstBlockTensorPrefix+attentionNormWeightTensor]
		cohere2MTPOnly = cohere2HasMTP && !hasTrunk
		if cohere2HasMTP {
			trunkBlockCount++
		}
	}
	mtpOnly := draftPlan.Kind == DraftSingleCatalog && draftPlan.SessionEligible()
	if mtpOnly {
		_, hasTrunk := catalog.tensors[firstBlockTensorPrefix+attentionNormWeightTensor]
		mtpOnly = !hasTrunk
	}
	if mtpOnly {
		trunkBlockCount = tensor.FirstOffset
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
	for block := uint32(tensor.FirstOffset); block < trunkBlockCount; block++ {
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
		if profile.Has(ArchitectureBiasFreeProjections) {
			for _, name := range projectionBiasNames {
				if _, ok := catalog.tensors[prefix+name]; ok {
					return Weights{}, fmt.Errorf(
						"tensor %q requires unsupported %s projection biases",
						prefix+name, spec.Architecture,
					)
				}
			}
		}
		for _, unsupported := range longRoPEFactorNames {
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
			_, hasQNorm := catalog.tensors[prefix+attentionQueryNormTensor]
			_, hasKNorm := catalog.tensors[prefix+attentionKeyNormTensor]
			if hasQNorm != hasKNorm {
				return Weights{}, errors.New("MPT Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				if queryLength != uint64(spec.EmbeddingLength) || keyLength != uint64(spec.EmbeddingLength) {
					return Weights{}, errors.New("MPT Q/K norm requires full-width Q/K projections")
				}
				if normErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredF32TensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm, uint64(spec.EmbeddingLength)),
					requiredF32TensorPointer(attentionKeyNormTensor, &layer.AttentionKNorm, uint64(spec.EmbeddingLength)),
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
			if scaleErr := bindTensorProgram(catalog, prefix, []tensorBinding{
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
			if spec.RopeDimensionCount > tensor.FirstOffset {
				rotaryDimensions = spec.RopeDimensionCount
			}
			if rotaryDimensions%rotaryPairAlignment != tensor.FirstOffset {
				return Weights{}, fmt.Errorf("tensor %q has odd rotary width %d", ropeFactors.Name, rotaryDimensions)
			}
			if validateErr := validateTensorInfo(
				ropeFactors, []dtype.Type{dtype.F32},
				[]uint64{uint64(rotaryDimensions / rotaryPairAlignment)},
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
			if block > tensor.FirstOffset {
				if !hasLong {
					longFactors, hasLong = l.catalog.tensor(firstBlockTensorPrefix + "rope_factors_long.weight")
				}
				if !hasShort {
					shortFactors, hasShort = l.catalog.tensor(firstBlockTensorPrefix + "rope_factors_short.weight")
				}
			}
			if hasLong != hasShort {
				return Weights{}, fmt.Errorf("%s LongRoPE factor tensors must both be present or absent", spec.Architecture)
			}
			if spec.RopeScalingType == ropeScalingLongRoPE && !hasLong {
				return Weights{}, fmt.Errorf("%s LongRoPE factor tensors are missing", spec.Architecture)
			}
			if hasLong {
				for _, item := range []gguf.TensorInfo{longFactors, shortFactors} {
					if itemErr := validateTensorInfo(
						item, []dtype.Type{dtype.F32},
						[]uint64{uint64(spec.RopeDimensionCount / rotaryPairAlignment)},
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
			norm := optionalTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, uint64(spec.EmbeddingLength))
			norm.optional = block == tensor.FirstOffset
			if normErr := bindTensorProgram(catalog, prefix, []tensorBinding{norm}); normErr != nil {
				return Weights{}, normErr
			}
		} else if normPlan.PreAttention &&
			(!layerPlan.DeciSparse || spec.LayerHasAttention(block)) &&
			!spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() {
			requirements := []tensorBinding{
				requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm,
					uint64(spec.EmbeddingLength)),
			}
			if spec.RequiresLayerNormBias() {
				requirements = append(requirements, requiredTensorPointer(
					"attn_norm.bias", &layer.AttentionNormBias, uint64(spec.EmbeddingLength)))
			}
			if normErr := bindTensorProgram(catalog, prefix, requirements); normErr != nil {
				return Weights{}, normErr
			}
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
					tensor.PairedExtent*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
				requirements := append(
					groupedSelectiveScanTensorRequirements(spec, layer, true),
					optionalTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, convDimension),
				)
				if itemErr := bindTensorProgram(catalog, prefix, requirements); itemErr != nil {
					return Weights{}, itemErr
				}
				continue
			}
			if !spec.LayerHasFeedForward(block) {
				if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, uint64(spec.EmbeddingLength), queryLength),
					requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
					requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
					requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, attentionOutputLength, uint64(spec.EmbeddingLength)),
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
				if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp, uint64(spec.EmbeddingLength), uint64(spec.LayerFeedForwardLength(block))),
					requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown, uint64(spec.LayerFeedForwardLength(block)), uint64(spec.EmbeddingLength)),
					optionalF32TensorPointer("ffn_up.bias", &layer.FeedForwardUpBias, uint64(spec.LayerFeedForwardLength(block))),
					optionalF32TensorPointer("ffn_down.bias", &layer.FeedForwardDownBias, uint64(spec.EmbeddingLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
				continue
			}
			moeWidth := uint64(spec.EmbeddingLength)
			if spec.MoELatentSize > tensor.FirstOffset {
				moeWidth = uint64(spec.MoELatentSize)
				if latentErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredTensorPointer("ffn_latent_down.weight", &layer.FeedForwardLatentDown,
						uint64(spec.EmbeddingLength), moeWidth),
					requiredTensorPointer("ffn_latent_up.weight", &layer.FeedForwardLatentUp,
						moeWidth, uint64(spec.EmbeddingLength)),
				}); latentErr != nil {
					return Weights{}, latentErr
				}
			}
			if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
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
			if subNormErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer("attn_sub_norm.weight", &layer.AttentionSubNorm,
					uint64(spec.EmbeddingLength)),
				requiredTensorPointer("ffn_sub_norm.weight", &layer.FeedForwardSubNorm,
					uint64(spec.FeedForwardLength)),
			}); subNormErr != nil {
				return Weights{}, subNormErr
			}
			if scaleErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				optionalF32TensorPointer("attn_q.scale", &layer.AttentionQScale, tensor.SingletonExtent),
				optionalF32TensorPointer("attn_k.scale", &layer.AttentionKScale, tensor.SingletonExtent),
				optionalF32TensorPointer("attn_v.scale", &layer.AttentionVScale, tensor.SingletonExtent),
				optionalF32TensorPointer("attn_output.scale", &layer.AttentionOutputScale, tensor.SingletonExtent),
				optionalF32TensorPointer("ffn_gate.scale", &layer.FeedForwardGateScale, tensor.SingletonExtent),
				optionalF32TensorPointer("ffn_up.scale", &layer.FeedForwardUpScale, tensor.SingletonExtent),
				optionalF32TensorPointer("ffn_down.scale", &layer.FeedForwardDownScale, tensor.SingletonExtent),
			}); scaleErr != nil {
				return Weights{}, scaleErr
			}
		}
		if layerPlan.DeciSparse && !spec.LayerHasAttention(block) {
			// Attention-free layer.
		} else if layerPlan.DeciSparse && !spec.LayerHasKVHeads(block) {
			if outputErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput,
					uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)),
			}); outputErr != nil {
				return Weights{}, outputErr
			}
		} else if handled, mixerErr := loadRecurrentMixerLayer(
			catalog, prefix, spec, layer, layerPlan,
			queryLength, keyLength, valueLength, attentionOutputLength,
		); mixerErr != nil {
			return Weights{}, mixerErr
		} else if handled {
		} else if profile.Has(ArchitectureSharedKV) {
			requirements := []tensorBinding{
				requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ,
					uint64(spec.EmbeddingLength), queryLength),
			}
			if spec.LayerHasKV(block) {
				requirements = append(requirements, requiredTensorPointer(
					attentionKeyWeightTensor, &layer.AttentionK, uint64(spec.EmbeddingLength), keyLength))
				if _, ok := catalog.tensors[prefix+attentionValueWeightTensor]; ok {
					requirements = append(requirements, requiredTensorPointer(
						attentionValueWeightTensor, &layer.AttentionV, uint64(spec.EmbeddingLength), valueLength))
				} else if layerPlan.splitProjection() {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+attentionValueWeightTensor)
				}
			}
			requirements = append(requirements, requiredTensorPointer(
				attentionOutputWeightTensor, &layer.AttentionOutput,
				attentionOutputLength, uint64(spec.EmbeddingLength)))
			if attentionErr := bindTensorProgram(catalog, prefix, requirements); attentionErr != nil {
				return Weights{}, attentionErr
			}
		} else if layerPlan.Attention == AttentionLatent || layerPlan.Attention == AttentionSparseLatent {
			if itemErr := loadLatentAttentionCatalog(
				catalog, prefix, spec, layer, layerPlan,
				queryLength, uint64(spec.HeadCount)*uint64(spec.ValueLength),
			); itemErr != nil {
				return Weights{}, itemErr
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
			requirements := []tensorBinding{
				requiredTensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm, shape...),
			}
			if spec.LayerHasKV(block) {
				requirements = append(requirements,
					requiredTensorPointer(attentionKeyNormTensor, &layer.AttentionKNorm, shape...))
			}
			if normErr := bindTensorProgram(catalog, prefix, requirements); normErr != nil {
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
			if gateErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				optionalTensorPointer("attn_gate.weight", &layer.AttentionOutputGate,
					uint64(spec.EmbeddingLength), uint64(spec.LayerHeadCount(block))),
			}); gateErr != nil {
				return Weights{}, gateErr
			}
		}
		if layerPlan.AttentionGraph.UseSinks {
			sinkRequirement := optionalF32TensorPointer(
				"attn_sinks.weight", &layer.AttentionSinks, uint64(spec.LayerHeadCount(block)))
			requirements := []tensorBinding{sinkRequirement}
			if layerPlan.DenseWeights.requireAttentionSinks {
				requirements[tensor.FirstOffset] = requiredF32TensorPointer(
					"attn_sinks.weight", &layer.AttentionSinks, uint64(spec.LayerHeadCount(block)))
				requirements = append(requirements, requiredTensorPointer(
					postAttentionNormWeightTensor, &layer.AttentionPostNorm, uint64(spec.EmbeddingLength)))
			}
			if sinkErr := bindTensorProgram(catalog, prefix, requirements); sinkErr != nil {
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
			if gateErr := bindTensorProgram(catalog, prefix, []tensorBinding{
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
			if biasErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				optionalTensorPointer("attn_q_norm.bias", &layer.AttentionQNormBias,
					uint64(spec.KeyLength), uint64(spec.HeadCount)),
				optionalTensorPointer("attn_k_norm.bias", &layer.AttentionKNormBias,
					uint64(spec.KeyLength), uint64(spec.HeadCountKV)),
			}); biasErr != nil {
				return Weights{}, biasErr
			}
		}
		if profile.LayerTopology == LayerTopologyCausalPostQKNormSkip {
			if normErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm,
					tensor.SingletonExtent, uint64(spec.HeadCount)),
				requiredTensorPointer("layer_out_scale.weight", &layer.LayerOutputScale, tensor.SingletonExtent),
			}); normErr != nil {
				return Weights{}, normErr
			}
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
			if biasErr := bindTensorProgram(catalog, prefix, []tensorBinding{
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
			(!layerPlan.DeciSparse || spec.LayerHasKVHeads(block)) {
			if biasErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				optionalF32TensorPointer("attn_q.bias", &layer.AttentionQBias, layer.AttentionQ.MatrixRows()),
				optionalF32TensorPointer("attn_k.bias", &layer.AttentionKBias, layer.AttentionK.MatrixRows()),
				optionalF32TensorPointer("attn_v.bias", &layer.AttentionVBias, layer.AttentionV.MatrixRows()),
			}); biasErr != nil {
				return Weights{}, biasErr
			}
		}
		if normPlan.PostAttention {
			normNames := normPlan.PostNormTensors()
			normNames.FeedForwardWeight = catalog.selectName(
				prefix, normNames.FeedForwardWeight, normNames.FeedForwardAlternate,
			)
			width := uint64(spec.EmbeddingLength)
			requirements := []tensorBinding{
				requiredTensorPointer(normNames.AttentionWeight, &layer.AttentionPostNorm, width),
				requiredTensorPointer(normNames.FeedForwardWeight, &layer.FeedForwardPostNorm, width),
			}
			if normNames.AttentionBias != "" {
				requirements = append(requirements,
					requiredTensorPointer(normNames.AttentionBias, &layer.AttentionPostNormBias, width),
					requiredTensorPointer(normNames.FeedForwardBias, &layer.FeedForwardPostNormBias, width),
				)
			}
			if normErr := bindTensorProgram(catalog, prefix, requirements); normErr != nil {
				return Weights{}, normErr
			}
		}
		if profile.Has(ArchitecturePerLayerEmbeddings) {
			if profile.LayerTopology == LayerTopologySharedKVAdapter {
				if scaleErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					optionalF32TensorPointer(
						"layer_output_scale.weight", &layer.LayerOutputScale, tensor.SingletonExtent,
					),
				}); scaleErr != nil {
					return Weights{}, scaleErr
				}
			}
			if spec.EmbeddingPerLayer > tensor.FirstOffset {
				if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredTensorPointer("per_layer_inp_gate.weight", &layer.PerLayerInputGate, uint64(spec.EmbeddingLength), uint64(spec.EmbeddingPerLayer)),
					requiredTensorPointer("per_layer_proj.weight", &layer.PerLayerProjection, uint64(spec.EmbeddingPerLayer), uint64(spec.EmbeddingLength)),
					requiredTensorPointer("per_layer_post_norm.weight", &layer.PerLayerPostNorm, uint64(spec.EmbeddingLength)),
				}); itemErr != nil {
					return Weights{}, itemErr
				}
			}
		}
		if layerPlan.splitProjection() {
			if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
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
				requiredTensorPointer(feedForwardNormName, &layer.FeedForwardNorm, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("ffn_norm.bias", &layer.FeedForwardNormBias, uint64(spec.EmbeddingLength)),
				"StableLM FFN norm bias has no weight",
			); normErr != nil {
				return Weights{}, normErr
			}
		} else if !layerPlan.DenseWeights.requireExpertProjectionBiases && normPlan.PreFeedForward &&
			(!layerPlan.DeciSparse || spec.LayerHasFeedForward(block)) &&
			profile.Residual != ResidualParallel &&
			!spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() {
			requirements := []tensorBinding{
				requiredTensorPointer(feedForwardNormName, &layer.FeedForwardNorm,
					uint64(spec.EmbeddingLength)),
			}
			if spec.RequiresLayerNormBias() {
				requirements = append(requirements, requiredTensorPointer(
					"ffn_norm.bias", &layer.FeedForwardNormBias, uint64(spec.EmbeddingLength)))
			}
			if normErr := bindTensorProgram(catalog, prefix, requirements); normErr != nil {
				return Weights{}, normErr
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
		prefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, tensor.FirstOffset))
		mtp := &SingleDraftWeights{}
		mtp.MTPOnly = mtpOnly
		mtp.Layer.Recurrent = false
		shapes := spec.TensorShapes(tensor.FirstOffset)
		queryLength := shapes.QueryProjectionWidth()
		keyLength := shapes.KeyProjectionWidth()
		valueLength := shapes.ValueProjectionWidth()
		if loadErr := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer(attentionNormWeightTensor, &mtp.Layer.AttentionNorm, uint64(spec.EmbeddingLength)),
			requiredTensorPointer(postAttentionNormWeightTensor, &mtp.Layer.FeedForwardNorm, uint64(spec.EmbeddingLength)),
			requiredTensorPointer(attentionQueryWeightTensor, &mtp.Layer.AttentionQ, uint64(spec.EmbeddingLength), uint64(draftPlan.QueryCopies)*queryLength),
			requiredTensorPointer(attentionKeyWeightTensor, &mtp.Layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
			requiredTensorPointer(attentionValueWeightTensor, &mtp.Layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
			requiredTensorPointer(attentionOutputWeightTensor, &mtp.Layer.AttentionOutput, queryLength, uint64(spec.EmbeddingLength)),
			requiredTensorPointer(feedForwardGateWeightTensor, &mtp.Layer.FeedForwardGate, uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensorPointer(feedForwardUpWeightTensor, &mtp.Layer.FeedForwardUp, uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensorPointer(feedForwardDownWeightTensor, &mtp.Layer.FeedForwardDown, uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)),
			requiredTensorPointer(attentionQueryNormTensor, &mtp.Layer.AttentionQNorm, uint64(spec.KeyLength)),
			requiredTensorPointer(attentionKeyNormTensor, &mtp.Layer.AttentionKNorm, uint64(spec.KeyLength)),
		}); loadErr != nil {
			return Weights{}, loadErr
		}
		if loadErr := loadMTPCommonWeights(catalog, prefix, spec, mtpCommonDestinations{
			ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
			tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
		}); loadErr != nil {
			return Weights{}, loadErr
		}
		result.SingleCatalogDraft = mtp
	}
	if (draftPlan.Kind == DraftAppendedMultiCarry || draftPlan.Kind == DraftAppendedMulti) &&
		draftPlan.HasHead(tensor.FirstOffset) {
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
		block := draftPlan.Block(spec.BlockCount, tensor.FirstOffset)
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
			result.Layers = result.Layers[:tensor.FirstOffset]
		} else {
			result.Layers = result.Layers[:spec.BlockCount]
		}
	}
	if draftPlan.Kind == DraftAppendedSingle && draftPlan.HasHead(tensor.FirstOffset) {
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
				if loadErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredTensorPointer("layer_output_norm.weight", &mtp.LayerOutputNorm,
						uint64(spec.EmbeddingLength)),
				}); loadErr != nil {
					return Weights{}, loadErr
				}
			}
		}
		result.Layers = result.Layers[:spec.BlockCount]
	}
	return result, nil
}
