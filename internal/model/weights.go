package model

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
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
	optional bool,
) error {
	if optional {
		_, hasQuery := catalog.tensor(prefix + attentionQueryNormTensor)
		_, hasKey := catalog.tensor(prefix + attentionKeyNormTensor)
		if hasQuery != hasKey {
			return fmt.Errorf("tensors %q and %q must both be present or absent",
				prefix+attentionQueryNormTensor, prefix+attentionKeyNormTensor)
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
) error {
	_, hasWeight := catalog.tensor(prefix + weight.name)
	_, hasBias := catalog.tensor(prefix + bias.name)
	if !hasWeight {
		if hasBias {
			return fmt.Errorf("tensor %q requires tensor %q", prefix+bias.name, prefix+weight.name)
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

func mtpCommonBindings(
	spec Spec,
	destination mtpCommonDestinations,
) []tensorBinding {
	width := uint64(spec.EmbeddingLength)
	return []tensorBinding{
		requiredTensor("nextn.eh_proj.weight", destination.ehProjection, tensor.PairedExtent*width, width),
		requiredTensor("nextn.enorm.weight", destination.embeddingNorm, width),
		requiredTensor("nextn.hnorm.weight", destination.hiddenNorm, width),
		optionalTensorPointer("nextn.embed_tokens.weight", destination.tokenEmbedding, width, uint64(spec.VocabularySize)),
		optionalTensorPointer("nextn.shared_head_norm.weight", destination.outputNorm, width),
		optionalTensorPointer("nextn.shared_head_head.weight", destination.output, width, uint64(spec.VocabularySize)),
	}
}

func loadSharedExpertWeights(
	catalog weightCatalog,
	prefix string,
	width uint64,
	spec Spec,
	layer *LayerWeights,
	policy sharedExpertCatalogPolicy,
) error {
	requirements := []tensorBinding{
		requiredTensorPointer("ffn_up_shexp.weight", &layer.FeedForwardSharedUp, width, uint64(spec.SharedExpertFF)),
		requiredTensorPointer("ffn_down_shexp.weight", &layer.FeedForwardSharedDown, uint64(spec.SharedExpertFF), width),
	}
	if policy != sharedExpertCatalogUngated {
		requirements = append(requirements, requiredTensorPointer(
			"ffn_gate_shexp.weight", &layer.FeedForwardSharedGate, width, uint64(spec.SharedExpertFF),
		))
	}
	if policy == sharedExpertCatalogGated {
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
	catalog           weightCatalog
	spec              Spec
	profile           ArchitectureProfile
	draftPlan         DraftPlan
	normPlan          NormalizationPlan
	trunkBlockCount   uint32
	hasOptionalDraft  bool
	optionalDraftOnly bool
	mtpOnly           bool
}

func compileModelTensorBindings(
	catalog weightCatalog,
	spec Spec,
	profile ArchitectureProfile,
	normPlan NormalizationPlan,
	result *Weights,
) ([]tensorBinding, error) {
	var bindings []tensorBinding
	alternate := ""
	if profile.ModelCatalog.TiedTokenEmbedding {
		alternate = outputWeightTensor
	}
	bindings = append(bindings, requiredTensor(
		catalog.selectName("", tokenEmbeddingWeightTensor, alternate),
		&result.TokenEmbedding,
		uint64(spec.EmbeddingLength), uint64(spec.VocabularySize),
	))
	if profile.ModelCatalog.PositionEmbedding != positionEmbeddingAbsent {
		position := optionalTensorPointer(
			"position_embd.weight", &result.PositionEmbedding,
			uint64(spec.EmbeddingLength), uint64(spec.ContextLength),
		)
		position.optional = profile.ModelCatalog.PositionEmbedding == positionEmbeddingOptional
		bindings = append(bindings, position)
	}
	outputLayerNorm := normPlan.PostNormLayout == PostNormLayoutOutputLayer
	if outputLayerNorm {
		tokenTypes := optionalTensorPointer(
			"token_types.weight", &result.TokenTypeEmbedding,
			uint64(spec.EmbeddingLength), uint64(spec.TokenTypeCount),
		)
		tokenTypes.optional = !profile.ModelCatalog.RequireTokenTypes
		bindings = append(bindings, tokenTypes)
	}
	if outputLayerNorm || profile.ModelCatalog.TokenNorm != tokenNormAbsent {
		bindings = append(bindings, requiredTensorPointer(
			"token_embd_norm.weight", &result.TokenEmbeddingNorm, uint64(spec.EmbeddingLength),
		))
	}
	if outputLayerNorm || profile.ModelCatalog.TokenNorm == tokenNormAffine {
		bindings = append(bindings, requiredTensorPointer(
			"token_embd_norm.bias", &result.TokenEmbeddingNormBias, uint64(spec.EmbeddingLength),
		))
	}
	if !spec.UsesUnweightedLayerNorm() && !spec.UsesUnweightedRMSNorm() && profile.OutputNorm != OutputNormAbsent {
		bindings = append(bindings, requiredTensor(
			profile.OutputNormTensor(), &result.OutputNorm, uint64(spec.EmbeddingLength),
		))
	}
	if profile.OutputNorm != OutputNormAbsent && (spec.RequiresLayerNormBias() || profile.ModelCatalog.OptionalOutputNormBias) {
		bias := optionalTensorPointer("output_norm.bias", &result.OutputNormBias, uint64(spec.EmbeddingLength))
		bias.optional = !spec.RequiresLayerNormBias()
		bindings = append(bindings, bias)
	}
	if !profile.ModelCatalog.SkipOutput {
		bindings = append(bindings, optionalTensorPointer(
			outputWeightTensor, &result.Output, uint64(spec.EmbeddingLength), uint64(spec.VocabularySize),
		))
	}
	bindings = append(bindings, optionalF32TensorPointer(
		"output.bias", &result.OutputBias, uint64(spec.VocabularySize),
	))
	if profile.Has(ArchitectureClassifierHead) {
		outputCount := uint64(tensor.SingletonExtent)
		if len(spec.ClassifierLabels) > tensor.FirstOffset {
			outputCount = uint64(len(spec.ClassifierLabels))
		}
		bindings = append(bindings, optionalTensorPointer(
			"cls.output.weight", &result.ClassifierOutput, uint64(spec.EmbeddingLength), outputCount,
		))
	}
	if profile.LayerTopology == LayerTopologyBidirectionalQKNorm {
		for _, projection := range []struct {
			name        string
			input       uint32
			output      uint32
			destination **gguf.TensorInfo
		}{
			{
				name:        "dense_2.weight",
				input:       spec.Dense2FeatureIn,
				output:      spec.Dense2FeatureOut,
				destination: &result.Dense2Output,
			},
			{
				name:        "dense_3.weight",
				input:       spec.Dense3FeatureIn,
				output:      spec.Dense3FeatureOut,
				destination: &result.Dense3Output,
			},
		} {
			if _, ok := catalog.tensor(projection.name); !ok {
				continue
			}
			if projection.input == tensor.FirstOffset || projection.output == tensor.FirstOffset {
				return nil, fmt.Errorf("tensor %q has no shape metadata", projection.name)
			}
			bindings = append(bindings, requiredTensorPointer(
				projection.name, projection.destination, uint64(projection.input), uint64(projection.output),
			))
		}
	}
	if profile.Has(ArchitecturePerLayerEmbeddings) && spec.EmbeddingPerLayer > tensor.FirstOffset {
		perLayerWidth := uint64(spec.EmbeddingPerLayer) * uint64(spec.BlockCount)
		bindings = append(bindings,
			requiredTensorPointer("per_layer_token_embd.weight", &result.PerLayerTokenEmbedding,
				perLayerWidth, uint64(spec.VocabularySize)),
			requiredTensorPointer("per_layer_model_proj.weight", &result.PerLayerModelProjection,
				uint64(spec.EmbeddingLength), perLayerWidth),
			requiredTensorPointer("per_layer_proj_norm.weight", &result.PerLayerProjectionNorm,
				uint64(spec.EmbeddingPerLayer)),
		)
	}
	if profile.LayerTopology == LayerTopologySplitProjection {
		shape := []uint64{
			uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength),
			uint64(spec.AltUpCount - tensor.SingletonExtent),
		}
		bindings = append(bindings,
			requiredTensorPointer("altup_proj.weight", &result.AltUpProjection, shape...),
			requiredTensorPointer("altup_unembd_proj.weight", &result.AltUpUnembedding, shape...),
		)
	}
	return bindings, nil
}

func (l *layerCatalogLoader) loadModelCatalog(result Weights) (Weights, error) {
	catalog := l.catalog
	spec, profile, draftPlan, normPlan := l.spec, l.profile, l.draftPlan, l.normPlan
	bindings, err := compileModelTensorBindings(catalog, spec, profile, normPlan, &result)
	if err != nil {
		return Weights{}, err
	}
	if err := bindTensorProgram(catalog, "", bindings); err != nil {
		return Weights{}, err
	}
	if profile.Has(ArchitectureRequiresOutput) && result.Output == nil {
		return Weights{}, fmt.Errorf("required tensor %q is missing", outputWeightTensor)
	}
	if profile.ModelCatalog.RequireOutputBias && result.OutputBias == nil {
		return Weights{}, errors.New(`required tensor "output.bias" is missing`)
	}

	trunkBlockCount := spec.BlockCount
	if draftPlan.AppendedBlocks {
		trunkBlockCount += draftPlan.Heads
	}
	hasOptionalDraft := false
	optionalDraftOnly := false
	if draftPlan.Kind == DraftOptionalSingleCatalog && draftPlan.SessionEligible() {
		mtpPrefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, tensor.FirstOffset))
		_, hasOptionalDraft = catalog.tensors[mtpPrefix+"nextn.eh_proj.weight"]
		_, hasTrunk := catalog.tensors[firstBlockTensorPrefix+attentionNormWeightTensor]
		optionalDraftOnly = hasOptionalDraft && !hasTrunk
		if hasOptionalDraft {
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
	l.hasOptionalDraft = hasOptionalDraft
	l.optionalDraftOnly = optionalDraftOnly
	l.mtpOnly = mtpOnly
	return result, nil
}

func (l *layerCatalogLoader) loadLayerCatalogs(result Weights) (Weights, error) {
	catalog := l.catalog
	spec, profile, normPlan := l.spec, l.profile, l.normPlan
	trunkBlockCount, optionalDraftOnly := l.trunkBlockCount, l.optionalDraftOnly
	for block := uint32(tensor.FirstOffset); block < trunkBlockCount; block++ {
		if optionalDraftOnly && block < spec.BlockCount {
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
				return Weights{}, errors.New("optional Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				if queryLength != uint64(spec.EmbeddingLength) || keyLength != uint64(spec.EmbeddingLength) {
					return Weights{}, errors.New("optional Q/K norm requires full-width Q/K projections")
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
				return Weights{}, errors.New("optional Q norm bias has no weight")
			} else if _, hasKBias := catalog.tensors[prefix+"attn_k_norm.bias"]; hasKBias {
				return Weights{}, errors.New("optional K norm bias has no weight")
			}
			if scaleErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				optionalF32TensorPointer("ffn_act.scales", &layer.FeedForwardActivationScale,
					uint64(spec.FeedForwardLength)),
			}); scaleErr != nil {
				return Weights{}, scaleErr
			}
		}
		if rotaryErr := layerPlan.rotaryCatalog.bind(catalog, spec.Architecture, layer); rotaryErr != nil {
			return Weights{}, rotaryErr
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
			bindings, compileErr := compileTokenShiftRecurrentBindings(catalog, prefix, spec, layer, block, mixer)
			if compileErr != nil {
				return Weights{}, compileErr
			}
			if mixerErr := bindTensorProgram(catalog, prefix, bindings); mixerErr != nil {
				return Weights{}, mixerErr
			}
			continue
		}
		if layerPlan.DenseWeights.useSecondaryAttentionNorm {
			if normErr := loadOptionalWeightBias(
				catalog, prefix,
				requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, uint64(spec.EmbeddingLength)),
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
		if layerPlan.Composition == LayerCompositionFeedForwardOnly {
			// No attention catalog.
		} else if layerPlan.DeciSparse && !spec.LayerHasAttention(block) {
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
			bindings := compileLatentAttentionBindings(
				catalog, prefix, spec, layer, layerPlan, queryLength,
				uint64(spec.HeadCount)*uint64(spec.ValueLength),
			)
			if itemErr := bindTensorProgram(catalog, prefix, bindings); itemErr != nil {
				return Weights{}, itemErr
			}
		} else {
			bindings, compileErr := compileStandardAttentionBindings(
				catalog, prefix, spec, layer, queryLength, keyLength, valueLength, attentionOutputLength,
			)
			if compileErr != nil {
				return Weights{}, compileErr
			}
			if attentionErr := bindTensorProgram(catalog, prefix, bindings); attentionErr != nil {
				return Weights{}, attentionErr
			}
		}
		qkPlan := layerPlan.QKPreprocess
		if !layer.Recurrent && (qkPlan.Heads == qkNormWeighted || qkPlan.PostRotary == qkNormWeighted) {
			shape := []uint64{uint64(spec.KeyLength)}
			if normErr := loadQKNormPair(catalog, prefix, layer, shape, shape, false); normErr != nil {
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
			if normErr := loadQKNormPair(catalog, prefix, layer, shape, shape, true); normErr != nil {
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
			if normErr := loadQKNormPair(
				catalog, prefix, layer,
				[]uint64{uint64(spec.KeyLength), uint64(spec.HeadCount)},
				[]uint64{uint64(spec.KeyLength), uint64(spec.HeadCountKV)}, qkPlan.Heads == qkNormLayer,
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
				catalog, prefix, layer, []uint64{queryLength}, []uint64{keyLength}, false,
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
				); normErr != nil {
					return Weights{}, normErr
				}
			}
			if layer.AttentionKNorm != nil && keyLength != uint64(spec.EmbeddingLength) {
				return Weights{}, errors.New("optional K norm requires full-width KV projection")
			}
			if normErr := loadOptionalWeightBias(
				catalog, prefix,
				requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, uint64(spec.EmbeddingLength)),
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
		if layerPlan.Composition == LayerCompositionAttentionOnly ||
			layerPlan.Composition == LayerCompositionRecurrentOnly ||
			layerPlan.Mixer == recurrentMixerSelectiveScan || layerPlan.Mixer == recurrentMixerGroupedSelectiveScan {
			continue
		}
		feedForwardNormName := normPlan.FeedForwardNormTensor()
		if layerPlan.ResidualStages.kind == residualStable {
			if normErr := loadOptionalWeightBias(
				catalog, prefix,
				requiredTensorPointer(feedForwardNormName, &layer.FeedForwardNorm, uint64(spec.EmbeddingLength)),
				requiredTensorPointer("ffn_norm.bias", &layer.FeedForwardNormBias, uint64(spec.EmbeddingLength)),
			); normErr != nil {
				return Weights{}, normErr
			}
		} else if !layerPlan.DenseWeights.requireExpertProjectionBiases &&
			layerPlan.DenseWeights.requireFeedForwardNorm &&
			(!layerPlan.DeciSparse || spec.LayerHasFeedForward(block)) {
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
	hasOptionalDraft, optionalDraftOnly := l.hasOptionalDraft, l.optionalDraftOnly
	if draftPlan.Kind == DraftSingleCatalog && draftPlan.SessionEligible() {
		prefix := fmt.Sprintf("blk.%d.", draftPlan.Block(spec.BlockCount, tensor.FirstOffset))
		mtp := &SingleDraftWeights{}
		mtp.MTPOnly = mtpOnly
		mtp.Layer.Recurrent = false
		shapes := spec.TensorShapes(tensor.FirstOffset)
		queryLength := shapes.QueryProjectionWidth()
		keyLength := shapes.KeyProjectionWidth()
		valueLength := shapes.ValueProjectionWidth()
		bindings := []tensorBinding{
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
		}
		bindings = append(bindings, mtpCommonBindings(spec, mtpCommonDestinations{
			ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
			tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
		})...)
		if loadErr := bindTensorProgram(catalog, prefix, bindings); loadErr != nil {
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
			if loadErr := bindTensorProgram(catalog, prefix, mtpCommonBindings(spec, mtpCommonDestinations{
				ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
				tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
			})); loadErr != nil {
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
	if hasOptionalDraft {
		block := draftPlan.Block(spec.BlockCount, tensor.FirstOffset)
		prefix := fmt.Sprintf("blk.%d.", block)
		mtp := &SingleDraftWeights{MTPOnly: optionalDraftOnly, Layer: result.Layers[block]}
		if loadErr := bindTensorProgram(catalog, prefix, mtpCommonBindings(spec, mtpCommonDestinations{
			ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
			tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
		})); loadErr != nil {
			return Weights{}, loadErr
		}
		result.OptionalCatalogDraft = mtp
		if optionalDraftOnly {
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
			bindings := mtpCommonBindings(spec, mtpCommonDestinations{
				ehProjection: &mtp.EHProjection, embeddingNorm: &mtp.EmbeddingNorm, hiddenNorm: &mtp.HiddenNorm,
				tokenEmbedding: &mtp.TokenEmbedding, outputNorm: &mtp.OutputNorm, output: &mtp.Output,
			})
			if profile.ModelCatalog.DraftLayerOutputNorm {
				bindings = append(bindings,
					requiredTensorPointer("layer_output_norm.weight", &mtp.LayerOutputNorm,
						uint64(spec.EmbeddingLength)),
				)
			}
			if loadErr := bindTensorProgram(catalog, prefix, bindings); loadErr != nil {
				return Weights{}, loadErr
			}
		}
		result.Layers = result.Layers[:spec.BlockCount]
	}
	return result, nil
}
