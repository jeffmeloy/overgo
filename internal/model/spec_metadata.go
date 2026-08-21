package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

const (
	yarnLogFactorStep   = float32(0.1)
	yarnDefaultBetaFast = float32(32)
	yarnDefaultBetaSlow = float32(1)
)

type specMetadata struct {
	values       map[string]gguf.Value
	architecture string
	prefix       string
	profile      ArchitectureProfile
}

type specReadState struct {
	declaredBlockCount uint32
	declaredExperts    bool
}

type metadataCardinality uint8

const (
	metadataScalar metadataCardinality = iota
	metadataLayer
	metadataLayerCompatible
	metadataFixedOne
	metadataFixedZero
	metadataMirrorHeads
	metadataMirrorHeadsOptional
	metadataHybridLayers
)

// MetadataShapePolicy: base FFN and attention metadata layout.
type MetadataShapePolicy struct {
	FeedForward               metadataCardinality
	Heads                     metadataCardinality
	KVHeads                   metadataCardinality
	PreserveLayerKV           bool
	InferRecurrentFromZeroFFN bool
	GroupNorm                 bool
	MLAHeadLengths            bool
	SWAHeadLengths            bool
	DraftBeforeShape          bool
	PreserveDraftLayers       bool
}

func newSpecMetadataWithProfile(file *gguf.File, resolved *ArchitectureProfile) (specMetadata, error) {
	if file == nil {
		return specMetadata{}, errors.New("model file is nil")
	}
	values := make(map[string]gguf.Value, len(file.Metadata))
	for _, item := range file.Metadata {
		values[item.Key] = item.Value
	}
	architecture, err := required[string](values, "general.architecture", gguf.ValueTypeString)
	if err != nil {
		return specMetadata{}, err
	}
	var profile ArchitectureProfile
	if resolved == nil {
		var supported bool
		profile, supported = LookupArchitecture(architecture)
		if !supported {
			return specMetadata{}, &UnsupportedArchitectureError{Architecture: architecture}
		}
	} else {
		profile = *resolved
		if err := ValidateArchitectureProfile(profile); err != nil {
			return specMetadata{}, err
		}
		if profile.Name == "" || profile.Name != architecture {
			return specMetadata{}, fmt.Errorf(
				"model profile %q does not match architecture %q", profile.Name, architecture,
			)
		}
	}
	return specMetadata{
		values: values, architecture: architecture, prefix: architecture + ".", profile: profile,
	}, nil
}

func (m specMetadata) readBase(spec *Spec) (specReadState, error) {
	values, prefix := m.values, m.prefix
	if m.profile.Validation.Attention == AttentionValidationQKNormEpsilon {
		spec.SandwichNorm, _ = optional[bool](values, "chameleon.swin_norm", gguf.ValueTypeBool)
	}
	if value, ok := optional[string](values, "general.name", gguf.ValueTypeString); ok {
		spec.Name = value
	}
	pooling, _ := optional[uint32](values, prefix+"pooling_type", gguf.ValueTypeUint32)
	spec.PoolingType = PoolingType(pooling)
	if !spec.PoolingType.Valid() {
		return specReadState{}, fmt.Errorf("metadata %q has unsupported pooling type %d", prefix+"pooling_type", spec.PoolingType)
	}
	if labels, ok, err := optionalArray[string](
		values, prefix+"classifier.output_labels", gguf.ValueTypeString,
	); err != nil {
		return specReadState{}, err
	} else if ok {
		spec.ClassifierLabels = slices.Clone(labels)
	}
	if m.profile.Has(ArchitectureLatentKVLayout) {
		spec.VocabularySize, _ = optional[uint32](values, prefix+"vocab_size", gguf.ValueTypeUint32)
		if tokens, ok := values["tokenizer.ggml.tokens"]; ok && spec.VocabularySize == tensor.FirstOffset {
			if tokens.Type != gguf.ValueTypeArray || tokens.ArrayType != gguf.ValueTypeString {
				return specReadState{}, errors.New(`metadata "tokenizer.ggml.tokens" must be a string array`)
			}
			if tokens.Count() > int(math.MaxUint32) {
				return specReadState{}, errors.New("tokenizer vocabulary exceeds uint32")
			}
			spec.VocabularySize = uint32(tokens.Count())
		}
	}
	state := specReadState{}
	if m.profile.Validation.ExpertMetadata == ExpertMetadataWhenDeclared {
		count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		state.declaredExperts = ok && count > tensor.FirstOffset
	}
	var err error
	if spec.BlockCount, err = required[uint32](values, prefix+"block_count", gguf.ValueTypeUint32); err != nil {
		return specReadState{}, err
	}
	state.declaredBlockCount = spec.BlockCount
	if m.profile.Metadata.DraftBeforeShape {
		if err := m.readDraftLayers(spec); err != nil {
			return specReadState{}, err
		}
	}
	if err = readRequiredMetadataFields(
		values, prefix, gguf.ValueTypeUint32,
		metadataDestination("context_length", &spec.ContextLength),
		metadataDestination("embedding_length", &spec.EmbeddingLength),
	); err != nil {
		return specReadState{}, err
	}
	if err := m.readProfileMetadata(spec); err != nil {
		return specReadState{}, err
	}
	return state, nil
}

func (m specMetadata) readAttentionMetadata(spec *Spec, state specReadState) error {
	values, prefix := m.values, m.prefix
	policy := m.profile.Metadata
	var err error
	switch policy.FeedForward {
	case metadataLayer:
		spec.LayerFeedForward, err = requiredLayerUint32(values, prefix+"feed_forward_length", spec.BlockCount)
	case metadataLayerCompatible:
		spec.LayerFeedForward, err = requiredLayerUint32Compatible(values, prefix+"feed_forward_length", spec.BlockCount)
	default:
		spec.FeedForwardLength, err = required[uint32](values, prefix+"feed_forward_length", gguf.ValueTypeUint32)
	}
	if err != nil {
		return err
	}
	if len(spec.LayerFeedForward) > tensor.FirstOffset {
		spec.FeedForwardLength = firstPositive(spec.LayerFeedForward)
	}
	switch policy.Heads {
	case metadataFixedOne:
		spec.HeadCount = tensor.SingletonExtent
	case metadataLayer:
		spec.LayerHeadCounts, err = requiredLayerUint32(values, prefix+"attention.head_count", spec.BlockCount)
	case metadataLayerCompatible:
		spec.LayerHeadCounts, err = requiredLayerUint32Compatible(values, prefix+"attention.head_count", state.declaredBlockCount)
	default:
		spec.HeadCount, err = required[uint32](values, prefix+"attention.head_count", gguf.ValueTypeUint32)
	}
	if err != nil {
		return err
	}
	if len(spec.LayerHeadCounts) > tensor.FirstOffset {
		spec.HeadCount = firstPositive(spec.LayerHeadCounts)
	}
	switch policy.KVHeads {
	case metadataFixedOne:
		spec.HeadCountKV = tensor.SingletonExtent
	case metadataFixedZero:
		spec.HeadCountKV = tensor.FirstOffset
	case metadataMirrorHeads, metadataMirrorHeadsOptional:
		spec.HeadCountKV = spec.HeadCount
		if policy.KVHeads == metadataMirrorHeadsOptional {
			if value, ok := optional[uint32](values, prefix+"attention.head_count_kv", gguf.ValueTypeUint32); ok {
				spec.HeadCountKV = value
			}
		}
	case metadataLayer:
		spec.LayerKVHeadCounts, err = requiredLayerUint32(values, prefix+"attention.head_count_kv", state.declaredBlockCount)
		spec.HeadCountKV = firstPositive(spec.LayerKVHeadCounts)
	case metadataLayerCompatible:
		spec.LayerKVHeadCounts, err = requiredLayerUint32Compatible(values, prefix+"attention.head_count_kv", state.declaredBlockCount)
		spec.HeadCountKV = firstPositive(spec.LayerKVHeadCounts)
		if policy.InferRecurrentFromZeroFFN && err == nil {
			spec.RecurrentLayers = make([]bool, spec.BlockCount)
			for block := uint32(tensor.FirstOffset); block < spec.BlockCount; block++ {
				spec.RecurrentLayers[block] = spec.LayerKVHeadCounts[block] == tensor.FirstOffset &&
					spec.LayerFeedForward[block] == tensor.FirstOffset
			}
		}
	case metadataHybridLayers:
		var counts []uint32
		counts, err = requiredArray[uint32](values, prefix+"attention.head_count_kv", gguf.ValueTypeUint32)
		if err == nil && len(counts) != int(spec.BlockCount) {
			return fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.head_count_kv", len(counts), spec.BlockCount)
		}
		if err == nil {
			spec.RecurrentLayers = make([]bool, len(counts))
			if policy.PreserveLayerKV {
				spec.LayerKVHeadCounts = slices.Clone(counts)
			}
			for index, count := range counts {
				if count == tensor.FirstOffset {
					spec.RecurrentLayers[index] = true
					continue
				}
				if spec.HeadCountKV == tensor.FirstOffset {
					spec.HeadCountKV = count
				} else if spec.HeadCountKV != count {
					return errors.New("hybrid attention layers use differing positive KV head counts")
				}
			}
		}
	default:
		spec.HeadCountKV, err = required[uint32](values, prefix+"attention.head_count_kv", gguf.ValueTypeUint32)
	}
	if err != nil {
		return err
	}
	if policy.GroupNorm {
		if spec.GroupNormEpsilon, err = required[float32](values, prefix+"attention.group_norm_epsilon", gguf.ValueTypeFloat32); err != nil {
			return err
		}
		if spec.GroupNormGroups, err = required[uint32](values, prefix+"attention.group_norm_groups", gguf.ValueTypeUint32); err != nil {
			return err
		}
	}
	spec.KeyLength, _ = optional[uint32](values, prefix+"attention.key_length", gguf.ValueTypeUint32)
	spec.ValueLength, _ = optional[uint32](values, prefix+"attention.value_length", gguf.ValueTypeUint32)
	if policy.GroupNorm {
		spec.KeyLength, spec.ValueLength = spec.PosNetEmbeddingLength, spec.PosNetEmbeddingLength
	}
	if policy.MLAHeadLengths {
		if value, ok := optional[uint32](values, prefix+"attention.key_length_mla", gguf.ValueTypeUint32); ok {
			spec.KeyLength = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.value_length_mla", gguf.ValueTypeUint32); ok {
			spec.ValueLength = value
		}
	}
	if spec.HeadCount > tensor.FirstOffset &&
		(spec.KeyLength == tensor.FirstOffset || spec.ValueLength == tensor.FirstOffset) {
		if spec.EmbeddingLength%spec.HeadCount != tensor.FirstOffset {
			return errors.New("embedding length is not divisible by attention head count")
		}
		headLength := spec.EmbeddingLength / spec.HeadCount
		if spec.KeyLength == tensor.FirstOffset {
			spec.KeyLength = headLength
		}
		if spec.ValueLength == tensor.FirstOffset {
			spec.ValueLength = headLength
		}
	}
	if policy.SWAHeadLengths {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("attention.key_length_swa", &spec.KeyLengthSWA),
			metadataDestination("attention.value_length_swa", &spec.ValueLengthSWA),
		); err != nil {
			return err
		}
	}
	return nil
}

func (m specMetadata) readPosition(spec *Spec) error {
	values, architecture, prefix, profile := m.values, m.architecture, m.prefix, m.profile
	validation := profile.Validation
	if !spec.RopeDisabled && profile.Forward.Operation != ForwardOperationEncoder {
		optionalBase := validation.optionalRopeBase()
		if optionalBase {
			spec.RopeFrequencyBase = profile.MetadataDefaults.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencyBase = value
			}
		} else if value, err := required[float32](values, prefix+"rope.freq_base", gguf.ValueTypeFloat32); err != nil {
			return err
		} else {
			spec.RopeFrequencyBase = value
		}
		if scalingType, ok := optional[string](values, prefix+"rope.scaling.type", gguf.ValueTypeString); ok &&
			scalingType != "" && scalingType != "none" {
			kind := ropeScalingKind(scalingType)
			gatedDeltaMulti := profile.Attention == AttentionGatedDelta && profile.Has(ArchitectureMultiAxisPositions)
			longRoPE := profile.Has(ArchitectureLongRoPE) && kind == ropeScalingLongRoPE
			yarn := kind == ropeScalingYaRN && (profile.Has(ArchitectureLatentKVLayout) ||
				validation.MLA == MLAValidationCompressedHyper || validation.supportsYaRN())
			if gatedDeltaMulti || kind != ropeScalingLinear && !longRoPE && !yarn {
				return fmt.Errorf("model architecture %q uses unsupported RoPE scaling type %q", architecture, scalingType)
			}
			spec.RopeScalingType = kind
			if kind == ropeScalingLinear || kind == ropeScalingYaRN {
				value, err := required[float32](values, prefix+"rope.scaling.factor", gguf.ValueTypeFloat32)
				if err != nil {
					return err
				}
				spec.RopeScalingFactor = value
			}
			if kind == ropeScalingYaRN {
				value, err := required[uint32](values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32)
				if err != nil {
					return err
				}
				spec.OriginalContextLength = value
				spec.YaRNExtFactor = tensor.UnitScale
				spec.YaRNAttentionFactor =
					tensor.UnitScale / (tensor.UnitScale + yarnLogFactorStep*float32(math.Log(float64(spec.RopeScalingFactor))))
				spec.YaRNBetaFast, spec.YaRNBetaSlow = yarnDefaultBetaFast, yarnDefaultBetaSlow
				if positiveFinite(profile.MetadataDefaults.YaRNBetaFast) {
					spec.YaRNBetaFast = profile.MetadataDefaults.YaRNBetaFast
				}
				for _, field := range []metadataField[float32]{
					metadataDestination("rope.scaling.yarn_ext_factor", &spec.YaRNExtFactor),
					metadataDestination("rope.scaling.yarn_attn_factor", &spec.YaRNAttentionFactor),
					metadataDestination("rope.scaling.yarn_beta_fast", &spec.YaRNBetaFast),
					metadataDestination("rope.scaling.yarn_beta_slow", &spec.YaRNBetaSlow),
				} {
					if value, ok := optional[float32](
						values, prefix+field.key, gguf.ValueTypeFloat32,
					); ok {
						*field.destination = value
					}
				}
			}
		}
	}
	if profile.readsMetadata(MetadataReadALiBi) || profile.EncoderOperator.usesALiBiQKNorm() {
		if !profile.readsMetadata(MetadataReadZeroALiBiDefault) {
			spec.MaxALiBiBias = profile.MetadataDefaults.MaxALiBiBias
		}
		if !profile.EncoderOperator.usesALiBiQKNorm() {
			if value, ok := optional[float32](values, prefix+"attention.max_alibi_bias", gguf.ValueTypeFloat32); ok {
				spec.MaxALiBiBias = value
			}
		}
	}
	if profile.DenseWeights.AllowActivationScale {
		spec.AttentionClamp, _ = optional[float32](values, prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32)
	}
	if validation.Hybrid == HybridValidationModelFeedForwardExperts {
		value, err := required[float32](values, prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32)
		if err != nil {
			return err
		}
		spec.AttentionClamp = value
	}
	return nil
}

func (m specMetadata) readDraftLayers(spec *Spec) error {
	if m.profile.DraftKind == DraftNone {
		return nil
	}
	key := m.prefix + "nextn_predict_layers"
	nextN, _ := optional[uint32](m.values, key, gguf.ValueTypeUint32)
	if nextN == tensor.FirstOffset {
		return nil
	}
	if nextN >= spec.BlockCount ||
		(m.profile.DraftKind == DraftSingleCatalog || m.profile.DraftKind == DraftOptionalSingleCatalog) &&
			nextN != singleDraftHeadCount {
		return errors.New("draft layer count is invalid")
	}
	spec.NextNPredictLayers = nextN
	spec.BlockCount -= nextN
	if limit := int(spec.BlockCount); !m.profile.Metadata.PreserveDraftLayers && len(spec.LayerKVHeadCounts) > limit {
		spec.LayerKVHeadCounts = spec.LayerKVHeadCounts[:limit]
	}
	return nil
}

// metadataReadMode: missing-key handling for one compiled scalar read.
type metadataReadMode uint8

const (
	metadataReadRequired metadataReadMode = iota
	metadataReadAssignZero
	metadataReadKeepCurrent
)

// metadataOpKind: shape of one compiled metadata operation.
type metadataOpKind uint8

const (
	metadataOpUint32 metadataOpKind = iota
	metadataOpFloat32
	metadataOpBool
	metadataOpSetUint32
	metadataOpSetBool
	metadataOpDefaults
	metadataOpInverseKeyScale
	metadataOpEpsilonEither
	metadataOpRopeSections
	metadataOpSlidingPatternType
	metadataOpHalveFeedForward
	metadataOpExpertFeedForwardFromModel
	metadataOpSharedFeedForwardFromModel
	metadataOpSharedFeedForwardFromExpert
	metadataOpSharedFeedForwardScale
	metadataOpSharedFeedForwardPolicy
)

// metadataOp: one compiled read/derive/validate/bind operation. field names
// a Spec destination resolved through the exported field set.
type metadataOp struct {
	kind  metadataOpKind
	key   string
	field string
	mode  metadataReadMode
	value uint32
	flag  bool
}

func (s *Spec) fieldDestination(field string) any {
	return reflect.ValueOf(s).Elem().FieldByName(field).Addr().Interface()
}

func setU32(field string, value uint32) metadataOp {
	return metadataOp{kind: metadataOpSetUint32, field: field, value: value}
}

// at: binds a scalar-read template to a metadata key and Spec field.
func (o metadataOp) at(key, field string) metadataOp {
	o.key, o.field = key, field
	return o
}

// scalar-read templates: kind and missing-key mode without a binding.
var (
	opReqU32   = metadataOp{kind: metadataOpUint32}
	opZeroU32  = metadataOp{kind: metadataOpUint32, mode: metadataReadAssignZero}
	opKeepU32  = metadataOp{kind: metadataOpUint32, mode: metadataReadKeepCurrent}
	opReqF32   = metadataOp{kind: metadataOpFloat32}
	opZeroF32  = metadataOp{kind: metadataOpFloat32, mode: metadataReadAssignZero}
	opKeepF32  = metadataOp{kind: metadataOpFloat32, mode: metadataReadKeepCurrent}
	opReqBool  = metadataOp{kind: metadataOpBool}
	opZeroBool = metadataOp{kind: metadataOpBool, mode: metadataReadAssignZero}
)

var (
	ropeDimensionRequired     = opReqU32.at("rope.dimension_count", "RopeDimensionCount")
	ropeDimensionAssigned     = opZeroU32.at("rope.dimension_count", "RopeDimensionCount")
	expertFeedForwardRequired = opReqU32.at("expert_feed_forward_length", "ExpertFeedForward")
	expertGatingRequired      = opReqU32.at("expert_gating_func", "ExpertGatingFunc")
	sharedWidthOptional       = opKeepU32.at("expert_shared_feed_forward_length", "SharedExpertFF")
	leadingDenseOptional      = opZeroU32.at("leading_dense_block_count", "LeadingDenseBlocks")
	expertNormAlways          = metadataOp{kind: metadataOpSetBool, field: "ExpertWeightsNorm", flag: true}
	expertNormOptional        = opZeroBool.at("expert_weights_norm", "ExpertWeightsNorm")
	sectionsRequired          = metadataOp{kind: metadataOpRopeSections, key: "rope.dimension_sections"}
	sectionsOptional          = metadataOp{kind: metadataOpRopeSections, key: "rope.dimension_sections", mode: metadataReadKeepCurrent}
)

// cohere2CoreProgram: shared Cohere2 and Cohere2-MoE core reads.
var cohere2CoreProgram = []metadataOp{
	opReqF32.at("logit_scale", "LogitScale"),
	ropeDimensionRequired,
	{kind: metadataOpSlidingPatternType, key: "attention.sliding_window_pattern"},
}

// coreAttentionPrograms: architecture-core reads keyed by attention contract.
var coreAttentionPrograms = map[AttentionValidationPolicy][]metadataOp{
	AttentionValidationFullRotary:                   {opReqF32.at("logit_scale", "LogitScale")},
	AttentionValidationRequiredSlidingRotary:        cohere2CoreProgram,
	AttentionValidationRequiredSlidingRotaryExperts: cohere2CoreProgram,
	AttentionValidationPartialRotaryRequired:        {ropeDimensionRequired},
	AttentionValidationPartialRotaryFixed:           {ropeDimensionRequired},
	AttentionValidationScaledPartialRotary: {
		ropeDimensionRequired,
		opReqU32.at("rope.scaling.original_context_length", "OriginalContextLength"),
	},
	AttentionValidationSlidingRotaryEmbeddingProjection: {
		opReqU32.at("attention.sliding_window", "SlidingWindow"),
		opZeroU32.at("dense_2_feat_in", "Dense2FeatureIn"),
		opZeroU32.at("dense_2_feat_out", "Dense2FeatureOut"),
		opZeroU32.at("dense_3_feat_in", "Dense3FeatureIn"),
		opZeroU32.at("dense_3_feat_out", "Dense3FeatureOut"),
	},
	AttentionValidationOptionalRotaryBase: {
		ropeDimensionAssigned,
		opReqBool.at("use_parallel_residual", "ParallelResidual"),
	},
	AttentionValidationHalvedFeedForward:           {{kind: metadataOpHalveFeedForward}},
	AttentionValidationOptionalRopeSections:        {sectionsOptional},
	AttentionValidationOptionalRopeSectionsExperts: {sectionsOptional},
	AttentionValidationOptionalRotaryBaseGQA:       {ropeDimensionAssigned},
}

// coreFlagPrograms: architecture-core reads keyed by metadata-read facts.
var coreFlagPrograms = []struct {
	flag    MetadataReadPolicy
	program []metadataOp
}{
	{MetadataReadGPTJRotary, []metadataOp{ropeDimensionRequired}},
	{MetadataReadCommandRLogits, []metadataOp{opZeroF32.at("logit_scale", "LogitScale")}},
	{MetadataReadVisualSections, []metadataOp{sectionsRequired}},
	{MetadataReadQwen3VLDeepstack, []metadataOp{opKeepU32.at("n_deepstack_layers", "DeepstackLayerCount")}},
	{MetadataReadOLMoClamp, []metadataOp{opKeepF32.at("attention.clamp_kqv", "AttentionClamp")}},
}

// expertProductProgram: shared-product MoE width reads with optional gating
// and weights-norm overrides.
func expertProductProgram(gating, norm bool) []metadataOp {
	program := []metadataOp{expertFeedForwardRequired, opReqU32.at("expert_shared_count", "SharedExpertCount")}
	if gating {
		program = append(program, expertGatingRequired)
	}
	program = append(program, metadataOp{kind: metadataOpSharedFeedForwardFromExpert}, metadataOp{kind: metadataOpSharedFeedForwardScale})
	if norm {
		program = append(program, expertNormOptional)
	}
	return append(program, leadingDenseOptional)
}

// expertHybridPrograms: expert-stage reads keyed by hybrid contract.
var expertHybridPrograms = map[HybridValidationPolicy][]metadataOp{
	HybridValidationSigmoidExperts:            {setU32("ExpertGatingFunc", expertGatingSigmoid), expertNormAlways},
	HybridValidationRequiredExpertFeedForward: {expertFeedForwardRequired, expertNormAlways},
	HybridValidationSharedExpertProduct: {
		expertFeedForwardRequired,
		{kind: metadataOpSharedFeedForwardPolicy},
		expertNormAlways,
	},
	HybridValidationModelFeedForwardExperts: {{kind: metadataOpExpertFeedForwardFromModel}, expertNormAlways},
	HybridValidationDualExpertProduct: {
		{kind: metadataOpExpertFeedForwardFromModel},
		expertNormAlways,
		expertGatingRequired,
	},
	HybridValidationWeightedExpertProduct:              expertProductProgram(true, true),
	HybridValidationProductSharedExperts:               expertProductProgram(false, true),
	HybridValidationProductExperts:                     expertProductProgram(false, false),
	HybridValidationAlternatingShortConvolutionExperts: {expertFeedForwardRequired, expertGatingRequired, leadingDenseOptional},
	HybridValidationScaledSharedExperts: {
		expertFeedForwardRequired,
		opReqU32.at("expert_shared_count", "SharedExpertCount"),
		{kind: metadataOpSharedFeedForwardFromExpert},
		sharedWidthOptional,
		{kind: metadataOpSharedFeedForwardScale},
		expertGatingRequired,
		expertNormOptional,
		leadingDenseOptional,
	},
	HybridValidationNormalizedSharedExperts: {
		{kind: metadataOpExpertFeedForwardFromModel, flag: true},
		setU32("SharedExpertCount", tensor.SingletonExtent),
		{kind: metadataOpSharedFeedForwardFromModel},
		sharedWidthOptional,
	},
}

// expertAttentionPrograms: expert-stage reads keyed by attention contract.
var expertAttentionPrograms = map[AttentionValidationPolicy][]metadataOp{
	AttentionValidationPeriodicExperts: {
		expertFeedForwardRequired,
		opReqU32.at("interleave_moe_layer_step", "MoELayerStep"),
		leadingDenseOptional,
		opZeroU32.at("expert_shared_feed_forward_length", "SharedExpertFF"),
		expertNormAlways,
	},
}

var rwkv6Reads = []metadataOp{
	opReqU32.at("time_mix_extra_dim", "TimeMixExtraDim"),
	opReqU32.at("time_decay_extra_dim", "TimeDecayExtraDim"),
	opZeroU32.at("rescale_every_n_layers", "RescaleEvery"),
}

var rwkv7Reads = []metadataOp{
	opReqU32.at("attention.decay_lora_rank", "DecayLoRARank"),
	opReqU32.at("attention.iclr_lora_rank", "ICLRLoRARank"),
	opReqU32.at("attention.value_residual_mix_lora_rank", "ValueMixLoRARank"),
	opZeroU32.at("attention.gate_lora_rank", "GateLoRARank"),
}

// tokenShiftProgram: WKV reads plus token-shift default.
func tokenShiftProgram(reads []metadataOp, shift uint32) []metadataOp {
	program := append([]metadataOp{opReqU32.at("wkv.head_size", "WKVHeadSize")}, reads...)
	return append(program, setU32("TokenShiftCount", shift), opKeepU32.at("token_shift_count", "TokenShiftCount"))
}

// ssmProgram: state-space reads; grouped layouts include group count.
func ssmProgram(grouped bool) []metadataOp {
	program := []metadataOp{
		opReqU32.at("ssm.conv_kernel", "SSMConvKernel"),
		opReqU32.at("ssm.inner_size", "SSMInnerSize"),
		opReqU32.at("ssm.state_size", "SSMStateSize"),
		opReqU32.at("ssm.time_step_rank", "SSMTimeStepRank"),
	}
	if grouped {
		return append(program, opReqU32.at("ssm.group_count", "SSMGroupCount"))
	}
	return append(program, setU32("SSMGroupCount", tensor.SingletonExtent), opZeroBool.at("ssm.dt_b_c_rms", "SSMDtBCNorm"))
}

// coreRecurrentPrograms: architecture-core reads keyed by recurrent contract.
var coreRecurrentPrograms = map[RecurrentValidationPolicy][]metadataOp{
	RecurrentValidationTimeMixV6:                        tokenShiftProgram(rwkv6Reads, tensor.PairedExtent),
	RecurrentValidationTimeMixV6SharedKV:                tokenShiftProgram(rwkv6Reads, tensor.SingletonExtent),
	RecurrentValidationTimeMixV7Gated:                   tokenShiftProgram(rwkv7Reads, tensor.PairedExtent),
	RecurrentValidationTimeMixV7:                        tokenShiftProgram(rwkv7Reads, tensor.SingletonExtent),
	RecurrentValidationUngroupedStateSpace:              ssmProgram(false),
	RecurrentValidationStateSpaceAttentionExperts:       ssmProgram(false),
	RecurrentValidationGroupedStateSpace:                ssmProgram(true),
	RecurrentValidationGroupedStateSpaceOptionalExperts: ssmProgram(true),
	RecurrentValidationUngroupedScheduledStateSpace:     ssmProgram(true),
	RecurrentValidationScheduledStateSpaceDense:         ssmProgram(true),
	RecurrentValidationScheduledStateSpaceExperts:       ssmProgram(true),
	RecurrentValidationGroupedStateSpaceAttention:       ssmProgram(true),
}

// lfm2RuntimeProgram: short-convolution cache and sliding-window reads.
var lfm2RuntimeProgram = []metadataOp{
	opReqU32.at("shortconv.l_cache", "ShortConvCacheLength"),
	opKeepU32.at("attention.sliding_window", "SlidingWindow"),
}

// compileRuntimeProgram: ordered runtime metadata operations from a profile.
func compileRuntimeProgram(profile ArchitectureProfile) []metadataOp {
	switch profile.Validation.Hybrid {
	case HybridValidationAlternatingShortConvolution, HybridValidationAlternatingShortConvolutionExperts:
		return lfm2RuntimeProgram
	case HybridValidationSelectedSoftmaxExperts:
		return []metadataOp{opReqU32.at("attention.sliding_window", "SlidingWindow")}
	}
	return nil
}

// compileArchitectureCoreProgram: ordered core metadata operations from a profile.
func compileArchitectureCoreProgram(profile ArchitectureProfile) []metadataOp {
	validation := profile.Validation
	var program []metadataOp
	switch {
	case validation.Attention == AttentionValidationRequiredSlidingRotaryExperts:
		program = append(program, metadataOp{kind: metadataOpEpsilonEither})
	case profile.Normalization == NormalizationLayer,
		profile.Normalization == NormalizationUnweightedLayer,
		profile.Normalization == NormalizationWeightOnlyLayer:
		program = append(program, opReqF32.at("attention.layer_norm_epsilon", "LayerNormEpsilon"))
	default:
		program = append(program, opReqF32.at("attention.layer_norm_rms_epsilon", "RMSNormEpsilon"))
	}
	program = append(program,
		opZeroF32.at("final_logit_softcapping", "FinalLogitSoftcap"),
		opZeroF32.at("attn_logit_softcapping", "AttentionSoftcap"),
	)
	if profile.readsMetadata(MetadataReadJaisScale) {
		program = append(program, metadataOp{kind: metadataOpInverseKeyScale})
	}
	program = append(program, opKeepF32.at("attention.scale", "AttentionScale"), metadataOp{kind: metadataOpDefaults})
	program = append(program, coreAttentionPrograms[validation.Attention]...)
	for _, entry := range coreFlagPrograms {
		if profile.readsMetadata(entry.flag) {
			program = append(program, entry.program...)
		}
	}
	if profile.MetadataDefaults.NoRopeLayerStep > tensor.FirstOffset {
		program = append(program, setU32("NoRopeLayerStep", profile.MetadataDefaults.NoRopeLayerStep))
	}
	return append(program, coreRecurrentPrograms[validation.Recurrent]...)
}

// compileExpertProgram: ordered expert metadata operations from a profile.
func compileExpertProgram(profile ArchitectureProfile) []metadataOp {
	validation := profile.Validation
	program := append([]metadataOp(nil), expertHybridPrograms[validation.Hybrid]...)
	program = append(program, expertAttentionPrograms[validation.Attention]...)
	if validation.Encoder == EncoderValidationRotaryPeriodicExperts {
		program = append(program, opReqU32.at("moe_every_n_layers", "MoELayerStep"))
	}
	return program
}

func runMetadataRead[T any](
	values map[string]gguf.Value, key string, valueType gguf.ValueType,
	mode metadataReadMode, destination *T,
) error {
	switch mode {
	case metadataReadRequired:
		value, err := required[T](values, key, valueType)
		if err != nil {
			return err
		}
		*destination = value
	case metadataReadAssignZero:
		*destination, _ = optional[T](values, key, valueType)
	default:
		if value, ok := optional[T](values, key, valueType); ok {
			*destination = value
		}
	}
	return nil
}

func (m specMetadata) readSectionsOp(spec *Spec, op metadataOp) error {
	key := m.prefix + op.key
	var sections []int32
	var err error
	if op.mode == metadataReadRequired {
		sections, err = requiredArray[int32](m.values, key, gguf.ValueTypeInt32)
	} else {
		var present bool
		sections, present, err = optionalArray[int32](m.values, key, gguf.ValueTypeInt32)
		if err == nil && !present {
			return nil
		}
	}
	if err != nil {
		return err
	}
	if len(sections) != len(spec.RopeSections) {
		return fmt.Errorf("metadata %q has %d values, need %d", key, len(sections), len(spec.RopeSections))
	}
	copy(spec.RopeSections[:], sections)
	return nil
}

// runMetadataProgram: neutral executor for compiled metadata operations.
func (m specMetadata) runMetadataProgram(spec Spec, program []metadataOp) (Spec, error) {
	values, prefix := m.values, m.prefix
	for _, op := range program {
		var err error
		switch op.kind {
		case metadataOpUint32:
			err = runMetadataRead(values, prefix+op.key, gguf.ValueTypeUint32, op.mode, spec.fieldDestination(op.field).(*uint32))
		case metadataOpFloat32:
			err = runMetadataRead(values, prefix+op.key, gguf.ValueTypeFloat32, op.mode, spec.fieldDestination(op.field).(*float32))
		case metadataOpBool:
			err = runMetadataRead(values, prefix+op.key, gguf.ValueTypeBool, op.mode, spec.fieldDestination(op.field).(*bool))
		case metadataOpSetUint32:
			*spec.fieldDestination(op.field).(*uint32) = op.value
		case metadataOpSetBool:
			*spec.fieldDestination(op.field).(*bool) = op.flag
		case metadataOpDefaults:
			m.profile.MetadataDefaults.read(values, prefix, &spec)
		case metadataOpInverseKeyScale:
			spec.AttentionScale = tensor.UnitScale / float32(spec.KeyLength)
		case metadataOpEpsilonEither:
			if value, ok := optional[float32](values, prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32); ok {
				spec.RMSNormEpsilon = value
				spec.profile.Normalization = NormalizationRMS
			} else if value, ok := optional[float32](values, prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32); ok {
				spec.LayerNormEpsilon = value
			} else {
				err = errors.New("Cohere2-MoE norm epsilon is missing")
			}
		case metadataOpRopeSections:
			err = m.readSectionsOp(&spec, op)
		case metadataOpSlidingPatternType:
			if pattern, ok := values[prefix+op.key]; ok && pattern.Type != gguf.ValueTypeUint32 &&
				(pattern.Type != gguf.ValueTypeArray || pattern.ArrayType != gguf.ValueTypeBool) {
				err = errors.New("Cohere2 sliding attention pattern has an invalid type")
			}
		case metadataOpHalveFeedForward:
			if spec.FeedForwardLength == tensor.FirstOffset || spec.FeedForwardLength%tensor.PairedExtent != tensor.FirstOffset {
				err = errors.New("Qwen feed-forward length must be positive and even")
			} else {
				spec.FeedForwardLength /= tensor.PairedExtent
			}
		case metadataOpExpertFeedForwardFromModel:
			if !op.flag || spec.ExpertFeedForward == tensor.FirstOffset {
				spec.ExpertFeedForward = spec.FeedForwardLength
			}
		case metadataOpSharedFeedForwardFromModel:
			spec.SharedExpertFF = spec.FeedForwardLength
		case metadataOpSharedFeedForwardFromExpert:
			spec.SharedExpertFF = spec.ExpertFeedForward
		case metadataOpSharedFeedForwardScale:
			spec.SharedExpertFF *= spec.SharedExpertCount
		case metadataOpSharedFeedForwardPolicy:
			spec.SharedExpertFF, err = m.profile.MetadataDefaults.readSharedExpertFeedForward(values, prefix, spec)
		}
		if err != nil {
			return Spec{}, err
		}
	}
	return spec, nil
}

func (m specMetadata) readProfileMetadata(spec *Spec) error {
	values, prefix := m.values, m.prefix
	validation := m.profile.Validation
	defaults := m.profile.MetadataDefaults
	var err error
	switch {
	case m.profile.Forward.Session == ForwardSessionPairedProjection:
		if spec.TargetHiddenSize, err = required[uint32](values, prefix+"embedding_length_out", gguf.ValueTypeUint32); err != nil {
			return err
		}
		nextN, nextErr := required[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32)
		if nextErr != nil {
			return nextErr
		}
		if nextN != spec.BlockCount {
			return errors.New("paired-projection layer count must match block count")
		}
	case validation.Attention == AttentionValidationSharedKVAlternatingState:
		spec.AltUpCount = defaults.AlternateStateCount
		spec.LaurelRank = defaults.LowRankResidualWidth
		spec.EmbeddingPerLayer = defaults.PerLayerEmbeddingWidth
		spec.KVFromStart = defaults.SharedKVStartLayer
		spec.SparseLayerCount = defaults.SparseLayerCount
		spec.SparsityStdMultiplier = defaults.SparsityStdMultiplier
		if spec.BlockCount >= spec.KVFromStart {
			spec.SharedKVLayers = spec.BlockCount - spec.KVFromStart
		}
	case m.profile.Forward.Operation == ForwardOperationAudioTokens:
		spec.OutputEmbeddingLength = spec.EmbeddingLength
		if spec.EmbeddingLength, err = required[uint32](values, prefix+"features_length", gguf.ValueTypeUint32); err != nil {
			return err
		}
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("posnet.embedding_length", &spec.PosNetEmbeddingLength),
			metadataDestination("posnet.block_count", &spec.PosNetBlockCount),
			metadataDestination("convnext.embedding_length", &spec.ConvNextEmbeddingLength),
			metadataDestination("convnext.block_count", &spec.ConvNextBlockCount),
		); err != nil {
			return err
		}
	case validation.Recurrent == RecurrentValidationTargetLayerBlock:
		if spec.TargetLayers, err = requiredArray[int32](values, prefix+"target_layers", gguf.ValueTypeInt32); err != nil {
			return err
		}
		spec.DFlashBlockSize = defaults.DraftBlockSize
		if value, ok := optional[uint32](values, prefix+"block_size", gguf.ValueTypeUint32); ok {
			spec.DFlashBlockSize = value
		}
	case validation.Recurrent == RecurrentValidationSingleBlockTarget:
		if spec.TargetLayers, err = requiredArray[int32](values, prefix+"target_layers", gguf.ValueTypeInt32); err != nil {
			return err
		}
		if spec.TargetHiddenSize, err = required[uint32](values, prefix+"target_hidden_size", gguf.ValueTypeUint32); err != nil {
			return err
		}
		spec.NormBeforeResidual, _ = optional[bool](values, prefix+"norm_before_residual", gguf.ValueTypeBool)
	}
	return nil
}
