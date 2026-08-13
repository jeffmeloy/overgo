package model

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/gguf"
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
	llamaMoE           bool
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
}

func newSpecMetadata(file *gguf.File) (specMetadata, error) {
	return newSpecMetadataWithProfile(file, nil)
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
	if m.profile.Validation.Attention == AttentionValidationChameleon {
		spec.QKNormEpsilon = chameleonQKNormEpsilon
		spec.SandwichNorm, _ = optional[bool](values, "chameleon.swin_norm", gguf.ValueTypeBool)
	}
	if value, ok := optional[string](values, "general.name", gguf.ValueTypeString); ok {
		spec.Name = value
	}
	if spec.PoolingType, _ = optional[uint32](values, prefix+"pooling_type", gguf.ValueTypeUint32); spec.PoolingType > 4 {
		return specReadState{}, fmt.Errorf("metadata %q has unsupported pooling type %d", prefix+"pooling_type", spec.PoolingType)
	}
	if labels, ok, err := optionalArray[string](
		values, prefix+"classifier.output_labels", gguf.ValueTypeString,
	); err != nil {
		return specReadState{}, err
	} else if ok {
		spec.ClassifierLabels = slices.Clone(labels)
	}
	if m.profile.Has(ArchitectureDeepSeek2Layout) {
		spec.VocabularySize, _ = optional[uint32](values, prefix+"vocab_size", gguf.ValueTypeUint32)
		if tokens, ok := values["tokenizer.ggml.tokens"]; ok && spec.VocabularySize == 0 {
			if tokens.Type != gguf.ValueTypeArray || tokens.ArrayType != gguf.ValueTypeString {
				return specReadState{}, errors.New(`metadata "tokenizer.ggml.tokens" must be a string array`)
			}
			if tokens.Count() > int(^uint32(0)) {
				return specReadState{}, errors.New("tokenizer vocabulary exceeds uint32")
			}
			spec.VocabularySize = uint32(tokens.Count())
		}
	}
	state := specReadState{}
	if m.profile.Validation.Hybrid == HybridValidationLlama {
		count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		state.llamaMoE = ok && count > 0
	}
	var err error
	if spec.BlockCount, err = required[uint32](values, prefix+"block_count", gguf.ValueTypeUint32); err != nil {
		return specReadState{}, err
	}
	state.declaredBlockCount = spec.BlockCount
	if err := m.readDraftLayers(spec); err != nil {
		return specReadState{}, err
	}
	if err = readRequiredMetadataFields(
		values, prefix, gguf.ValueTypeUint32,
		metadataDestination("context_length", &spec.ContextLength),
		metadataDestination("embedding_length", &spec.EmbeddingLength),
	); err != nil {
		return specReadState{}, err
	}
	if err := m.readFamilyShape(spec); err != nil {
		return specReadState{}, err
	}
	return state, nil
}

func (m specMetadata) readAttentionShape(spec *Spec, state specReadState) error {
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
	if len(spec.LayerFeedForward) > 0 {
		spec.FeedForwardLength = firstPositive(spec.LayerFeedForward)
	}
	switch policy.Heads {
	case metadataFixedOne:
		spec.HeadCount = 1
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
	if len(spec.LayerHeadCounts) > 0 {
		spec.HeadCount = firstPositive(spec.LayerHeadCounts)
	}
	switch policy.KVHeads {
	case metadataFixedOne:
		spec.HeadCountKV = 1
	case metadataFixedZero:
		spec.HeadCountKV = 0
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
			for block := uint32(0); block < spec.BlockCount; block++ {
				spec.RecurrentLayers[block] = spec.LayerKVHeadCounts[block] == 0 && spec.LayerFeedForward[block] == 0
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
				if count == 0 {
					spec.RecurrentLayers[index] = true
					continue
				}
				if spec.HeadCountKV == 0 {
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
	if spec.HeadCount > 0 && (spec.KeyLength == 0 || spec.ValueLength == 0) {
		if spec.EmbeddingLength%spec.HeadCount != 0 {
			return errors.New("embedding length is not divisible by attention head count")
		}
		headLength := spec.EmbeddingLength / spec.HeadCount
		if spec.KeyLength == 0 {
			spec.KeyLength = headLength
		}
		if spec.ValueLength == 0 {
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
	if profile.readsMetadata(MetadataReadBaichuanBlocks) && spec.BlockCount == 40 {
		spec.RopeDisabled, spec.MaxALiBiBias = true, 8
	}
	if !spec.RopeDisabled && profile.Forward.Operation != ForwardOperationEncoder {
		optionalBase := validation.optionalRopeBase()
		if optionalBase {
			spec.RopeFrequencyBase = 10000
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
			qwenGDNMulti := profile.Attention == AttentionQwenGDN && profile.Has(ArchitectureMultiAxisPositions)
			longRoPE := profile.Has(ArchitectureLongRoPE) && scalingType == "longrope"
			yarn := scalingType == "yarn" && (profile.Has(ArchitectureDeepSeek2Layout) ||
				validation.MLA == MLAValidationDeepSeek4 || validation.supportsYaRN())
			if qwenGDNMulti || scalingType != "linear" && !longRoPE && !yarn {
				return fmt.Errorf("model architecture %q uses unsupported RoPE scaling type %q", architecture, scalingType)
			}
			spec.RopeScalingType = scalingType
			if scalingType == "linear" || scalingType == "yarn" {
				value, err := required[float32](values, prefix+"rope.scaling.factor", gguf.ValueTypeFloat32)
				if err != nil {
					return err
				}
				spec.RopeScalingFactor = value
			}
			if scalingType == "yarn" {
				value, err := required[uint32](values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32)
				if err != nil {
					return err
				}
				spec.OriginalContextLength = value
				spec.YaRNExtFactor = 1
				spec.YaRNAttentionFactor =
					1 / (1 + yarnLogFactorStep*float32(math.Log(float64(spec.RopeScalingFactor))))
				spec.YaRNBetaFast, spec.YaRNBetaSlow = yarnDefaultBetaFast, yarnDefaultBetaSlow
				if validation.Hybrid == HybridValidationGrok {
					spec.YaRNBetaFast = 8
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
			spec.MaxALiBiBias = 8
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
	if validation.Hybrid == HybridValidationDBRX {
		value, err := required[float32](values, prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32)
		if err != nil {
			return err
		}
		spec.AttentionClamp = value
	}
	return nil
}

func (m specMetadata) readDraftLayers(spec *Spec) error {
	key := m.prefix + "nextn_predict_layers"
	nextN, _ := optional[uint32](m.values, key, gguf.ValueTypeUint32)
	validation := m.profile.Validation
	switch {
	case validation.hybridOneOf(HybridValidationQwen35, HybridValidationQwen35MoE):
		if nextN == 0 {
			return nil
		}
		if nextN != 1 || nextN >= spec.BlockCount {
			return errors.New("Qwen3.5 NextN/MTP layer count is invalid")
		}
	case validation.MLA == MLAValidationDeepSeek32 ||
		m.profile.readsMetadata(MetadataReadGLMDSAGating):
		label := "GLM-DSA"
		if validation.MLA == MLAValidationDeepSeek32 {
			spec.LayerNormEpsilon = deepSeek32LayerNormEpsilon
			label = "DeepSeek 3.2"
		}
		if nextN == 0 {
			return nil
		}
		if nextN >= spec.BlockCount {
			return fmt.Errorf("%s NextN/MTP layer count is invalid", label)
		}
	default:
		return nil
	}
	spec.NextNPredictLayers = nextN
	spec.BlockCount -= nextN
	return nil
}

func (m specMetadata) readFamilyShape(spec *Spec) error {
	values, prefix := m.values, m.prefix
	validation := m.profile.Validation
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
			return errors.New("Gemma 4 assistant NextN layer count must match block count")
		}
	case validation.Attention == AttentionValidationGemma3N:
		spec.AltUpCount = 4
		spec.LaurelRank = 64
		spec.EmbeddingPerLayer = 256
		spec.KVFromStart = 20
		spec.SparseLayerCount = 10
		spec.SparsityStdMultiplier = 1.6448533535003662
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
	case validation.Recurrent == RecurrentValidationDFlash:
		if spec.TargetLayers, err = requiredArray[int32](values, prefix+"target_layers", gguf.ValueTypeInt32); err != nil {
			return err
		}
		spec.DFlashBlockSize = 16
		if value, ok := optional[uint32](values, prefix+"block_size", gguf.ValueTypeUint32); ok {
			spec.DFlashBlockSize = value
		}
	case validation.Recurrent == RecurrentValidationEagle3:
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
