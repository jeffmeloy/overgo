package model

import (
	"errors"
	"fmt"
	"slices"

	"llamacpp2go/internal/gguf"
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

func newSpecMetadata(file *gguf.File) (specMetadata, error) {
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
	profile, supported := LookupArchitecture(architecture)
	if !supported {
		return specMetadata{}, &UnsupportedArchitectureError{Architecture: architecture}
	}
	return specMetadata{
		values: values, architecture: architecture, prefix: architecture + ".", profile: profile,
	}, nil
}

func (m specMetadata) readBase(spec *Spec) (specReadState, error) {
	values, architecture, prefix := m.values, m.architecture, m.prefix
	if architecture == "chameleon" {
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
	if architecture == "llama" || architecture == "llama-embed" {
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
	if spec.ContextLength, err = required[uint32](values, prefix+"context_length", gguf.ValueTypeUint32); err != nil {
		return specReadState{}, err
	}
	if spec.EmbeddingLength, err = required[uint32](values, prefix+"embedding_length", gguf.ValueTypeUint32); err != nil {
		return specReadState{}, err
	}
	if err := m.readFamilyShape(spec); err != nil {
		return specReadState{}, err
	}
	return state, nil
}

func (m specMetadata) readDraftLayers(spec *Spec) error {
	key := m.prefix + "nextn_predict_layers"
	nextN, _ := optional[uint32](m.values, key, gguf.ValueTypeUint32)
	switch m.architecture {
	case "qwen35", "qwen35moe":
		if nextN == 0 {
			return nil
		}
		if nextN != 1 || nextN >= spec.BlockCount {
			return errors.New("Qwen3.5 NextN/MTP layer count is invalid")
		}
	case "deepseek32", "glm-dsa":
		if m.architecture == "deepseek32" {
			spec.LayerNormEpsilon = deepSeek32LayerNormEpsilon
		}
		if nextN == 0 {
			return nil
		}
		if nextN >= spec.BlockCount {
			return fmt.Errorf("%s NextN/MTP layer count is invalid", map[string]string{
				"deepseek32": "DeepSeek 3.2", "glm-dsa": "GLM-DSA",
			}[m.architecture])
		}
	default:
		return nil
	}
	spec.NextNPredictLayers = nextN
	spec.BlockCount -= nextN
	return nil
}

func (m specMetadata) readFamilyShape(spec *Spec) error {
	values, architecture, prefix := m.values, m.architecture, m.prefix
	var err error
	switch architecture {
	case "gemma4-assistant":
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
	case "gemma3n":
		spec.AltUpCount = 4
		spec.LaurelRank = 64
		spec.EmbeddingPerLayer = 256
		spec.KVFromStart = 20
		spec.SparseLayerCount = 10
		spec.SparsityStdMultiplier = 1.6448533535003662
		if spec.BlockCount >= spec.KVFromStart {
			spec.SharedKVLayers = spec.BlockCount - spec.KVFromStart
		}
	case "wavtokenizer-dec":
		spec.OutputEmbeddingLength = spec.EmbeddingLength
		if spec.EmbeddingLength, err = required[uint32](values, prefix+"features_length", gguf.ValueTypeUint32); err != nil {
			return err
		}
		for key, destination := range map[string]*uint32{
			"posnet.embedding_length": &spec.PosNetEmbeddingLength, "posnet.block_count": &spec.PosNetBlockCount,
			"convnext.embedding_length": &spec.ConvNextEmbeddingLength, "convnext.block_count": &spec.ConvNextBlockCount,
		} {
			if *destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32); err != nil {
				return err
			}
		}
	case "dflash":
		if spec.TargetLayers, err = requiredArray[int32](values, prefix+"target_layers", gguf.ValueTypeInt32); err != nil {
			return err
		}
		spec.DFlashBlockSize = 16
		if value, ok := optional[uint32](values, prefix+"block_size", gguf.ValueTypeUint32); ok {
			spec.DFlashBlockSize = value
		}
	case "eagle3":
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
