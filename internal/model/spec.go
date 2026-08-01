package model

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/gguf"
)

// Spec: contains common transformer metadata needed to construct model
type Spec struct {
	Architecture            string
	Name                    string
	BlockCount              uint32
	ContextLength           uint32
	EmbeddingLength         uint32
	FeedForwardLength       uint32
	HeadCount               uint32
	HeadCountKV             uint32
	KeyLength               uint32
	ValueLength             uint32
	KeyLengthSWA            uint32
	ValueLengthSWA          uint32
	RopeFrequencyBase       float32
	RopeFrequencySWA        float32
	RopeScalingType         string
	RopeScalingFactor       float32
	RopeAttentionFactor     float32
	RopeYaRNLogMultiplier   float32
	OriginalContextLength   uint32
	AttentionScale          float32
	AttentionTempScale      float32
	AttentionTempFloor      uint32
	AttentionTempOffset     float32
	AttentionValueScale     float32
	AttentionClamp          float32
	MaxALiBiBias            float32
	EmbeddingScale          float32
	ResidualScale           float32
	LogitScale              float32
	AttentionSoftcap        float32
	FinalLogitSoftcap       float32
	RMSNormEpsilon          float32
	LayerNormEpsilon        float32
	VocabularySize          uint32
	TokenTypeCount          uint32
	ExpertCount             uint32
	ExpertUsedCount         uint32
	ExpertFeedForward       uint32
	MoELatentSize           uint32
	ExpertChunkFeedForward  uint32
	ExpertWeightsScale      float32
	ExpertGroupScale        float32
	ExpertsPerGroup         uint32
	LeadingDenseBlocks      uint32
	MoELayerStep            uint32
	SharedExpertFF          uint32
	SharedExpertCount       uint32
	ExpertGatingFunc        uint32
	ExpertWeightsNorm       bool
	ShortConvCacheLength    uint32
	QLoRARank               uint32
	KVLoRARank              uint32
	QKNormEpsilon           float32
	SlidingWindow           uint32
	SlidingPattern          uint32
	RelativeBuckets         uint32
	DecoderBlockCount       uint32
	DecoderStartTokenID     uint32
	OutputEmbeddingLength   uint32
	PosNetEmbeddingLength   uint32
	PosNetBlockCount        uint32
	ConvNextEmbeddingLength uint32
	ConvNextBlockCount      uint32
	GroupNormGroups         uint32
	GroupNormEpsilon        float32
	DFlashBlockSize         uint32
	NextNPredictLayers      uint32
	TargetLayers            []int32
	TargetHiddenSize        uint32
	NormBeforeResidual      bool
	NoRopeLayerStep         uint32
	HiddenActivation        string
	Dense2FeatureIn         uint32
	Dense2FeatureOut        uint32
	Dense3FeatureIn         uint32
	Dense3FeatureOut        uint32
	PoolingType             uint32
	ClassifierLabels        []string
	RopeDisabled            bool
	ParallelResidual        bool
	NonCausalAttention      bool
	SandwichNorm            bool
	YaRNExtFactor           float32
	YaRNAttentionFactor     float32
	YaRNBetaFast            float32
	YaRNBetaSlow            float32
	RopeDimensionSWA        uint32
	LayerHeadCounts         []uint32
	LayerKVHeadCounts       []uint32
	LayerFeedForward        []uint32
	SlidingLayers           []bool
	LayerSwiGLUClamp        []float32
	LayerSharedSwiGLUClamp  []float32
	XIELUAlphaN             []float32
	XIELUAlphaP             []float32
	XIELUBeta               []float32
	XIELUEpsilon            []float32

	// Qwen3.5 hybrid recurrent-attention metadata
	RopeDimensionCount    uint32
	RopeSections          [4]int32
	SSMConvKernel         uint32
	SSMInnerSize          uint32
	SSMStateSize          uint32
	SSMTimeStepRank       uint32
	SSMGroupCount         uint32
	KDAHeadDim            uint32
	SSMDtBCNorm           bool
	FullAttentionInterval uint32
	DeepstackLayerCount   uint32
	DeepstackMapping      []int32
	EmbeddingPerLayer     uint32
	SharedKVLayers        uint32
	KVFromStart           uint32
	AltUpCount            uint32
	AltUpActive           uint32
	LaurelRank            uint32
	SparseLayerCount      uint32
	SparsityStdMultiplier float32
	RecurrentLayers       []bool
	IndexerHeadCount      uint32
	IndexerKeyLength      uint32
	IndexerTopK           uint32
	IndexerFullLayers     []bool
	AttentionOutputGroups uint32
	AttentionOutputRank   uint32
	CompressRopeBase      float32
	CompressRatios        []uint32
	HyperConnectionCount  uint32
	HyperSinkhornIters    uint32
	HyperConnectionEps    float32
	HashLayerCount        uint32

	// RWKV recurrent metadata
	WKVHeadSize       uint32
	TimeMixExtraDim   uint32
	TimeDecayExtraDim uint32
	RescaleEvery      uint32
	TokenShiftCount   uint32
	DecayLoRARank     uint32
	ICLRLoRARank      uint32
	ValueMixLoRARank  uint32
	GateLoRARank      uint32
}

// UnsupportedArchitectureError: identifies valid GGUF architecture that
// runtime cannot execute yet
type UnsupportedArchitectureError struct {
	Architecture string
}

func isDeepSeek2Family(architecture string) bool {
	return architecture == "deepseek2" || architecture == "deepseek32" || architecture == "mistral4"
}

func isDSAArchitecture(architecture string) bool {
	return architecture == "deepseek32" || architecture == "glm-dsa"
}

func isMLAArchitecture(architecture string) bool {
	return architecture == "plm" || architecture == "minicpm3" ||
		isDeepSeek2Family(architecture) || architecture == "glm-dsa"
}

// LayerHasFullIndexer: DSA full-indexer predicate.
func (s Spec) LayerHasFullIndexer(layer uint32) bool {
	return isDSAArchitecture(s.Architecture) && int(layer) < len(s.IndexerFullLayers) && s.IndexerFullLayers[layer]
}

func (e *UnsupportedArchitectureError) Error() string {
	return fmt.Sprintf("model architecture %q is not supported", e.Architecture)
}

// ReadSpec: validates common metadata for initial Llama and Qwen3
// architecture families
func ReadSpec(file *gguf.File) (Spec, error) {
	if file == nil {
		return Spec{}, errors.New("model file is nil")
	}
	values := make(map[string]gguf.Value, len(file.Metadata))
	for _, item := range file.Metadata {
		values[item.Key] = item.Value
	}
	architecture, err := required[string](values, "general.architecture", gguf.ValueTypeString)
	if err != nil {
		return Spec{}, err
	}
	if architecture != "llama" && architecture != "llama4" && architecture != "llama-embed" && architecture != "internlm2" && architecture != "jais" &&
		architecture != "arcee" &&
		architecture != "apertus" &&
		architecture != "arctic" &&
		architecture != "baichuan" &&
		architecture != "bailingmoe" &&
		architecture != "bailingmoe2" &&
		architecture != "bert" &&
		architecture != "bitnet" &&
		architecture != "bloom" &&
		architecture != "codeshell" &&
		architecture != "chameleon" &&
		architecture != "chatglm" &&
		architecture != "cogvlm" &&
		architecture != "dream" &&
		architecture != "deepseek" &&
		architecture != "deepseek2" &&
		architecture != "deepseek32" &&
		architecture != "deepseek4" &&
		architecture != "deepseek2-ocr" &&
		architecture != "glm-dsa" &&
		architecture != "deci" &&
		architecture != "dbrx" &&
		architecture != "dots1" &&
		architecture != "ernie4_5" &&
		architecture != "ernie4_5-moe" &&
		architecture != "eurobert" &&
		architecture != "cohere2" &&
		architecture != "cohere2moe" &&
		architecture != "command-r" &&
		architecture != "jais2" &&
		architecture != "afmoe" &&
		architecture != "laguna" &&
		architecture != "lfm2" &&
		architecture != "lfm2moe" &&
		architecture != "llada" &&
		architecture != "llada-moe" &&
		architecture != "xverse" &&
		architecture != "exaone" && architecture != "olmo2" &&
		architecture != "exaone4" &&
		architecture != "exaone-moe" &&
		architecture != "smollm3" &&
		architecture != "smallthinker" &&
		architecture != "minicpm" &&
		architecture != "minicpm3" &&
		architecture != "minimax-m2" &&
		architecture != "mimo2" &&
		architecture != "step35" &&
		architecture != "granite" &&
		architecture != "granitehybrid" &&
		architecture != "granitemoe" &&
		architecture != "glm4" &&
		architecture != "glm4moe" &&
		architecture != "gpt2" &&
		architecture != "gpt-oss" &&
		architecture != "gptneox" &&
		architecture != "grovemoe" &&
		architecture != "grok" &&
		architecture != "hunyuan-dense" &&
		architecture != "hunyuan_vl" &&
		architecture != "hunyuan-moe" &&
		architecture != "hy_v3" &&
		architecture != "jamba" &&
		architecture != "jina-bert-v2" &&
		architecture != "jina-bert-v3" &&
		architecture != "maincoder" &&
		architecture != "mamba" &&
		architecture != "mamba2" &&
		architecture != "mellum" &&
		architecture != "mistral3" &&
		architecture != "mistral4" &&
		architecture != "modern-bert" &&
		architecture != "mpt" &&
		architecture != "nemotron" &&
		architecture != "nemotron_h" &&
		architecture != "nemotron_h_moe" &&
		architecture != "neo-bert" &&
		architecture != "nomic-bert" &&
		architecture != "nomic-bert-moe" &&
		architecture != "olmo" &&
		architecture != "olmoe" &&
		architecture != "openelm" &&
		architecture != "orion" &&
		architecture != "paddleocr" &&
		architecture != "pangu-embedded" &&
		architecture != "phi2" &&
		architecture != "phi3" &&
		architecture != "phimoe" &&
		architecture != "plamo" &&
		architecture != "plamo2" &&
		architecture != "plamo3" &&
		architecture != "plm" &&
		architecture != "qwen" &&
		architecture != "seed_oss" &&
		architecture != "stablelm" &&
		architecture != "starcoder" &&
		architecture != "starcoder2" &&
		architecture != "qwen2" &&
		architecture != "qwen3" &&
		architecture != "qwen3moe" &&
		architecture != "qwen2moe" &&
		architecture != "qwen2vl" &&
		architecture != "qwen3vl" &&
		architecture != "qwen3vlmoe" &&
		architecture != "qwen3next" && architecture != "qwen35" && architecture != "qwen35moe" && architecture != "gemma" && architecture != "gemma-embedding" &&
		architecture != "kimi-linear" &&
		architecture != "refact" &&
		architecture != "rnd1" &&
		architecture != "rwkv6" &&
		architecture != "rwkv6qwen2" &&
		architecture != "rwkv7" &&
		architecture != "arwkv7" &&
		architecture != "gemma2" &&
		architecture != "gemma3" && architecture != "gemma3n" && architecture != "gemma4" && architecture != "gemma4-assistant" &&
		architecture != "falcon" &&
		architecture != "falcon-h1" &&
		architecture != "talkie" &&
		architecture != "t5" &&
		architecture != "t5encoder" &&
		architecture != "wavtokenizer-dec" &&
		architecture != "dflash" &&
		architecture != "eagle3" {
		return Spec{}, &UnsupportedArchitectureError{Architecture: architecture}
	}
	spec := Spec{Architecture: architecture}
	if architecture == "bert" || architecture == "dream" || architecture == "eurobert" || architecture == "gemma-embedding" || architecture == "jina-bert-v2" || architecture == "jina-bert-v3" || architecture == "llada" || architecture == "llada-moe" || architecture == "llama-embed" || architecture == "modern-bert" || architecture == "neo-bert" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" || architecture == "rnd1" || architecture == "wavtokenizer-dec" || architecture == "dflash" {
		spec.NonCausalAttention = true
	}
	if architecture == "bert" || architecture == "jina-bert-v2" {
		spec.RopeDisabled = true
	}
	if architecture == "mamba" || architecture == "mamba2" || architecture == "jamba" || architecture == "kimi-linear" || architecture == "rwkv6" || architecture == "rwkv6qwen2" || architecture == "rwkv7" || architecture == "arwkv7" || architecture == "t5" || architecture == "wavtokenizer-dec" ||
		architecture == "nemotron_h" || architecture == "nemotron_h_moe" {
		spec.RopeDisabled = true
	}
	if architecture == "chameleon" {
		spec.QKNormEpsilon = 1e-5
		spec.SandwichNorm, _ = optional[bool](values, "chameleon.swin_norm", gguf.ValueTypeBool)
	}
	if value, ok := optional[string](values, "general.name", gguf.ValueTypeString); ok {
		spec.Name = value
	}
	prefix := architecture + "."
	if spec.PoolingType, _ = optional[uint32](values, prefix+"pooling_type", gguf.ValueTypeUint32); spec.PoolingType > 4 {
		return Spec{}, fmt.Errorf("metadata %q has unsupported pooling type %d", prefix+"pooling_type", spec.PoolingType)
	}
	if labels, ok, labelsErr := optionalArray[string](
		values, prefix+"classifier.output_labels", gguf.ValueTypeString,
	); labelsErr != nil {
		return Spec{}, labelsErr
	} else if ok {
		spec.ClassifierLabels = append([]string(nil), labels...)
	}
	if isDeepSeek2Family(architecture) || architecture == "glm-dsa" {
		spec.VocabularySize, _ = optional[uint32](values, prefix+"vocab_size", gguf.ValueTypeUint32)
		if tokens, ok := values["tokenizer.ggml.tokens"]; ok && spec.VocabularySize == 0 {
			if tokens.Type != gguf.ValueTypeArray || tokens.ArrayType != gguf.ValueTypeString {
				return Spec{}, errors.New(`metadata "tokenizer.ggml.tokens" must be a string array`)
			}
			if tokens.Count() > int(^uint32(0)) {
				return Spec{}, errors.New("tokenizer vocabulary exceeds uint32")
			}
			spec.VocabularySize = uint32(tokens.Count())
		}
	}
	isLlamaMoE := false
	if architecture == "llama" || architecture == "llama-embed" {
		if count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); ok && count > 0 {
			isLlamaMoE = true
		}
	}
	if spec.BlockCount, err = required[uint32](values, prefix+"block_count", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	declaredBlockCount := spec.BlockCount
	if architecture == "qwen35" || architecture == "qwen35moe" {
		spec.NextNPredictLayers, _ = optional[uint32](
			values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32,
		)
		if spec.NextNPredictLayers > 0 {
			if spec.NextNPredictLayers != 1 || spec.NextNPredictLayers >= spec.BlockCount {
				return Spec{}, errors.New("Qwen3.5 NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= spec.NextNPredictLayers
		}
	}
	if architecture == "deepseek32" {
		spec.LayerNormEpsilon = 1e-6
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("DeepSeek 3.2 NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= nextN
		}
	}
	if spec.ContextLength, err = required[uint32](values, prefix+"context_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if spec.EmbeddingLength, err = required[uint32](values, prefix+"embedding_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if architecture == "gemma4-assistant" {
		if spec.TargetHiddenSize, err = required[uint32](values, prefix+"embedding_length_out", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		nextN, nextErr := required[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32)
		if nextErr != nil {
			return Spec{}, nextErr
		}
		if nextN != spec.BlockCount {
			return Spec{}, errors.New("Gemma 4 assistant NextN layer count must match block count")
		}
	}
	if architecture == "gemma3n" {
		spec.AltUpCount = 4
		spec.AltUpActive = 0
		spec.LaurelRank = 64
		spec.EmbeddingPerLayer = 256
		spec.KVFromStart = 20
		spec.SparseLayerCount = 10
		spec.SparsityStdMultiplier = 1.6448533535003662
		if spec.BlockCount >= spec.KVFromStart {
			spec.SharedKVLayers = spec.BlockCount - spec.KVFromStart
		}
	}
	if architecture == "wavtokenizer-dec" {
		spec.OutputEmbeddingLength = spec.EmbeddingLength
		if spec.EmbeddingLength, err = required[uint32](values, prefix+"features_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		for key, destination := range map[string]*uint32{
			"posnet.embedding_length":   &spec.PosNetEmbeddingLength,
			"posnet.block_count":        &spec.PosNetBlockCount,
			"convnext.embedding_length": &spec.ConvNextEmbeddingLength,
			"convnext.block_count":      &spec.ConvNextBlockCount,
		} {
			if *destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		}
	}
	if architecture == "dflash" {
		if spec.TargetLayers, err = requiredArray[int32](values, prefix+"target_layers", gguf.ValueTypeInt32); err != nil {
			return Spec{}, err
		}
		spec.DFlashBlockSize = 16
		if value, ok := optional[uint32](values, prefix+"block_size", gguf.ValueTypeUint32); ok {
			spec.DFlashBlockSize = value
		}
	}
	if architecture == "eagle3" {
		if spec.TargetLayers, err = requiredArray[int32](values, prefix+"target_layers", gguf.ValueTypeInt32); err != nil {
			return Spec{}, err
		}
		if spec.TargetHiddenSize, err = required[uint32](values, prefix+"target_hidden_size", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.NormBeforeResidual, _ = optional[bool](values, prefix+"norm_before_residual", gguf.ValueTypeBool)
	}
	if architecture == "gemma4" || architecture == "nemotron_h" || architecture == "nemotron_h_moe" {
		if spec.LayerFeedForward, err = requiredLayerUint32Compatible(
			values, prefix+"feed_forward_length", spec.BlockCount,
		); err != nil {
			return Spec{}, err
		}
		spec.FeedForwardLength = firstPositive(spec.LayerFeedForward)
	} else if architecture == "deci" || architecture == "openelm" || architecture == "plamo3" {
		if spec.LayerFeedForward, err = requiredLayerUint32(values, prefix+"feed_forward_length", spec.BlockCount); err != nil {
			return Spec{}, err
		}
		spec.FeedForwardLength = firstPositive(spec.LayerFeedForward)
	} else if spec.FeedForwardLength, err = required[uint32](values, prefix+"feed_forward_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if architecture == "wavtokenizer-dec" {
		spec.HeadCount = 1
	} else if architecture == "step35" || architecture == "nemotron_h" || architecture == "nemotron_h_moe" {
		if spec.LayerHeadCounts, err = requiredLayerUint32Compatible(
			values, prefix+"attention.head_count", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
		spec.HeadCount = firstPositive(spec.LayerHeadCounts)
	} else if architecture == "deci" || architecture == "laguna" || architecture == "openelm" || architecture == "plamo3" {
		if spec.LayerHeadCounts, err = requiredLayerUint32(
			values, prefix+"attention.head_count", spec.BlockCount,
		); err != nil {
			return Spec{}, err
		}
		spec.HeadCount = firstPositive(spec.LayerHeadCounts)
	} else if spec.HeadCount, err = required[uint32](values, prefix+"attention.head_count", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if architecture == "wavtokenizer-dec" {
		spec.HeadCountKV = 1
	} else if architecture == "mamba" || architecture == "mamba2" {
		spec.HeadCountKV = 0
	} else if architecture == "bert" || architecture == "gemma-embedding" || architecture == "jina-bert-v2" || architecture == "jina-bert-v3" || architecture == "modern-bert" || architecture == "neo-bert" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" || architecture == "t5" || architecture == "t5encoder" || architecture == "bloom" || architecture == "gpt2" || architecture == "jais" || architecture == "mpt" || architecture == "qwen" ||
		architecture == "starcoder" || architecture == "gptneox" || architecture == "falcon" {
		spec.HeadCountKV = spec.HeadCount
		if architecture == "gptneox" || architecture == "falcon" || architecture == "gemma-embedding" || architecture == "jina-bert-v2" || architecture == "jina-bert-v3" || architecture == "mpt" || architecture == "neo-bert" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" {
			if value, ok := optional[uint32](
				values,
				prefix+"attention.head_count_kv",
				gguf.ValueTypeUint32,
			); ok {
				spec.HeadCountKV = value
			}
		}
	} else if architecture == "mimo2" || architecture == "step35" || architecture == "gemma4" || architecture == "nemotron_h" || architecture == "nemotron_h_moe" {
		if spec.LayerKVHeadCounts, err = requiredLayerUint32Compatible(
			values, prefix+"attention.head_count_kv", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
		spec.HeadCountKV = firstPositive(spec.LayerKVHeadCounts)
		if architecture == "nemotron_h" || architecture == "nemotron_h_moe" {
			spec.RecurrentLayers = make([]bool, spec.BlockCount)
			for block := uint32(0); block < spec.BlockCount; block++ {
				spec.RecurrentLayers[block] = spec.LayerKVHeadCounts[block] == 0 && spec.LayerFeedForward[block] == 0
			}
		}
	} else if architecture == "deci" || architecture == "laguna" || architecture == "openelm" || architecture == "plamo3" {
		if spec.LayerKVHeadCounts, err = requiredLayerUint32(
			values, prefix+"attention.head_count_kv", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
		spec.HeadCountKV = firstPositive(spec.LayerKVHeadCounts)
	} else if architecture == "lfm2" || architecture == "lfm2moe" || architecture == "jamba" || architecture == "granitehybrid" || architecture == "plamo2" || architecture == "kimi-linear" {
		counts, countErr := requiredArray[uint32](
			values, prefix+"attention.head_count_kv", gguf.ValueTypeUint32,
		)
		if countErr != nil {
			return Spec{}, countErr
		}
		if len(counts) != int(spec.BlockCount) {
			return Spec{}, fmt.Errorf(
				"metadata %q has %d values, need %d",
				prefix+"attention.head_count_kv", len(counts), spec.BlockCount,
			)
		}
		spec.RecurrentLayers = make([]bool, len(counts))
		if architecture == "jamba" || architecture == "granitehybrid" || architecture == "plamo2" || architecture == "kimi-linear" {
			spec.LayerKVHeadCounts = append([]uint32(nil), counts...)
		}
		for index, count := range counts {
			if count == 0 {
				spec.RecurrentLayers[index] = true
				continue
			}
			if spec.HeadCountKV == 0 {
				spec.HeadCountKV = count
			} else if spec.HeadCountKV != count {
				return Spec{}, errors.New("hybrid attention layers use differing positive KV head counts")
			}
		}
	} else {
		if spec.HeadCountKV, err = required[uint32](values, prefix+"attention.head_count_kv", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "wavtokenizer-dec" {
		if spec.GroupNormEpsilon, err = required[float32](values, prefix+"attention.group_norm_epsilon", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if spec.GroupNormGroups, err = required[uint32](values, prefix+"attention.group_norm_groups", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	spec.KeyLength, _ = optional[uint32](values, prefix+"attention.key_length", gguf.ValueTypeUint32)
	spec.ValueLength, _ = optional[uint32](values, prefix+"attention.value_length", gguf.ValueTypeUint32)
	if architecture == "wavtokenizer-dec" {
		spec.KeyLength = spec.PosNetEmbeddingLength
		spec.ValueLength = spec.PosNetEmbeddingLength
	}
	if isDeepSeek2Family(architecture) || architecture == "glm-dsa" || architecture == "kimi-linear" {
		if value, ok := optional[uint32](values, prefix+"attention.key_length_mla", gguf.ValueTypeUint32); ok {
			spec.KeyLength = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.value_length_mla", gguf.ValueTypeUint32); ok {
			spec.ValueLength = value
		}
	}
	if architecture != "mamba" && architecture != "mamba2" && (spec.KeyLength == 0 || spec.ValueLength == 0) {
		if spec.HeadCount == 0 || spec.EmbeddingLength%spec.HeadCount != 0 {
			return Spec{}, errors.New("embedding length is not divisible by attention head count")
		}
		headLength := spec.EmbeddingLength / spec.HeadCount
		if spec.KeyLength == 0 {
			spec.KeyLength = headLength
		}
		if spec.ValueLength == 0 {
			spec.ValueLength = headLength
		}
	}
	if architecture == "gemma4" || architecture == "gemma4-assistant" {
		if spec.KeyLengthSWA, err = required[uint32](values, prefix+"attention.key_length_swa", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ValueLengthSWA, err = required[uint32](values, prefix+"attention.value_length_swa", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "baichuan" && spec.BlockCount == 40 {
		spec.RopeDisabled = true
		spec.MaxALiBiBias = 8
	}
	if architecture == "bloom" || architecture == "gpt2" || architecture == "jais" || architecture == "mpt" ||
		architecture == "refact" || architecture == "starcoder" {
		// architectures use ALiBi or learned absolute rows instead of RoPE
		spec.RopeDisabled = true
	} else if !spec.RopeDisabled && architecture != "t5encoder" {
		if architecture == "gptneox" || architecture == "falcon" || architecture == "deepseek2-ocr" || architecture == "gemma-embedding" || architecture == "jina-bert-v3" || architecture == "modern-bert" || architecture == "neo-bert" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" {
			spec.RopeFrequencyBase = 10000
			if value, ok := optional[float32](
				values,
				prefix+"rope.freq_base",
				gguf.ValueTypeFloat32,
			); ok {
				spec.RopeFrequencyBase = value
			}
		} else if architecture == "minicpm3" {
			spec.RopeFrequencyBase = 10000
			if value, ok := optional[float32](values, prefix+"rope.freq_base", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencyBase = value
			}
		} else if spec.RopeFrequencyBase, err = required[float32](values, prefix+"rope.freq_base", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if scalingType, ok := optional[string](
			values,
			prefix+"rope.scaling.type",
			gguf.ValueTypeString,
		); ok && scalingType != "" && scalingType != "none" {
			if architecture == "qwen35" || architecture == "qwen35moe" ||
				(scalingType != "linear" && !(supportsLongRoPE(architecture) && scalingType == "longrope")) {
				if (!isDeepSeek2Family(architecture) && architecture != "deepseek4" && architecture != "glm-dsa" && architecture != "laguna" && architecture != "grok" && architecture != "mellum" && architecture != "llama" && architecture != "llama-embed" && architecture != "minicpm" && architecture != "mistral3") || scalingType != "yarn" {
					return Spec{}, fmt.Errorf(
						"model architecture %q uses unsupported RoPE scaling type %q",
						architecture,
						scalingType,
					)
				}
			}
			spec.RopeScalingType = scalingType
			if scalingType == "linear" || scalingType == "yarn" {
				if spec.RopeScalingFactor, err = required[float32](
					values,
					prefix+"rope.scaling.factor",
					gguf.ValueTypeFloat32,
				); err != nil {
					return Spec{}, err
				}
			}
			if scalingType == "yarn" {
				if spec.OriginalContextLength, err = required[uint32](
					values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
				); err != nil {
					return Spec{}, err
				}
				spec.YaRNExtFactor = 1
				// ggml's YaRN primitive applies its logarithmic magnitude factor
				// internally; llama.cpp cancels it in context parameters for
				// ordinary YaRN models, leaving rotation interpolation without
				// unintended residual-vector scale
				spec.YaRNAttentionFactor = 1 / (1 + 0.1*float32(math.Log(float64(spec.RopeScalingFactor))))
				spec.YaRNBetaFast = 32
				if architecture == "grok" {
					spec.YaRNBetaFast = 8
				}
				spec.YaRNBetaSlow = 1
				for key, destination := range map[string]*float32{
					"rope.scaling.yarn_ext_factor":  &spec.YaRNExtFactor,
					"rope.scaling.yarn_attn_factor": &spec.YaRNAttentionFactor,
					"rope.scaling.yarn_beta_fast":   &spec.YaRNBetaFast,
					"rope.scaling.yarn_beta_slow":   &spec.YaRNBetaSlow,
				} {
					if value, ok := optional[float32](values, prefix+key, gguf.ValueTypeFloat32); ok {
						*destination = value
					}
				}
			}
		}
	}
	if architecture == "bloom" || architecture == "jais" || architecture == "jina-bert-v2" || architecture == "mpt" || architecture == "refact" {
		if architecture != "mpt" {
			spec.MaxALiBiBias = 8
		}
		if architecture != "jina-bert-v2" {
			if value, ok := optional[float32](
				values, prefix+"attention.max_alibi_bias", gguf.ValueTypeFloat32,
			); ok {
				spec.MaxALiBiBias = value
			}
		}
	}
	if architecture == "mpt" {
		if clamp, ok := optional[float32](
			values, prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32,
		); ok {
			spec.AttentionClamp = clamp
		}
	}
	if architecture == "dbrx" {
		if spec.AttentionClamp, err = required[float32](
			values, prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "refact" {
		if expertCount, ok := optional[uint32](
			values, prefix+"expert_count", gguf.ValueTypeUint32,
		); ok && expertCount > 0 {
			return Spec{}, errors.New("Refact expert layers are not supported")
		}
	}
	if architecture == "cohere2moe" {
		if value, ok := optional[float32](values, prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32); ok {
			spec.RMSNormEpsilon = value
		} else if value, ok := optional[float32](values, prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32); ok {
			spec.LayerNormEpsilon = value
		} else {
			return Spec{}, errors.New("Cohere2-MoE norm epsilon is missing")
		}
	} else if spec.UsesLayerNorm() || spec.UsesWeightOnlyLayerNorm() || spec.UsesUnweightedLayerNorm() {
		if spec.LayerNormEpsilon, err = required[float32](
			values,
			prefix+"attention.layer_norm_epsilon",
			gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
	} else {
		if spec.RMSNormEpsilon, err = required[float32](
			values,
			prefix+"attention.layer_norm_rms_epsilon",
			gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
	}
	spec.FinalLogitSoftcap, _ = optional[float32](
		values,
		prefix+"final_logit_softcapping",
		gguf.ValueTypeFloat32,
	)
	spec.AttentionSoftcap, _ = optional[float32](
		values,
		prefix+"attn_logit_softcapping",
		gguf.ValueTypeFloat32,
	)
	if architecture == "jais" {
		spec.AttentionScale = 1 / float32(spec.KeyLength)
	}
	if value, ok := optional[float32](values, prefix+"attention.scale", gguf.ValueTypeFloat32); ok {
		spec.AttentionScale = value
	}
	if architecture == "grok" {
		spec.AttentionSoftcap = 30
		if value, ok := optional[float32](values, prefix+"attn_logit_softcapping", gguf.ValueTypeFloat32); ok {
			spec.AttentionSoftcap = value
		}
		spec.AttentionScale = 0.08838834764831845
		if value, ok := optional[float32](values, prefix+"attention.output_scale", gguf.ValueTypeFloat32); ok {
			spec.AttentionScale = value
		}
		spec.EmbeddingScale = 78.38367176906169
		if value, ok := optional[float32](values, prefix+"embedding_scale", gguf.ValueTypeFloat32); ok {
			spec.EmbeddingScale = value
		}
		spec.LogitScale = 0.5773502691896257
		if value, ok := optional[float32](values, prefix+"logit_scale", gguf.ValueTypeFloat32); ok {
			spec.LogitScale = value
		}
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "talkie" {
		if spec.LogitScale, err = required[float32](values, prefix+"logit_scale", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "dflash" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if window, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok && window > 0 {
			spec.SlidingWindow = window
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if pattern, patternOK := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); patternOK {
				spec.SlidingPattern = pattern
			} else if layers, layersOK, layersErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); layersErr != nil {
				return Spec{}, layersErr
			} else if layersOK {
				if len(layers) != int(spec.BlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
				}
				spec.SlidingLayers = append([]bool(nil), layers...)
			}
		}
	}
	if architecture == "eagle3" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "gemma3n" {
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "cogvlm" {
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "minicpm" || architecture == "minicpm3" {
		spec.EmbeddingScale = 12
		spec.ResidualScale = float32(1.4 / math.Sqrt(float64(spec.BlockCount)))
		spec.LogitScale = 256 / float32(spec.EmbeddingLength)
		if architecture == "minicpm3" {
			spec.OriginalContextLength = spec.ContextLength
			spec.RopeAttentionFactor = 1
			if value, ok := optional[uint32](values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32); ok {
				spec.OriginalContextLength = value
			}
			if value, ok := optional[float32](values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32); ok {
				spec.RopeAttentionFactor = value
			}
		}
		if value, ok := optional[float32](
			values,
			prefix+"embedding_scale",
			gguf.ValueTypeFloat32,
		); ok {
			spec.EmbeddingScale = value
		}
		if value, ok := optional[float32](
			values,
			prefix+"residual_scale",
			gguf.ValueTypeFloat32,
		); ok {
			spec.ResidualScale = value
		}
		if value, ok := optional[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		); ok {
			spec.LogitScale = value
		}
	}
	if architecture == "granite" || architecture == "granitemoe" || architecture == "granitehybrid" {
		if architecture == "granitehybrid" {
			spec.LogitScale, _ = optional[float32](values, prefix+"logit_scale", gguf.ValueTypeFloat32)
		} else if spec.LogitScale, err = required[float32](
			values, prefix+"logit_scale", gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
		spec.EmbeddingScale, _ = optional[float32](
			values,
			prefix+"embedding_scale",
			gguf.ValueTypeFloat32,
		)
		spec.ResidualScale, _ = optional[float32](
			values,
			prefix+"residual_scale",
			gguf.ValueTypeFloat32,
		)
		spec.RopeDimensionCount = spec.KeyLength
		spec.RopeDimensionCount, _ = optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		)
		if spec.RopeDimensionCount == 0 {
			spec.RopeDimensionCount = spec.KeyLength
		}
		spec.OriginalContextLength = spec.ContextLength
		spec.OriginalContextLength, _ = optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		)
		if spec.OriginalContextLength == 0 {
			spec.OriginalContextLength = spec.ContextLength
		}
		spec.RopeAttentionFactor = 1
		spec.RopeAttentionFactor, _ = optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		)
		if spec.RopeAttentionFactor == 0 {
			spec.RopeAttentionFactor = 1
		}
		ropeEnabled := true
		if value, ok := optional[bool](
			values,
			prefix+"rope.scaling.finetuned",
			gguf.ValueTypeBool,
		); ok {
			ropeEnabled = value
		}
		spec.RopeDisabled = !ropeEnabled
		if architecture == "granite" {
			if expertCount, ok := optional[uint32](
				values,
				prefix+"expert_count",
				gguf.ValueTypeUint32,
			); ok {
				spec.ExpertCount = expertCount
			}
		}
		if mapping, ok, mappingErr := optionalArray[int32](
			values,
			prefix+"deepstack_mapping",
			gguf.ValueTypeInt32,
		); mappingErr != nil {
			return Spec{}, mappingErr
		} else if ok && len(mapping) > 0 {
			if architecture != "granite" {
				return Spec{}, errors.New("Granite deepstack mapping requires granite architecture")
			}
			if len(mapping) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("Granite deepstack mapping has %d entries, need %d", len(mapping), spec.BlockCount)
			}
			unique := make(map[int32]struct{})
			for _, index := range mapping {
				if index < -1 {
					return Spec{}, errors.New("Granite deepstack mapping index is invalid")
				}
				if index >= 0 {
					unique[index] = struct{}{}
				}
			}
			for index := range unique {
				if uint32(index) > uint32(len(unique)) {
					return Spec{}, errors.New("Granite deepstack mapping index exceeds stream count")
				}
			}
			spec.DeepstackLayerCount = uint32(len(unique))
			spec.DeepstackMapping = append([]int32(nil), mapping...)
		}
	}
	if architecture == "cohere2" || architecture == "cohere2moe" {
		if spec.LogitScale, err = required[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if pattern, ok := values[prefix+"attention.sliding_window_pattern"]; ok && pattern.Type != gguf.ValueTypeUint32 &&
			(architecture != "cohere2moe" || pattern.Type != gguf.ValueTypeArray || pattern.ArrayType != gguf.ValueTypeBool) {
			return Spec{}, errors.New("Cohere2 array sliding attention patterns are not supported")
		}
	}
	if architecture == "stablelm" {
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "phi2" {
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "phi3" || architecture == "phimoe" {
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.OriginalContextLength, err = required[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeAttentionFactor = value
		}
	}
	if architecture == "pangu-embedded" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32); ok {
			spec.OriginalContextLength = value
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32); ok {
			spec.RopeAttentionFactor = value
		}
	}
	if architecture == "modern-bert" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.RopeFrequencySWA = 10000
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 3
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
		}
		spec.HiddenActivation = "gelu"
		if value, ok := optional[string](values, prefix+"hidden_act", gguf.ValueTypeString); ok {
			spec.HiddenActivation = value
		}
		if spec.HiddenActivation != "gelu" && spec.HiddenActivation != "silu" {
			return Spec{}, fmt.Errorf("ModernBERT hidden activation %q is unsupported", spec.HiddenActivation)
		}
	}
	if architecture == "gemma-embedding" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.RopeFrequencySWA = 10000
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = 6
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = value
		}
		spec.Dense2FeatureIn, _ = optional[uint32](values, prefix+"dense_2_feat_in", gguf.ValueTypeUint32)
		spec.Dense2FeatureOut, _ = optional[uint32](values, prefix+"dense_2_feat_out", gguf.ValueTypeUint32)
		spec.Dense3FeatureIn, _ = optional[uint32](values, prefix+"dense_3_feat_in", gguf.ValueTypeUint32)
		spec.Dense3FeatureOut, _ = optional[uint32](values, prefix+"dense_3_feat_out", gguf.ValueTypeUint32)
	}
	if architecture == "deci" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); ok {
			spec.OriginalContextLength = value
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeAttentionFactor = value
		}
	}
	if architecture == "apertus" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); ok {
			spec.OriginalContextLength = value
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeAttentionFactor = value
		}
		for key, destination := range map[string]*[]float32{
			"xielu.alpha_n": &spec.XIELUAlphaN,
			"xielu.alpha_p": &spec.XIELUAlphaP,
			"xielu.beta":    &spec.XIELUBeta,
			"xielu.eps":     &spec.XIELUEpsilon,
		} {
			*destination, err = requiredLayerFloat32(values, prefix+key, spec.BlockCount)
			if err != nil {
				return Spec{}, err
			}
		}
	}
	if architecture == "gptneox" {
		spec.RopeDimensionCount, _ = optional[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		)
		if spec.ParallelResidual, err = required[bool](
			values,
			prefix+"use_parallel_residual",
			gguf.ValueTypeBool,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "glm4" || architecture == "glm4moe" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](
			values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32,
		); ok && nextN > 0 {
			if architecture == "glm4" {
				return Spec{}, errors.New("GLM4 NextN/MTP layers are not supported")
			}
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("GLM4-MoE NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= nextN
		}
		if sections, ok, sectionsErr := optionalArray[int32](
			values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32,
		); sectionsErr != nil {
			return Spec{}, sectionsErr
		} else if ok {
			if len(sections) != 4 {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need 4", prefix+"rope.dimension_sections", len(sections))
			}
			copy(spec.RopeSections[:], sections)
		}
	}
	if architecture == "mimo2" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("MiMo2 NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= nextN
			spec.LayerKVHeadCounts = spec.LayerKVHeadCounts[:spec.BlockCount]
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		patternKey := prefix + "attention.sliding_window_pattern"
		if pattern, ok := optional[uint32](values, patternKey, gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = pattern
		} else {
			pattern, exists := values[patternKey]
			if !exists || pattern.Type != gguf.ValueTypeArray {
				return Spec{}, fmt.Errorf("required metadata %q is missing", patternKey)
			}
			spec.SlidingLayers = make([]bool, spec.BlockCount)
			switch pattern.ArrayType {
			case gguf.ValueTypeBool:
				layers, valid := pattern.Data.([]bool)
				if !valid || len(layers) != int(declaredBlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has invalid layer values", patternKey)
				}
				copy(spec.SlidingLayers, layers)
			case gguf.ValueTypeUint32:
				layers, valid := pattern.Data.([]uint32)
				if !valid || len(layers) != int(declaredBlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has invalid layer values", patternKey)
				}
				for index := range spec.SlidingLayers {
					spec.SlidingLayers[index] = layers[index] != 0
				}
			case gguf.ValueTypeInt32:
				layers, valid := pattern.Data.([]int32)
				if !valid || len(layers) != int(declaredBlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has invalid layer values", patternKey)
				}
				for index := range spec.SlidingLayers {
					if layers[index] < 0 {
						return Spec{}, fmt.Errorf("metadata %q has a negative layer value", patternKey)
					}
					spec.SlidingLayers[index] = layers[index] != 0
				}
			default:
				return Spec{}, fmt.Errorf("metadata %q must be an integer or bool array", patternKey)
			}
		}
		if value, ok := optional[float32](values, prefix+"attention.value_scale", gguf.ValueTypeFloat32); ok && value != 1 {
			spec.AttentionValueScale = value
		}
	}
	if architecture == "step35" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("Step3.5 NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.SlidingLayers, err = requiredLayerBoolCompatible(
			values, prefix+"attention.sliding_window_pattern", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
		if spec.LayerSwiGLUClamp, err = optionalLayerFloat32(
			values, prefix+"swiglu_clamp_exp", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
		if spec.LayerSharedSwiGLUClamp, err = optionalLayerFloat32(
			values, prefix+"swiglu_clamp_shexp", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "exaone4" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](
			values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32,
		); ok && nextN > 0 {
			return Spec{}, errors.New("EXAONE 4 NextN/MTP layers are not supported")
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](
			values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.BlockCount == 64 {
			spec.SlidingWindow = 4096
		}
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](
				values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32,
			); ok {
				spec.SlidingPattern = value
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
		}
	}
	if architecture == "exaone-moe" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("EXAONE-MoE NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= nextN
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = 4
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = value
		} else if layers, ok, arrayErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); arrayErr != nil {
			return Spec{}, arrayErr
		} else if ok {
			if len(layers) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
			}
			spec.SlidingLayers = append([]bool(nil), layers...)
		}
	}
	if architecture == "bailingmoe2" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("BailingMoE2 NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= nextN
		}
	}
	if architecture == "falcon" {
		spec.RopeDimensionCount, _ = optional[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		)
	}
	if architecture == "command-r" {
		spec.LogitScale, _ = optional[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		)
	}
	if architecture == "baichuan" && spec.BlockCount != 32 && spec.BlockCount != 40 {
		return Spec{}, errors.New("Baichuan block count must select the 32-layer RoPE or 40-layer ALiBi variant")
	}
	if architecture == "mistral3" {
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); ok {
			spec.OriginalContextLength = value
		}
		spec.AttentionTempScale, _ = optional[float32](
			values, prefix+"attention.temperature_scale", gguf.ValueTypeFloat32,
		)
		if spec.AttentionTempScale != 0 {
			spec.AttentionTempFloor = spec.OriginalContextLength
		}
		if spec.RopeScalingType == "yarn" {
			spec.RopeYaRNLogMultiplier, _ = optional[float32](
				values, prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32,
			)
			rawAttentionFactor := float32(1)
			if value, ok := optional[float32](
				values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
			); ok {
				rawAttentionFactor = value
			}
			denominator := float32(1)
			if spec.RopeYaRNLogMultiplier != 0 {
				denominator += 0.1 * spec.RopeYaRNLogMultiplier *
					float32(math.Log(float64(spec.RopeScalingFactor)))
			}
			spec.YaRNAttentionFactor = rawAttentionFactor / denominator
		}
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		if spec.ExpertCount > 0 {
			if spec.ExpertUsedCount, err = required[uint32](
				values, prefix+"expert_used_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsNorm = true
			spec.ExpertWeightsScale = 1
			if value, ok := optional[float32](
				values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32,
			); ok && value != 0 {
				spec.ExpertWeightsScale = value
			}
		}
	}
	if architecture == "qwen" {
		if spec.FeedForwardLength == 0 || spec.FeedForwardLength%2 != 0 {
			return Spec{}, errors.New("Qwen feed-forward length must be positive and even")
		}
		spec.FeedForwardLength /= 2
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "chatglm" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "hunyuan-dense" || architecture == "hunyuan_vl" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if sections, ok, sectionsErr := optionalArray[int32](values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32); sectionsErr != nil {
			return Spec{}, sectionsErr
		} else if ok {
			if len(sections) != 4 {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need 4", prefix+"rope.dimension_sections", len(sections))
			}
			copy(spec.RopeSections[:], sections)
		}
		if alpha, ok := optional[float32](values, prefix+"rope.scaling.alpha", gguf.ValueTypeFloat32); ok && alpha != 0 {
			if alpha < 0 || spec.KeyLength <= 2 || math.IsNaN(float64(alpha)) || math.IsInf(float64(alpha), 0) {
				return Spec{}, errors.New("Hunyuan-Dense XDRoPE alpha is invalid")
			}
			exponent := float64(spec.KeyLength) / float64(spec.KeyLength-2)
			spec.RopeFrequencyBase *= float32(math.Pow(float64(alpha), exponent))
		}
	}
	if architecture == "paddleocr" || architecture == "qwen2vl" || architecture == "qwen3vl" || architecture == "qwen3vlmoe" {
		spec.RopeDimensionCount = spec.KeyLength
		sections, sectionsErr := requiredArray[int32](
			values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32,
		)
		if sectionsErr != nil {
			return Spec{}, sectionsErr
		}
		if len(sections) != len(spec.RopeSections) {
			return Spec{}, fmt.Errorf(
				"metadata %q has %d values, need %d",
				prefix+"rope.dimension_sections", len(sections), len(spec.RopeSections),
			)
		}
		copy(spec.RopeSections[:], sections)
	}
	if architecture == "qwen3vl" || architecture == "qwen3vlmoe" {
		if value, ok := optional[uint32](values, prefix+"n_deepstack_layers", gguf.ValueTypeUint32); ok {
			spec.DeepstackLayerCount = value
		}
	}
	if architecture == "smollm3" {
		spec.NoRopeLayerStep = 4
	}
	if architecture == "smallthinker" {
		spec.RopeDimensionCount = spec.KeyLength
		spec.NoRopeLayerStep = spec.BlockCount
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok && value > 0 {
			spec.SlidingWindow = 4096
			spec.SlidingPattern = 4
			if pattern, patternOK := optional[uint32](
				values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32,
			); patternOK {
				spec.SlidingPattern = pattern
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if frequency, frequencyOK := optional[float32](
				values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32,
			); frequencyOK {
				spec.RopeFrequencySWA = frequency
			}
		}
	}
	if architecture == "mellum" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			} else if layers, ok, arrayErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); arrayErr != nil {
				return Spec{}, arrayErr
			} else if ok {
				if len(layers) != int(spec.BlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
				}
				spec.SlidingLayers = append([]bool(nil), layers...)
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
		}
	}
	if architecture == "plamo3" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 8
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			} else if layers, ok, arrayErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); arrayErr != nil {
				return Spec{}, arrayErr
			} else if ok {
				if len(layers) != int(spec.BlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
				}
				spec.SlidingLayers = append([]bool(nil), layers...)
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
		}
	}
	if architecture == "afmoe" {
		spec.NoRopeLayerStep = 4
	}
	if architecture == "olmo" {
		if clamp, ok := optional[float32](
			values,
			prefix+"attention.clamp_kqv",
			gguf.ValueTypeFloat32,
		); ok {
			spec.AttentionClamp = clamp
		}
	}
	if architecture == "t5" || architecture == "t5encoder" {
		if spec.RelativeBuckets, err = required[uint32](
			values,
			prefix+"attention.relative_buckets_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if architecture == "t5" {
			spec.DecoderBlockCount = spec.BlockCount
			if value, ok := optional[uint32](values, prefix+"decoder_block_count", gguf.ValueTypeUint32); ok {
				spec.DecoderBlockCount = value
			}
			spec.DecoderStartTokenID, _ = optional[uint32](values, prefix+"decoder_start_token_id", gguf.ValueTypeUint32)
		}
	}
	if architecture == "qwen3next" || architecture == "qwen35" || architecture == "qwen35moe" {
		if architecture == "qwen3next" {
			spec.RopeDimensionCount = spec.KeyLength
			if value, ok := optional[uint32](
				values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
			); ok {
				spec.RopeDimensionCount = value
			}
		} else {
			if spec.RopeDimensionCount, err = required[uint32](
				values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			sections, sectionsErr := requiredArray[int32](
				values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32,
			)
			if sectionsErr != nil {
				return Spec{}, sectionsErr
			}
			if len(sections) != len(spec.RopeSections) {
				return Spec{}, fmt.Errorf(
					"metadata %q has %d values, need %d",
					prefix+"rope.dimension_sections", len(sections), len(spec.RopeSections),
				)
			}
			copy(spec.RopeSections[:], sections)
		}
		for key, destination := range map[string]*uint32{
			"ssm.conv_kernel":    &spec.SSMConvKernel,
			"ssm.inner_size":     &spec.SSMInnerSize,
			"ssm.state_size":     &spec.SSMStateSize,
			"ssm.time_step_rank": &spec.SSMTimeStepRank,
			"ssm.group_count":    &spec.SSMGroupCount,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.FullAttentionInterval = 4
		if interval, ok := optional[uint32](
			values,
			prefix+"full_attention_interval",
			gguf.ValueTypeUint32,
		); ok {
			spec.FullAttentionInterval = interval
		}
		if recurrent, ok, recurrentErr := optionalArray[bool](
			values,
			prefix+"attention.recurrent_layers",
			gguf.ValueTypeBool,
		); recurrentErr != nil {
			return Spec{}, recurrentErr
		} else if ok {
			if len(recurrent) != int(spec.BlockCount) && len(recurrent) != int(declaredBlockCount) {
				return Spec{}, fmt.Errorf(
					"metadata %q has %d values, need %d or %d",
					prefix+"attention.recurrent_layers",
					len(recurrent),
					spec.BlockCount,
					declaredBlockCount,
				)
			}
			spec.RecurrentLayers = append([]bool(nil), recurrent[:spec.BlockCount]...)
		}
	}
	if architecture == "kimi-linear" {
		if spec.KVLoRARank, err = required[uint32](
			values, prefix+"attention.kv_lora_rank", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.QLoRARank, _ = optional[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32)
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SSMConvKernel, err = required[uint32](
			values, prefix+"ssm.conv_kernel", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.KDAHeadDim, err = required[uint32](
			values, prefix+"kda.head_dim", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SSMInnerSize = spec.HeadCount * spec.KDAHeadDim
		spec.SSMStateSize = spec.KDAHeadDim
		spec.SSMTimeStepRank = spec.HeadCount
		spec.SSMGroupCount = spec.HeadCount
	}
	if architecture == "rwkv6" || architecture == "rwkv6qwen2" {
		for key, destination := range map[string]*uint32{
			"wkv.head_size":        &spec.WKVHeadSize,
			"time_mix_extra_dim":   &spec.TimeMixExtraDim,
			"time_decay_extra_dim": &spec.TimeDecayExtraDim,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.RescaleEvery, _ = optional[uint32](values, prefix+"rescale_every_n_layers", gguf.ValueTypeUint32)
		spec.TokenShiftCount = 1
		if architecture == "rwkv6" {
			spec.TokenShiftCount = 2
		}
		if value, ok := optional[uint32](values, prefix+"token_shift_count", gguf.ValueTypeUint32); ok {
			spec.TokenShiftCount = value
		}
	}
	if architecture == "rwkv7" || architecture == "arwkv7" {
		for key, destination := range map[string]*uint32{
			"wkv.head_size":                          &spec.WKVHeadSize,
			"attention.decay_lora_rank":              &spec.DecayLoRARank,
			"attention.iclr_lora_rank":               &spec.ICLRLoRARank,
			"attention.value_residual_mix_lora_rank": &spec.ValueMixLoRARank,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.GateLoRARank, _ = optional[uint32](values, prefix+"attention.gate_lora_rank", gguf.ValueTypeUint32)
		spec.TokenShiftCount = 1
		if architecture == "rwkv7" {
			spec.TokenShiftCount = 2
		}
		if value, ok := optional[uint32](values, prefix+"token_shift_count", gguf.ValueTypeUint32); ok {
			spec.TokenShiftCount = value
		}
	}
	if architecture == "mamba" || architecture == "mamba2" || architecture == "jamba" || architecture == "granitehybrid" || architecture == "plamo2" || architecture == "nemotron_h" || architecture == "nemotron_h_moe" || architecture == "falcon-h1" {
		for key, destination := range map[string]*uint32{
			"ssm.conv_kernel":    &spec.SSMConvKernel,
			"ssm.inner_size":     &spec.SSMInnerSize,
			"ssm.state_size":     &spec.SSMStateSize,
			"ssm.time_step_rank": &spec.SSMTimeStepRank,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		if architecture == "mamba" || architecture == "jamba" {
			spec.SSMGroupCount = 1
			spec.SSMDtBCNorm, _ = optional[bool](values, prefix+"ssm.dt_b_c_rms", gguf.ValueTypeBool)
		} else if spec.SSMGroupCount, err = required[uint32](values, prefix+"ssm.group_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "falcon-h1" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "plamo2" {
		spec.AttentionScale = float32(1 / math.Sqrt(float64(spec.ValueLength)))
	}
	if architecture == "granitehybrid" {
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		if spec.ExpertCount > 0 {
			if spec.ExpertUsedCount, err = required[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsScale = 1
			spec.ExpertWeightsNorm = true
			spec.SharedExpertFF, _ = optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32)
		}
	}
	if architecture == "nemotron_h_moe" {
		for key, destination := range map[string]*uint32{
			"expert_count":                      &spec.ExpertCount,
			"expert_used_count":                 &spec.ExpertUsedCount,
			"expert_feed_forward_length":        &spec.ExpertFeedForward,
			"expert_shared_feed_forward_length": &spec.SharedExpertFF,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.ExpertWeightsScale = 1
		if value, ok := optional[float32](values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32); ok {
			spec.ExpertWeightsScale = value
		}
		spec.MoELatentSize, _ = optional[uint32](values, prefix+"moe_latent_size", gguf.ValueTypeUint32)
		spec.ExpertGatingFunc = 2
	}
	if isLlamaMoE || architecture == "llama4" || architecture == "gpt-oss" || architecture == "arctic" || architecture == "bailingmoe" || architecture == "bailingmoe2" || architecture == "cohere2moe" || architecture == "deepseek" || architecture == "deepseek2-ocr" || architecture == "dbrx" || architecture == "dots1" || architecture == "ernie4_5-moe" || architecture == "glm4moe" || architecture == "granitemoe" || (architecture == "granite" && spec.ExpertCount > 0) || architecture == "grovemoe" || architecture == "grok" || architecture == "hunyuan-moe" || architecture == "hy_v3" || architecture == "jamba" || architecture == "kimi-linear" || architecture == "llada-moe" || architecture == "mellum" || architecture == "mimo2" || architecture == "step35" || architecture == "minimax-m2" || architecture == "nomic-bert-moe" || architecture == "qwen3moe" || architecture == "qwen3vlmoe" || architecture == "qwen3next" || architecture == "qwen35moe" || architecture == "qwen2moe" || architecture == "olmoe" || architecture == "phimoe" || architecture == "exaone-moe" || architecture == "rnd1" || architecture == "afmoe" || architecture == "laguna" || architecture == "lfm2moe" || architecture == "smallthinker" {
		if spec.ExpertCount, err = required[uint32](
			values, prefix+"expert_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.ExpertUsedCount, err = required[uint32](
			values, prefix+"expert_used_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.ExpertUsedCount == 0 {
			return Spec{}, errors.New("expert used count is zero")
		}
		spec.ExpertFeedForward = spec.FeedForwardLength / spec.ExpertUsedCount
		if value, ok := optional[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); ok {
			spec.ExpertFeedForward = value
		}
		if architecture == "arctic" {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		if architecture == "nomic-bert-moe" {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		if architecture == "jamba" {
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsNorm = false
		}
		if architecture == "kimi-linear" {
			if spec.ExpertFeedForward, err = required[uint32](
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
			if spec.ExpertGatingFunc, err = required[uint32](
				values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertWeightsNorm = true
		}
		if architecture == "llama4" {
			if spec.ExpertFeedForward, err = required[uint32](
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			if spec.MoELayerStep, err = required[uint32](
				values, prefix+"interleave_moe_layer_step", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.SharedExpertFF = spec.ExpertFeedForward
			spec.ExpertGatingFunc = 2
		}
		spec.ExpertWeightsScale = 1
		if architecture == "gpt-oss" {
			spec.ExpertGatingFunc = 3
			spec.ExpertWeightsNorm = false
		}
		if value, ok := optional[float32](
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32,
		); ok {
			spec.ExpertWeightsScale = value
		}
		if architecture == "qwen3next" || architecture == "qwen35moe" {
			spec.SharedExpertFF = spec.FeedForwardLength
			if value, ok := optional[uint32](
				values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
			); ok {
				spec.SharedExpertFF = value
			}
		}
	}
	if isDeepSeek2Family(architecture) || architecture == "glm-dsa" {
		if architecture == "deepseek32" {
			if spec.ExpertCount, err = required[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		} else {
			spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertCount > 0 {
			if spec.ExpertUsedCount, err = required[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		}
		spec.ExpertWeightsScale = 1
		if value, ok := optional[float32](values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32); ok {
			spec.ExpertWeightsScale = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.ExpertGatingFunc = 1
		if architecture == "glm-dsa" {
			spec.ExpertGatingFunc = 2
		}
		if architecture == "deepseek32" {
			if spec.ExpertGatingFunc, err = required[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		} else if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		} else if (spec.BlockCount == 47 || spec.BlockCount == 48) && spec.VocabularySize == 154880 {
			spec.ExpertGatingFunc = 2
		}
	}
	if architecture == "mimo2" {
		spec.ExpertGatingFunc = 2
		spec.ExpertWeightsNorm = true
	}
	if architecture == "gemma4" {
		if count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); ok && count > 0 {
			spec.ExpertCount = count
			if spec.ExpertUsedCount, err = required[uint32](
				values, prefix+"expert_used_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			if spec.ExpertFeedForward, err = required[uint32](
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertWeightsScale = 1
			spec.ExpertWeightsNorm = true
		}
	}
	if architecture == "step35" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF, _ = optional[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		)
		spec.LeadingDenseBlocks, _ = optional[uint32](
			values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32,
		)
		spec.MoELayerStep = 1
		if value, ok := optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); ok {
			spec.MoELayerStep = value
		}
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
	}
	if architecture == "glm4moe" {
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertFeedForward > 0 && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("GLM4-MoE shared expert width overflows")
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
	}
	if architecture == "grovemoe" {
		spec.ExpertChunkFeedForward = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"expert_chunk_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.ExpertChunkFeedForward = value
		}
		if spec.ExpertGroupScale, err = required[float32](values, prefix+"expert_group_scale", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertsPerGroup, err = required[uint32](values, prefix+"experts_per_group", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "cohere2moe" {
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN != 1 || nextN >= spec.BlockCount {
				return Spec{}, errors.New("Cohere2-MoE NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
		}
		if spec.LeadingDenseBlocks, err = required[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		if value, ok := optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); ok {
			spec.SharedExpertCount = value
		}
		if spec.SharedExpertCount > 0 {
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
				spec.SharedExpertFF = value
			}
		}
	}
	if architecture == "hy_v3" {
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("HY-V3 NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "deepseek2-ocr" {
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("DeepSeek2-OCR shared expert width overflows")
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = 1
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "neo-bert" {
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "jina-bert-v3" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); ok && count > 0 {
			return Spec{}, errors.New("JinaBERT v3 expert layers are not supported")
		}
		if cadence, ok := optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); ok && cadence > 0 {
			return Spec{}, errors.New("JinaBERT v3 MoE cadence is not supported")
		}
	}
	if architecture == "nomic-bert" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if cadence, ok := optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); ok && cadence > 0 {
			return Spec{}, errors.New("NomicBERT MoE cadence requires nomic-bert-moe architecture")
		}
	}
	if architecture == "nomic-bert-moe" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if spec.MoELayerStep, err = required[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "ernie4_5-moe" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.MoELayerStep, err = required[uint32](values, prefix+"interleave_moe_layer_step", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		spec.SharedExpertFF, _ = optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32)
		spec.ExpertWeightsNorm = true
	}
	if architecture == "mellum" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm = true
	}
	if architecture == "hunyuan-moe" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.FeedForwardLength
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
		spec.ExpertWeightsNorm = true
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "grok" {
		if _, ok := optional[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); !ok {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.ExpertWeightsNorm = true
	}
	if architecture == "dbrx" {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
	}
	if architecture == "granitemoe" || (architecture == "granite" && spec.ExpertCount > 0) {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		spec.SharedExpertFF, _ = optional[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		)
	}
	if architecture == "smallthinker" {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "dots1" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm, _ = optional[bool](
			values, prefix+"expert_weights_norm", gguf.ValueTypeBool,
		)
		spec.LeadingDenseBlocks, _ = optional[uint32](
			values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32,
		)
	}
	if architecture == "minimax-m2" {
		expertWidth, widthErr := required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		)
		if widthErr != nil {
			return Spec{}, widthErr
		}
		if expertWidth != spec.FeedForwardLength {
			return Spec{}, errors.New("MiniMax-M2 expert width differs from packed tensor width")
		}
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "bailingmoe" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "deepseek" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "lfm2moe" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "bailingmoe2" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		sharedWidth := spec.ExpertFeedForward
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			sharedWidth = value
		}
		spec.SharedExpertFF = sharedWidth * spec.SharedExpertCount
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "qwen2moe" {
		if spec.ExpertFeedForward == 0 {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.SharedExpertCount = 1
		spec.SharedExpertFF = spec.FeedForwardLength
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
	}
	if isLlamaMoE || architecture == "olmoe" || architecture == "phimoe" {
		spec.ExpertFeedForward = spec.FeedForwardLength
	}
	if architecture == "afmoe" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if value, ok := optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32); ok {
			spec.LeadingDenseBlocks = value
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
		}
	}
	if architecture == "exaone-moe" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
		spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.ExpertGatingFunc, err = required[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
	}
	if architecture == "laguna" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.LeadingDenseBlocks, err = required[uint32](
			values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertFF, err = required[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = 2
		spec.ExpertGatingFunc, _ = optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32)
		if spec.ExpertGatingFunc == 0 {
			spec.ExpertGatingFunc = 2
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		sharedCount := uint32(1)
		if value, ok := optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); ok {
			sharedCount = value
		}
		if sharedCount != 1 {
			return Spec{}, errors.New("Laguna requires exactly one shared expert")
		}
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
			spec.RopeDimensionSWA = spec.KeyLength
			if value, ok := optional[uint32](values, prefix+"rope.dimension_count_swa", gguf.ValueTypeUint32); ok {
				spec.RopeDimensionSWA = value
			}
		}
	}
	if architecture == "lfm2" || architecture == "lfm2moe" {
		if spec.ShortConvCacheLength, err = required[uint32](
			values, prefix+"shortconv.l_cache", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
	}
	if isMLAArchitecture(architecture) {
		if architecture == "minicpm3" {
			if spec.QLoRARank, err = required[uint32](
				values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
		}
		if isDeepSeek2Family(architecture) || architecture == "glm-dsa" {
			lite := spec.BlockCount == 26 || spec.BlockCount == 27 || (spec.BlockCount == 48 && spec.VocabularySize == 128256)
			if !lite {
				if spec.QLoRARank, err = required[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32); err != nil {
					return Spec{}, err
				}
			}
		}
		if spec.KVLoRARank, err = required[uint32](
			values, prefix+"attention.kv_lora_rank", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if isDeepSeek2Family(architecture) || architecture == "glm-dsa" {
			spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
			if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
			if spec.ExpertFeedForward > 0 && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
				return Spec{}, errors.New("DeepSeek2 shared expert width overflows")
			}
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			if value, ok := optional[float32](values, prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32); ok {
				spec.RopeYaRNLogMultiplier = value / 0.1
			}
			if spec.RopeScalingType == "yarn" && spec.RopeScalingFactor > 0 {
				rawAttentionFactor := float32(1)
				if value, ok := optional[float32](values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32); ok {
					rawAttentionFactor = value
				}
				spec.YaRNAttentionFactor = rawAttentionFactor /
					(1 + 0.1*float32(math.Log(float64(spec.RopeScalingFactor))))
			}
			spec.AttentionTempScale, _ = optional[float32](values, prefix+"attention.temperature_scale", gguf.ValueTypeFloat32)
			spec.AttentionTempFloor, _ = optional[uint32](values, prefix+"attention.temperature_length", gguf.ValueTypeUint32)
		}
	}
	if isDSAArchitecture(architecture) {
		sections, hasSections, sectionsErr := optionalArray[int32](values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32)
		if sectionsErr != nil {
			return Spec{}, sectionsErr
		}
		if hasSections && len(sections) != len(spec.RopeSections) {
			return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"rope.dimension_sections", len(sections), len(spec.RopeSections))
		}
		if hasSections {
			copy(spec.RopeSections[:], sections)
		}
		for key, destination := range map[string]*uint32{
			"attention.indexer.head_count": &spec.IndexerHeadCount,
			"attention.indexer.key_length": &spec.IndexerKeyLength,
			"attention.indexer.top_k":      &spec.IndexerTopK,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.IndexerFullLayers = make([]bool, spec.BlockCount)
		if architecture == "deepseek32" || spec.ContextLength < 1048576 {
			for index := range spec.IndexerFullLayers {
				spec.IndexerFullLayers[index] = true
			}
		} else {
			for index := range spec.IndexerFullLayers {
				spec.IndexerFullLayers[index] = index < 2 || (index >= 2 && (index-2)%4 == 0)
			}
		}
		indexerTypesKey := prefix + "attention.indexer.types"
		if value, present := values[indexerTypesKey]; present && value.Type == gguf.ValueTypeUint32 {
			typeValue, valid := value.Data.(uint32)
			if !valid || typeValue > 1 {
				return Spec{}, fmt.Errorf("metadata %q must be 0 or 1", indexerTypesKey)
			}
			for index := range spec.IndexerFullLayers {
				spec.IndexerFullLayers[index] = typeValue == 1
			}
		} else if types, ok, arrayErr := optionalArray[uint32](values, indexerTypesKey, gguf.ValueTypeUint32); arrayErr != nil {
			return Spec{}, arrayErr
		} else if ok {
			if len(types) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", indexerTypesKey, len(types), spec.BlockCount)
			}
			for index, typeValue := range types {
				if typeValue > 1 {
					return Spec{}, fmt.Errorf("metadata %q value %d is not 0 or 1", indexerTypesKey, typeValue)
				}
				spec.IndexerFullLayers[index] = typeValue == 1
			}
		}
	}
	if architecture == "deepseek4" {
		if spec.QLoRARank, err = required[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		for key, destination := range map[string]*uint32{
			"attention.sliding_window":             &spec.SlidingWindow,
			"attention.indexer.head_count":         &spec.IndexerHeadCount,
			"attention.indexer.key_length":         &spec.IndexerKeyLength,
			"attention.indexer.top_k":              &spec.IndexerTopK,
			"attention.output_group_count":         &spec.AttentionOutputGroups,
			"attention.output_lora_rank":           &spec.AttentionOutputRank,
			"hyper_connection.count":               &spec.HyperConnectionCount,
			"hyper_connection.sinkhorn_iterations": &spec.HyperSinkhornIters,
			"hash_layer_count":                     &spec.HashLayerCount,
			"expert_count":                         &spec.ExpertCount,
			"expert_used_count":                    &spec.ExpertUsedCount,
			"expert_feed_forward_length":           &spec.ExpertFeedForward,
			"expert_shared_count":                  &spec.SharedExpertCount,
			"expert_gating_func":                   &spec.ExpertGatingFunc,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		for key, destination := range map[string]*float32{
			"attention.compress_rope_freq_base": &spec.CompressRopeBase,
			"hyper_connection.epsilon":          &spec.HyperConnectionEps,
			"expert_weights_scale":              &spec.ExpertWeightsScale,
		} {
			*destination, err = required[float32](values, prefix+key, gguf.ValueTypeFloat32)
			if err != nil {
				return Spec{}, err
			}
		}
		if spec.ExpertWeightsNorm, err = required[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); err != nil {
			return Spec{}, err
		}
		if spec.CompressRatios, err = requiredArray[uint32](values, prefix+"attention.compress_ratios", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if len(spec.CompressRatios) < int(spec.BlockCount) {
			return Spec{}, errors.New("DeepSeek 4 compression schedule is shorter than its block count")
		}
		spec.CompressRatios = spec.CompressRatios[:spec.BlockCount]
		if spec.LayerSwiGLUClamp, err = optionalLayerFloat32(values, prefix+"swiglu_clamp_exp", spec.BlockCount); err != nil {
			return Spec{}, err
		}
		if len(spec.LayerSwiGLUClamp) == 0 {
			return Spec{}, errors.New("DeepSeek 4 expert SwiGLU clamp is missing")
		}
		if spec.LayerSharedSwiGLUClamp, err = optionalLayerFloat32(values, prefix+"swiglu_clamp_shexp", spec.BlockCount); err != nil {
			return Spec{}, err
		}
		if len(spec.LayerSharedSwiGLUClamp) == 0 {
			spec.LayerSharedSwiGLUClamp = append([]float32(nil), spec.LayerSwiGLUClamp...)
		}
		if spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("DeepSeek 4 shared expert width overflows")
		}
		spec.SharedExpertFF = spec.SharedExpertCount * spec.ExpertFeedForward
	}
	if architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma3n" || architecture == "gemma4" || architecture == "gemma4-assistant" ||
		architecture == "olmo2" || architecture == "cohere2" || architecture == "cohere2moe" {
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if architecture == "gemma3" || architecture == "gemma3n" || architecture == "gemma4" || architecture == "gemma4-assistant" {
			spec.RopeFrequencySWA = 10000
		}
		if value, ok := optional[float32](
			values,
			prefix+"rope.freq_base_swa",
			gguf.ValueTypeFloat32,
		); ok {
			spec.RopeFrequencySWA = value
		}
		if architecture == "gemma2" {
			spec.SlidingWindow = 4096
		}
		if value, ok := optional[uint32](
			values,
			prefix+"attention.sliding_window",
			gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if architecture == "gemma2" {
				spec.SlidingPattern = 2
			}
			if architecture == "gemma3" {
				spec.SlidingPattern = 6
			}
			if architecture == "gemma3n" {
				spec.SlidingPattern = 5
			}
			if architecture == "gemma4" || architecture == "gemma4-assistant" {
				if spec.SlidingLayers, err = requiredLayerBoolCompatible(
					values, prefix+"attention.sliding_window_pattern", spec.BlockCount,
				); err != nil {
					return Spec{}, err
				}
			}
			if value, ok := optional[uint32](
				values,
				prefix+"attention.sliding_window_pattern",
				gguf.ValueTypeUint32,
			); ok {
				spec.SlidingPattern = value
			}
			if architecture == "cohere2moe" {
				if layers, ok, layersErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); layersErr != nil {
					return Spec{}, layersErr
				} else if ok {
					if len(layers) != int(spec.BlockCount) {
						return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
					}
					spec.SlidingLayers = append([]bool(nil), layers...)
				}
			}
			if architecture == "cohere2" {
				spec.NoRopeLayerStep = spec.SlidingPattern
			}
		}
	}
	if architecture == "llama4" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		window, hasWindow := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32)
		if hasWindow && window == 0 {
			spec.NoRopeLayerStep = 0
		} else {
			spec.SlidingWindow = 8192
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
			spec.AttentionTempFloor = 8192
			spec.AttentionTempScale = 0.1
			spec.AttentionTempOffset = 1
		}
	}
	if architecture == "gpt-oss" {
		spec.RopeDimensionCount = spec.KeyLength
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = 2
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = value
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
	}
	if architecture == "gemma4" {
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionSWA, err = required[uint32](
			values, prefix+"rope.dimension_count_swa", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.EmbeddingPerLayer, err = required[uint32](
			values, prefix+"embedding_length_per_layer_input", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedKVLayers, _ = optional[uint32](
			values, prefix+"attention.shared_kv_layers", gguf.ValueTypeUint32,
		)
		spec.AttentionScale = 1
	}
	if architecture == "gemma4-assistant" {
		spec.RopeDimensionCount = spec.KeyLength
		spec.RopeDimensionSWA = spec.KeyLengthSWA
		spec.AttentionScale = 1
	}
	if spec.RopeScalingType == "longrope" &&
		(architecture == "llama" || architecture == "llama-embed" ||
			architecture == "minicpm" || architecture == "mistral3") {
		spec.RopeDimensionCount = spec.KeyLength
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); ok {
			spec.OriginalContextLength = value
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeAttentionFactor = value
		}
	}
	if spec.RopeScalingType == "yarn" &&
		(architecture == "llama" || architecture == "llama-embed" || architecture == "minicpm") {
		rawAttentionFactor := float32(1)
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			rawAttentionFactor = value
		}
		spec.YaRNAttentionFactor = rawAttentionFactor
	}
	spec.VocabularySize, _ = optional[uint32](values, prefix+"vocab_size", gguf.ValueTypeUint32)
	if tokens, ok := values["tokenizer.ggml.tokens"]; ok && spec.VocabularySize == 0 {
		if tokens.Type != gguf.ValueTypeArray || tokens.ArrayType != gguf.ValueTypeString {
			return Spec{}, errors.New(`metadata "tokenizer.ggml.tokens" must be a string array`)
		}
		if tokens.Count() > int(^uint32(0)) {
			return Spec{}, errors.New("tokenizer vocabulary exceeds uint32")
		}
		spec.VocabularySize = uint32(tokens.Count())
	}
	if architecture == "bert" || architecture == "jina-bert-v2" || architecture == "jina-bert-v3" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" {
		if spec.TokenTypeCount, err = required[uint32](values, "tokenizer.ggml.token_type_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if err := spec.validate(); err != nil {
		return Spec{}, err
	}
	return spec, nil
}

func (s Spec) IsRecurrentLayer(block uint32) bool {
	if block >= s.BlockCount {
		return false
	}
	if len(s.RecurrentLayers) == int(s.BlockCount) {
		return s.RecurrentLayers[block]
	}
	return (s.Architecture == "qwen3next" || s.Architecture == "qwen35" || s.Architecture == "qwen35moe") &&
		s.FullAttentionInterval > 0 &&
		(block+1)%s.FullAttentionInterval != 0
}

func (s Spec) IsInterleavedMoELayer(block uint32) bool {
	if s.Architecture == "nomic-bert-moe" {
		return block < s.BlockCount && s.MoELayerStep > 1 && block%s.MoELayerStep == 1
	}
	return (s.Architecture == "ernie4_5-moe" && block >= s.LeadingDenseBlocks || s.Architecture == "llama4") &&
		block < s.BlockCount && s.MoELayerStep > 0 && (block+1)%s.MoELayerStep == 0
}

func (s Spec) IsSlidingLayer(block uint32) bool {
	if s.Architecture == "lfm2" || s.Architecture == "lfm2moe" {
		return block < s.BlockCount && s.SlidingWindow > 0 && !s.IsRecurrentLayer(block)
	}
	if s.Architecture == "laguna" || s.Architecture == "modern-bert" || s.Architecture == "smallthinker" {
		return block < s.BlockCount &&
			s.SlidingWindow > 0 &&
			s.SlidingPattern > 0 &&
			block%s.SlidingPattern != 0
	}
	if block < uint32(len(s.SlidingLayers)) {
		return s.SlidingLayers[block]
	}
	return usesSlidingAttention(s.Architecture) &&
		block < s.BlockCount &&
		s.SlidingWindow > 0 &&
		s.SlidingPattern > 0 &&
		block%s.SlidingPattern < s.SlidingPattern-1
}

// LayerHeadCount: returns query-head count selected for layer; Laguna
// stores this metadata as either scalar or one value per layer
func (s Spec) LayerHeadCount(block uint32) uint32 {
	if block < uint32(len(s.LayerHeadCounts)) {
		return s.LayerHeadCounts[block]
	}
	return s.HeadCount
}

// LayerKVHeadCount: returns key/value-head count selected for layer
func (s Spec) LayerKVHeadCount(block uint32) uint32 {
	if block < uint32(len(s.LayerKVHeadCounts)) {
		return s.LayerKVHeadCounts[block]
	}
	return s.HeadCountKV
}

func (s Spec) LayerFeedForwardLength(block uint32) uint32 {
	if block < uint32(len(s.LayerFeedForward)) {
		return s.LayerFeedForward[block]
	}
	return s.FeedForwardLength
}

func (s Spec) LayerKeyLength(block uint32) uint32 {
	if (s.Architecture == "gemma4" || s.Architecture == "gemma4-assistant") && s.IsSlidingLayer(block) {
		return s.KeyLengthSWA
	}
	return s.KeyLength
}

func (s Spec) LayerValueLength(block uint32) uint32 {
	if (s.Architecture == "gemma4" || s.Architecture == "gemma4-assistant") && s.IsSlidingLayer(block) {
		return s.ValueLengthSWA
	}
	return s.ValueLength
}

func (s Spec) LayerHasKV(block uint32) bool {
	return (s.Architecture != "gemma4" && s.Architecture != "gemma3n") || block < s.BlockCount-s.SharedKVLayers
}

func (s Spec) LayerSharedKVSource(block uint32) uint32 {
	start := s.BlockCount - s.SharedKVLayers
	if s.IsSlidingLayer(block) {
		return start - 2
	}
	return start - 1
}

func (s Spec) LayerRopeDimensionCount(block uint32) uint32 {
	if (s.Architecture == "gemma4" || s.Architecture == "gemma4-assistant") && s.IsSlidingLayer(block) {
		return s.RopeDimensionSWA
	}
	if s.Architecture == "step35" && !s.IsSlidingLayer(block) {
		return s.RopeDimensionCount / 2
	}
	return s.RopeDimensionCount
}

func (s Spec) LayerExpertSwiGLUClamp(block uint32) float32 {
	if block < uint32(len(s.LayerSwiGLUClamp)) {
		return s.LayerSwiGLUClamp[block]
	}
	return 0
}

func (s Spec) LayerSharedSwiGLUClampLimit(block uint32) float32 {
	if block < uint32(len(s.LayerSharedSwiGLUClamp)) {
		return s.LayerSharedSwiGLUClamp[block]
	}
	return 0
}

func (s Spec) UsesRoPE(block uint32) bool {
	if s.Architecture == "cohere2moe" {
		return !s.RopeDisabled && block < s.BlockCount &&
			(block < s.LeadingDenseBlocks || s.IsSlidingLayer(block))
	}
	if s.Architecture == "smallthinker" {
		return !s.RopeDisabled && block < s.BlockCount &&
			(s.SlidingWindow == 0 || s.NoRopeLayerStep == 0 || block%s.NoRopeLayerStep != 0)
	}
	if s.Architecture == "exaone-moe" {
		return s.IsSlidingLayer(block)
	}
	blockCount := s.BlockCount
	if s.Architecture == "step35" || s.Architecture == "hy_v3" {
		blockCount += s.NextNPredictLayers
	}
	return !s.RopeDisabled &&
		(blockCount == 0 || block < blockCount) &&
		(s.NoRopeLayerStep == 0 || (block+1)%s.NoRopeLayerStep != 0)
}

func (s Spec) InputEmbeddingScale() float32 {
	if s.EmbeddingScale > 0 {
		return s.EmbeddingScale
	}
	if s.Architecture == "afmoe" || isGemmaArchitecture(s.Architecture) {
		return float32(math.Sqrt(float64(s.EmbeddingLength)))
	}
	return 1
}

func (s Spec) OutputLogitMultiplier() float32 {
	if s.LogitScale > 0 {
		if s.Architecture == "cohere2" || s.Architecture == "cohere2moe" || s.Architecture == "command-r" || s.Architecture == "grok" || s.Architecture == "talkie" {
			return s.LogitScale
		}
		return 1 / s.LogitScale
	}
	return 1
}

func (s Spec) IsEncoderOnly() bool {
	return s.Architecture == "bert" || s.Architecture == "eurobert" || s.Architecture == "jina-bert-v2" || s.Architecture == "jina-bert-v3" ||
		s.Architecture == "gemma-embedding" ||
		s.Architecture == "llama-embed" ||
		s.Architecture == "modern-bert" ||
		s.Architecture == "neo-bert" || s.Architecture == "nomic-bert" ||
		s.Architecture == "nomic-bert-moe" ||
		s.Architecture == "t5encoder" ||
		s.Architecture == "wavtokenizer-dec"
}

func (s Spec) UsesLayerNorm() bool {
	return s.Architecture == "bert" ||
		s.Architecture == "dbrx" ||
		s.Architecture == "falcon" ||
		s.Architecture == "jais" ||
		s.Architecture == "jina-bert-v2" ||
		s.Architecture == "jina-bert-v3" ||
		s.Architecture == "nemotron" ||
		s.Architecture == "nomic-bert" ||
		s.Architecture == "nomic-bert-moe" ||
		s.Architecture == "jais2" ||
		s.Architecture == "orion" ||
		s.Architecture == "rwkv6" ||
		s.Architecture == "rwkv7" ||
		s.Architecture == "stablelm" ||
		s.Architecture == "wavtokenizer-dec" ||
		s.Architecture == "mpt" ||
		usesSequentialGELU(s.Architecture)
}

func (s Spec) RequiresLayerNormBias() bool {
	return s.Architecture == "phimoe" ||
		(s.UsesLayerNorm() && s.Architecture != "dbrx" && s.Architecture != "mpt")
}

func (s Spec) UsesUnweightedLayerNorm() bool {
	return s.Architecture == "olmo"
}

func (s Spec) UsesUnweightedRMSNorm() bool {
	return s.Architecture == "talkie"
}

func (s Spec) UsesWeightOnlyLayerNorm() bool {
	return s.Architecture == "cohere2" || s.Architecture == "command-r" || s.Architecture == "modern-bert" ||
		(s.Architecture == "cohere2moe" && s.LayerNormEpsilon > 0)
}

func (s Spec) validate() error {
	if (s.Architecture == "mpt" || s.Architecture == "olmo") &&
		(s.AttentionClamp < 0 || math.IsNaN(float64(s.AttentionClamp)) ||
			math.IsInf(float64(s.AttentionClamp), 0)) {
		return fmt.Errorf("%s attention QKV clamp is invalid", s.Architecture)
	}
	attentionFree := s.Architecture == "mamba" || s.Architecture == "mamba2"
	switch {
	case s.BlockCount == 0:
		return errors.New("model block count is zero")
	case s.ContextLength == 0:
		return errors.New("model context length is zero")
	case s.EmbeddingLength == 0:
		return errors.New("model embedding length is zero")
	case !attentionFree && s.FeedForwardLength == 0:
		return errors.New("model feed-forward length is zero")
	case !attentionFree && s.HeadCount == 0:
		return errors.New("model attention head count is zero")
	case !attentionFree && s.HeadCountKV == 0:
		return errors.New("model KV head count is zero")
	case !attentionFree && s.HeadCount%s.HeadCountKV != 0:
		return errors.New("attention head count is not divisible by KV head count")
	case !attentionFree && (s.KeyLength == 0 || s.ValueLength == 0):
		return errors.New("model attention key/value length is zero")
	case s.Architecture != "t5encoder" && !s.RopeDisabled && s.RopeFrequencyBase <= 0:
		return errors.New("model RoPE frequency base must be positive")
	case (s.UsesLayerNorm() || s.UsesWeightOnlyLayerNorm() || s.UsesUnweightedLayerNorm()) && s.LayerNormEpsilon <= 0:
		return errors.New("model LayerNorm epsilon must be positive")
	case !s.UsesLayerNorm() && !s.UsesWeightOnlyLayerNorm() && !s.UsesUnweightedLayerNorm() && s.RMSNormEpsilon <= 0:
		return errors.New("model RMSNorm epsilon must be positive")
	}
	if (s.Architecture == "t5" || s.Architecture == "t5encoder") && s.RelativeBuckets == 0 {
		return errors.New("T5 encoder relative attention bucket count is zero")
	}
	if s.Architecture == "t5" && s.DecoderBlockCount == 0 {
		return errors.New("T5 decoder block count is zero")
	}
	if s.Architecture == "wavtokenizer-dec" {
		if s.OutputEmbeddingLength == 0 || s.PosNetBlockCount != 6 || s.ConvNextBlockCount == 0 ||
			s.PosNetEmbeddingLength == 0 || s.PosNetEmbeddingLength != s.ConvNextEmbeddingLength ||
			s.GroupNormGroups == 0 || s.GroupNormEpsilon <= 0 ||
			s.PosNetEmbeddingLength%s.GroupNormGroups != 0 {
			return errors.New("WavTokenizer metadata is invalid")
		}
	}
	if s.Architecture == "dflash" {
		if len(s.TargetLayers) == 0 || s.DFlashBlockSize < 2 {
			return errors.New("DFlash target-layer metadata is invalid")
		}
		for _, layer := range s.TargetLayers {
			if layer < 0 {
				return errors.New("DFlash target layer is negative")
			}
		}
	}
	if s.Architecture == "eagle3" {
		if s.BlockCount != 1 || len(s.TargetLayers) != 3 || s.TargetHiddenSize == 0 {
			return errors.New("Eagle3 target-layer metadata is invalid")
		}
		for _, layer := range s.TargetLayers {
			if layer < 0 {
				return errors.New("Eagle3 target layer is negative")
			}
		}
	}
	if s.Architecture == "mamba" {
		switch {
		case s.FeedForwardLength != 0 || s.HeadCount != 0 || s.HeadCountKV != 0 || s.KeyLength != 0 || s.ValueLength != 0:
			return errors.New("Mamba attention metadata must be zero")
		case s.SSMConvKernel < 2 || s.SSMInnerSize != 2*s.EmbeddingLength || s.SSMStateSize == 0 || s.SSMTimeStepRank == 0:
			return errors.New("Mamba SSM metadata is invalid")
		}
	}
	if s.Architecture == "mamba2" {
		switch {
		case s.FeedForwardLength != 0 || s.HeadCount != 0 || s.HeadCountKV != 0 || s.KeyLength != 0 || s.ValueLength != 0:
			return errors.New("Mamba2 attention metadata must be zero")
		case s.SSMConvKernel < 2 || s.SSMInnerSize == 0 || s.SSMStateSize == 0 ||
			s.SSMTimeStepRank == 0 || s.SSMGroupCount == 0 ||
			s.SSMInnerSize%s.SSMTimeStepRank != 0 || s.SSMInnerSize%s.SSMGroupCount != 0 ||
			s.SSMTimeStepRank%s.SSMGroupCount != 0:
			return errors.New("Mamba2 SSM metadata is invalid")
		}
	}
	if s.Architecture == "falcon-h1" {
		switch {
		case s.SSMConvKernel < 2 || s.SSMInnerSize == 0 || s.SSMStateSize == 0 ||
			s.SSMTimeStepRank == 0 || s.SSMGroupCount == 0 ||
			s.SSMInnerSize%s.SSMTimeStepRank != 0 || s.SSMInnerSize%s.SSMGroupCount != 0 ||
			s.SSMTimeStepRank%s.SSMGroupCount != 0:
			return errors.New("Falcon-H1 SSM metadata is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("Falcon-H1 rotary/head dimensions are invalid")
		}
	}
	if s.Architecture == "rwkv6" || s.Architecture == "rwkv6qwen2" {
		switch {
		case s.WKVHeadSize == 0 || s.EmbeddingLength%s.WKVHeadSize != 0 || s.HeadCount != s.EmbeddingLength/s.WKVHeadSize:
			return fmt.Errorf("%s WKV head metadata is invalid", s.Architecture)
		case s.TimeMixExtraDim == 0 || s.TimeDecayExtraDim == 0 ||
			(s.Architecture == "rwkv6" && s.TokenShiftCount != 2) ||
			(s.Architecture == "rwkv6qwen2" && s.TokenShiftCount != 1):
			return fmt.Errorf("%s time-mix metadata is invalid", s.Architecture)
		case s.KeyLength != s.WKVHeadSize || s.ValueLength != s.WKVHeadSize:
			return fmt.Errorf("%s head dimensions are invalid", s.Architecture)
		case s.Architecture == "rwkv6" && s.HeadCountKV != s.HeadCount:
			return errors.New("rwkv6 KV head count must equal query head count")
		}
	}
	if s.Architecture == "rwkv7" || s.Architecture == "arwkv7" {
		switch {
		case s.WKVHeadSize == 0 || s.EmbeddingLength%s.WKVHeadSize != 0 || s.HeadCount != s.EmbeddingLength/s.WKVHeadSize:
			return fmt.Errorf("%s WKV head metadata is invalid", s.Architecture)
		case s.HeadCountKV != s.HeadCount || s.KeyLength != s.WKVHeadSize || s.ValueLength != s.WKVHeadSize:
			return fmt.Errorf("%s head dimensions are invalid", s.Architecture)
		case s.DecayLoRARank == 0 || s.ICLRLoRARank == 0 || s.ValueMixLoRARank == 0:
			return fmt.Errorf("%s time-mix LoRA metadata is invalid", s.Architecture)
		case s.Architecture == "rwkv7" && s.GateLoRARank == 0:
			return errors.New("rwkv7 gate LoRA metadata is invalid")
		case (s.Architecture == "rwkv7" && s.TokenShiftCount != 2) ||
			(s.Architecture == "arwkv7" && s.TokenShiftCount != 1):
			return fmt.Errorf("%s token-shift metadata is invalid", s.Architecture)
		}
	}
	if s.Architecture == "jamba" {
		switch {
		case s.SSMConvKernel < 2 || s.SSMInnerSize != 2*s.EmbeddingLength ||
			s.SSMStateSize == 0 || s.SSMTimeStepRank == 0:
			return errors.New("Jamba SSM metadata is invalid")
		case len(s.RecurrentLayers) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("Jamba layer schedule is invalid")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0:
			return errors.New("Jamba expert metadata is invalid")
		}
	}
	if s.Architecture == "granitehybrid" {
		switch {
		case s.SSMConvKernel < 2 || s.SSMInnerSize != 2*s.EmbeddingLength ||
			s.SSMStateSize == 0 || s.SSMTimeStepRank == 0 || s.SSMGroupCount == 0 ||
			s.SSMInnerSize%s.SSMTimeStepRank != 0 || s.SSMInnerSize%s.SSMGroupCount != 0 ||
			s.SSMTimeStepRank%s.SSMGroupCount != 0:
			return errors.New("Granite Hybrid SSM metadata is invalid")
		case len(s.RecurrentLayers) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("Granite Hybrid layer schedule is invalid")
		case s.ExpertCount > 0 && (s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0):
			return errors.New("Granite Hybrid expert metadata is invalid")
		case s.ExpertCount == 0 && (s.ExpertUsedCount != 0 || s.ExpertFeedForward != 0 || s.SharedExpertFF != 0):
			return errors.New("dense Granite Hybrid expert metadata is inconsistent")
		}
	}
	if s.Architecture == "plamo2" {
		switch {
		case s.SSMConvKernel < 2 || s.SSMInnerSize == 0 || s.SSMStateSize == 0 ||
			s.SSMTimeStepRank == 0 || s.SSMGroupCount != 0 ||
			s.SSMInnerSize%s.SSMTimeStepRank != 0:
			return errors.New("PLaMo2 SSM metadata is invalid")
		case len(s.RecurrentLayers) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("PLaMo2 layer schedule is invalid")
		}
	}
	if s.Architecture == "nemotron_h" || s.Architecture == "nemotron_h_moe" {
		switch {
		case s.SSMConvKernel < 2 || s.SSMInnerSize == 0 || s.SSMStateSize == 0 ||
			s.SSMTimeStepRank == 0 || s.SSMGroupCount == 0 ||
			s.SSMInnerSize%s.SSMTimeStepRank != 0 || s.SSMInnerSize%s.SSMGroupCount != 0 ||
			s.SSMTimeStepRank%s.SSMGroupCount != 0:
			return errors.New("Nemotron-H SSM metadata is invalid")
		case len(s.LayerHeadCounts) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) || len(s.RecurrentLayers) != int(s.BlockCount):
			return errors.New("Nemotron-H layer schedule is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			heads, kvHeads, ff := s.LayerHeadCount(block), s.LayerKVHeadCount(block), s.LayerFeedForwardLength(block)
			if ff > 0 {
				continue
			}
			if s.IsRecurrentLayer(block) {
				if heads != 0 || kvHeads != 0 {
					return errors.New("Nemotron-H recurrent layer schedule is invalid")
				}
				continue
			}
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 {
				return errors.New("Nemotron-H attention layer schedule is invalid")
			}
		}
		if s.Architecture == "nemotron_h_moe" {
			if s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
				s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 ||
				s.ExpertWeightsScale <= 0 {
				return errors.New("Nemotron-H MoE metadata is invalid")
			}
		} else if s.ExpertCount != 0 || s.ExpertUsedCount != 0 || s.ExpertFeedForward != 0 ||
			s.SharedExpertFF != 0 || s.MoELatentSize != 0 {
			return errors.New("dense Nemotron-H expert metadata is inconsistent")
		}
	}
	if s.Architecture == "bert" &&
		(s.TokenTypeCount == 0 || s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength) {
		return errors.New("BERT metadata is invalid")
	}
	if s.Architecture == "jina-bert-v2" &&
		(s.TokenTypeCount == 0 || s.KeyLength != s.ValueLength || s.MaxALiBiBias != 8) {
		return errors.New("JinaBERT v2 metadata is invalid")
	}
	if s.Architecture == "jina-bert-v3" &&
		(s.TokenTypeCount == 0 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength) {
		return errors.New("JinaBERT v3 metadata is invalid")
	}
	if s.Architecture == "neo-bert" &&
		(s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("NeoBERT attention metadata is invalid")
	}
	if s.Architecture == "nomic-bert" &&
		(s.TokenTypeCount == 0 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength) {
		return errors.New("NomicBERT metadata is invalid")
	}
	if s.Architecture == "nomic-bert-moe" &&
		(s.TokenTypeCount == 0 || s.MoELayerStep < 2 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength) {
		return errors.New("NomicBERT-MoE metadata is invalid")
	}
	if s.Architecture == "qwen3next" || s.Architecture == "qwen35" || s.Architecture == "qwen35moe" {
		switch {
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Qwen hybrid rotary dimension count is invalid")
		case s.SSMConvKernel == 0:
			return errors.New("Qwen hybrid SSM convolution kernel is zero")
		case s.SSMInnerSize == 0:
			return errors.New("Qwen hybrid SSM inner size is zero")
		case s.SSMStateSize == 0:
			return errors.New("Qwen hybrid SSM state size is zero")
		case s.SSMTimeStepRank == 0 || s.SSMInnerSize%s.SSMTimeStepRank != 0:
			return errors.New("Qwen hybrid SSM inner size is not divisible by time-step rank")
		case s.SSMInnerSize/s.SSMTimeStepRank != s.SSMStateSize:
			return errors.New("Qwen hybrid SSM value-head width differs from state size")
		case s.SSMGroupCount == 0 || s.SSMTimeStepRank%s.SSMGroupCount != 0:
			return errors.New("Qwen hybrid SSM value heads are not divisible by key groups")
		case s.FullAttentionInterval == 0:
			return errors.New("Qwen hybrid full-attention interval is zero")
		}
		if s.Architecture != "qwen3next" {
			var sectionPairs int64
			for _, section := range s.RopeSections {
				if section < 0 {
					return errors.New("Qwen3.5 RoPE section count is negative")
				}
				sectionPairs += int64(section)
			}
			if sectionPairs == 0 || sectionPairs > int64(s.RopeDimensionCount/2) {
				return errors.New("Qwen3.5 RoPE sections exceed rotary pair count")
			}
		}
	}
	if (s.Architecture == "qwen3next" || s.Architecture == "qwen35moe") &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 ||
			s.ExpertWeightsScale == 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Qwen3.5-MoE expert metadata is invalid")
	}
	if (s.Architecture == "qwen3moe" || s.Architecture == "qwen3vlmoe" || s.Architecture == "rnd1") &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale == 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Qwen3-MoE expert metadata is invalid")
	}
	if s.Architecture == "grovemoe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertChunkFeedForward == 0 ||
			s.ExpertsPerGroup == 0 || s.ExpertCount%s.ExpertsPerGroup != 0 ||
			s.ExpertWeightsScale == 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0) || math.IsNaN(float64(s.ExpertGroupScale)) ||
			math.IsInf(float64(s.ExpertGroupScale), 0)) {
		return errors.New("GroveMoE expert metadata is invalid")
	}
	if s.Architecture == "mimo2" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale == 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("MiMo2 expert metadata is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("MiMo2 rotary dimension count is invalid")
		case s.SlidingWindow == 0 || (len(s.SlidingLayers) == 0 && s.SlidingPattern == 0) || s.RopeFrequencySWA <= 0:
			return errors.New("MiMo2 sliding-attention metadata is invalid")
		case math.IsNaN(float64(s.AttentionValueScale)) || math.IsInf(float64(s.AttentionValueScale), 0):
			return errors.New("MiMo2 attention value scale is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			kvHeads := s.LayerKVHeadCount(block)
			if kvHeads == 0 || s.HeadCount%kvHeads != 0 {
				return errors.New("MiMo2 layer KV head count is invalid")
			}
		}
	}
	if s.Architecture == "step35" {
		layerCount := int(s.BlockCount + s.NextNPredictLayers)
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Step3.5 expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("Step3.5 expert routing function is unsupported")
		case len(s.LayerHeadCounts) != layerCount || len(s.LayerKVHeadCounts) != layerCount:
			return errors.New("Step3.5 layer head metadata is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%4 != 0:
			return errors.New("Step3.5 rotary dimension count is invalid")
		case s.SlidingWindow == 0 || len(s.SlidingLayers) != layerCount || s.RopeFrequencySWA <= 0:
			return errors.New("Step3.5 sliding-attention metadata is invalid")
		case len(s.LayerSwiGLUClamp) != 0 && len(s.LayerSwiGLUClamp) != layerCount:
			return errors.New("Step3.5 expert clamp metadata is invalid")
		case len(s.LayerSharedSwiGLUClamp) != 0 && len(s.LayerSharedSwiGLUClamp) != layerCount:
			return errors.New("Step3.5 shared-expert clamp metadata is invalid")
		}
		for block := uint32(0); block < uint32(layerCount); block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 {
				return errors.New("Step3.5 layer head count is invalid")
			}
			for _, limit := range []float32{
				s.LayerExpertSwiGLUClamp(block), s.LayerSharedSwiGLUClampLimit(block),
			} {
				if limit < 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
					return errors.New("Step3.5 SwiGLU clamp is invalid")
				}
			}
		}
	}
	if s.Architecture == "llada-moe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("LLaDA-MoE expert metadata is invalid")
	}
	if s.Architecture == "qwen2moe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 ||
			s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Qwen2-MoE expert metadata is invalid")
	}
	if s.Architecture == "arctic" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Arctic expert metadata is invalid")
	}
	if s.Architecture == "bailingmoe" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("BailingMoE expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("BailingMoE shared expert width overflows")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("BailingMoE expert weight scale is invalid")
		}
	}
	if s.Architecture == "deepseek" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DeepSeek expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DeepSeek shared expert width overflows")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek expert weight scale is invalid")
		}
	}
	if (s.Architecture == "granitemoe" || s.Architecture == "granite" && s.ExpertCount > 0) &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("GraniteMoE expert metadata is invalid")
	}
	if s.Architecture == "dbrx" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			s.AttentionClamp < 0 || math.IsNaN(float64(s.AttentionClamp)) ||
			math.IsInf(float64(s.AttentionClamp), 0) || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("DBRX expert or attention metadata is invalid")
	}
	if s.Architecture == "grok" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("Grok expert metadata is invalid")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Grok expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Grok rotary/head dimensions are invalid")
		case s.AttentionScale <= 0 || s.AttentionSoftcap <= 0:
			return errors.New("Grok attention scaling metadata is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 ||
				s.YaRNExtFactor < 0 || s.YaRNAttentionFactor <= 0 ||
				s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
				math.IsNaN(float64(s.RopeScalingFactor)) || math.IsInf(float64(s.RopeScalingFactor), 0) ||
				math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
				math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
				math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
				math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0)):
			return errors.New("Grok YaRN metadata is invalid")
		}
	}
	if s.Architecture == "mellum" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("Mellum expert metadata is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("Mellum rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.RopeFrequencySWA <= 0 ||
			(len(s.SlidingLayers) == 0 && s.SlidingPattern < 2)):
			return errors.New("Mellum sliding-attention metadata is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 || s.YaRNExtFactor < 0 ||
				s.YaRNAttentionFactor <= 0 || s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0):
			return errors.New("Mellum YaRN metadata is invalid")
		}
	}
	if s.Architecture == "hunyuan-moe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 ||
			s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("Hunyuan-MoE metadata is invalid")
	}
	if s.Architecture == "hy_v3" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("HY-V3 expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("HY-V3 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("HY-V3 expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("HY-V3 rotary/head dimensions are invalid")
		}
	}
	if s.Architecture == "deepseek2-ocr" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek2-OCR leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DeepSeek2-OCR expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("DeepSeek2-OCR expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek2-OCR expert weight scale is invalid")
		case math.Abs(float64(s.RopeFrequencyBase-10000)) >= 1e-4 || s.RopeScalingType != "" ||
			s.AttentionScale != 0:
			return errors.New("DeepSeek2-OCR attention scaling is unsupported")
		case s.HeadCountKV != s.HeadCount || s.RopeDimensionCount != s.KeyLength ||
			s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("DeepSeek2-OCR attention metadata is invalid")
		}
	}
	if s.Architecture == "smallthinker" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("SmallThinker expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("SmallThinker expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("SmallThinker expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("SmallThinker rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0):
			return errors.New("SmallThinker sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "dots1" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DOTS1 leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DOTS1 expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DOTS1 shared expert width overflows")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("DOTS1 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DOTS1 expert weight scale is invalid")
		case s.RopeDimensionCount > 0 && s.RopeDimensionCount != s.KeyLength:
			return errors.New("DOTS1 rotary dimension must equal key length")
		case s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength:
			return errors.New("DOTS1 requires full-head matching key/value attention")
		}
	}
	if s.Architecture == "minimax-m2" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("MiniMax-M2 expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("MiniMax-M2 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("MiniMax-M2 expert weight scale is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("MiniMax-M2 rotary/head dimensions are invalid")
		}
	}
	if (s.Architecture == "granite" || s.Architecture == "granitemoe" || s.Architecture == "granitehybrid") &&
		s.RopeScalingType == "longrope" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("Granite LongRoPE metadata is invalid")
	}
	if s.Architecture == "bailingmoe2" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("BailingMoE2 leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("BailingMoE2 expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("BailingMoE2 expert routing function is unsupported")
		case s.SharedExpertFF%s.SharedExpertCount != 0:
			return errors.New("BailingMoE2 shared expert width is invalid")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("BailingMoE2 expert weight scale is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("BailingMoE2 rotary/head dimensions are invalid")
		}
	}
	if s.Architecture == "olmoe" &&
		(s.HeadCountKV != s.HeadCount || s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("OLMoE expert or attention metadata is invalid")
	}
	if (s.Architecture == "llama" || s.Architecture == "llama-embed") && s.ExpertCount > 0 &&
		(s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Llama MoE expert metadata is invalid")
	}
	if s.RopeScalingType == "longrope" &&
		(s.Architecture == "llama" || s.Architecture == "llama-embed" ||
			s.Architecture == "minicpm" || s.Architecture == "mistral3") &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return fmt.Errorf("%s LongRoPE metadata is invalid", s.Architecture)
	}
	if s.RopeScalingType == "yarn" &&
		(s.Architecture == "llama" || s.Architecture == "llama-embed" ||
			s.Architecture == "minicpm" || s.Architecture == "mistral3") &&
		(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 ||
			s.YaRNExtFactor < 0 || s.YaRNAttentionFactor <= 0 ||
			s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
			math.IsNaN(float64(s.RopeScalingFactor)) || math.IsInf(float64(s.RopeScalingFactor), 0) ||
			math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
			math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
			math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
			math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0)) {
		return fmt.Errorf("%s YaRN metadata is invalid", s.Architecture)
	}
	if s.Architecture == "llama4" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 || s.MoELayerStep == 0:
			return errors.New("Llama 4 expert metadata is invalid")
		case s.ExpertGatingFunc != 2 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Llama 4 expert routing metadata is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("Llama 4 rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0 ||
			s.AttentionTempFloor == 0 || s.AttentionTempScale <= 0 ||
			math.IsNaN(float64(s.AttentionTempScale)) || math.IsInf(float64(s.AttentionTempScale), 0) ||
			math.IsNaN(float64(s.AttentionTempOffset)) || math.IsInf(float64(s.AttentionTempOffset), 0)):
			return errors.New("Llama 4 chunked-attention metadata is invalid")
		}
	}
	if s.Architecture == "gpt-oss" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertGatingFunc != 3 ||
			s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0) ||
			s.SlidingWindow == 0 || s.SlidingPattern < 2 || s.RopeDimensionCount != s.KeyLength ||
			s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0 || s.RopeFrequencySWA <= 0) {
		return errors.New("GPT-OSS metadata is invalid")
	}
	if s.Architecture == "phimoe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("PhiMoE expert metadata is invalid")
	}
	if s.Architecture == "laguna" {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) {
			return errors.New("Laguna per-layer attention head metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 {
				return fmt.Errorf("Laguna layer %d attention head metadata is invalid", block)
			}
		}
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("Laguna leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("Laguna expert metadata is invalid")
		case s.ExpertGatingFunc != 2:
			return errors.New("Laguna requires sigmoid expert routing")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Laguna expert weight scale is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Laguna full-attention rotary dimension count is invalid")
		case s.RopeScalingType != "yarn":
			return errors.New("Laguna full-attention layers require YaRN RoPE")
		case s.KeyLength != s.ValueLength:
			return errors.New("Laguna requires matching attention key and value lengths")
		case s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 ||
			s.YaRNExtFactor < 0 || s.YaRNAttentionFactor <= 0 ||
			s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
			math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
			math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
			math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
			math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0):
			return errors.New("Laguna YaRN metadata is invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0 ||
			s.RopeDimensionSWA == 0 || s.RopeDimensionSWA > s.KeyLength || s.RopeDimensionSWA%2 != 0):
			return errors.New("Laguna sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "afmoe" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("AFMoE leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0:
			return errors.New("AFMoE expert metadata is invalid")
		case s.ExpertGatingFunc != 2:
			return errors.New("AFMoE requires sigmoid expert routing")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("AFMoE expert weight scale is invalid")
		case s.SharedExpertCount > 0 && s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("AFMoE shared expert width overflows")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("AFMoE rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0):
			return errors.New("AFMoE sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "exaone-moe" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("EXAONE-MoE leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("EXAONE-MoE expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("EXAONE-MoE expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("EXAONE-MoE expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("EXAONE-MoE rotary/head dimensions are invalid")
		case s.SlidingWindow == 0 || (len(s.SlidingLayers) == 0 && s.SlidingPattern < 2) || s.RopeFrequencySWA <= 0:
			return errors.New("EXAONE-MoE sliding attention metadata is invalid")
		}
	}
	if s.Architecture == "lfm2" || s.Architecture == "lfm2moe" {
		if s.ShortConvCacheLength < 2 || len(s.RecurrentLayers) != int(s.BlockCount) {
			return errors.New("LFM2 short-convolution metadata is invalid")
		}
		var recurrent, attention bool
		for _, item := range s.RecurrentLayers {
			recurrent = recurrent || item
			attention = attention || !item
		}
		if !recurrent || !attention {
			return errors.New("LFM2 requires both convolution and attention layers")
		}
		if s.Architecture == "lfm2moe" {
			switch {
			case s.LeadingDenseBlocks >= s.BlockCount:
				return errors.New("LFM2-MoE leading dense block count leaves no MoE layers")
			case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
				s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
				return errors.New("LFM2-MoE expert metadata is invalid")
			case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
				return errors.New("LFM2-MoE expert routing function is unsupported")
			case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
				math.IsInf(float64(s.ExpertWeightsScale), 0):
				return errors.New("LFM2-MoE expert weight scale is invalid")
			}
		}
	}
	if (isMLAArchitecture(s.Architecture) || s.Architecture == "kimi-linear") &&
		(s.KVLoRARank == 0 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount >= s.KeyLength ||
			(!isDeepSeek2Family(s.Architecture) && s.Architecture != "glm-dsa" && s.Architecture != "kimi-linear" && s.HeadCountKV != s.HeadCount) ||
			((isDeepSeek2Family(s.Architecture) || s.Architecture == "glm-dsa" || s.Architecture == "kimi-linear") && s.HeadCountKV != 1 && s.HeadCountKV != s.HeadCount)) {
		return errors.New("MLA metadata is invalid")
	}
	if isDSAArchitecture(s.Architecture) {
		switch {
		case s.QLoRARank == 0:
			return errors.New("DSA query LoRA rank is missing")
		case s.IndexerHeadCount == 0 || s.IndexerKeyLength == 0 || s.IndexerTopK == 0 ||
			s.IndexerTopK > s.ContextLength || s.IndexerKeyLength < s.RopeDimensionCount ||
			s.IndexerKeyLength&(s.IndexerKeyLength-1) != 0:
			return errors.New("DSA indexer metadata is invalid")
		case len(s.IndexerFullLayers) != int(s.BlockCount) || !s.IndexerFullLayers[0]:
			return errors.New("DSA indexer schedule is invalid")
		case s.Architecture == "deepseek32" && (s.BlockCount != 62 || s.LayerNormEpsilon != 1e-6):
			return errors.New("DeepSeek 3.2 layer metadata is invalid")
		}
		var sectionPairs int32
		for _, section := range s.RopeSections {
			if section < 0 {
				return errors.New("DSA RoPE section is negative")
			}
			sectionPairs += section
		}
		if sectionPairs > int32(s.RopeDimensionCount/2) {
			return errors.New("DSA RoPE sections are invalid")
		}
		seenFull := false
		for _, full := range s.IndexerFullLayers {
			if full {
				seenFull = true
			} else if !seenFull {
				return errors.New("DSA shared indexer precedes every full indexer")
			}
		}
		if s.Architecture == "deepseek32" {
			for _, full := range s.IndexerFullLayers {
				if !full {
					return errors.New("DeepSeek 3.2 requires a full indexer in every layer")
				}
			}
		}
	}
	if s.Architecture == "deepseek4" {
		switch {
		case s.BlockCount != 43 || s.HeadCountKV != 1 || s.KeyLength != s.ValueLength:
			return errors.New("DeepSeek 4 layer metadata is invalid")
		case s.QLoRARank == 0 || s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("DeepSeek 4 attention dimensions are invalid")
		case s.SlidingWindow == 0 || s.CompressRopeBase <= 0 ||
			math.IsNaN(float64(s.CompressRopeBase)) || math.IsInf(float64(s.CompressRopeBase), 0):
			return errors.New("DeepSeek 4 compressed-attention metadata is invalid")
		case s.AttentionOutputGroups == 0 || s.HeadCount%s.AttentionOutputGroups != 0 || s.AttentionOutputRank == 0:
			return errors.New("DeepSeek 4 output LoRA metadata is invalid")
		case s.HyperConnectionCount != 4 || s.HyperSinkhornIters == 0 || s.HyperConnectionEps <= 0 ||
			math.IsNaN(float64(s.HyperConnectionEps)) || math.IsInf(float64(s.HyperConnectionEps), 0):
			return errors.New("DeepSeek 4 hyper-connection metadata is invalid")
		case s.IndexerHeadCount == 0 || s.IndexerKeyLength < s.RopeDimensionCount || s.IndexerTopK == 0 ||
			s.IndexerTopK > s.ContextLength || s.IndexerKeyLength&(s.IndexerKeyLength-1) != 0:
			return errors.New("DeepSeek 4 indexer metadata is invalid")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 || s.SharedExpertFF == 0:
			return errors.New("DeepSeek 4 expert metadata is invalid")
		case s.ExpertGatingFunc != uint32(4):
			return errors.New("DeepSeek 4 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek 4 expert weight scale is invalid")
		case s.HashLayerCount > s.BlockCount || len(s.CompressRatios) != int(s.BlockCount) ||
			len(s.LayerSwiGLUClamp) != int(s.BlockCount) || len(s.LayerSharedSwiGLUClamp) != int(s.BlockCount):
			return errors.New("DeepSeek 4 layer schedule is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 || s.YaRNExtFactor < 0 ||
				s.YaRNAttentionFactor <= 0 || s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0):
			return errors.New("DeepSeek 4 YaRN metadata is invalid")
		}
		for block, ratio := range s.CompressRatios {
			if ratio != 0 && ratio != 4 && ratio != 128 {
				return fmt.Errorf("DeepSeek 4 layer %d compression ratio is invalid", block)
			}
			for _, limit := range []float32{s.LayerSwiGLUClamp[block], s.LayerSharedSwiGLUClamp[block]} {
				if limit < 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
					return fmt.Errorf("DeepSeek 4 layer %d SwiGLU clamp is invalid", block)
				}
			}
		}
	}
	if s.Architecture == "kimi-linear" {
		switch {
		case len(s.RecurrentLayers) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("Kimi Linear layer schedule is invalid")
		case s.SSMConvKernel < 2 || s.KDAHeadDim == 0 || s.SSMInnerSize != s.HeadCount*s.KDAHeadDim:
			return errors.New("Kimi Linear KDA metadata is invalid")
		case s.LeadingDenseBlocks >= s.BlockCount || s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 ||
			s.SharedExpertCount == 0 || s.SharedExpertFF == 0:
			return errors.New("Kimi Linear expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("Kimi Linear expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Kimi Linear expert weight scale is invalid")
		}
		var recurrent, attention bool
		for _, item := range s.RecurrentLayers {
			recurrent = recurrent || item
			attention = attention || !item
		}
		if !recurrent || !attention {
			return errors.New("Kimi Linear requires KDA and MLA layers")
		}
	}
	if isDeepSeek2Family(s.Architecture) || s.Architecture == "glm-dsa" {
		lite := s.BlockCount == 26 || s.BlockCount == 27 || (s.BlockCount == 48 && s.VocabularySize == 128256)
		switch {
		case s.ExpertCount == 0 && s.LeadingDenseBlocks != s.BlockCount:
			return errors.New("dense DeepSeek2 requires every block to be dense")
		case s.ExpertCount > 0 && s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek2 leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 && (s.ExpertUsedCount != 0 || s.SharedExpertCount != 0 || s.SharedExpertFF != 0):
			return errors.New("dense DeepSeek2 expert metadata is inconsistent")
		case s.ExpertCount > 0 && (s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 || s.SharedExpertFF == 0):
			return errors.New("DeepSeek2 expert metadata is invalid")
		case s.ExpertCount > 0 && s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DeepSeek2 shared expert width overflows")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek2 expert weight scale is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("DeepSeek2 expert routing function is unsupported")
		case !lite && s.QLoRARank == 0:
			return errors.New("DeepSeek2 query LoRA rank is missing")
		case s.RopeDimensionCount%2 != 0:
			return errors.New("DeepSeek2 rotary dimension is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 || s.YaRNExtFactor < 0 ||
				s.YaRNAttentionFactor <= 0 || s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
				math.IsNaN(float64(s.RopeScalingFactor)) || math.IsInf(float64(s.RopeScalingFactor), 0) ||
				math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
				math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
				math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
				math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0)):
			return errors.New("DeepSeek2 YaRN metadata is invalid")
		case math.IsNaN(float64(s.RopeYaRNLogMultiplier)) || math.IsInf(float64(s.RopeYaRNLogMultiplier), 0):
			return errors.New("DeepSeek2 YaRN log multiplier is invalid")
		case s.AttentionTempScale != 0 && (s.AttentionTempScale <= 0 || s.AttentionTempFloor == 0 ||
			math.IsNaN(float64(s.AttentionTempScale)) || math.IsInf(float64(s.AttentionTempScale), 0)):
			return errors.New("DeepSeek2 attention temperature metadata is invalid")
		}
	}
	if s.Architecture == "mistral3" {
		switch {
		case s.AttentionTempScale != 0 &&
			(s.AttentionTempScale <= 0 || s.AttentionTempFloor == 0 ||
				math.IsNaN(float64(s.AttentionTempScale)) || math.IsInf(float64(s.AttentionTempScale), 0)):
			return errors.New("Mistral 3 attention temperature metadata is invalid")
		case s.ExpertCount == 0 && (s.ExpertUsedCount != 0 || s.ExpertFeedForward != 0):
			return errors.New("dense Mistral 3 expert metadata is inconsistent")
		case s.ExpertCount > 0 &&
			(s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
				s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 ||
				s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
				math.IsInf(float64(s.ExpertWeightsScale), 0)):
			return errors.New("Mistral 3 expert metadata is invalid")
		case s.RopeScalingType == "yarn" &&
			(math.IsNaN(float64(s.RopeYaRNLogMultiplier)) || math.IsInf(float64(s.RopeYaRNLogMultiplier), 0)):
			return errors.New("Mistral 3 YaRN metadata is invalid")
		}
	}
	if s.Architecture == "minicpm3" &&
		(s.QLoRARank == 0 || s.ResidualScale <= 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.ResidualScale)) ||
			math.IsInf(float64(s.ResidualScale), 0) || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("MiniCPM3 metadata is invalid")
	}
	if s.Architecture == "chameleon" && s.QKNormEpsilon <= 0 {
		return errors.New("Chameleon Q/K LayerNorm epsilon must be positive")
	}
	if s.Architecture == "paddleocr" || s.Architecture == "qwen2vl" || s.Architecture == "qwen3vl" || s.Architecture == "qwen3vlmoe" {
		var sectionPairs int32
		for _, section := range s.RopeSections {
			if section < 0 {
				return fmt.Errorf("%s MRoPE section count is negative", s.Architecture)
			}
			sectionPairs += section
		}
		if s.RopeDimensionCount == 0 || s.RopeDimensionCount%2 != 0 ||
			s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || sectionPairs == 0 ||
			sectionPairs > int32(s.RopeDimensionCount/2) {
			return fmt.Errorf("%s MRoPE metadata is invalid", s.Architecture)
		}
		if (s.Architecture == "qwen3vl" || s.Architecture == "qwen3vlmoe") && s.DeepstackLayerCount > s.BlockCount {
			return errors.New("Qwen3-VL deepstack layer count exceeds block count")
		}
	}
	if s.Architecture == "jais2" && s.HeadCountKV != s.HeadCount {
		return errors.New("Jais2 requires matching attention and KV head counts")
	}
	if s.Architecture == "openelm" {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("OpenELM per-layer metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 || s.LayerFeedForwardLength(block) == 0 {
				return fmt.Errorf("OpenELM layer %d dimensions are invalid", block)
			}
		}
	}
	if s.Architecture == "plamo3" {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("PLaMo 3 per-layer metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 || s.LayerFeedForwardLength(block) == 0 {
				return fmt.Errorf("PLaMo 3 layer %d dimensions are invalid", block)
			}
		}
		if s.RopeDimensionCount != s.KeyLength || s.KeyLength%2 != 0 ||
			(s.SlidingWindow > 0 && (s.RopeFrequencySWA <= 0 ||
				(len(s.SlidingLayers) == 0 && s.SlidingPattern < 2))) {
			return errors.New("PLaMo 3 rotary or sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "deci" {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("Deci per-layer metadata is invalid")
		}
		var fullAttention bool
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 && kvHeads != 0 || kvHeads > 0 && (heads == 0 || heads%kvHeads != 0) {
				return fmt.Errorf("Deci layer %d attention head metadata is invalid", block)
			}
			fullAttention = fullAttention || kvHeads > 0
		}
		if !fullAttention {
			return errors.New("Deci requires at least one full-attention layer")
		}
		if s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength {
			return errors.New("Deci rotary/head dimensions are invalid")
		}
		if s.RopeScalingType == "longrope" &&
			(s.OriginalContextLength == 0 || s.RopeAttentionFactor <= 0 ||
				math.IsNaN(float64(s.RopeAttentionFactor)) || math.IsInf(float64(s.RopeAttentionFactor), 0)) {
			return errors.New("Deci LongRoPE metadata is invalid")
		}
	}
	if s.Architecture == "gemma3" {
		switch {
		case s.RopeFrequencySWA <= 0:
			return errors.New("Gemma 3 sliding RoPE frequency base must be positive")
		case s.RopeScalingFactor <= 0:
			return errors.New("Gemma 3 RoPE scaling factor must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("Gemma 3 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "gemma3n" {
		switch {
		case s.BlockCount != 30 && s.BlockCount != 35:
			return errors.New("Gemma 3n block count must be 30 or 35")
		case s.KVFromStart != 20 || s.SharedKVLayers != s.BlockCount-s.KVFromStart:
			return errors.New("Gemma 3n shared-KV boundary is invalid")
		case s.AltUpCount != 4 || s.AltUpActive != 0 || s.LaurelRank != 64 || s.EmbeddingPerLayer != 256:
			return errors.New("Gemma 3n AltUp/Laurel dimensions are invalid")
		case s.SparseLayerCount != 10 || s.SparsityStdMultiplier != 1.6448533535003662:
			return errors.New("Gemma 3n sparsity parameters are invalid")
		case s.KeyLength == 0 || s.KeyLength != s.ValueLength || s.HeadCount == 0 ||
			s.HeadCountKV == 0 || s.HeadCount%s.HeadCountKV != 0:
			return errors.New("Gemma 3n attention dimensions are invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength%2 != 0 ||
			s.RopeFrequencySWA <= 0 || s.SlidingWindow == 0 || s.SlidingPattern != 5:
			return errors.New("Gemma 3n rotary/sliding metadata is invalid")
		case s.FinalLogitSoftcap <= 0:
			return errors.New("Gemma 3n final logit softcap must be positive")
		}
	}
	if s.Architecture == "gemma4" {
		switch {
		case s.KeyLength == 0 || s.ValueLength == 0 || s.KeyLength != s.ValueLength ||
			s.KeyLengthSWA == 0 || s.ValueLengthSWA == 0 || s.KeyLengthSWA != s.ValueLengthSWA:
			return errors.New("Gemma 4 attention head dimensions are invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.RopeDimensionSWA == 0 || s.RopeDimensionSWA > s.KeyLengthSWA || s.RopeDimensionSWA%2 != 0:
			return errors.New("Gemma 4 rotary dimensions are invalid")
		case s.RopeFrequencySWA <= 0 || s.SlidingWindow == 0 || len(s.SlidingLayers) != int(s.BlockCount):
			return errors.New("Gemma 4 sliding-attention metadata is invalid")
		case s.SharedKVLayers > 0 && (s.SharedKVLayers >= s.BlockCount || s.BlockCount-s.SharedKVLayers < 2):
			return errors.New("Gemma 4 shared-KV layer count is invalid")
		case len(s.LayerFeedForward) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("Gemma 4 layer metadata is invalid")
		case s.ExpertCount > 0 && (s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0):
			return errors.New("Gemma 4 expert metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			if s.LayerFeedForwardLength(block) == 0 || s.LayerKVHeadCount(block) == 0 ||
				s.HeadCount%s.LayerKVHeadCount(block) != 0 {
				return errors.New("Gemma 4 per-layer dimensions are invalid")
			}
		}
	}
	if s.Architecture == "gemma4-assistant" {
		switch {
		case s.TargetHiddenSize == 0 || s.TargetHiddenSize == s.EmbeddingLength:
			return errors.New("Gemma 4 assistant target hidden size is invalid")
		case s.KeyLength == 0 || s.ValueLength == 0 || s.KeyLength != s.ValueLength ||
			s.KeyLengthSWA == 0 || s.ValueLengthSWA == 0 || s.KeyLengthSWA != s.ValueLengthSWA:
			return errors.New("Gemma 4 assistant attention head dimensions are invalid")
		case s.HeadCount == 0 || s.HeadCountKV == 0 || s.HeadCount%s.HeadCountKV != 0:
			return errors.New("Gemma 4 assistant attention head counts are invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.RopeDimensionSWA == 0 || s.RopeDimensionSWA > s.KeyLengthSWA || s.RopeDimensionSWA%2 != 0:
			return errors.New("Gemma 4 assistant rotary dimensions are invalid")
		case s.RopeFrequencySWA <= 0 || s.SlidingWindow == 0 || len(s.SlidingLayers) != int(s.BlockCount):
			return errors.New("Gemma 4 assistant sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "gemma2" {
		switch {
		case s.RopeFrequencySWA <= 0:
			return errors.New("Gemma 2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("Gemma 2 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "olmo2" {
		switch {
		case s.SlidingWindow > 0 && s.RopeFrequencySWA <= 0:
			return errors.New("OLMo2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("OLMo2 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "cohere2" || s.Architecture == "cohere2moe" {
		switch {
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Cohere2 rotary dimension count is invalid")
		case s.RopeFrequencySWA <= 0:
			return errors.New("Cohere2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow == 0:
			return errors.New("Cohere2 sliding attention window is zero")
		case s.SlidingPattern < 2:
			return errors.New("Cohere2 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "cohere2moe" &&
		(s.LeadingDenseBlocks >= s.BlockCount || s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertFeedForward == 0 ||
			(s.ExpertGatingFunc != 2) || (s.SharedExpertCount > 0 && s.SharedExpertFF == 0)) {
		return errors.New("Cohere2-MoE expert metadata is invalid")
	}
	if s.Architecture == "ernie4_5-moe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertFeedForward == 0 || s.MoELayerStep == 0 || s.LeadingDenseBlocks >= s.BlockCount) {
		return errors.New("ERNIE 4.5 MoE expert metadata is invalid")
	}
	if s.Architecture == "stablelm" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("StableLM rotary dimension count is invalid")
	}
	if s.Architecture == "phi2" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("Phi-2 rotary dimension count is invalid")
	}
	if (s.Architecture == "phi3" || s.Architecture == "phimoe") &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("Phi-3 RoPE metadata is invalid")
	}
	if s.Architecture == "pangu-embedded" &&
		(s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("Pangu Embedded RoPE metadata is invalid")
	}
	if s.Architecture == "modern-bert" {
		switch {
		case s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength:
			return errors.New("ModernBERT requires full-head matching key/value attention")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("ModernBERT rotary dimension count is invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0):
			return errors.New("ModernBERT sliding attention metadata is invalid")
		}
	}
	if s.Architecture == "gemma-embedding" {
		switch {
		case s.KeyLength != s.ValueLength:
			return errors.New("Gemma embedding requires matching key/value head widths")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("Gemma embedding rotary dimension count is invalid")
		case s.SlidingWindow == 0 || s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0:
			return errors.New("Gemma embedding sliding attention metadata is invalid")
		case s.Dense2FeatureIn > 0 && s.Dense2FeatureIn != s.EmbeddingLength:
			return errors.New("Gemma embedding dense-2 input width must match embedding length")
		case s.Dense3FeatureOut > 0 && s.Dense3FeatureOut != s.EmbeddingLength:
			return errors.New("Gemma embedding dense-3 output width must match embedding length")
		}
	}
	if s.Architecture == "talkie" &&
		(s.KeyLength != s.ValueLength || s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0) {
		return errors.New("Talkie attention metadata is invalid")
	}
	if s.Architecture == "apertus" {
		if s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.OriginalContextLength == 0 || s.RopeAttentionFactor <= 0 ||
			math.IsNaN(float64(s.RopeAttentionFactor)) || math.IsInf(float64(s.RopeAttentionFactor), 0) {
			return errors.New("Apertus RoPE metadata is invalid")
		}
		for name, values := range map[string][]float32{
			"alpha_n": s.XIELUAlphaN,
			"alpha_p": s.XIELUAlphaP,
			"beta":    s.XIELUBeta,
			"epsilon": s.XIELUEpsilon,
		} {
			if len(values) != int(s.BlockCount) {
				return fmt.Errorf("Apertus xIELU %s has %d values, need %d", name, len(values), s.BlockCount)
			}
			for _, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return fmt.Errorf("Apertus xIELU %s values must be finite", name)
				}
			}
		}
	}
	if s.Architecture == "gptneox" && s.RopeDimensionCount > 0 &&
		(s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0) {
		return errors.New("GPT-NeoX rotary dimension count is invalid")
	}
	if s.Architecture == "qwen" &&
		(s.HeadCountKV != s.HeadCount || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0) {
		return errors.New("Qwen attention metadata is invalid")
	}
	if s.Architecture == "chatglm" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength) {
		return errors.New("ChatGLM attention metadata is invalid")
	}
	if s.Architecture == "cogvlm" &&
		(s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength ||
			uint64(s.KeyLength)*uint64(s.HeadCount) != uint64(s.EmbeddingLength) ||
			s.RopeDimensionCount != s.KeyLength) {
		return errors.New("CogVLM attention metadata is invalid")
	}
	if (s.Architecture == "hunyuan-dense" || s.Architecture == "hunyuan_vl") &&
		(s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0 || s.RopeFrequencyBase <= 0 ||
			math.IsNaN(float64(s.RopeFrequencyBase)) || math.IsInf(float64(s.RopeFrequencyBase), 0)) {
		return fmt.Errorf("%s attention metadata is invalid", s.Architecture)
	}
	if s.Architecture == "hunyuan-dense" || s.Architecture == "hunyuan_vl" {
		for _, section := range s.RopeSections {
			if section < 0 {
				return errors.New("Hunyuan MRoPE section count is negative")
			}
		}
	}
	if (s.Architecture == "glm4" || s.Architecture == "glm4moe") &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("GLM4 rotary dimension count is invalid")
	}
	if s.Architecture == "glm4" || s.Architecture == "glm4moe" {
		for _, section := range s.RopeSections {
			if section < 0 {
				return errors.New("GLM4 MRoPE section count is negative")
			}
		}
	}
	if s.Architecture == "glm4moe" &&
		(s.LeadingDenseBlocks >= s.BlockCount || s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 ||
			s.SharedExpertCount == 0 || s.SharedExpertFF == 0 ||
			(s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2) ||
			s.ExpertWeightsScale == 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("GLM4-MoE expert metadata is invalid")
	}
	if s.Architecture == "exaone4" {
		switch {
		case s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("EXAONE 4 rotary dimension count must equal the key length")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("EXAONE 4 sliding attention pattern must be at least 2")
		case s.SlidingWindow > 0 && s.RopeFrequencySWA <= 0:
			return errors.New("EXAONE 4 sliding RoPE frequency base must be positive")
		}
	}
	if s.Architecture == "falcon" && s.RopeDimensionCount > 0 &&
		s.RopeDimensionCount != s.KeyLength {
		return errors.New("Falcon rotary dimension count must equal the key length")
	}
	if s.RopeScalingType == "linear" && s.RopeScalingFactor <= 0 {
		return errors.New("linear RoPE scaling factor must be positive")
	}
	if s.FinalLogitSoftcap < 0 ||
		math.IsNaN(float64(s.FinalLogitSoftcap)) ||
		math.IsInf(float64(s.FinalLogitSoftcap), 0) {
		return errors.New("final logit softcap must be finite and non-negative")
	}
	if s.AttentionSoftcap < 0 ||
		math.IsNaN(float64(s.AttentionSoftcap)) ||
		math.IsInf(float64(s.AttentionSoftcap), 0) {
		return errors.New("attention logit softcap must be finite and non-negative")
	}
	if s.AttentionScale < 0 ||
		math.IsNaN(float64(s.AttentionScale)) ||
		math.IsInf(float64(s.AttentionScale), 0) {
		return errors.New("attention scale must be finite and non-negative")
	}
	if s.MaxALiBiBias < 0 || math.IsNaN(float64(s.MaxALiBiBias)) ||
		math.IsInf(float64(s.MaxALiBiBias), 0) {
		return errors.New("maximum ALiBi bias must be finite and non-negative")
	}
	if s.EmbeddingScale < 0 ||
		math.IsNaN(float64(s.EmbeddingScale)) ||
		math.IsInf(float64(s.EmbeddingScale), 0) {
		return errors.New("embedding scale must be finite and non-negative")
	}
	if s.ResidualScale < 0 ||
		math.IsNaN(float64(s.ResidualScale)) ||
		math.IsInf(float64(s.ResidualScale), 0) {
		return errors.New("residual scale must be finite and non-negative")
	}
	if s.LogitScale < 0 ||
		math.IsNaN(float64(s.LogitScale)) ||
		math.IsInf(float64(s.LogitScale), 0) ||
		((s.Architecture == "minicpm" || s.Architecture == "granite" || s.Architecture == "granitemoe" || s.Architecture == "talkie") &&
			s.LogitScale == 0) {
		return errors.New("logit scale must be finite and positive when required")
	}
	return nil
}

func isGemmaArchitecture(architecture string) bool {
	return architecture == "gemma" || architecture == "gemma-embedding" || architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma3n" || architecture == "gemma4"
}

func hasPostNorm(architecture string) bool {
	return architecture == "afmoe" || architecture == "bert" || architecture == "jina-bert-v2" || architecture == "jina-bert-v3" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" || architecture == "exaone4" || architecture == "gemma2" ||
		architecture == "gemma-embedding" || architecture == "gemma3" || architecture == "gemma3n" || architecture == "gemma4" || architecture == "glm4" || architecture == "grok" || architecture == "plamo2" || architecture == "plamo3"
}

func usesSlidingAttention(architecture string) bool {
	return architecture == "afmoe" || (architecture == "gemma-embedding" || architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma3n" || architecture == "gemma4" || architecture == "gemma4-assistant") ||
		architecture == "exaone4" || architecture == "exaone-moe" || architecture == "gpt-oss" || architecture == "llama4" || architecture == "olmo2" ||
		architecture == "cohere2" || architecture == "cohere2moe" || architecture == "mellum" || architecture == "mimo2" ||
		architecture == "plamo3" || architecture == "smallthinker" || architecture == "step35" || architecture == "dflash"
}

func usesPostOnlyNorm(architecture string) bool {
	return architecture == "bert" || architecture == "jina-bert-v2" || architecture == "jina-bert-v3" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" || architecture == "exaone4"
}

func usesNormalRoPE(architecture string) bool {
	return architecture == "llama" || architecture == "llama4" || architecture == "gpt-oss" || architecture == "llama-embed" || architecture == "eagle3" ||
		architecture == "arctic" ||
		architecture == "deci" ||
		architecture == "llada" ||
		architecture == "internlm2" ||
		architecture == "arcee" ||
		architecture == "baichuan" ||
		architecture == "bailingmoe" ||
		architecture == "deepseek" ||
		architecture == "deepseek4" ||
		architecture == "ernie4_5" ||
		architecture == "ernie4_5-moe" ||
		architecture == "cohere2" ||
		architecture == "cohere2moe" ||
		architecture == "command-r" ||
		architecture == "chameleon" ||
		architecture == "chatglm" ||
		architecture == "cogvlm" ||
		architecture == "granite" ||
		architecture == "granitehybrid" ||
		architecture == "granitemoe" ||
		architecture == "hunyuan-dense" ||
		architecture == "hunyuan_vl" ||
		architecture == "glm4" ||
		architecture == "glm4moe" ||
		architecture == "minicpm" ||
		architecture == "olmo" ||
		architecture == "maincoder" ||
		architecture == "neo-bert" ||
		architecture == "mistral3" ||
		architecture == "plamo2" ||
		architecture == "smollm3" ||
		architecture == "xverse"
}

func usesParallelResidual(architecture string) bool {
	return architecture == "cohere2" || architecture == "cohere2moe" || architecture == "command-r" || architecture == "falcon" ||
		architecture == "phi2" || architecture == "plamo"
}

func usesSequentialGELU(architecture string) bool {
	return architecture == "bloom" || architecture == "codeshell" || architecture == "gpt2" ||
		architecture == "gptneox" || architecture == "phi2" ||
		architecture == "starcoder" || architecture == "starcoder2"
}

func usesGateFreeFFN(architecture string) bool {
	return architecture == "apertus" || usesFusedGateUp(architecture) ||
		usesSquaredReLU(architecture) || usesGELU(architecture)
}

func usesFusedGateUp(architecture string) bool {
	return architecture == "chatglm" || architecture == "glm4" || architecture == "modern-bert" || architecture == "neo-bert" || architecture == "phi3" || architecture == "plamo2" || architecture == "plamo3"
}

func supportsLongRoPE(architecture string) bool {
	return architecture == "apertus" || architecture == "deci" || architecture == "granite" || architecture == "granitehybrid" || architecture == "granitemoe" ||
		architecture == "llama" || architecture == "llama-embed" || architecture == "minicpm" || architecture == "minicpm3" || architecture == "mistral3" ||
		architecture == "pangu-embedded" || architecture == "phi3" || architecture == "phimoe" || architecture == "step35"
}

func firstPositive(values []uint32) uint32 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func usesGELU(architecture string) bool {
	return architecture == "bert" || architecture == "falcon" || architecture == "jina-bert-v3" || architecture == "nomic-bert-moe" || architecture == "mpt" || usesSequentialGELU(architecture)
}

func usesSquaredReLU(architecture string) bool {
	return architecture == "arcee" || architecture == "jais2" || architecture == "nemotron" || architecture == "plm"
}

func required[T any](values map[string]gguf.Value, key string, valueType gguf.ValueType) (T, error) {
	value, ok := values[key]
	if !ok {
		var zero T
		return zero, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type != valueType {
		var zero T
		return zero, fmt.Errorf("metadata %q has type %s, need %s", key, value.Type, valueType)
	}
	typed, ok := value.Data.(T)
	if !ok {
		var zero T
		return zero, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	return typed, nil
}

func optional[T any](values map[string]gguf.Value, key string, valueType gguf.ValueType) (T, bool) {
	value, ok := values[key]
	if !ok || value.Type != valueType {
		var zero T
		return zero, false
	}
	typed, ok := value.Data.(T)
	return typed, ok
}

func requiredArray[T any](
	values map[string]gguf.Value,
	key string,
	elementType gguf.ValueType,
) ([]T, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != elementType {
		return nil, fmt.Errorf(
			"metadata %q must be an array of %s",
			key,
			elementType,
		)
	}
	typed, ok := value.Data.([]T)
	if !ok {
		return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	return typed, nil
}

func requiredLayerFloat32(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]float32, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type == gguf.ValueTypeFloat32 {
		scalar, ok := value.Data.(float32)
		if !ok {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		result := make([]float32, count)
		for index := range result {
			result[index] = scalar
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeFloat32 {
		return nil, fmt.Errorf("metadata %q must be a float32 or float32 array", key)
	}
	items, ok := value.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	if len(items) != int(count) {
		return nil, fmt.Errorf("metadata %q has %d values, need %d", key, len(items), count)
	}
	return append([]float32(nil), items...), nil
}

func optionalLayerFloat32(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]float32, error) {
	if _, ok := values[key]; !ok {
		return nil, nil
	}
	return requiredLayerFloat32(values, key, count)
}

func requiredLayerUint32(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]uint32, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type == gguf.ValueTypeUint32 {
		scalar, ok := value.Data.(uint32)
		if !ok {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		result := make([]uint32, count)
		for index := range result {
			result[index] = scalar
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeUint32 {
		return nil, fmt.Errorf("metadata %q must be a uint32 or uint32 array", key)
	}
	items, ok := value.Data.([]uint32)
	if !ok {
		return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	if len(items) != int(count) {
		return nil, fmt.Errorf("metadata %q has %d values, need %d", key, len(items), count)
	}
	return append([]uint32(nil), items...), nil
}

func requiredLayerUint32Compatible(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]uint32, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type == gguf.ValueTypeUint32 {
		return requiredLayerUint32(values, key, count)
	}
	if value.Type == gguf.ValueTypeInt32 {
		scalar, valid := value.Data.(int32)
		if !valid || scalar < 0 {
			return nil, fmt.Errorf("metadata %q has an invalid scalar value", key)
		}
		result := make([]uint32, count)
		for index := range result {
			result[index] = uint32(scalar)
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray ||
		(value.ArrayType != gguf.ValueTypeUint32 && value.ArrayType != gguf.ValueTypeInt32) {
		return nil, fmt.Errorf("metadata %q must be a uint32 or integer array", key)
	}
	result := make([]uint32, count)
	switch value.ArrayType {
	case gguf.ValueTypeUint32:
		items, valid := value.Data.([]uint32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		copy(result, items)
	case gguf.ValueTypeInt32:
		items, valid := value.Data.([]int32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		for index, item := range items {
			if item < 0 {
				return nil, fmt.Errorf("metadata %q has a negative layer value", key)
			}
			result[index] = uint32(item)
		}
	}
	return result, nil
}

func requiredLayerBoolCompatible(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]bool, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	result := make([]bool, count)
	if value.Type == gguf.ValueTypeBool {
		scalar, valid := value.Data.(bool)
		if !valid {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		for index := range result {
			result[index] = scalar
		}
		return result, nil
	}
	if value.Type == gguf.ValueTypeUint32 {
		scalar, valid := value.Data.(uint32)
		if !valid {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		for index := range result {
			result[index] = scalar != 0
		}
		return result, nil
	}
	if value.Type == gguf.ValueTypeInt32 {
		scalar, valid := value.Data.(int32)
		if !valid || scalar < 0 {
			return nil, fmt.Errorf("metadata %q has an invalid scalar value", key)
		}
		for index := range result {
			result[index] = scalar != 0
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray {
		return nil, fmt.Errorf("metadata %q must be a bool, uint32, or compatible array", key)
	}
	switch value.ArrayType {
	case gguf.ValueTypeBool:
		items, valid := value.Data.([]bool)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		copy(result, items)
	case gguf.ValueTypeUint32:
		items, valid := value.Data.([]uint32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		for index, item := range items {
			result[index] = item != 0
		}
	case gguf.ValueTypeInt32:
		items, valid := value.Data.([]int32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		for index, item := range items {
			if item < 0 {
				return nil, fmt.Errorf("metadata %q has a negative layer value", key)
			}
			result[index] = item != 0
		}
	default:
		return nil, fmt.Errorf("metadata %q must use bool or integer layer values", key)
	}
	return result, nil
}

func optionalArray[T any](
	values map[string]gguf.Value,
	key string,
	elementType gguf.ValueType,
) ([]T, bool, error) {
	value, ok := values[key]
	if !ok {
		return nil, false, nil
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != elementType {
		return nil, false, fmt.Errorf(
			"metadata %q must be an array of %s",
			key,
			elementType,
		)
	}
	typed, ok := value.Data.([]T)
	if !ok {
		return nil, false, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	return typed, true, nil
}
