package model

import (
	"sort"
	"strings"
)

// ArchitectureFamily: primary runtime dispatch family.
type ArchitectureFamily uint8

const (
	ArchitectureFamilyAttention ArchitectureFamily = iota
	ArchitectureFamilyMoE
	ArchitectureFamilyRecurrent
	ArchitectureFamilyHybrid
	ArchitectureFamilyEncoder
	ArchitectureFamilyEncoderDecoder
	ArchitectureFamilyDiffusion
	ArchitectureFamilyDraft
)

type DraftKind uint8

const (
	DraftNone DraftKind = iota
	DraftQwen35MTP
	DraftStep35MTP
	DraftHYV3MTP
	DraftNextNMTP
	DraftCohere2MTP
)

// ForwardPolicy: public inference entry route.
type ForwardPolicy uint8

const (
	ForwardCached ForwardPolicy = iota
	ForwardNonCausal
	ForwardDFlash
	ForwardEagle3
	ForwardGemma4Assistant
	ForwardWavTokenizer
	ForwardT5Encoder
	ForwardT5
)

// NormalizationPolicy: model normalization contract.
type NormalizationPolicy uint8

const (
	NormalizationRMS NormalizationPolicy = iota
	NormalizationLayer
	NormalizationUnweightedLayer
	NormalizationUnweightedRMS
	NormalizationWeightOnlyLayer
)

// PositionPolicy: rotary layout contract.
type PositionPolicy uint8

const (
	PositionNeoX PositionPolicy = iota
	PositionNormal
)

// ResidualPolicy: block residual ordering.
type ResidualPolicy uint8

const (
	ResidualSequential ResidualPolicy = iota
	ResidualParallel
)

// FeedForwardPolicy: dense activation/layout contract.
type FeedForwardPolicy uint8

const (
	FeedForwardSwiGLU FeedForwardPolicy = iota
	FeedForwardSequentialGELU
	FeedForwardFusedGateUp
	FeedForwardGELU
	FeedForwardSquaredReLU
	FeedForwardGateFreeSiLU
)

// AttentionPolicy: primary attention implementation.
type AttentionPolicy uint8

const (
	AttentionStandard AttentionPolicy = iota
	AttentionMLA
	AttentionDSA
	AttentionQwenGDN
	AttentionLFM2
)

// ArchitectureCapability: orthogonal runtime behavior.
type ArchitectureCapability uint64

const (
	ArchitectureNonCausal ArchitectureCapability = 1 << iota
	ArchitectureRoPEDisabled
	ArchitectureMoE
	ArchitectureRecurrent
	ArchitectureEncoderOnly
	ArchitectureDiffusion
	ArchitectureMultimodal
	ArchitectureDeepSeek2
	ArchitectureDSA
	ArchitectureMLA
	ArchitectureGemma
	ArchitecturePostNorm
	ArchitecturePostOnlyNorm
	ArchitectureSlidingAttention
	ArchitectureNormalRoPE
	ArchitectureParallelResidual
	ArchitectureSequentialGELU
	ArchitectureFusedGateUp
	ArchitectureLongRoPE
	ArchitectureGELU
	ArchitectureSquaredReLU
	ArchitectureQwenGDN
	ArchitectureLFM2
	ArchitectureMultiAxisPositions
	ArchitectureDeepstack
	ArchitectureRequiresOutput
	ArchitectureClassifierHead
	ArchitectureBiasFreeProjections
	ArchitectureFusedQKV
	ArchitectureRequiresFusedQKV
	ArchitectureRequiresFusedQKVBias
	ArchitectureRejectsOrphanFusedQKVBias
	ArchitectureSharedKV
	ArchitectureAltUp
	ArchitecturePerLayerEmbeddings
	ArchitectureEmbeddingSkip
)

// ArchitectureProfile: registry entry and capability set.
type ArchitectureProfile struct {
	Name          string
	Family        ArchitectureFamily
	GraphFamily   ArchitectureFamily
	CatalogFamily ArchitectureFamily
	DraftKind     DraftKind
	Forward       ForwardPolicy
	Capabilities  ArchitectureCapability
	Normalization NormalizationPolicy
	Position      PositionPolicy
	Residual      ResidualPolicy
	FeedForward   FeedForwardPolicy
	Attention     AttentionPolicy
}

// Has: capability predicate.
func (p ArchitectureProfile) Has(capability ArchitectureCapability) bool {
	return p.Capabilities&capability != 0
}

// AppendsDraftBlocks: catalog-visible draft tail
func (p ArchitectureProfile) AppendsDraftBlocks() bool {
	switch p.DraftKind {
	case DraftStep35MTP, DraftHYV3MTP, DraftNextNMTP:
		return true
	default:
		return false
	}
}

// LookupArchitecture: registered profile lookup.
func LookupArchitecture(name string) (ArchitectureProfile, bool) {
	profile, ok := architectureRegistry[name]
	return profile, ok
}

// SupportedArchitectures: sorted registered names.
func SupportedArchitectures() []string {
	names := make([]string, 0, len(architectureRegistry))
	for name := range architectureRegistry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Profile: registered profile; zero profile for invalid specs.
func (s Spec) Profile() ArchitectureProfile {
	profile, _ := LookupArchitecture(s.Architecture)
	return profile
}

var architectureRegistry = buildArchitectureRegistry()

func buildArchitectureRegistry() map[string]ArchitectureProfile {
	names := strings.Fields(`
		afmoe apertus arcee arctic arwkv7 baichuan bailingmoe bailingmoe2
		bert bitnet bloom chameleon chatglm codeshell cogvlm cohere2 cohere2moe
		command-r dbrx deci deepseek deepseek2 deepseek2-ocr deepseek32 deepseek4
		dflash dots1 dream eagle3 ernie4_5 ernie4_5-moe eurobert exaone
		exaone-moe exaone4 falcon falcon-h1 gemma gemma-embedding gemma2 gemma3
		gemma3n gemma4 gemma4-assistant glm-dsa glm4 glm4moe gpt-oss gpt2
		gptj gptneox granite granitehybrid granitemoe grok grovemoe hunyuan-dense
		hunyuan-moe hunyuan_vl hy_v3 internlm2 jais jais2 jamba jina-bert-v2
		jina-bert-v3 kimi-linear laguna lfm2 lfm2moe llama llama-embed llama4
		llada llada-moe maincoder mamba mamba2 mellum mimo2 minicpm minicpm3
		minimax-m2 mistral3 mistral4 modern-bert mpt nemotron nemotron_h
		nemotron_h_moe neo-bert nomic-bert nomic-bert-moe olmo olmo2 olmoe
		openelm orion paddleocr pangu-embedded phi2 phi3 phimoe plamo plamo2
		plamo3 plm qwen qwen2 qwen2moe qwen2vl qwen3 qwen35 qwen35moe
		qwen3moe qwen3next qwen3vl qwen3vlmoe refact rnd1 rwkv6 rwkv6qwen2
		rwkv7 seed_oss smallthinker smollm3 stablelm starcoder starcoder2 step35
		t5 t5encoder talkie wavtokenizer-dec xverse
	`)
	registry := make(map[string]ArchitectureProfile, len(names))
	for _, name := range names {
		registry[name] = ArchitectureProfile{
			Name:          name,
			Family:        ArchitectureFamilyAttention,
			GraphFamily:   ArchitectureFamilyAttention,
			CatalogFamily: ArchitectureFamilyAttention,
		}
	}
	setCapabilities := func(capabilities ArchitectureCapability, names ...string) {
		for _, name := range names {
			profile, ok := registry[name]
			if !ok {
				panic("unknown architecture profile: " + name)
			}
			profile.Capabilities |= capabilities
			registry[name] = profile
		}
	}
	setFamily := func(family ArchitectureFamily, names ...string) {
		for _, name := range names {
			profile, ok := registry[name]
			if !ok {
				panic("unknown architecture profile: " + name)
			}
			profile.Family = family
			profile.GraphFamily = family
			profile.CatalogFamily = family
			registry[name] = profile
		}
	}
	setDraftKind := func(kind DraftKind, names ...string) {
		for _, name := range names {
			profile, ok := registry[name]
			if !ok {
				panic("unknown architecture profile: " + name)
			}
			profile.DraftKind = kind
			registry[name] = profile
		}
	}
	setForward := func(policy ForwardPolicy, names ...string) {
		for _, name := range names {
			profile, ok := registry[name]
			if !ok {
				panic("unknown architecture profile: " + name)
			}
			profile.Forward = policy
			registry[name] = profile
		}
	}
	setNormalization := func(policy NormalizationPolicy, names ...string) {
		for _, name := range names {
			profile, ok := registry[name]
			if !ok {
				panic("unknown architecture profile: " + name)
			}
			profile.Normalization = policy
			registry[name] = profile
		}
	}
	setFeedForward := func(policy FeedForwardPolicy, names ...string) {
		for _, name := range names {
			profile, ok := registry[name]
			if !ok {
				panic("unknown architecture profile: " + name)
			}
			profile.FeedForward = policy
			registry[name] = profile
		}
	}

	setDraftKind(DraftQwen35MTP, "qwen35", "qwen35moe")
	setDraftKind(DraftStep35MTP, "step35")
	setDraftKind(DraftHYV3MTP, "hy_v3")
	setDraftKind(DraftCohere2MTP, "cohere2moe")
	setDraftKind(DraftNextNMTP,
		"glm4", "glm4moe", "exaone4", "exaone-moe", "mimo2", "bailingmoe2", "deepseek32", "glm-dsa",
	)

	setCapabilities(ArchitectureNonCausal,
		"bert", "dream", "eurobert", "gemma-embedding", "jina-bert-v2",
		"jina-bert-v3", "llada", "llada-moe", "llama-embed", "modern-bert",
		"neo-bert", "nomic-bert", "nomic-bert-moe", "rnd1", "wavtokenizer-dec",
		"dflash",
	)
	setCapabilities(ArchitectureRoPEDisabled,
		"bert", "jina-bert-v2", "mamba", "mamba2", "jamba", "kimi-linear",
		"rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7", "t5", "wavtokenizer-dec",
		"nemotron_h", "nemotron_h_moe",
	)
	setCapabilities(ArchitectureMoE,
		"afmoe", "bailingmoe", "bailingmoe2", "cohere2moe", "dbrx", "deepseek",
		"deepseek2", "deepseek2-ocr", "deepseek32", "deepseek4", "dots1",
		"ernie4_5-moe", "exaone-moe", "glm-dsa", "glm4moe", "gpt-oss",
		"granitemoe", "grok", "grovemoe", "hunyuan-moe", "hy_v3", "lfm2moe",
		"llada-moe", "llama4", "minimax-m2", "nemotron_h_moe", "nomic-bert-moe",
		"olmoe", "phimoe", "qwen2moe", "qwen3moe", "qwen3vlmoe", "qwen35moe",
	)
	setCapabilities(ArchitectureRecurrent,
		"arwkv7", "falcon-h1", "granitehybrid", "jamba", "kimi-linear", "lfm2",
		"lfm2moe", "mamba", "mamba2", "nemotron_h", "nemotron_h_moe", "plamo2",
		"qwen3next", "qwen35", "qwen35moe", "rwkv6", "rwkv6qwen2", "rwkv7",
	)
	setCapabilities(ArchitectureEncoderOnly,
		"bert", "eurobert", "gemma-embedding", "jina-bert-v2", "jina-bert-v3",
		"llama-embed", "modern-bert", "neo-bert", "nomic-bert", "nomic-bert-moe",
		"t5encoder", "wavtokenizer-dec",
	)
	setCapabilities(ArchitectureDiffusion, "dream", "llada", "llada-moe", "rnd1")
	setCapabilities(ArchitectureMultimodal,
		"chameleon", "cogvlm", "deepseek2-ocr", "gemma3", "gemma3n", "gemma4",
		"hunyuan_vl", "paddleocr", "qwen2vl", "qwen3vl", "qwen3vlmoe",
	)
	setCapabilities(ArchitectureDeepSeek2, "deepseek2", "deepseek32", "mistral4")
	setCapabilities(ArchitectureDSA, "deepseek32", "glm-dsa")
	setCapabilities(ArchitectureMLA,
		"deepseek2", "deepseek32", "glm-dsa", "minicpm3", "mistral4", "plm",
	)
	setCapabilities(ArchitectureGemma,
		"gemma", "gemma-embedding", "gemma2", "gemma3", "gemma3n", "gemma4",
	)
	setCapabilities(ArchitecturePostNorm,
		"afmoe", "bert", "exaone4", "gemma-embedding", "gemma2", "gemma3",
		"gemma3n", "gemma4", "glm4", "grok", "jina-bert-v2", "jina-bert-v3",
		"nomic-bert", "nomic-bert-moe", "plamo2", "plamo3",
	)
	setCapabilities(ArchitecturePostOnlyNorm,
		"bert", "exaone4", "jina-bert-v2", "jina-bert-v3", "nomic-bert",
		"nomic-bert-moe",
	)
	setCapabilities(ArchitectureSlidingAttention,
		"afmoe", "cohere2", "cohere2moe", "dflash", "exaone-moe", "exaone4",
		"gemma-embedding", "gemma2", "gemma3", "gemma3n", "gemma4",
		"gemma4-assistant", "gpt-oss", "llama4", "mellum", "mimo2", "olmo2",
		"plamo3", "smallthinker", "step35",
	)
	setCapabilities(ArchitectureNormalRoPE,
		"arcee", "arctic", "baichuan", "bailingmoe", "chameleon", "chatglm",
		"cogvlm", "cohere2", "cohere2moe", "command-r", "deci", "deepseek",
		"deepseek4", "eagle3", "ernie4_5", "ernie4_5-moe", "glm4", "glm4moe",
		"gpt-oss", "gptj", "granite", "granitehybrid", "granitemoe", "hunyuan-dense",
		"hunyuan_vl", "internlm2", "llada", "llama", "llama-embed", "llama4",
		"maincoder", "minicpm", "mistral3", "neo-bert", "olmo", "plamo2",
		"smollm3", "xverse",
	)
	setCapabilities(ArchitectureParallelResidual,
		"cohere2", "cohere2moe", "command-r", "falcon", "gptj", "phi2", "plamo",
	)
	setCapabilities(ArchitectureSequentialGELU,
		"bloom", "codeshell", "gpt2", "gptneox", "phi2", "starcoder", "starcoder2",
	)
	setCapabilities(ArchitectureFusedGateUp,
		"chatglm", "glm4", "modern-bert", "neo-bert", "phi3", "plamo2", "plamo3",
	)
	setCapabilities(ArchitectureLongRoPE,
		"apertus", "deci", "granite", "granitehybrid", "granitemoe", "llama",
		"llama-embed", "minicpm", "minicpm3", "mistral3", "pangu-embedded",
		"phi3", "phimoe", "step35",
	)
	setCapabilities(ArchitectureGELU,
		"bert", "falcon", "gptj", "jina-bert-v3", "mpt", "nomic-bert-moe",
	)
	setCapabilities(ArchitectureSquaredReLU, "arcee", "jais2", "nemotron", "plm")
	setCapabilities(ArchitectureQwenGDN, "qwen3next", "qwen35", "qwen35moe")
	setCapabilities(ArchitectureLFM2, "lfm2", "lfm2moe")
	setCapabilities(ArchitectureMultiAxisPositions,
		"glm4", "glm4moe", "hunyuan-dense", "hunyuan_vl", "paddleocr", "qwen2vl",
		"qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe",
	)
	setCapabilities(ArchitectureDeepstack, "granite", "qwen3vl", "qwen3vlmoe")
	setCapabilities(ArchitectureRequiresOutput,
		"apertus", "arwkv7", "baichuan", "bailingmoe", "bailingmoe2", "codeshell",
		"dbrx", "dots1", "gpt-oss", "gptj", "gptneox", "internlm2", "jais",
		"kimi-linear", "llada-moe", "mellum", "mimo2", "minimax-m2", "nemotron",
		"olmo2", "orion", "phi2", "phimoe", "plamo", "qwen", "rwkv6",
		"rwkv6qwen2", "rwkv7", "stablelm", "step35", "talkie", "xverse",
	)
	setCapabilities(ArchitectureClassifierHead, "qwen3", "qwen3vl")
	setCapabilities(ArchitectureBiasFreeProjections, "cogvlm", "qwen3next", "qwen35", "qwen35moe")
	setCapabilities(ArchitectureFusedQKV,
		"apertus", "bailingmoe2", "bert", "bloom", "chatglm", "cogvlm", "cohere2moe",
		"deci", "dbrx", "dots1", "ernie4_5", "ernie4_5-moe", "eurobert", "exaone4",
		"falcon", "gemma-embedding", "glm4", "glm4moe", "gpt2", "gptneox", "grok",
		"hunyuan-dense", "hunyuan_vl", "hy_v3", "jais", "jina-bert-v2", "jina-bert-v3",
		"mimo2", "minimax-m2", "modern-bert", "mpt", "neo-bert", "nomic-bert",
		"nomic-bert-moe", "openelm", "paddleocr", "pangu-embedded", "phi2", "phi3",
		"phimoe", "plamo2", "plamo3", "qwen", "qwen2vl", "qwen3vl", "qwen3vlmoe",
		"refact", "smallthinker", "starcoder", "step35", "talkie",
	)
	setCapabilities(ArchitectureRequiresFusedQKV,
		"bailingmoe2", "bloom", "cogvlm", "dbrx", "falcon", "gpt2", "gptneox",
		"jais", "modern-bert", "mpt", "neo-bert", "qwen", "starcoder",
	)
	setCapabilities(ArchitectureRequiresFusedQKVBias,
		"bloom", "gpt2", "gptneox", "jais", "qwen", "starcoder",
	)
	setCapabilities(ArchitectureRejectsOrphanFusedQKVBias,
		"apertus", "exaone4", "glm4", "phi2", "phi3", "phimoe", "smallthinker",
	)
	setCapabilities(ArchitectureSharedKV, "gemma3n", "gemma4")
	setCapabilities(ArchitectureAltUp, "gemma3n")
	setCapabilities(ArchitecturePerLayerEmbeddings, "gemma3n", "gemma4")
	setCapabilities(ArchitectureEmbeddingSkip, "talkie")

	setNormalization(NormalizationLayer,
		"bert", "bloom", "codeshell", "dbrx", "falcon", "gpt2", "gptj",
		"gptneox", "jais", "jais2", "jina-bert-v2", "jina-bert-v3", "mpt",
		"nemotron", "nomic-bert", "nomic-bert-moe", "orion", "phi2", "rwkv6",
		"rwkv7", "stablelm", "starcoder", "starcoder2", "wavtokenizer-dec",
	)
	setNormalization(NormalizationUnweightedLayer, "olmo")
	setNormalization(NormalizationUnweightedRMS, "talkie")
	setNormalization(NormalizationWeightOnlyLayer,
		"cohere2", "cohere2moe", "command-r", "modern-bert",
	)
	setFeedForward(FeedForwardSequentialGELU,
		"bloom", "codeshell", "gpt2", "gptneox", "phi2", "starcoder", "starcoder2",
	)
	setFeedForward(FeedForwardFusedGateUp,
		"chatglm", "glm4", "modern-bert", "neo-bert", "phi3", "plamo2", "plamo3",
	)
	setFeedForward(FeedForwardGELU,
		"bert", "falcon", "gptj", "jina-bert-v3", "mpt", "nomic-bert-moe",
	)
	setFeedForward(FeedForwardSquaredReLU, "arcee", "jais2", "nemotron", "plm")
	setFeedForward(FeedForwardGateFreeSiLU, "apertus")

	setFamily(ArchitectureFamilyMoE,
		"afmoe", "bailingmoe", "bailingmoe2", "cohere2moe", "dbrx", "deepseek",
		"deepseek2", "deepseek2-ocr", "deepseek32", "deepseek4", "dots1",
		"ernie4_5-moe", "exaone-moe", "glm-dsa", "glm4moe", "gpt-oss",
		"granitemoe", "grok", "grovemoe", "hunyuan-moe", "hy_v3", "llama4",
		"minimax-m2", "nomic-bert-moe", "olmoe", "phimoe", "qwen2moe",
		"qwen3moe", "qwen3vlmoe",
	)
	setFamily(ArchitectureFamilyRecurrent,
		"arwkv7", "kimi-linear", "mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7",
	)
	setFamily(ArchitectureFamilyHybrid,
		"falcon-h1", "granitehybrid", "jamba", "lfm2", "lfm2moe", "nemotron_h",
		"nemotron_h_moe", "plamo2", "qwen3next", "qwen35", "qwen35moe",
	)
	setFamily(ArchitectureFamilyEncoder,
		"bert", "eurobert", "gemma-embedding", "jina-bert-v2", "jina-bert-v3",
		"llama-embed", "modern-bert", "neo-bert", "nomic-bert", "nomic-bert-moe",
		"t5encoder", "wavtokenizer-dec",
	)
	setFamily(ArchitectureFamilyEncoderDecoder, "t5")
	setFamily(ArchitectureFamilyDiffusion, "dream", "llada", "llada-moe", "rnd1")
	setFamily(ArchitectureFamilyDraft, "dflash", "eagle3", "gemma4-assistant")
	setForward(ForwardDFlash, "dflash")
	setForward(ForwardEagle3, "eagle3")
	setForward(ForwardGemma4Assistant, "gemma4-assistant")
	setForward(ForwardWavTokenizer, "wavtokenizer-dec")
	setForward(ForwardT5Encoder, "t5encoder")
	setForward(ForwardT5, "t5")
	for name, profile := range registry {
		if profile.Forward == ForwardCached && profile.Has(ArchitectureNonCausal) {
			profile.Forward = ForwardNonCausal
		}
		if profile.Has(ArchitectureNormalRoPE) {
			profile.Position = PositionNormal
		}
		if profile.Has(ArchitectureParallelResidual) {
			profile.Residual = ResidualParallel
		}
		switch {
		case profile.Has(ArchitectureDSA):
			profile.Attention = AttentionDSA
		case profile.Has(ArchitectureMLA):
			profile.Attention = AttentionMLA
		case profile.Has(ArchitectureQwenGDN):
			profile.Attention = AttentionQwenGDN
		case profile.Has(ArchitectureLFM2):
			profile.Attention = AttentionLFM2
		}
		registry[name] = profile
	}
	return registry
}
