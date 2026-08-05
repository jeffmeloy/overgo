package model

import (
	"sort"
	"strings"

	"llamacpp2go/internal/tensor"
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

// DraftSessionPolicy: runtime coordinator cardinality.
type DraftSessionPolicy uint8

const (
	DraftSessionNone DraftSessionPolicy = iota
	DraftSessionSingle
	DraftSessionMulti
)

// DraftPlan: compiled catalog and session contract.
type DraftPlan struct {
	Kind            DraftKind
	Heads           uint32
	Label           string
	AppendedBlocks  bool
	SingleCatalog   bool
	OptionalCatalog bool
	SupportsMTPOnly bool
	Session         DraftSessionPolicy
}

// HasHead: bounded catalog head.
func (p DraftPlan) HasHead(offset uint32) bool {
	if p.Kind == DraftNone || p.Heads == 0 || p.SingleCatalog && p.Heads != 1 {
		return false
	}
	return offset < p.Heads
}

// SessionEligible: supported runtime coordinator cardinality.
func (p DraftPlan) SessionEligible() bool {
	if !p.HasHead(0) {
		return false
	}
	return p.Session != DraftSessionSingle || p.Heads == 1
}

// Block: physical draft block index.
func (p DraftPlan) Block(trunk, offset uint32) uint32 {
	return trunk + offset
}

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

// OutputNormPolicy: final normalization tensor ownership.
type OutputNormPolicy uint8

const (
	OutputNormModel OutputNormPolicy = iota
	OutputNormAbsent
	OutputNormEncoder
	OutputNormDecoder
	OutputNormTokenEmbedding
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

// EmbeddingOverridePolicy: projected-embedding application order.
type EmbeddingOverridePolicy uint8

const (
	EmbeddingOverrideStandard EmbeddingOverridePolicy = iota
	EmbeddingOverrideCogVLM
	EmbeddingOverrideRawScaled
	EmbeddingOverrideDeepstackBase
)

// DeepstackPolicy: projected-stream layer placement.
type DeepstackPolicy uint8

const (
	DeepstackNone DeepstackPolicy = iota
	DeepstackMappedBefore
	DeepstackSequentialAfter
)

// AttentionBlockPolicy: projected attention-mask contract.
type AttentionBlockPolicy uint8

const (
	AttentionBlocksNone AttentionBlockPolicy = iota
	AttentionBlocksUncached
)

// AuxiliaryFlow: transient cross-layer value contract.
type AuxiliaryFlow uint8

const (
	AuxiliaryNone AuxiliaryFlow = iota
	AuxiliaryRWKVValue
	AuxiliaryDSATopK
)

// AttentionTemperaturePolicy: scheduled query-scale contract.
type AttentionTemperaturePolicy uint8

const (
	AttentionTemperatureNone AttentionTemperaturePolicy = iota
	AttentionTemperatureConfigured
	AttentionTemperatureNoRoPE
)

// PostNormLayoutPolicy: post-attention/FFN tensor namespace.
type PostNormLayoutPolicy uint8

const (
	PostNormLayoutStandard PostNormLayoutPolicy = iota
	PostNormLayoutBERT
	PostNormLayoutGrok
)

// FeedForwardNormLayoutPolicy: pre-FFN tensor namespace.
type FeedForwardNormLayoutPolicy uint8

const (
	FeedForwardNormLayoutStandard FeedForwardNormLayoutPolicy = iota
	FeedForwardNormLayoutBare
	FeedForwardNormLayoutAttentionOutput
	FeedForwardNormLayoutAttentionPost
	FeedForwardNormLayoutPostAttention
)

// NormalizationPlan: compiled operation, placement, bias, and tensor layout.
type NormalizationPlan struct {
	Operation         NormalizationPolicy
	PreAttention      bool
	PreFeedForward    bool
	PostAttention     bool
	PostFeedForward   bool
	Bias              bool
	PostNormLayout    PostNormLayoutPolicy
	FeedForwardLayout FeedForwardNormLayoutPolicy
}

// Weighted: learned normalization weight required.
func (p NormalizationPlan) Weighted() bool {
	return p.Operation != NormalizationUnweightedLayer && p.Operation != NormalizationUnweightedRMS
}

// PostNormTensors: post-norm tensor namespace.
func (p NormalizationPlan) PostNormTensors() PostNormTensorNames {
	switch p.PostNormLayout {
	case PostNormLayoutBERT:
		return PostNormTensorNames{
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			AttentionBias: "attn_output_norm.bias", FeedForwardBias: "layer_output_norm.bias",
		}
	case PostNormLayoutGrok:
		return PostNormTensorNames{
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			FeedForwardFallback: "ffn_post_norm.weight",
		}
	default:
		return PostNormTensorNames{
			AttentionWeight: "post_attention_norm.weight", FeedForwardWeight: "post_ffw_norm.weight",
		}
	}
}

// FeedForwardNormTensor: pre-FFN tensor namespace.
func (p NormalizationPlan) FeedForwardNormTensor() string {
	switch p.FeedForwardLayout {
	case FeedForwardNormLayoutBare:
		return "ffn_norm"
	case FeedForwardNormLayoutAttentionOutput:
		return "attn_output_norm.weight"
	case FeedForwardNormLayoutAttentionPost:
		return "attn_post_norm.weight"
	case FeedForwardNormLayoutPostAttention:
		return "post_attention_norm.weight"
	default:
		return "ffn_norm.weight"
	}
}

// PostNormTensorNames: post-norm tensor catalog entry.
type PostNormTensorNames struct {
	AttentionWeight     string
	FeedForwardWeight   string
	FeedForwardFallback string
	AttentionBias       string
	FeedForwardBias     string
}

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
	ArchitectureBERTNormLayout
	ArchitectureDeepSeek2Layout
)

// ArchitectureProfile: registry entry and capability set.
type ArchitectureProfile struct {
	Name            string
	Family          ArchitectureFamily
	GraphFamily     ArchitectureFamily
	CatalogFamily   ArchitectureFamily
	DraftKind       DraftKind
	Forward         ForwardPolicy
	OutputNorm      OutputNormPolicy
	Capabilities    ArchitectureCapability
	Normalization   NormalizationPolicy
	Position        PositionPolicy
	Residual        ResidualPolicy
	FeedForward     FeedForwardPolicy
	Attention       AttentionPolicy
	Overrides       EmbeddingOverridePolicy
	Deepstack       DeepstackPolicy
	AttentionBlocks AttentionBlockPolicy
	Auxiliary       AuxiliaryFlow
	Temperature     AttentionTemperaturePolicy
	PostNormLayout  PostNormLayoutPolicy
	FFNNormLayout   FeedForwardNormLayoutPolicy
	Block           BlockPolicy
	RecurrentBlock  BlockPolicy
	Cache           CachePolicy
	RecurrentCache  CachePolicy
	CacheFallback   CacheFallbackPolicy
	DenseGraph      DenseGraphPolicy
	DenseStages     DenseStagePolicy
	DenseWeights    DenseWeightPolicy
	Rotary          RotaryPolicy
	AttentionGraph  AttentionGraphPolicy
	Experts         ExpertPolicy
	Cadence         LayerCadencePolicy
	DeciSparse      bool
}

// Has: capability predicate.
func (p ArchitectureProfile) Has(capability ArchitectureCapability) bool {
	return p.Capabilities&capability != 0
}

// AppendsDraftBlocks: catalog-visible draft tail
func (p ArchitectureProfile) AppendsDraftBlocks() bool {
	return p.DraftPlan(1).AppendedBlocks
}

// HasDraftHead: bounded draft-head policy.
func (p ArchitectureProfile) HasDraftHead(kind DraftKind, count, offset uint32) bool {
	plan := p.DraftPlan(count)
	return plan.Kind == kind && plan.HasHead(offset)
}

// HasSingleDraft: single-head draft policy.
func (p ArchitectureProfile) HasSingleDraft(kind DraftKind, count uint32) bool {
	plan := p.DraftPlan(count)
	return plan.Kind == kind && plan.HasHead(0) && count == 1
}

// DraftPlan: architecture draft catalog/session descriptor.
func (p ArchitectureProfile) DraftPlan(heads uint32) DraftPlan {
	plan := DraftPlan{Kind: p.DraftKind, Heads: heads}
	switch p.DraftKind {
	case DraftQwen35MTP:
		plan.Label, plan.SingleCatalog = "Qwen3.5 MTP", true
		plan.SupportsMTPOnly, plan.Session = true, DraftSessionSingle
	case DraftStep35MTP:
		plan.Label, plan.AppendedBlocks, plan.Session = "Step3.5 MTP", true, DraftSessionMulti
	case DraftHYV3MTP:
		plan.Label, plan.AppendedBlocks, plan.Session = "HY-V3 MTP", true, DraftSessionMulti
	case DraftNextNMTP:
		plan.Label, plan.AppendedBlocks, plan.Session = "NextN MTP", true, DraftSessionSingle
	case DraftCohere2MTP:
		plan.Label, plan.SingleCatalog, plan.OptionalCatalog = "Cohere2-MoE MTP", true, true
		plan.SupportsMTPOnly, plan.Session = true, DraftSessionSingle
	}
	return plan
}

// OutputNormTensor: final normalization tensor name.
func (p ArchitectureProfile) OutputNormTensor() string {
	switch p.OutputNorm {
	case OutputNormAbsent:
		return ""
	case OutputNormEncoder:
		return "enc.output_norm.weight"
	case OutputNormDecoder:
		return "dec.output_norm.weight"
	case OutputNormTokenEmbedding:
		return "token_embd_norm.weight"
	default:
		return "output_norm.weight"
	}
}

// PostNormTensors: post-norm tensor namespace.
func (p ArchitectureProfile) PostNormTensors() PostNormTensorNames {
	return NormalizationPlan{PostNormLayout: p.PostNormLayout}.PostNormTensors()
}

// FeedForwardNormTensor: pre-FFN tensor namespace.
func (p ArchitectureProfile) FeedForwardNormTensor() string {
	return NormalizationPlan{FeedForwardLayout: p.FFNNormLayout}.FeedForwardNormTensor()
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
	update := func(names []string, apply func(*ArchitectureProfile)) {
		for _, name := range names {
			profile, ok := registry[name]
			if !ok {
				panic("unknown architecture profile: " + name)
			}
			apply(&profile)
			registry[name] = profile
		}
	}
	setCapabilities := func(capabilities ArchitectureCapability, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Capabilities |= capabilities })
	}
	setFamily := func(family ArchitectureFamily, names ...string) {
		update(names, func(profile *ArchitectureProfile) {
			profile.Family = family
			profile.GraphFamily = family
			profile.CatalogFamily = family
		})
	}
	setDraftKind := func(kind DraftKind, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.DraftKind = kind })
	}
	setForward := func(policy ForwardPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Forward = policy })
	}
	setOutputNorm := func(policy OutputNormPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.OutputNorm = policy })
	}
	setNormalization := func(policy NormalizationPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Normalization = policy })
	}
	setFeedForward := func(policy FeedForwardPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.FeedForward = policy })
	}
	setOverrides := func(policy EmbeddingOverridePolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Overrides = policy })
	}
	setDeepstack := func(policy DeepstackPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Deepstack = policy })
	}
	setAttentionBlocks := func(policy AttentionBlockPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.AttentionBlocks = policy })
	}
	setAuxiliary := func(policy AuxiliaryFlow, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Auxiliary = policy })
	}
	setTemperature := func(policy AttentionTemperaturePolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Temperature = policy })
	}
	setPostNormLayout := func(policy PostNormLayoutPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.PostNormLayout = policy })
	}
	setFFNNormLayout := func(policy FeedForwardNormLayoutPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.FFNNormLayout = policy })
	}
	setBlock := func(policy BlockPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Block = policy })
	}
	setRecurrentBlock := func(policy BlockPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.RecurrentBlock = policy })
	}
	setCache := func(policy CachePolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Cache = policy })
	}
	setRecurrentCache := func(policy CachePolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.RecurrentCache = policy })
	}
	setCacheFallback := func(policy CacheFallbackPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.CacheFallback = policy })
	}
	setDenseGraph := func(policy DenseGraphPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.DenseGraph = policy })
	}
	setExperts := func(policy ExpertPolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Experts = policy })
	}
	updateExperts := func(names []string, apply func(*ExpertPolicy)) {
		update(names, func(profile *ArchitectureProfile) { apply(&profile.Experts) })
	}
	setCadence := func(policy LayerCadencePolicy, names ...string) {
		update(names, func(profile *ArchitectureProfile) { profile.Cadence = policy })
	}
	setExpertCatalog := func(policy expertCatalogPolicy, names ...string) {
		updateExperts(names, func(experts *ExpertPolicy) { experts.Catalog = policy })
	}
	updateDenseStages := func(names []string, apply func(*DenseStagePolicy)) {
		update(names, func(profile *ArchitectureProfile) { apply(&profile.DenseStages) })
	}
	updateDenseWeights := func(names []string, apply func(*DenseWeightPolicy)) {
		update(names, func(profile *ArchitectureProfile) { apply(&profile.DenseWeights) })
	}
	updateRotary := func(names []string, apply func(*RotaryPolicy)) {
		update(names, func(profile *ArchitectureProfile) { apply(&profile.Rotary) })
	}
	updateAttentionGraph := func(names []string, apply func(*AttentionGraphPolicy)) {
		update(names, func(profile *ArchitectureProfile) { apply(&profile.AttentionGraph) })
	}

	setDraftKind(DraftQwen35MTP, "qwen35", "qwen35moe")
	setDraftKind(DraftStep35MTP, "step35")
	setDraftKind(DraftHYV3MTP, "hy_v3")
	setDraftKind(DraftCohere2MTP, "cohere2moe")
	setDraftKind(DraftNextNMTP,
		"glm4", "glm4moe", "exaone4", "exaone-moe", "mimo2", "bailingmoe2", "deepseek32", "glm-dsa",
	)
	setOverrides(EmbeddingOverrideCogVLM, "cogvlm")
	setOverrides(EmbeddingOverrideRawScaled, "gemma3n", "gemma4")
	setOverrides(EmbeddingOverrideDeepstackBase, "granite")
	setDeepstack(DeepstackMappedBefore, "granite")
	setDeepstack(DeepstackSequentialAfter, "qwen3vl", "qwen3vlmoe")
	setAttentionBlocks(AttentionBlocksUncached, "gemma4")
	setAuxiliary(AuxiliaryRWKVValue, "arwkv7", "rwkv7")
	setAuxiliary(AuxiliaryDSATopK, "glm-dsa")
	setTemperature(AttentionTemperatureConfigured, "deepseek2", "mistral3", "mistral4")
	setTemperature(AttentionTemperatureNoRoPE, "llama4")
	setPostNormLayout(PostNormLayoutGrok, "grok")
	setFFNNormLayout(FeedForwardNormLayoutBare, "falcon-h1")
	setFFNNormLayout(FeedForwardNormLayoutAttentionOutput, "dbrx")
	setFFNNormLayout(FeedForwardNormLayoutAttentionPost, "glm4moe")
	setFFNNormLayout(FeedForwardNormLayoutPostAttention, "qwen3next", "qwen35", "qwen35moe", "seed_oss")

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
	setCadence(LayerCadencePolicy{Recurrent: recurrentCadenceAttentionInterval},
		"qwen3next", "qwen35", "qwen35moe")
	setCadence(LayerCadencePolicy{MoE: moeCadenceOffsetOne}, "jina-bert-v3", "nomic-bert-moe")
	setCadence(LayerCadencePolicy{MoE: moeCadenceAfterDense}, "ernie4_5-moe")
	setCadence(LayerCadencePolicy{MoE: moeCadenceEvery, Sliding: slidingCadenceExceptLast}, "llama4")
	setCadence(LayerCadencePolicy{Sliding: slidingCadenceNonRecurrent}, "lfm2", "lfm2moe")
	setCadence(LayerCadencePolicy{Sliding: slidingCadenceExceptFirst}, "laguna", "modern-bert", "smallthinker")
	setCadence(LayerCadencePolicy{Sliding: slidingCadenceExceptLast},
		"afmoe", "cohere2", "cohere2moe", "dflash", "exaone-moe", "exaone4",
		"gemma-embedding", "gemma2", "gemma3", "gemma3n", "gemma4",
		"gemma4-assistant", "gpt-oss", "mellum", "mimo2", "olmo2",
		"plamo3", "step35",
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
	setCapabilities(ArchitectureBERTNormLayout,
		"bert", "jina-bert-v2", "jina-bert-v3", "nomic-bert", "nomic-bert-moe",
	)
	setCapabilities(ArchitectureDeepSeek2Layout, "deepseek2", "deepseek32", "mistral4", "glm-dsa")

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
	setOutputNorm(OutputNormEncoder, "t5encoder", "neo-bert")
	setOutputNorm(OutputNormDecoder, "t5")
	setOutputNorm(OutputNormTokenEmbedding, "lfm2", "lfm2moe")

	setBlock(BlockDeepSeek4, "deepseek4")
	setBlock(BlockMamba, "mamba")
	setBlock(BlockMamba2, "mamba2")
	setBlock(BlockFalconH1, "falcon-h1")
	setBlock(BlockNemotronH, "nemotron_h", "nemotron_h_moe")
	setBlock(BlockKimiLinear, "kimi-linear")
	setRecurrentBlock(BlockJamba, "jamba")
	setRecurrentBlock(BlockGraniteHybrid, "granitehybrid")
	setRecurrentBlock(BlockPLaMo2, "plamo2")

	setCache(CacheDeepSeek4, "deepseek4")
	setCache(CacheFalconH1, "falcon-h1")
	setCache(CacheT5, "t5")
	setCache(CacheRWKV6, "rwkv6")
	setCache(CacheRWKV6Qwen2, "rwkv6qwen2")
	setCache(CacheRWKV7, "rwkv7", "arwkv7")
	setCache(CacheMamba, "mamba")
	setCache(CacheMamba2, "mamba2")
	setRecurrentCache(CacheKimiLinear, "kimi-linear")
	setRecurrentCache(CacheQwenGDN, "qwen3next", "qwen35", "qwen35moe")
	setRecurrentCache(CacheMamba2, "granitehybrid", "nemotron_h", "nemotron_h_moe")
	setRecurrentCache(CacheMamba, "jamba", "plamo2")
	setRecurrentCache(CacheLFM2, "lfm2", "lfm2moe")
	setCacheFallback(CacheFallbackFeedForward, "nemotron_h", "nemotron_h_moe")
	setCacheFallback(CacheFallbackMissingKV, "deci")

	setDenseGraph(DenseGraphBERT, "bert", "jina-bert-v2", "jina-bert-v3", "nomic-bert", "nomic-bert-moe")
	setDenseGraph(DenseGraphModernBERT, "modern-bert")
	setDenseGraph(DenseGraphGemmaEmbedding, "gemma-embedding")
	setDenseGraph(DenseGraphTalkie, "talkie")
	setDenseGraph(DenseGraphGemma4, "gemma4")
	setDenseGraph(DenseGraphGemma3n, "gemma3n")
	setDenseGraph(DenseGraphRWKV6, "rwkv6")
	setDenseGraph(DenseGraphRWKV6Qwen2, "rwkv6qwen2")
	setDenseGraph(DenseGraphRWKV7, "rwkv7", "arwkv7")

	updateDenseStages([]string{"olmo2", "olmoe", "minimax-m2"}, func(policy *DenseStagePolicy) {
		policy.QK.Projection = qkNormWeighted
	})
	updateDenseStages([]string{"mpt"}, func(policy *DenseStagePolicy) {
		policy.QK.Projection = qkNormConfigured
	})
	updateDenseStages([]string{
		"apertus", "afmoe", "bailingmoe2", "dots1", "dflash", "exaone4",
		"exaone-moe", "grovemoe", "hy_v3", "llada-moe", "mellum", "openelm",
		"plamo2", "plamo3", "qwen3", "qwen3moe", "qwen3vl", "qwen3vlmoe",
		"rnd1", "laguna", "lfm2", "lfm2moe", "gemma3",
	}, func(policy *DenseStagePolicy) { policy.QK.Heads = qkNormWeighted })
	updateDenseStages([]string{"glm4moe", "step35"}, func(policy *DenseStagePolicy) {
		policy.QK.Heads = qkNormOptionalWeighted
	})
	updateDenseStages([]string{"command-r"}, func(policy *DenseStagePolicy) {
		policy.QK.Heads = qkNormConfiguredNoBias
		policy.QKHeadsMinBlocks = 64
	})
	updateDenseStages([]string{"chameleon"}, func(policy *DenseStagePolicy) {
		policy.QK.Heads = qkNormAffine
	})
	updateDenseStages([]string{"stablelm"}, func(policy *DenseStagePolicy) {
		policy.QK.Heads = qkNormLayer
	})
	updateDenseStages([]string{"maincoder", "hunyuan-moe", "hunyuan-dense", "hunyuan_vl"}, func(policy *DenseStagePolicy) {
		policy.QK.PostRotary = qkNormWeighted
	})
	updateDenseStages([]string{"llama4"}, func(policy *DenseStagePolicy) {
		policy.PostRotaryRMSNon128 = true
		policy.QueryScale = queryScalePolicyTemperatureWithoutRoPE
	})
	updateDenseStages([]string{"laguna"}, func(policy *DenseStagePolicy) {
		policy.AttentionGate = attentionGateSoftplus
		policy.AttentionHeadGate = true
		policy.AttentionFlatGateElse = true
	})
	updateDenseStages([]string{"afmoe"}, func(policy *DenseStagePolicy) {
		policy.AttentionGate = attentionGateSigmoid
		policy.AttentionFlatGate = true
	})
	updateDenseStages([]string{"step35"}, func(policy *DenseStagePolicy) {
		policy.AttentionGate = attentionGateSigmoid
		policy.AttentionHeadGate = true
	})
	updateDenseStages([]string{"bitnet"}, func(policy *DenseStagePolicy) {
		policy.AttentionSubNorm = true
	})
	updateDenseStages([]string{"mimo2"}, func(policy *DenseStagePolicy) {
		policy.AttentionValueScale = true
	})
	updateDenseStages([]string{"falcon"}, func(policy *DenseStagePolicy) {
		policy.Residual = residualFalcon
	})
	updateDenseStages([]string{"gptneox"}, func(policy *DenseStagePolicy) {
		policy.Residual = residualOriginalNorm
		policy.ResidualParallelOnly = true
	})
	updateDenseStages([]string{"stablelm"}, func(policy *DenseStagePolicy) {
		policy.Residual = residualStable
	})
	updateDenseStages([]string{"gpt-oss"}, func(policy *DenseStagePolicy) {
		policy.Residual = residualGPTOSS
	})
	updateDenseStages([]string{"mistral3"}, func(policy *DenseStagePolicy) {
		policy.QueryScale = queryScalePolicyConfiguredTemperature
	})
	updateDenseStages([]string{"phi2", "phi3", "phimoe"}, func(policy *DenseStagePolicy) {
		policy.QueryScale = queryScalePolicyPreDot
	})
	updateDenseStages([]string{
		"gemma", "gemma2", "gemma3", "gemma3n", "gemma4", "gemma-embedding",
	}, func(policy *DenseStagePolicy) { policy.QueryScale = queryScalePolicyGemma })
	updateDenseStages([]string{"gemma2"}, func(policy *DenseStagePolicy) {
		policy.GemmaSpecial = true
	})
	updateDenseWeights([]string{
		"granitemoe", "granitehybrid", "grok", "ernie4_5-moe", "refact", "granite",
	}, func(policy *DenseWeightPolicy) { policy.AllowUngatedExperts = true })
	updateDenseWeights([]string{"glm4moe", "laguna", "afmoe", "lfm2moe", "minimax-m2"}, func(policy *DenseWeightPolicy) {
		policy.RequireExpertBias = true
	})
	updateDenseWeights([]string{"olmo2"}, func(policy *DenseWeightPolicy) {
		policy.RequirePostNorm = true
	})
	updateDenseWeights([]string{"bitnet"}, func(policy *DenseWeightPolicy) {
		policy.RequireSubNorm = true
	})
	updateDenseWeights([]string{"phimoe", "pangu-embedded", "gpt-oss"}, func(policy *DenseWeightPolicy) {
		policy.RequireAttentionOutputBias = true
	})
	updateDenseWeights([]string{"stablelm", "mpt"}, func(policy *DenseWeightPolicy) {
		policy.ValidateOptionalQKNorm = true
	})
	updateDenseWeights([]string{"falcon"}, func(policy *DenseWeightPolicy) {
		policy.ValidateFalconNorm = true
	})
	updateDenseWeights([]string{"laguna", "afmoe"}, func(policy *DenseWeightPolicy) {
		policy.RequireAttentionGate = true
	})
	updateDenseWeights([]string{"gpt-oss"}, func(policy *DenseWeightPolicy) {
		policy.RequireOpenAIBiases = true
		policy.RequireAttentionSinks = true
		policy.SkipFeedForwardNorm = true
	})
	updateDenseWeights([]string{"stablelm"}, func(policy *DenseWeightPolicy) {
		policy.SkipFeedForwardNorm = true
	})
	updateRotary([]string{"paddleocr", "qwen2vl", "qwen3vl", "qwen3vlmoe"}, func(policy *RotaryPolicy) {
		policy.MultiAxis = multiAxisRotaryAlways
	})
	updateRotary([]string{"glm4", "glm4moe", "hunyuan-dense", "hunyuan_vl"}, func(policy *RotaryPolicy) {
		policy.MultiAxis = multiAxisRotaryWithSections
	})
	updateRotary([]string{"laguna"}, func(policy *RotaryPolicy) {
		policy.Kind = rotaryPolicyLaguna
	})
	updateRotary([]string{"grok", "mellum"}, func(policy *RotaryPolicy) {
		policy.Kind = rotaryPolicyGrokMellum
	})
	updateRotary([]string{"llama", "llama-embed", "minicpm", "mistral3"}, func(policy *RotaryPolicy) {
		policy.Kind = rotaryPolicyLlamaYaRN
	})
	updateRotary([]string{"cohere2", "cohere2moe", "llama4", "gpt-oss"}, func(policy *RotaryPolicy) {
		policy.SlidingFrequency = true
	})
	updateRotary([]string{"afmoe", "exaone-moe", "mimo2", "step35", "smallthinker", "plamo3"}, func(policy *RotaryPolicy) {
		policy.SlidingFrequency = true
	})
	updateRotary([]string{"mellum"}, func(policy *RotaryPolicy) {
		policy.SlidingFrequency = true
	})
	updateRotary([]string{"olmo2", "mellum"}, func(policy *RotaryPolicy) {
		policy.SlidingScaleReset = true
	})
	updateRotary([]string{"step35"}, func(policy *RotaryPolicy) {
		policy.FactorPairs = true
	})
	updateRotary([]string{"gemma3"}, func(policy *RotaryPolicy) {
		policy.Gemma3 = true
	})
	updateAttentionGraph([]string{"mimo2", "gpt-oss"}, func(policy *AttentionGraphPolicy) {
		policy.UseSinks = true
	})
	updateAttentionGraph([]string{"llama4"}, func(policy *AttentionGraphPolicy) {
		policy.ChunkedWindow = true
	})

	setExperts(ExpertPolicy{Composition: expertArctic}, "arctic")
	setExperts(ExpertPolicy{Composition: expertGrok, Activation: tensor.MoEActivationGELU}, "grok")
	setExperts(ExpertPolicy{Composition: expertGrouped}, "grovemoe")
	setExperts(ExpertPolicy{
		Composition: expertSharedAverage, Condition: expertCompositionWithShared,
		Normalization: expertNormalizeMetadata, Routing: expertRouteSigmoid,
	}, "cohere2moe")
	setExperts(ExpertPolicy{
		Composition: expertSharedLimited, Condition: expertCompositionWithShared,
		Normalization: expertNormalizeMetadata, SelectionBias: true, ClampSwiGLU: true,
	}, "step35")
	setExperts(ExpertPolicy{
		Composition: expertSharedGated, Normalization: expertNormalizeNever,
	}, "qwen2moe")
	setExperts(ExpertPolicy{Composition: expertSharedAdd},
		"hunyuan-moe", "llama4",
	)
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Normalization: expertNormalizeMetadata, SelectionBias: true,
	}, "hy_v3", "deepseek2-ocr")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Normalization: expertNormalizeMetadata,
	}, "bailingmoe")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Normalization: expertNormalizeMetadata, SelectionBias: true,
	}, "glm4moe")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Normalization: expertNormalizeNever,
	}, "deepseek")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Condition: expertCompositionUnlessSigmoidWithoutShared,
		Normalization: expertNormalizeMetadata, SelectionBias: true,
	}, "exaone-moe", "bailingmoe2", "lfm2moe", "dots1")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Condition: expertCompositionWithShared, SelectionBias: true,
	}, "ernie4_5-moe")
	setExperts(ExpertPolicy{SelectionBias: true}, "minimax-m2")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Condition: expertCompositionWithShared,
		Normalization: expertNormalizeMetadata, Routing: expertRouteSigmoid, SelectionBias: true,
	}, "laguna", "afmoe")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Condition: expertCompositionWithShared,
		ResidualScale: true,
	}, "granitemoe", "granitehybrid")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Condition: expertCompositionWithExpertsAndShared,
		ResidualScale: true,
	}, "granite")
	setExperts(ExpertPolicy{RouterInputOriginal: true, Activation: tensor.MoEActivationReLU}, "smallthinker")
	setExperts(ExpertPolicy{Normalization: expertNormalizeMetadata}, "jamba")
	setExperts(ExpertPolicy{Normalization: expertNormalizeNever}, "llada-moe", "olmoe")
	setExperts(ExpertPolicy{Routing: expertRouteSigmoid, SelectionBias: true}, "mimo2")
	setExperts(ExpertPolicy{
		Composition: expertSharedAdd, Normalization: expertNormalizeNever,
		Routing: expertRouteSigmoid,
	}, "llama4")
	setExperts(ExpertPolicy{
		Normalization: expertNormalizeNever, Routing: expertRouteSelectedSoftmax,
		Activation: tensor.MoEActivationSwiGLUOAI,
	}, "gpt-oss")
	setExperts(ExpertPolicy{Activation: tensor.MoEActivationGELU}, "gemma4")
	setExpertCatalog(expertCatalogAlways,
		"arctic", "bailingmoe", "dbrx", "grovemoe", "grok", "hunyuan-moe",
		"llada-moe", "mellum", "minimax-m2", "qwen3moe", "qwen3vlmoe",
		"qwen3next", "qwen35moe", "qwen2moe", "olmoe", "phimoe", "rnd1",
		"smallthinker", "granitemoe", "gpt-oss",
	)
	setExpertCatalog(expertCatalogWithExperts,
		"llama", "llama-embed", "mistral3", "refact", "granitehybrid", "granite",
	)
	setExpertCatalog(expertCatalogWithRouter,
		"jamba", "mimo2", "step35", "gemma4", "hy_v3", "llama4",
	)
	setExpertCatalog(expertCatalogAfterDense,
		"glm4moe", "cohere2moe", "dots1", "deepseek", "bailingmoe2", "lfm2moe",
		"afmoe", "laguna", "deepseek2-ocr", "deepseek2", "deepseek32", "mistral4", "glm-dsa",
	)
	setExpertCatalog(expertCatalogInterleaved, "nomic-bert-moe", "ernie4_5-moe", "jina-bert-v3")
	setExpertCatalog(expertCatalogAfterDenseExceptNextN, "exaone-moe")
	updateExperts([]string{
		"cohere2moe", "deepseek2", "deepseek32", "mistral4", "glm-dsa",
		"deepseek2-ocr", "gemma4", "hy_v3", "qwen3next", "qwen35moe",
	}, func(policy *ExpertPolicy) { policy.FusedGateUp = true })
	updateExperts([]string{
		"granitemoe", "granitehybrid", "granite", "grok", "ernie4_5-moe",
		"jina-bert-v3", "nomic-bert-moe", "refact",
	}, func(policy *ExpertPolicy) { policy.OptionalGate = true })
	update([]string{"deci"}, func(profile *ArchitectureProfile) { profile.DeciSparse = true })
	for name, profile := range registry {
		if profile.Has(ArchitectureBERTNormLayout) {
			profile.OutputNorm = OutputNormAbsent
			profile.PostNormLayout = PostNormLayoutBERT
		}
		if profile.Forward == ForwardCached && profile.Has(ArchitectureNonCausal) {
			profile.Forward = ForwardNonCausal
		}
		if profile.Has(ArchitectureNormalRoPE) {
			profile.Position = PositionNormal
			if profile.Rotary.Kind == rotaryPolicyDefault {
				profile.Rotary.Kind = rotaryPolicyNormal
			}
		}
		if profile.Has(ArchitectureGemma) {
			profile.Rotary.Kind = rotaryPolicyGemma
		}
		if profile.Has(ArchitectureParallelResidual) {
			profile.Residual = ResidualParallel
		}
		switch {
		case profile.Has(ArchitectureDSA):
			profile.Attention = AttentionDSA
			profile.Block = BlockDSA
		case profile.Has(ArchitectureMLA):
			profile.Attention = AttentionMLA
			profile.Block = BlockMLA
		case profile.Has(ArchitectureQwenGDN):
			profile.Attention = AttentionQwenGDN
		case profile.Has(ArchitectureLFM2):
			profile.Attention = AttentionLFM2
		}
		registry[name] = profile
	}
	return registry
}
