package model

import (
	"sort"
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
	FeedForwardXIELU
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
	ArchitectureDiscreteImageTokens
)

// ArchitectureProfile: registry entry and capability set.
type ArchitectureProfile struct {
	Name             string
	Family           ArchitectureFamily
	GraphFamily      ArchitectureFamily
	CatalogFamily    ArchitectureFamily
	DraftKind        DraftKind
	Forward          ForwardProgram
	OutputNorm       OutputNormPolicy
	Capabilities     ArchitectureCapability
	Normalization    NormalizationPolicy
	Position         PositionPolicy
	Residual         ResidualPolicy
	FeedForward      FeedForwardPolicy
	Attention        AttentionPolicy
	Overrides        EmbeddingOverridePolicy
	Deepstack        DeepstackPolicy
	AttentionBlocks  AttentionBlockPolicy
	Auxiliary        AuxiliaryFlow
	Temperature      AttentionTemperaturePolicy
	PostNormLayout   PostNormLayoutPolicy
	FFNNormLayout    FeedForwardNormLayoutPolicy
	Cache            CachePolicy
	RecurrentCache   CachePolicy
	CacheFallback    CacheFallbackPolicy
	DenseGraph       DenseGraphPolicy
	DenseStages      DenseStagePolicy
	DenseWeights     DenseWeightPolicy
	ModelCatalog     ModelCatalogPolicy
	Rotary           RotaryPolicy
	AttentionGraph   AttentionGraphPolicy
	Experts          ExpertPolicy
	Metadata         MetadataShapePolicy
	MetadataRead     MetadataReadPolicy
	MetadataDefaults MetadataDefaultPolicy
	Validation       ValidationPolicy
	EncoderGraph     EncoderGraphPolicy
	MLAVariant       mlaVariantPolicy
	Cadence          LayerCadencePolicy
	Runtime          RuntimePolicy
	DeciSparse       bool
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

// ResolvedProfile: bound policy; bootstrap fallback.
func (s Spec) ResolvedProfile() (ArchitectureProfile, bool) {
	if s.profile != nil {
		return *s.profile, s.profile.Name == s.Architecture
	}
	return LookupArchitecture(s.Architecture)
}

// Profile: resolved policy; zero profile for invalid specs.
func (s Spec) Profile() ArchitectureProfile {
	profile, _ := s.ResolvedProfile()
	return profile
}

func (s Spec) withProfile(profile ArchitectureProfile) Spec {
	s.profile = &profile
	return s
}

var architectureRegistry = mustLoadArchitectureRegistry()
