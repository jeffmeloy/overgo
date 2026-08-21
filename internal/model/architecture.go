package model

import (
	"reflect"
	"sort"
)

type DraftKind uint8

const (
	DraftNone DraftKind = iota
	DraftSingleCatalog
	DraftAppendedMultiCarry
	DraftAppendedMulti
	DraftAppendedSingle
	DraftOptionalSingleCatalog
)

const (
	firstDraftHead uint32 = iota
	singleDraftHeadCount
)

// DraftSessionPolicy: runtime coordinator cardinality.
type DraftSessionPolicy uint8

const (
	DraftSessionNone DraftSessionPolicy = iota
	DraftSessionSingle
	DraftSessionMulti
)

// DraftNormalizationPolicy: draft projection normalization operator.
type DraftNormalizationPolicy uint8

const (
	DraftNormalizationWeightedRMS DraftNormalizationPolicy = iota
	DraftNormalizationArchitecture
)

// DraftPlan: compiled catalog and session contract.
type DraftPlan struct {
	Kind            DraftKind
	Heads           uint32
	AppendedBlocks  bool
	SingleCatalog   bool
	OptionalCatalog bool
	SupportsMTPOnly bool
	CarryRawHidden  bool
	ScaleLogits     bool
	Normalization   DraftNormalizationPolicy
	Session         DraftSessionPolicy
	QueryCopies     uint32
}

// HasHead: bounded catalog head.
func (p DraftPlan) HasHead(offset uint32) bool {
	if p.Kind == DraftNone || p.Heads == firstDraftHead || p.SingleCatalog && p.Heads != singleDraftHeadCount {
		return false
	}
	return offset < p.Heads
}

// SessionEligible: supported runtime coordinator cardinality.
func (p DraftPlan) SessionEligible() bool {
	if !p.HasHead(firstDraftHead) {
		return false
	}
	return p.Session != DraftSessionSingle || p.Heads == singleDraftHeadCount
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
	// NormalizationMAD: mean-absolute-deviation normalization, the
	// constructed (scratch) topology's norm. Registered so constructed
	// components speak the same policy vocabulary as inherited ones.
	NormalizationMAD
)

// PositionPolicy: rotary layout contract.
type PositionPolicy uint8

const (
	PositionNeoX PositionPolicy = iota
	PositionNormal
	// PositionLearnedAbsolute: learned absolute position embeddings plus
	// optional lag bias, the constructed topology's position contract.
	PositionLearnedAbsolute
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
	FeedForwardGEGLU
	// FeedForwardReLU: plain gateless ReLU MLP, the constructed topology's
	// feed-forward contract.
	FeedForwardReLU
)

func (p FeedForwardPolicy) separateGate() bool {
	return p == FeedForwardSwiGLU || p == FeedForwardGEGLU
}

func (p FeedForwardPolicy) upProjectionCopies() uint64 {
	if p == FeedForwardFusedGateUp {
		return fusedGateUpProjectionCopies
	}
	return separateProjectionCopies
}

const (
	separateProjectionCopies    = uint64(1)
	fusedGateUpProjectionCopies = 2 * separateProjectionCopies
)

// AttentionPolicy: primary attention implementation.
type AttentionPolicy uint8

const (
	AttentionStandard AttentionPolicy = iota
	AttentionLatent
	AttentionSparseLatent
	AttentionGatedDelta
	AttentionShortConvolution
)

// EmbeddingOverridePolicy: projected-embedding application order.
type EmbeddingOverridePolicy uint8

const (
	EmbeddingOverrideStandard EmbeddingOverridePolicy = iota
	EmbeddingOverrideVisualSpan
	EmbeddingOverrideRawScaled
	EmbeddingOverrideMappedBase
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
	AuxiliaryRecurrentValue
	AuxiliarySparseTopK
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
	PostNormLayoutOutputLayer
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
	RMSBias           bool
	Epsilon           float32
	PostNormLayout    PostNormLayoutPolicy
	FeedForwardLayout FeedForwardNormLayoutPolicy
}

// PostNormTensors: post-norm tensor namespace.
func (p NormalizationPlan) PostNormTensors() PostNormTensorNames {
	switch p.PostNormLayout {
	case PostNormLayoutOutputLayer:
		return PostNormTensorNames{
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			AttentionBias: "attn_output_norm.bias", FeedForwardBias: "layer_output_norm.bias",
		}
	case PostNormLayoutGrok:
		return PostNormTensorNames{
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			FeedForwardAlternate: "ffn_post_norm.weight",
		}
	default:
		return PostNormTensorNames{
			AttentionWeight: postAttentionNormWeightTensor, FeedForwardWeight: "post_ffw_norm.weight",
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
		return postAttentionNormWeightTensor
	default:
		return feedForwardNormWeightTensor
	}
}

// PostNormTensorNames: post-norm tensor catalog entry.
type PostNormTensorNames struct {
	AttentionWeight      string
	FeedForwardWeight    string
	FeedForwardAlternate string
	AttentionBias        string
	FeedForwardBias      string
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
	ArchitectureLatentYaRNQuery
	ArchitectureSparseLatent
	ArchitectureLatent
	architectureReservedGEGLU
	ArchitecturePostNorm
	ArchitecturePostOnlyNorm
	ArchitectureNormalRoPE
	ArchitectureParallelResidual
	ArchitectureSequentialGELU
	ArchitectureFusedGateUp
	ArchitectureLongRoPE
	ArchitectureGELU
	ArchitectureSquaredReLU
	ArchitectureGatedDelta
	ArchitectureShortConvolution
	ArchitectureMultiAxisPositions
	ArchitectureRequiresOutput
	ArchitectureClassifierHead
	ArchitectureBiasFreeProjections
	ArchitectureFusedQKV
	ArchitectureRequiresFusedQKV
	ArchitectureRequiresFusedQKVBias
	ArchitectureRejectsOrphanFusedQKVBias
	ArchitectureSharedKV
	architectureReservedAltUp
	ArchitecturePerLayerEmbeddings
	ArchitectureEmbeddingSkip
	ArchitectureOutputLayerNormLayout
	ArchitectureLatentKVLayout
	ArchitectureDiscreteImageTokens
)

// ArchitectureProfile: registry entry and capability set.
type ArchitectureProfile struct {
	Name                      string
	DraftKind                 DraftKind
	DraftQueryCopies          uint32
	Forward                   ForwardProgram
	OutputNorm                OutputNormPolicy
	Capabilities              ArchitectureCapability
	Normalization             NormalizationPolicy
	NormalizationFromMetadata bool
	Position                  PositionPolicy
	Residual                  ResidualPolicy
	FeedForward               FeedForwardPolicy
	Attention                 AttentionPolicy
	Overrides                 EmbeddingOverridePolicy
	Deepstack                 DeepstackPolicy
	AttentionBlocks           AttentionBlockPolicy
	Auxiliary                 AuxiliaryFlow
	Temperature               AttentionTemperaturePolicy
	PostNormLayout            PostNormLayoutPolicy
	FFNNormLayout             FeedForwardNormLayoutPolicy
	Cache                     CachePolicy
	RecurrentCache            CachePolicy
	RecurrentMixer            RecurrentMixerPolicy
	CacheFallback             CacheFallbackPolicy
	LayerTopology             LayerTopologyPolicy
	DenseStages               DenseStagePolicy
	DenseWeights              DenseWeightPolicy
	ModelCatalog              ModelCatalogPolicy
	Rotary                    RotaryPolicy
	AttentionGraph            AttentionGraphPolicy
	Experts                   ExpertPolicy
	Metadata                  MetadataShapePolicy
	MetadataRead              MetadataReadPolicy
	MetadataDefaults          MetadataDefaultPolicy
	Validation                ValidationPolicy
	EncoderOperator           EncoderOperatorPolicy
	LatentAttention           latentAttentionPolicy
	Cadence                   LayerCadencePolicy
	Runtime                   RuntimePolicy
	DeciSparse                bool
}

// Has: capability predicate.
func (p ArchitectureProfile) Has(capability ArchitectureCapability) bool {
	return p.Capabilities&capability != 0
}

func (p ArchitectureProfile) clone() ArchitectureProfile {
	p.Forward = p.Forward.clone()
	return p
}

func (p ArchitectureProfile) equal(other ArchitectureProfile) bool {
	if !p.Forward.equal(other.Forward) {
		return false
	}
	p.Forward, other.Forward = ForwardProgram{}, ForwardProgram{}
	return reflect.DeepEqual(p, other)
}

// DraftPlan: architecture draft catalog/session descriptor.
func (p ArchitectureProfile) DraftPlan(heads uint32) DraftPlan {
	plan := DraftPlan{Kind: p.DraftKind, Heads: heads, QueryCopies: p.DraftQueryCopies}
	switch p.DraftKind {
	case DraftSingleCatalog:
		plan.SingleCatalog = true
		plan.SupportsMTPOnly, plan.Session = true, DraftSessionSingle
	case DraftAppendedMultiCarry:
		plan.AppendedBlocks, plan.Session = true, DraftSessionMulti
		plan.CarryRawHidden = true
	case DraftAppendedMulti:
		plan.AppendedBlocks, plan.Session = true, DraftSessionMulti
	case DraftAppendedSingle:
		plan.AppendedBlocks, plan.Session = true, DraftSessionSingle
	case DraftOptionalSingleCatalog:
		plan.SingleCatalog, plan.OptionalCatalog = true, true
		plan.SupportsMTPOnly, plan.Session = true, DraftSessionSingle
		plan.Normalization, plan.ScaleLogits = DraftNormalizationArchitecture, true
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
		return outputNormWeightTensor
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
	return profile.clone(), ok
}

// SupportedArchitectures: sorted registered names.
func SupportedArchitectures() []string {
	names := make([]string, len(architectureRegistry))
	var index int
	for name := range architectureRegistry {
		names[index] = name
		index++
	}
	sort.Strings(names)
	return names
}

func (s Spec) boundProfile() (ArchitectureProfile, bool) {
	if s.profile == nil {
		return ArchitectureProfile{}, false
	}
	return s.profile.clone(), s.profile.Name == s.Architecture
}

// Profile: bound policy; zero profile for invalid specs.
func (s Spec) Profile() ArchitectureProfile {
	profile, _ := s.boundProfile()
	return profile
}

func (s Spec) withProfile(profile ArchitectureProfile) Spec {
	profile = profile.clone()
	if profile.NormalizationFromMetadata &&
		!positiveFinite(s.LayerNormEpsilon) && positiveFinite(s.RMSNormEpsilon) {
		profile.Normalization = NormalizationRMS
	}
	s.profile = &profile
	return s
}

var architectureRegistry = mustLoadArchitectureRegistry()
