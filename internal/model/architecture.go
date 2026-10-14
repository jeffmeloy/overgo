package model

import (
	"reflect"
	"slices"
)

type DraftKind uint8

const (
	draftNone DraftKind = iota
	draftSingleCatalog
	draftAppendedMultiCarry
	draftAppendedMulti
	DraftAppendedSingle
	draftOptionalSingleCatalog
)

const (
	firstDraftHead uint32 = iota
	singleDraftHeadCount
)

// draftSessionPolicy: runtime coordinator cardinality.
type draftSessionPolicy uint8

const (
	draftSessionNone draftSessionPolicy = iota
	DraftSessionSingle
	DraftSessionMulti
)

// draftNormalizationPolicy: draft projection normalization operator.
type draftNormalizationPolicy uint8

const (
	draftNormalizationWeightedRMS draftNormalizationPolicy = iota
	draftNormalizationArchitecture
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
	Normalization   draftNormalizationPolicy
	Session         draftSessionPolicy
	QueryCopies     uint32
}

// HasHead: bounded catalog head.
func (p DraftPlan) HasHead(offset uint32) bool {
	if p.Kind == draftNone || p.Heads == firstDraftHead || p.SingleCatalog && p.Heads != singleDraftHeadCount {
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
	outputNormModel OutputNormPolicy = iota
	OutputNormAbsent
	outputNormEncoder
	outputNormDecoder
	outputNormTokenEmbedding
)

// NormalizationPolicy: model normalization contract.
type NormalizationPolicy uint8

const (
	NormalizationRMS NormalizationPolicy = iota
	normalizationLayer
	NormalizationUnweightedLayer
	NormalizationUnweightedRMS
	normalizationWeightOnlyLayer
	// NormalizationMAD: mean-absolute-deviation normalization, the
	// constructed (scratch) topology's norm. Registered so constructed
	// components speak the same policy vocabulary as inherited ones.
	NormalizationMAD
)

// PositionPolicy: rotary layout contract.
type PositionPolicy uint8

const (
	positionNeoX PositionPolicy = iota
	PositionNormal
	// PositionLearnedAbsolute: learned absolute position embeddings plus
	// optional lag bias, the constructed topology's position contract.
	PositionLearnedAbsolute
)

// ResidualPolicy: block residual ordering.
type ResidualPolicy uint8

const (
	ResidualSequential ResidualPolicy = iota
	residualParallel
)

// FeedForwardPolicy: dense activation/layout contract.
type FeedForwardPolicy uint8

const (
	feedForwardSwiGLU FeedForwardPolicy = iota
	feedForwardSequentialGELU
	feedForwardFusedGateUp
	feedForwardGELU
	feedForwardSquaredReLU
	feedForwardXIELU
	feedForwardGEGLU
	// FeedForwardReLU: plain gateless ReLU MLP, the constructed topology's
	// feed-forward contract.
	FeedForwardReLU
)

func (p FeedForwardPolicy) separateGate() bool {
	return p == feedForwardSwiGLU || p == feedForwardGEGLU
}

func (p FeedForwardPolicy) upProjectionCopies() uint64 {
	if p == feedForwardFusedGateUp {
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
	attentionStandard AttentionPolicy = iota
	attentionLatent
	AttentionSparseLatent
	AttentionGatedDelta
	attentionShortConvolution
)

// EmbeddingOverridePolicy: projected-embedding application order.
type EmbeddingOverridePolicy uint8

const (
	embeddingOverrideStandard EmbeddingOverridePolicy = iota
	EmbeddingOverrideVisualSpan
	EmbeddingOverrideRawScaled
	EmbeddingOverrideMappedBase
)

// DeepstackPolicy: projected-stream layer placement.
type DeepstackPolicy uint8

const (
	deepstackNone DeepstackPolicy = iota
	deepstackMappedBefore
	deepstackSequentialAfter
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
	auxiliaryRecurrentValue
	AuxiliarySparseTopK
)

// AttentionTemperaturePolicy: scheduled query-scale contract.
type AttentionTemperaturePolicy uint8

const (
	AttentionTemperatureNone AttentionTemperaturePolicy = iota
	AttentionTemperatureConfigured
	attentionTemperatureNoRoPE
)

// PostNormLayoutPolicy: post-attention/FFN tensor namespace.
type PostNormLayoutPolicy uint8

const (
	postNormLayoutStandard PostNormLayoutPolicy = iota
	postNormLayoutOutputLayer
	postNormLayoutGrok
)

// FeedForwardNormLayoutPolicy: pre-FFN tensor namespace.
type FeedForwardNormLayoutPolicy uint8

const (
	feedForwardNormLayoutStandard FeedForwardNormLayoutPolicy = iota
	feedForwardNormLayoutBare
	feedForwardNormLayoutAttentionOutput
	feedForwardNormLayoutAttentionPost
	feedForwardNormLayoutPostAttention
)

// normalizationPlan: compiled operation, placement, bias, and tensor layout.
type normalizationPlan struct {
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
func (p normalizationPlan) PostNormTensors() postNormTensorNames {
	switch p.PostNormLayout {
	case postNormLayoutOutputLayer:
		return postNormTensorNames{
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			AttentionBias: "attn_output_norm.bias", FeedForwardBias: "layer_output_norm.bias",
		}
	case postNormLayoutGrok:
		return postNormTensorNames{
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			FeedForwardAlternate: "ffn_post_norm.weight",
		}
	default:
		return postNormTensorNames{
			AttentionWeight: postAttentionNormWeightTensor, FeedForwardWeight: "post_ffw_norm.weight",
		}
	}
}

// FeedForwardNormTensor: pre-FFN tensor namespace.
func (p normalizationPlan) FeedForwardNormTensor() string {
	switch p.FeedForwardLayout {
	case feedForwardNormLayoutBare:
		return "ffn_norm"
	case feedForwardNormLayoutAttentionOutput:
		return "attn_output_norm.weight"
	case feedForwardNormLayoutAttentionPost:
		return "attn_post_norm.weight"
	case feedForwardNormLayoutPostAttention:
		return postAttentionNormWeightTensor
	default:
		return feedForwardNormWeightTensor
	}
}

// postNormTensorNames: post-norm tensor catalog entry.
type postNormTensorNames struct {
	AttentionWeight      string
	FeedForwardWeight    string
	FeedForwardAlternate string
	AttentionBias        string
	FeedForwardBias      string
}

// ArchitectureCapability: orthogonal runtime behavior.
type ArchitectureCapability uint64

const (
	architectureNonCausal ArchitectureCapability = 1 << iota
	architectureRoPEDisabled
	architectureMoE
	architectureRecurrent
	architectureEncoderOnly
	architectureDiffusion
	architectureMultimodal
	architectureLatentYaRNQuery
	architectureSparseLatent
	architectureLatent
	architectureReservedGEGLU
	architecturePostNorm
	architecturePostOnlyNorm
	architectureNormalRoPE
	architectureParallelResidual
	architectureSequentialGELU
	architectureFusedGateUp
	architectureLongRoPE
	architectureGELU
	architectureSquaredReLU
	architectureGatedDelta
	architectureShortConvolution
	architectureMultiAxisPositions
	architectureRequiresOutput
	architectureClassifierHead
	architectureBiasFreeProjections
	architectureFusedQKV
	architectureRequiresFusedQKV
	architectureRequiresFusedQKVBias
	architectureRejectsOrphanFusedQKVBias
	architectureSharedKV
	architectureReservedAltUp
	architecturePerLayerEmbeddings
	architectureEmbeddingSkip
	architectureOutputLayerNormLayout
	architectureLatentKVLayout
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
	DenseStages               denseStagePolicy
	DenseWeights              denseWeightPolicy
	ModelCatalog              modelCatalogPolicy
	Rotary                    rotaryPolicy
	AttentionGraph            attentionGraphPolicy
	Experts                   expertPolicy
	Metadata                  metadataShapePolicy
	MetadataRead              MetadataReadPolicy
	MetadataDefaults          metadataDefaultPolicy
	Validation                validationPolicy
	EncoderOperator           EncoderOperatorPolicy
	LatentAttention           latentAttentionPolicy
	Cadence                   layerCadencePolicy
	Runtime                   runtimePolicy
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
	case draftSingleCatalog:
		plan.SingleCatalog = true
		plan.SupportsMTPOnly, plan.Session = true, DraftSessionSingle
	case draftAppendedMultiCarry:
		plan.AppendedBlocks, plan.Session = true, DraftSessionMulti
		plan.CarryRawHidden = true
	case draftAppendedMulti:
		plan.AppendedBlocks, plan.Session = true, DraftSessionMulti
	case DraftAppendedSingle:
		plan.AppendedBlocks, plan.Session = true, DraftSessionSingle
	case draftOptionalSingleCatalog:
		plan.SingleCatalog, plan.OptionalCatalog = true, true
		plan.SupportsMTPOnly, plan.Session = true, DraftSessionSingle
		plan.Normalization, plan.ScaleLogits = draftNormalizationArchitecture, true
	}
	return plan
}

// OutputNormTensor: final normalization tensor name.
func (p ArchitectureProfile) OutputNormTensor() string {
	switch p.OutputNorm {
	case OutputNormAbsent:
		return ""
	case outputNormEncoder:
		return "enc.output_norm.weight"
	case outputNormDecoder:
		return "dec.output_norm.weight"
	case outputNormTokenEmbedding:
		return "token_embd_norm.weight"
	default:
		return outputNormWeightTensor
	}
}

// PostNormTensors: post-norm tensor namespace.
func (p ArchitectureProfile) PostNormTensors() postNormTensorNames {
	return normalizationPlan{PostNormLayout: p.PostNormLayout}.PostNormTensors()
}

// FeedForwardNormTensor: pre-FFN tensor namespace.
func (p ArchitectureProfile) FeedForwardNormTensor() string {
	return normalizationPlan{FeedForwardLayout: p.FFNNormLayout}.FeedForwardNormTensor()
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
	slices.Sort(names)
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
