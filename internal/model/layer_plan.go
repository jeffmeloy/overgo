package model

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
)

// BlockPolicy: compiled block-builder selection.
type BlockPolicy uint8

const (
	BlockDense BlockPolicy = iota
	BlockMamba
	BlockMamba2
	BlockFalconH1
	BlockJamba
	BlockGraniteHybrid
	BlockPLaMo2
	BlockNemotronH
	BlockKimiLinear
	BlockMLA
	BlockDSA
	BlockDeepSeek4
)

// RuntimeCacheBinding: indexed cache operand.
type RuntimeCacheBinding uint8

const (
	RuntimeCacheNone RuntimeCacheBinding = iota
	RuntimeCachePrimaryKey
	RuntimeCachePrimaryValue
	RuntimeCacheConvolution
	RuntimeCacheSSM
	RuntimeCacheIndexerKey
)

// RuntimeTensorBinding: indexed auxiliary tensor operand.
type RuntimeTensorBinding uint8

const (
	RuntimeTensorNone RuntimeTensorBinding = iota
	RuntimeTensorPerLayerInput
	RuntimeTensorCurrentPositions
)

// LayerOperator: semantic execution stage.
type LayerOperator uint8

const (
	LayerOperatorNone LayerOperator = iota
	LayerOperatorDenseAttention
	LayerOperatorDenseFeedForward
	LayerOperatorLinearAttention
	LayerOperatorLatentAttention
	LayerOperatorHyperAttention
	LayerOperatorHyperFeedForward
	LayerOperatorAttentionNorm
	LayerOperatorAttentionPostNorm
	LayerOperatorAttentionMix
	LayerOperatorHybridMix
	LayerOperatorRecurrentMix
	LayerOperatorFeedForwardNorm
	LayerOperatorFeedForwardMix
	LayerOperatorFeedForwardPostNorm
	LayerOperatorCacheSentinel
	LayerOperatorScale
	LayerOperatorResidual
	LayerOperatorRMSNorm
	LayerOperatorScaledSkip
	LayerOperatorOutputAdapter
	LayerOperatorTokenShiftMix
	LayerOperatorPeriodicScale
	LayerOperatorAttentionResidualNorm
	LayerOperatorInputResidualNorm
	LayerOperatorFeedForwardResidualNorm
)

// RecurrentMixPolicy: recurrent operator implementation.
type RecurrentMixPolicy uint8

const (
	RecurrentMixNone RecurrentMixPolicy = iota
	RecurrentMixMamba
	RecurrentMixMamba2
	RecurrentMixPLaMo2
	RecurrentMixGatedDelta
	RecurrentMixShortConvolution
	RecurrentMixDynamicWKV6
	RecurrentMixAffineWKV6
	RecurrentMixDynamicWKV7
)

// AttentionMixPolicy: attention operator implementation.
type AttentionMixPolicy uint8

const (
	AttentionMixNone AttentionMixPolicy = iota
	AttentionMixCausalProjection
	AttentionMixGatedProjection
	AttentionMixOutputProjection
	AttentionMixBidirectionalFusedQKV
	AttentionMixBidirectionalQKNorm
	AttentionMixCausalPostQKNorm
	AttentionMixSharedKVQKNorm
	AttentionMixBidirectionalEncoder
)

// HybridMixPolicy: parallel mixer implementation.
type HybridMixPolicy uint8

const (
	HybridMixNone HybridMixPolicy = iota
	HybridMixAttentionSSM
)

// FeedForwardMixPolicy: feed-forward operator implementation.
type FeedForwardMixPolicy uint8

const (
	FeedForwardMixNone FeedForwardMixPolicy = iota
	FeedForwardMixStandardSwiGLU
	FeedForwardMixFusedGLU
	FeedForwardMixSquaredReLU
	FeedForwardMixRoutedSquaredReLU
	FeedForwardMixRoutedSwiGLU
	FeedForwardMixGatedGELU
	FeedForwardMixParallelGatedGELU
	FeedForwardMixGatedTokenShiftSquaredReLU
	FeedForwardMixTokenShiftSquaredReLU
	FeedForwardMixEncoder
)

const (
	maxLayerCacheBindings  = 4
	maxLayerTensorBindings = 2
	maxLayerInstructions   = 8
)

// LayerOperatorInstruction: compiled operator and operand indexes.
type LayerOperatorInstruction struct {
	Operator               LayerOperator
	Attention              AttentionMixPolicy
	Hybrid                 HybridMixPolicy
	Recurrent              RecurrentMixPolicy
	FeedForward            FeedForwardMixPolicy
	RequireConvolutionBias bool
	CacheCount             uint8
	TensorCount            uint8
	Caches                 [maxLayerCacheBindings]RuntimeCacheBinding
	Tensors                [maxLayerTensorBindings]RuntimeTensorBinding
}

// LayerProgram: ordered fixed-capacity layer instructions.
type LayerProgram struct {
	Count        uint8
	Instructions [maxLayerInstructions]LayerOperatorInstruction
}

// LayerCompositionPolicy: semantic block-stage composition.
type LayerCompositionPolicy uint8

const (
	LayerCompositionStandard LayerCompositionPolicy = iota
	LayerCompositionRecurrentOnly
	LayerCompositionAttentionOnly
	LayerCompositionFeedForwardOnly
	LayerCompositionIdentity
)

func (p LayerProgram) Instruction(index int) (LayerOperatorInstruction, bool) {
	if index < 0 || index >= int(p.Count) || index >= len(p.Instructions) {
		return LayerOperatorInstruction{}, false
	}
	return p.Instructions[index], true
}

// CachePolicy: compiled layer-state layout.
type CachePolicy uint8

const (
	CacheAttention CachePolicy = iota
	CacheSentinel
	CacheMamba
	CacheMamba2
	CacheRWKV6
	CacheRWKV6Qwen2
	CacheRWKV7
	CacheKimiLinear
	CacheQwenGDN
	CacheLFM2
	CacheFalconH1
	CacheT5
	CacheDeepSeek4
)

func (p CachePolicy) PrimaryMode() CacheStateMode {
	switch p {
	case CacheMamba, CacheMamba2, CacheRWKV6, CacheRWKV6Qwen2, CacheRWKV7,
		CacheKimiLinear, CacheQwenGDN, CacheLFM2:
		return CacheStateFixed
	default:
		return CacheStateToken
	}
}

// CacheFallbackPolicy: non-primary per-layer cache selection.
type CacheFallbackPolicy uint8

const (
	CacheFallbackNone CacheFallbackPolicy = iota
	CacheFallbackFeedForward
	CacheFallbackMissingKV
)

// CacheWritePolicy: compiled cache graph capabilities.
type CacheWritePolicy uint8

const (
	CacheWriteFixed CacheWritePolicy = iota
	CacheWriteConcatOnly
	CacheWriteConcatOrAppend
)

// CachedGraphPolicy: cached decoder graph composition.
type CachedGraphPolicy uint8

const (
	CachedGraphLayered CachedGraphPolicy = iota
	CachedGraphDense
)

// DenseGraphPolicy: dense-family leaf graph.
type DenseGraphPolicy uint8

const (
	DenseGraphStandard DenseGraphPolicy = iota
	DenseGraphBERT
	DenseGraphModernBERT
	DenseGraphGemmaEmbedding
	DenseGraphTalkie
	DenseGraphGemma4
	DenseGraphGemma3n
	DenseGraphRWKV6
	DenseGraphRWKV6Qwen2
	DenseGraphRWKV7
)

// OutputHeadPolicy: compiled terminal projection source.
type OutputHeadPolicy uint8

const (
	OutputHeadTokenEmbedding OutputHeadPolicy = iota
	OutputHeadDedicated
)

// TerminalPlan: compiled final normalization and projection.
type TerminalPlan struct {
	Normalization OutputNormPolicy
	OutputHead    OutputHeadPolicy
}

// DeepstackSource: compiled projected-stream source.
type DeepstackSource int32

const (
	DeepstackSourceNone DeepstackSource = -2
	DeepstackSourceBase DeepstackSource = -1
)

// LayerPlan: derived layer execution contract.
type LayerPlan struct {
	Layer             uint32
	GraphFamily       ArchitectureFamily
	CatalogFamily     ArchitectureFamily
	Block             BlockPolicy
	Program           LayerProgram
	Composition       LayerCompositionPolicy
	Cache             CachePolicy
	Attention         AttentionPolicy
	Position          PositionPolicy
	Residual          ResidualPolicy
	FeedForward       FeedForwardPolicy
	CacheMode         CacheStateMode
	CacheWrite        CacheWritePolicy
	Recurrent         bool
	Sliding           bool
	UsesRoPE          bool
	MultiAxis         bool
	HasKV             bool
	SharedKV          bool
	KVSource          uint32
	DeepstackBefore   DeepstackSource
	DeepstackAfter    DeepstackSource
	AuxiliaryInput    AuxiliaryFlow
	AuxiliaryOutput   AuxiliaryFlow
	Temperature       AttentionTemperaturePolicy
	AttentionBlocks   AttentionBlockPolicy
	EmbeddingSkip     bool
	PerLayerInput     bool
	Normalization     NormalizationPlan
	Rotary            RotaryPlan
	AttentionGraph    AttentionGraphPlan
	Experts           MoEGraphPlan
	ExpertComposition ExpertCompositionPlan
	DenseWeights      DenseWeightPlan
	DenseGraph        DenseGraphPolicy
	DeciSparse        bool
	QKPreprocess      QKPreprocessPlan
	QueryScale        QueryScalePlan
	AttentionOutput   AttentionOutputPlan
	ResidualStages    ResidualStagePlan
}

// PlanLayer: derives graph and cache behavior once per layer.
func (s Spec) PlanLayer(layer uint32, recurrent bool) LayerPlan {
	profile := s.Profile()
	recurrent = recurrent || s.IsRecurrentLayer(layer)
	hasKV := s.LayerHasKV(layer)
	sharedKV := profile.Has(ArchitectureSharedKV) && !hasKV
	var kvSource uint32
	if sharedKV {
		kvSource = s.LayerSharedKVSource(layer)
	}
	deepstackBefore, deepstackAfter := deepstackSources(s, profile, layer)
	auxiliaryInput, auxiliaryOutput := auxiliaryFlow(s, profile, layer)
	temperature := profile.Temperature
	if temperature == AttentionTemperatureNoRoPE && s.UsesRoPE(layer) {
		temperature = AttentionTemperatureNone
	}
	normalization := s.NormPlan()
	cache := cachePolicy(s, profile, layer, recurrent)
	block := blockPolicy(profile, recurrent)
	composition := LayerCompositionStandard
	if block == BlockNemotronH {
		switch {
		case recurrent:
			composition = LayerCompositionRecurrentOnly
		case s.LayerFeedForwardLength(layer) > 0:
			composition = LayerCompositionFeedForwardOnly
		default:
			composition = LayerCompositionAttentionOnly
		}
	}
	deciSparse := profile.DeciSparse &&
		(s.LayerFeedForwardLength(layer) == 0 || s.LayerHeadCount(layer) == 0 ||
			s.LayerKVHeadCount(layer) == 0)
	if deciSparse {
		switch {
		case s.LayerFeedForwardLength(layer) == 0:
			composition = LayerCompositionIdentity
		case s.LayerHeadCount(layer) == 0:
			composition = LayerCompositionFeedForwardOnly
		}
	}
	residualStages := s.residualStagePlan(profile, normalization)
	if profile.DenseGraph == DenseGraphRWKV6Qwen2 {
		residualStages.residualScale = 0
		if s.RescaleEvery > 0 && (layer+1)%s.RescaleEvery == 0 {
			residualStages.residualScale = rwkvLayerRescale
		}
	}
	experts := s.moeGraphPlan(layer)
	if block == BlockNemotronH {
		experts.Routing = tensor.MoERoutingSigmoid
		experts.Activation = tensor.MoEActivationReLUSquared
		experts.SelectionBias = true
	}
	if profile.Attention == AttentionQwenGDN {
		experts.NormalizeTopKProb = true
	}
	if profile.Attention == AttentionLFM2 && recurrent {
		experts.NormalizeTopKProb = true
		experts.SelectionBias = true
	}
	cacheWrite := CacheWriteFixed
	if cache.PrimaryMode().TokenAligned() {
		cacheWrite = CacheWriteConcatOnly
		if block == BlockDense || profile.Attention == AttentionQwenGDN {
			cacheWrite = CacheWriteConcatOrAppend
		}
	}
	plan := LayerPlan{
		Layer:             layer,
		GraphFamily:       profile.GraphFamily,
		CatalogFamily:     profile.CatalogFamily,
		Attention:         profile.Attention,
		Position:          profile.Position,
		Residual:          profile.Residual,
		FeedForward:       profile.FeedForward,
		Block:             block,
		Composition:       composition,
		Cache:             cache,
		CacheMode:         cache.PrimaryMode(),
		CacheWrite:        cacheWrite,
		Recurrent:         recurrent,
		Sliding:           s.IsSlidingLayer(layer),
		UsesRoPE:          s.UsesRoPE(layer),
		MultiAxis:         profile.Has(ArchitectureMultiAxisPositions),
		HasKV:             hasKV,
		SharedKV:          sharedKV,
		KVSource:          kvSource,
		DeepstackBefore:   deepstackBefore,
		DeepstackAfter:    deepstackAfter,
		AuxiliaryInput:    auxiliaryInput,
		AuxiliaryOutput:   auxiliaryOutput,
		Temperature:       temperature,
		AttentionBlocks:   profile.AttentionBlocks,
		EmbeddingSkip:     profile.Has(ArchitectureEmbeddingSkip),
		PerLayerInput:     profile.Has(ArchitecturePerLayerEmbeddings) && s.EmbeddingPerLayer > 0,
		Normalization:     normalization,
		Rotary:            s.rotaryPlan(profile, layer),
		AttentionGraph:    s.attentionGraphPlan(layer),
		Experts:           experts,
		ExpertComposition: s.expertCompositionPlan(),
		DenseWeights:      s.denseWeightPlan(profile, layer),
		DenseGraph:        profile.DenseGraph,
		DeciSparse:        deciSparse,
		QKPreprocess:      s.qkPreprocessPlan(layer),
		QueryScale:        s.queryScalePlan(profile, layer),
		AttentionOutput:   s.attentionOutputPlan(normalization),
		ResidualStages:    residualStages,
	}
	plan.Program = compileLayerProgram(plan, profile)
	return plan
}

// SupportsCapacityCache: all token caches admit bounded append.
func (p ModelPlan) SupportsCapacityCache() bool {
	if len(p.layers) == 0 {
		return false
	}
	for _, layer := range p.layers {
		if layer.CacheMode.TokenAligned() &&
			(layer.CacheWrite != CacheWriteConcatOrAppend || layer.SharedKV) {
			return false
		}
	}
	return true
}

// SupportsMultiAxisPositions: model-level MRoPE contract.
func (s Spec) SupportsMultiAxisPositions() bool {
	return s.SupportsMultiAxisPositionsWithProfile(s.Profile())
}

// SupportsMultiAxisPositionsWithProfile: bound-profile MRoPE contract.
func (s Spec) SupportsMultiAxisPositionsWithProfile(profile ArchitectureProfile) bool {
	if !profile.Has(ArchitectureMultiAxisPositions) {
		return false
	}
	sections := false
	for _, count := range s.RopeSections {
		sections = sections || count > 0
	}
	if !sections {
		return false
	}
	return profile.Rotary.MultiAxis != multiAxisRotaryWithSections ||
		len(s.RopeSections) >= 2 && s.RopeSections[0] > 0 && s.RopeSections[1] > 0
}

func deepstackSources(s Spec, profile ArchitectureProfile, layer uint32) (DeepstackSource, DeepstackSource) {
	switch profile.Deepstack {
	case DeepstackMappedBefore:
		if layer == 0 || int(layer) >= len(s.DeepstackMapping) {
			return DeepstackSourceNone, DeepstackSourceNone
		}
		source := s.DeepstackMapping[layer]
		switch {
		case source < 0:
			return DeepstackSourceNone, DeepstackSourceNone
		case source == 0:
			return DeepstackSourceBase, DeepstackSourceNone
		default:
			return DeepstackSource(source - 1), DeepstackSourceNone
		}
	case DeepstackSequentialAfter:
		if layer < s.DeepstackLayerCount {
			return DeepstackSourceNone, DeepstackSource(layer)
		}
	}
	return DeepstackSourceNone, DeepstackSourceNone
}

func auxiliaryFlow(s Spec, profile ArchitectureProfile, layer uint32) (AuxiliaryFlow, AuxiliaryFlow) {
	switch profile.Auxiliary {
	case AuxiliaryRWKVValue:
		if layer == 0 {
			return AuxiliaryNone, AuxiliaryRWKVValue
		}
		return AuxiliaryRWKVValue, AuxiliaryNone
	case AuxiliaryDSATopK:
		if s.LayerHasFullIndexer(layer) {
			return AuxiliaryNone, AuxiliaryDSATopK
		}
		return AuxiliaryDSATopK, AuxiliaryDSATopK
	default:
		return AuxiliaryNone, AuxiliaryNone
	}
}

// ModelPlan: immutable model and per-layer execution contract.
type ModelPlan struct {
	profile      ArchitectureProfile
	layers       []LayerPlan
	draftLayers  []LayerPlan
	cacheSchemas []LayerCacheSchema
	cacheLayers  uint32
	cachedGraph  CachedGraphPolicy
	terminal     TerminalPlan
	draft        DraftPlan
}

// CompileModelPlan: resolves architecture decisions before execution.
func CompileModelPlan(spec Spec, weights Weights) (ModelPlan, error) {
	profile := spec.Profile()
	if profile.Name == "" {
		return ModelPlan{}, &UnsupportedArchitectureError{Architecture: spec.Architecture}
	}
	return CompileModelPlanWithProfile(spec, weights, profile)
}

// CompileModelPlanWithProfile: compiles a resolved policy without registry lookup.
func CompileModelPlanWithProfile(spec Spec, weights Weights, profile ArchitectureProfile) (ModelPlan, error) {
	if err := ValidateArchitectureProfile(profile); err != nil {
		return ModelPlan{}, err
	}
	if profile.Name == "" || profile.Name != spec.Architecture {
		return ModelPlan{}, fmt.Errorf(
			"model plan profile %q does not match architecture %q", profile.Name, spec.Architecture,
		)
	}
	if profile.Forward == ForwardCached && spec.NonCausalAttention {
		profile.Forward = ForwardNonCausal
	}
	spec = spec.withProfile(profile)
	layers := spec.BlockCount
	if profile.Family == ArchitectureFamilyEncoderDecoder && spec.DecoderBlockCount > layers {
		layers = spec.DecoderBlockCount
	}
	cacheLayers := spec.BlockCount
	if profile.Family == ArchitectureFamilyEncoderDecoder {
		cacheLayers = spec.DecoderBlockCount
	}
	plan := ModelPlan{
		profile: profile, layers: make([]LayerPlan, layers), cacheLayers: cacheLayers,
		terminal: TerminalPlan{Normalization: profile.OutputNorm},
		draft:    profile.DraftPlan(spec.NextNPredictLayers),
	}
	if weights.Output != nil {
		plan.terminal.OutputHead = OutputHeadDedicated
	}
	for layer := range layers {
		recurrent := int(layer) < len(weights.Layers) && weights.Layers[layer].Recurrent
		plan.layers[layer] = spec.PlanLayer(layer, recurrent)
	}
	cacheSchemas, cacheErr := compileCacheSchemas(spec, plan.layers, weights.Layers)
	if cacheErr != nil {
		if weights.TokenEmbedding.Name != "" {
			return ModelPlan{}, cacheErr
		}
	} else {
		plan.cacheSchemas = cacheSchemas
	}
	if plan.draft.AppendedBlocks {
		plan.draftLayers = make([]LayerPlan, plan.draft.Heads)
		executable := spec
		if plan.draft.Kind == DraftNextNMTP {
			executable.BlockCount += plan.draft.Heads
		}
		for offset := range plan.draftLayers {
			plan.draftLayers[offset] = executable.PlanLayer(spec.BlockCount+uint32(offset), false)
		}
	} else if plan.draft.Kind == DraftQwen35MTP && plan.draft.SessionEligible() {
		executable := qwen35MTPExecutableSpec(spec)
		plan.draftLayers = []LayerPlan{executable.PlanLayer(0, false)}
	}
	if err := validateModelPlan(spec, weights, plan); err != nil {
		return ModelPlan{}, err
	}
	plan.cachedGraph = cachedGraphPolicy(profile, plan.layers)
	return plan, nil
}

func validateModelPlan(spec Spec, weights Weights, plan ModelPlan) error {
	if plan.terminal.OutputHead > OutputHeadDedicated ||
		plan.terminal.Normalization != plan.profile.OutputNorm ||
		(plan.terminal.OutputHead == OutputHeadDedicated) != (weights.Output != nil) {
		return fmt.Errorf("model plan architecture %s has invalid terminal policy", spec.Architecture)
	}
	if plan.draft != plan.profile.DraftPlan(spec.NextNPredictLayers) {
		return fmt.Errorf("model plan architecture %s has invalid draft policy", spec.Architecture)
	}
	wantDraftLayers := 0
	if plan.draft.AppendedBlocks {
		wantDraftLayers = int(plan.draft.Heads)
	} else if plan.draft.Kind == DraftQwen35MTP && plan.draft.SessionEligible() {
		wantDraftLayers = 1
	}
	if len(plan.draftLayers) != wantDraftLayers {
		return fmt.Errorf("model plan architecture %s has invalid draft layers", spec.Architecture)
	}
	if len(plan.cacheSchemas) != 0 && len(plan.cacheSchemas) != len(plan.layers) {
		return fmt.Errorf("model plan architecture %s has invalid cache schemas", spec.Architecture)
	}
	for offset, layer := range plan.draftLayers {
		wantLayer := spec.BlockCount + uint32(offset)
		if plan.draft.Kind == DraftQwen35MTP {
			wantLayer = 0
		}
		if layer.Layer != wantLayer {
			return fmt.Errorf("model plan draft layer %d identity is inconsistent", offset)
		}
	}
	if spec.SharedKVLayers > 0 && (!plan.profile.Has(ArchitectureSharedKV) ||
		spec.SharedKVLayers >= spec.BlockCount) {
		return fmt.Errorf("model plan architecture %s has invalid shared-KV layer count %d", spec.Architecture, spec.SharedKVLayers)
	}
	producedAuxiliary := make(map[AuxiliaryFlow]bool)
	norm := spec.NormPlan()
	for index, layer := range plan.layers {
		if layer.Layer != uint32(index) || layer.GraphFamily != plan.profile.GraphFamily ||
			layer.CatalogFamily != plan.profile.CatalogFamily {
			return fmt.Errorf("model plan layer %d identity is inconsistent", index)
		}
		if layer.Program != compileLayerProgram(layer, plan.profile) {
			return fmt.Errorf("model plan layer %d operator program is inconsistent", index)
		}
		if layer.SharedKV {
			if layer.HasKV || layer.KVSource >= layer.Layer || int(layer.KVSource) >= len(plan.layers) ||
				!plan.layers[layer.KVSource].HasKV {
				return fmt.Errorf("model plan layer %d shared-KV source %d is invalid", index, layer.KVSource)
			}
		}
		for _, source := range []DeepstackSource{layer.DeepstackBefore, layer.DeepstackAfter} {
			if source >= 0 && uint32(source) >= spec.DeepstackLayerCount {
				return fmt.Errorf("model plan layer %d deepstack source %d exceeds %d streams", index, source, spec.DeepstackLayerCount)
			}
		}
		if layer.AuxiliaryInput != AuxiliaryNone && len(spec.IndexerFullLayers) > 0 &&
			!producedAuxiliary[layer.AuxiliaryInput] {
			return fmt.Errorf("model plan layer %d consumes auxiliary flow %d before production", index, layer.AuxiliaryInput)
		}
		if layer.AuxiliaryOutput != AuxiliaryNone {
			producedAuxiliary[layer.AuxiliaryOutput] = true
		}
		if layer.Cache == CacheDeepSeek4 && len(spec.CompressRatios) <= index {
			return fmt.Errorf("model plan layer %d has no DeepSeek4 compression ratio", index)
		}
		if layer.Normalization != norm {
			return fmt.Errorf("model plan layer %d normalization drifted from model policy", index)
		}
	}
	if spec.DeepstackLayerCount > 0 {
		switch plan.profile.Deepstack {
		case DeepstackMappedBefore:
			if len(spec.DeepstackMapping) < len(plan.layers) {
				return fmt.Errorf("model plan deepstack map has %d entries for %d layers", len(spec.DeepstackMapping), len(plan.layers))
			}
		case DeepstackSequentialAfter:
			if int(spec.DeepstackLayerCount) > len(plan.layers) {
				return fmt.Errorf("model plan has %d deepstack streams for %d layers", spec.DeepstackLayerCount, len(plan.layers))
			}
		default:
			return fmt.Errorf("model plan architecture %s has no deepstack policy", spec.Architecture)
		}
	}
	if plan.profile.AttentionBlocks != AttentionBlocksNone && !plan.profile.Has(ArchitectureMultimodal) {
		return fmt.Errorf("model plan architecture %s has attention blocks without multimodal support", spec.Architecture)
	}
	if spec.AttentionTempScale != 0 || spec.AttentionTempFloor != 0 || spec.AttentionTempOffset != 0 {
		if plan.profile.Temperature == AttentionTemperatureNone || spec.AttentionTempScale <= 0 ||
			spec.AttentionTempFloor == 0 || math.IsNaN(float64(spec.AttentionTempScale)) ||
			math.IsInf(float64(spec.AttentionTempScale), 0) || math.IsNaN(float64(spec.AttentionTempOffset)) ||
			math.IsInf(float64(spec.AttentionTempOffset), 0) {
			return fmt.Errorf("model plan architecture %s has an invalid attention temperature contract", spec.Architecture)
		}
	}
	if norm.PostNormLayout == PostNormLayoutBERT &&
		(norm.Operation != NormalizationLayer || norm.PreAttention || !norm.PostAttention || !norm.Bias) {
		return fmt.Errorf("model plan architecture %s has an invalid BERT normalization layout", spec.Architecture)
	}
	draft := plan.profile.DraftPlan(spec.NextNPredictLayers)
	if spec.NextNPredictLayers > 0 && (draft.Kind == DraftNone || !draft.HasHead(0)) {
		return fmt.Errorf("model plan architecture %s has no draft policy for %d heads", spec.Architecture, spec.NextNPredictLayers)
	}
	return nil
}

func cachedGraphPolicy(profile ArchitectureProfile, layers []LayerPlan) CachedGraphPolicy {
	if len(layers) == 0 || profile.Has(ArchitectureAltUp) ||
		profile.GraphFamily != ArchitectureFamilyAttention &&
			profile.GraphFamily != ArchitectureFamilyMoE {
		return CachedGraphLayered
	}
	for _, layer := range layers {
		if layer.Block != BlockDense || layer.Attention != AttentionStandard ||
			layer.Cache != CacheAttention && layer.Cache != CacheSentinel {
			return CachedGraphLayered
		}
	}
	return CachedGraphDense
}

// HasCache: reports a compiled cache policy.
func (p ModelPlan) HasCache(policy CachePolicy) bool {
	for _, layer := range p.layers {
		if layer.Cache == policy {
			return true
		}
	}
	return false
}

// Layer: bounds-checked layer contract.
func (p ModelPlan) Layer(layer int) (LayerPlan, error) {
	if layer < 0 || layer >= len(p.layers) {
		return LayerPlan{}, fmt.Errorf("model plan layer %d is outside [0,%d)", layer, len(p.layers))
	}
	return p.layers[layer], nil
}

// Profile: compiled architecture policy.
func (p ModelPlan) Profile() ArchitectureProfile { return p.profile }

// LayerCount: compiled trunk layer count.
func (p ModelPlan) LayerCount() int { return len(p.layers) }

// Layers: owned trunk layer-program copy.
func (p ModelPlan) Layers() []LayerPlan { return append([]LayerPlan(nil), p.layers...) }

// DraftLayer: bounds-checked appended draft contract.
func (p ModelPlan) DraftLayer(offset uint32) (LayerPlan, error) {
	if int(offset) >= len(p.draftLayers) {
		return LayerPlan{}, fmt.Errorf("model plan draft layer %d is unavailable", offset)
	}
	return p.draftLayers[offset], nil
}

// CacheSchema: materialized compiled layer-cache contract.
func (p ModelPlan) CacheSchema(layer int, tokens uint32) (LayerCacheSchema, error) {
	if layer < 0 || layer >= len(p.cacheSchemas) {
		return LayerCacheSchema{}, fmt.Errorf("model plan cache schema %d is unavailable", layer)
	}
	return p.cacheSchemas[layer].WithTokenCount(tokens), nil
}

// HasCacheSchemas: physical cache layout compiled.
func (p ModelPlan) HasCacheSchemas() bool {
	return len(p.layers) != 0 && len(p.cacheSchemas) == len(p.layers)
}

// CacheLayerCount: compiled cache-layer count.
func (p ModelPlan) CacheLayerCount() uint32 { return p.cacheLayers }

// CachedGraph: compiled cached graph policy.
func (p ModelPlan) CachedGraph() CachedGraphPolicy { return p.cachedGraph }

// Terminal: compiled output stages.
func (p ModelPlan) Terminal() TerminalPlan { return p.terminal }

// Draft: compiled speculative policy.
func (p ModelPlan) Draft() DraftPlan { return p.draft }

func blockPolicy(profile ArchitectureProfile, recurrent bool) BlockPolicy {
	if recurrent && profile.RecurrentBlock != BlockDense {
		return profile.RecurrentBlock
	}
	return profile.Block
}

func compileLayerProgram(plan LayerPlan, profile ArchitectureProfile) LayerProgram {
	block := plan.Block
	recurrent := plan.Recurrent
	composition := plan.Composition
	if profile.Attention == AttentionLFM2 && recurrent {
		return residualMixerProgram(
			recurrentLayerStage(RecurrentMixShortConvolution, false), FeedForwardMixStandardSwiGLU, false,
		)
	}
	if profile.DenseGraph == DenseGraphRWKV6Qwen2 {
		return residualMixerProgram(
			recurrentLayerStage(RecurrentMixDynamicWKV6, false), FeedForwardMixStandardSwiGLU,
			plan.ResidualStages.residualScale > 0,
		)
	}
	if profile.Attention == AttentionQwenGDN {
		mixer := attentionLayerStage(AttentionMixGatedProjection)
		if recurrent {
			mixer = recurrentLayerStage(RecurrentMixGatedDelta, false)
		}
		return residualMixerProgram(mixer, FeedForwardMixRoutedSwiGLU, false)
	}
	if block == BlockFalconH1 {
		return residualMixerProgram(
			hybridLayerStage(HybridMixAttentionSSM), FeedForwardMixStandardSwiGLU, false,
		)
	}
	if block == BlockGraniteHybrid {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(RecurrentMixMamba2, false),
			layerStage(LayerOperatorScale), layerStage(LayerOperatorResidual),
			layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(FeedForwardMixStandardSwiGLU),
			layerStage(LayerOperatorScale), layerStage(LayerOperatorResidual),
		)
	}
	if block == BlockJamba {
		return residualMixerProgram(
			recurrentLayerStage(RecurrentMixMamba, false), FeedForwardMixStandardSwiGLU, false,
		)
	}
	if block == BlockPLaMo2 {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(RecurrentMixPLaMo2, false),
			layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
			layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(FeedForwardMixFusedGLU),
			layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidual),
		)
	}
	if block == BlockNemotronH {
		switch composition {
		case LayerCompositionRecurrentOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(RecurrentMixMamba2, false),
				layerStage(LayerOperatorResidual),
			)
		case LayerCompositionAttentionOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(AttentionMixCausalProjection),
				layerStage(LayerOperatorResidual),
			)
		case LayerCompositionFeedForwardOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), cacheSentinelLayerStage(),
				feedForwardLayerStage(FeedForwardMixRoutedSquaredReLU), layerStage(LayerOperatorResidual),
			)
		default:
			return LayerProgram{}
		}
	}
	if block == BlockMamba || block == BlockMamba2 {
		recurrentPolicy := RecurrentMixMamba
		if block == BlockMamba2 {
			recurrentPolicy = RecurrentMixMamba2
		}
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm),
			recurrentLayerStage(recurrentPolicy, block == BlockMamba2),
			layerStage(LayerOperatorResidual),
		)
	}
	switch block {
	case BlockDense:
		if profile.DenseGraph == DenseGraphRWKV6 {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(RecurrentMixAffineWKV6, false),
				layerStage(LayerOperatorResidual), tokenShiftLayerStage(FeedForwardMixGatedTokenShiftSquaredReLU),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorPeriodicScale),
			)
		}
		if profile.DenseGraph == DenseGraphRWKV7 {
			stages := []LayerOperatorInstruction{
				layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(RecurrentMixDynamicWKV7, false),
				layerStage(LayerOperatorResidual),
			}
			if profile.Normalization == NormalizationLayer {
				stages = append(stages, tokenShiftLayerStage(FeedForwardMixTokenShiftSquaredReLU))
			} else {
				stages = append(stages, layerStage(LayerOperatorFeedForwardNorm),
					feedForwardLayerStage(FeedForwardMixStandardSwiGLU))
			}
			return newLayerProgram(append(stages, layerStage(LayerOperatorResidual))...)
		}
		if profile.DenseGraph == DenseGraphGemma4 {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(AttentionMixSharedKVQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				feedForwardLayerStage(FeedForwardMixParallelGatedGELU), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorOutputAdapter),
			)
		}
		if profile.DenseGraph == DenseGraphTalkie {
			return newLayerProgram(
				layerStage(LayerOperatorRMSNorm), attentionLayerStage(AttentionMixCausalPostQKNorm),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorRMSNorm),
				feedForwardLayerStage(FeedForwardMixStandardSwiGLU), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorScaledSkip),
			)
		}
		if profile.DenseGraph == DenseGraphGemmaEmbedding {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(AttentionMixBidirectionalQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(FeedForwardMixGatedGELU),
				layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidual),
			)
		}
		if profile.DenseGraph == DenseGraphModernBERT {
			return newLayerProgram(
				attentionLayerStage(AttentionMixBidirectionalFusedQKV), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(FeedForwardMixFusedGLU),
				layerStage(LayerOperatorResidual),
			)
		}
		if profile.DenseGraph == DenseGraphBERT {
			stages := []LayerOperatorInstruction{
				attentionLayerStage(AttentionMixBidirectionalEncoder),
				layerStage(LayerOperatorAttentionResidualNorm),
			}
			if profile.EncoderGraph.Kind == encoderGraphJinaV2 {
				stages = append(stages, layerStage(LayerOperatorInputResidualNorm))
			}
			return newLayerProgram(append(stages,
				feedForwardLayerStage(FeedForwardMixEncoder),
				layerStage(LayerOperatorFeedForwardResidualNorm),
			)...)
		}
		if plan.DeciSparse {
			stages := []LayerOperatorInstruction{cacheSentinelLayerStage()}
			if plan.Composition == LayerCompositionIdentity {
				return newLayerProgram(stages...)
			}
			if plan.Composition != LayerCompositionFeedForwardOnly {
				stages = append(stages, layerStage(LayerOperatorAttentionNorm),
					attentionLayerStageWithoutCache(AttentionMixOutputProjection),
					layerStage(LayerOperatorResidual))
			}
			return newLayerProgram(append(stages,
				layerStage(LayerOperatorFeedForwardNorm),
				feedForwardLayerStage(FeedForwardMixStandardSwiGLU),
				layerStage(LayerOperatorResidual),
			)...)
		}
		if profile.DenseGraph == DenseGraphStandard && !plan.DeciSparse {
			return newLayerProgram(
				leafLayerStage(
					LayerOperatorDenseAttention,
					[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
				),
				layerStage(LayerOperatorDenseFeedForward),
			)
		}
		return LayerProgram{}
	case BlockKimiLinear:
		if !recurrent {
			return latentLayerProgram(
				profile,
				[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
			)
		}
		return residualMixerProgram(
			leafLayerStage(
				LayerOperatorLinearAttention,
				[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
			),
			FeedForwardMixStandardSwiGLU, false,
		)
	case BlockMLA:
		return latentLayerProgram(
			profile,
			[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
		)
	case BlockDSA:
		return latentLayerProgram(
			profile,
			[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue, RuntimeCacheIndexerKey},
			[]RuntimeTensorBinding{RuntimeTensorPerLayerInput},
		)
	case BlockDeepSeek4:
		return newLayerProgram(
			leafLayerStage(
				LayerOperatorHyperAttention,
				[]RuntimeCacheBinding{RuntimeCachePrimaryKey},
				[]RuntimeTensorBinding{RuntimeTensorCurrentPositions},
			),
			layerStage(LayerOperatorHyperFeedForward),
		)
	default:
		return LayerProgram{}
	}
}

func residualMixerProgram(mixer LayerOperatorInstruction, feedForward FeedForwardMixPolicy, scale bool) LayerProgram {
	stages := []LayerOperatorInstruction{
		layerStage(LayerOperatorAttentionNorm), mixer, layerStage(LayerOperatorResidual),
		layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(feedForward), layerStage(LayerOperatorResidual),
	}
	if scale {
		stages = append(stages, layerStage(LayerOperatorScale))
	}
	return newLayerProgram(stages...)
}

func latentLayerProgram(
	profile ArchitectureProfile,
	caches []RuntimeCacheBinding,
	tensors []RuntimeTensorBinding,
) LayerProgram {
	mix := FeedForwardMixStandardSwiGLU
	if profile.FeedForward == FeedForwardSquaredReLU {
		mix = FeedForwardMixSquaredReLU
	}
	stages := []LayerOperatorInstruction{
		layerStage(LayerOperatorAttentionNorm),
		leafLayerStage(LayerOperatorLatentAttention, caches, tensors),
		layerStage(LayerOperatorResidual), layerStage(LayerOperatorFeedForwardNorm),
		feedForwardLayerStage(mix),
	}
	if profile.MLAVariant == mlaVariantMiniCPM3 {
		stages = append(stages, layerStage(LayerOperatorScale))
	}
	return newLayerProgram(append(stages, layerStage(LayerOperatorResidual))...)
}

func leafLayerProgram(
	operator LayerOperator,
	caches []RuntimeCacheBinding,
	tensors []RuntimeTensorBinding,
) LayerProgram {
	return newLayerProgram(leafLayerStage(operator, caches, tensors))
}

func leafLayerStage(
	operator LayerOperator,
	caches []RuntimeCacheBinding,
	tensors []RuntimeTensorBinding,
) LayerOperatorInstruction {
	instruction := layerStage(operator)
	instruction.CacheCount = uint8(len(caches))
	instruction.TensorCount = uint8(len(tensors))
	copy(instruction.Caches[:], caches)
	copy(instruction.Tensors[:], tensors)
	return instruction
}

func layerStage(operator LayerOperator) LayerOperatorInstruction {
	return LayerOperatorInstruction{Operator: operator}
}

func recurrentLayerStage(
	policy RecurrentMixPolicy,
	requireConvolutionBias bool,
) LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorRecurrentMix)
	instruction.Recurrent = policy
	instruction.RequireConvolutionBias = requireConvolutionBias
	instruction.CacheCount = 2
	instruction.Caches[0] = RuntimeCachePrimaryKey
	instruction.Caches[1] = RuntimeCachePrimaryValue
	return instruction
}

func attentionLayerStage(policy AttentionMixPolicy) LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorAttentionMix)
	instruction.Attention = policy
	instruction.CacheCount = 2
	instruction.Caches[0] = RuntimeCachePrimaryKey
	instruction.Caches[1] = RuntimeCachePrimaryValue
	return instruction
}

func attentionLayerStageWithoutCache(policy AttentionMixPolicy) LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorAttentionMix)
	instruction.Attention = policy
	return instruction
}

func hybridLayerStage(policy HybridMixPolicy) LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorHybridMix)
	instruction.Hybrid = policy
	instruction.CacheCount = 4
	instruction.Caches[0] = RuntimeCachePrimaryKey
	instruction.Caches[1] = RuntimeCachePrimaryValue
	instruction.Caches[2] = RuntimeCacheConvolution
	instruction.Caches[3] = RuntimeCacheSSM
	return instruction
}

func feedForwardLayerStage(policy FeedForwardMixPolicy) LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorFeedForwardMix)
	instruction.FeedForward = policy
	return instruction
}

func tokenShiftLayerStage(policy FeedForwardMixPolicy) LayerOperatorInstruction {
	instruction := feedForwardLayerStage(policy)
	instruction.Operator = LayerOperatorTokenShiftMix
	instruction.CacheCount = 1
	instruction.Caches[0] = RuntimeCachePrimaryKey
	return instruction
}

func cacheSentinelLayerStage() LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorCacheSentinel)
	instruction.CacheCount = 2
	instruction.Caches[0] = RuntimeCachePrimaryKey
	instruction.Caches[1] = RuntimeCachePrimaryValue
	return instruction
}

func newLayerProgram(stages ...LayerOperatorInstruction) LayerProgram {
	if len(stages) > maxLayerInstructions {
		return LayerProgram{}
	}
	program := LayerProgram{Count: uint8(len(stages))}
	copy(program.Instructions[:], stages)
	return program
}

func cachePolicy(spec Spec, profile ArchitectureProfile, layer uint32, recurrent bool) CachePolicy {
	if recurrent && profile.RecurrentCache != CacheAttention {
		return profile.RecurrentCache
	}
	if profile.Cache != CacheAttention {
		return profile.Cache
	}
	if profile.CacheFallback == CacheFallbackFeedForward && spec.LayerFeedForwardLength(layer) > 0 ||
		profile.CacheFallback == CacheFallbackMissingKV && spec.LayerKVHeadCount(layer) == 0 {
		return CacheSentinel
	}
	return CacheAttention
}
