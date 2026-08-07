package model

import (
	"fmt"
	"math"
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
	BlockQwenGDN
)

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
	cacheWrite := CacheWriteFixed
	if cache.PrimaryMode().TokenAligned() {
		cacheWrite = CacheWriteConcatOnly
		if block == BlockDense || profile.Attention == AttentionQwenGDN {
			cacheWrite = CacheWriteConcatOrAppend
		}
	}
	return LayerPlan{
		Layer:             layer,
		GraphFamily:       profile.GraphFamily,
		CatalogFamily:     profile.CatalogFamily,
		Attention:         profile.Attention,
		Position:          profile.Position,
		Residual:          profile.Residual,
		FeedForward:       profile.FeedForward,
		Block:             block,
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
		Experts:           s.moeGraphPlan(layer),
		ExpertComposition: s.expertCompositionPlan(),
		DenseWeights:      s.denseWeightPlan(profile, layer),
		DenseGraph:        profile.DenseGraph,
		DeciSparse: profile.DeciSparse &&
			(s.LayerFeedForwardLength(layer) == 0 || s.LayerHeadCount(layer) == 0 ||
				s.LayerKVHeadCount(layer) == 0),
		QKPreprocess:    s.qkPreprocessPlan(layer),
		QueryScale:      s.queryScalePlan(profile, layer),
		AttentionOutput: s.attentionOutputPlan(normalization),
		ResidualStages:  s.residualStagePlan(profile, normalization),
	}
}

// SupportsCapacityCache: all token caches admit bounded append.
func (p ModelPlan) SupportsCapacityCache() bool {
	if len(p.Layers) == 0 {
		return false
	}
	for _, layer := range p.Layers {
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
	Profile     ArchitectureProfile
	Layers      []LayerPlan
	CacheLayers uint32
	CachedGraph CachedGraphPolicy
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
		Profile: profile, Layers: make([]LayerPlan, layers), CacheLayers: cacheLayers,
	}
	for layer := range layers {
		recurrent := int(layer) < len(weights.Layers) && weights.Layers[layer].Recurrent
		plan.Layers[layer] = spec.PlanLayer(layer, recurrent)
	}
	if err := validateModelPlan(spec, plan); err != nil {
		return ModelPlan{}, err
	}
	plan.CachedGraph = cachedGraphPolicy(profile, plan.Layers)
	return plan, nil
}

func validateModelPlan(spec Spec, plan ModelPlan) error {
	if spec.SharedKVLayers > 0 && (!plan.Profile.Has(ArchitectureSharedKV) ||
		spec.SharedKVLayers >= spec.BlockCount) {
		return fmt.Errorf("model plan architecture %s has invalid shared-KV layer count %d", spec.Architecture, spec.SharedKVLayers)
	}
	producedAuxiliary := make(map[AuxiliaryFlow]bool)
	norm := spec.NormPlan()
	for index, layer := range plan.Layers {
		if layer.Layer != uint32(index) || layer.GraphFamily != plan.Profile.GraphFamily ||
			layer.CatalogFamily != plan.Profile.CatalogFamily {
			return fmt.Errorf("model plan layer %d identity is inconsistent", index)
		}
		if layer.SharedKV {
			if layer.HasKV || layer.KVSource >= layer.Layer || int(layer.KVSource) >= len(plan.Layers) ||
				!plan.Layers[layer.KVSource].HasKV {
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
		switch plan.Profile.Deepstack {
		case DeepstackMappedBefore:
			if len(spec.DeepstackMapping) < len(plan.Layers) {
				return fmt.Errorf("model plan deepstack map has %d entries for %d layers", len(spec.DeepstackMapping), len(plan.Layers))
			}
		case DeepstackSequentialAfter:
			if int(spec.DeepstackLayerCount) > len(plan.Layers) {
				return fmt.Errorf("model plan has %d deepstack streams for %d layers", spec.DeepstackLayerCount, len(plan.Layers))
			}
		default:
			return fmt.Errorf("model plan architecture %s has no deepstack policy", spec.Architecture)
		}
	}
	if plan.Profile.AttentionBlocks != AttentionBlocksNone && !plan.Profile.Has(ArchitectureMultimodal) {
		return fmt.Errorf("model plan architecture %s has attention blocks without multimodal support", spec.Architecture)
	}
	if spec.AttentionTempScale != 0 || spec.AttentionTempFloor != 0 || spec.AttentionTempOffset != 0 {
		if plan.Profile.Temperature == AttentionTemperatureNone || spec.AttentionTempScale <= 0 ||
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
	draft := plan.Profile.DraftPlan(spec.NextNPredictLayers)
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
	for _, layer := range p.Layers {
		if layer.Cache == policy {
			return true
		}
	}
	return false
}

// Layer: bounds-checked layer contract.
func (p ModelPlan) Layer(layer int) (LayerPlan, error) {
	if layer < 0 || layer >= len(p.Layers) {
		return LayerPlan{}, fmt.Errorf("model plan layer %d is outside [0,%d)", layer, len(p.Layers))
	}
	return p.Layers[layer], nil
}

func blockPolicy(profile ArchitectureProfile, recurrent bool) BlockPolicy {
	if profile.Attention == AttentionQwenGDN {
		return BlockQwenGDN
	}
	if recurrent && profile.RecurrentBlock != BlockDense {
		return profile.RecurrentBlock
	}
	return profile.Block
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
