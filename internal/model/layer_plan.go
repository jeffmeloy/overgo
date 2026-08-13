package model

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
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
	LayerOperatorAttentionInputNorm
	LayerOperatorLatentAttention
	LayerOperatorHyperAttention
	LayerOperatorHyperFeedForward
	LayerOperatorAttentionNorm
	LayerOperatorAttentionPostNorm
	LayerOperatorAttentionCausalProjection
	LayerOperatorAttentionGatedProjection
	LayerOperatorAttentionOutputProjection
	LayerOperatorAttentionBidirectionalFusedQKV
	LayerOperatorAttentionBidirectionalQKNorm
	LayerOperatorAttentionCausalPostQKNorm
	LayerOperatorAttentionSharedKVQKNorm
	LayerOperatorAttentionBidirectionalEncoder
	LayerOperatorAttentionPairedCausalProjection
	LayerOperatorAttentionSharedCacheQKNorm
	LayerOperatorAttentionPlannedProjection
	LayerOperatorHybridMix
	LayerOperatorRecurrentMix
	LayerOperatorFeedForwardNorm
	LayerOperatorFeedForwardStandardSwiGLU
	LayerOperatorFeedForwardFusedGLU
	LayerOperatorFeedForwardSquaredReLU
	LayerOperatorFeedForwardRoutedSquaredReLU
	LayerOperatorFeedForwardRoutedSwiGLU
	LayerOperatorFeedForwardGatedGELU
	LayerOperatorFeedForwardParallelGatedGELU
	LayerOperatorFeedForwardEncoder
	LayerOperatorFeedForwardPlanned
	LayerOperatorFeedForwardPostNorm
	LayerOperatorCacheSentinel
	LayerOperatorScale
	LayerOperatorResidual
	LayerOperatorRMSNorm
	LayerOperatorScaledSkip
	LayerOperatorOutputAdapter
	LayerOperatorGatedTokenShiftSquaredReLU
	LayerOperatorTokenShiftSquaredReLU
	LayerOperatorPeriodicScale
	LayerOperatorAttentionResidualNorm
	LayerOperatorInputResidualNorm
	LayerOperatorFeedForwardResidualNorm
	LayerOperatorPairedInputNorm
	LayerOperatorResidualScale
	LayerOperatorFeedForwardInputNorm
	LayerOperatorFeedForwardOutput
)

const (
	maxLayerCacheBindings  = 4
	maxLayerTensorBindings = 2
	maxLayerInstructions   = 8
)

// LayerOperatorInstruction: compiled operator and operand indexes.
type LayerOperatorInstruction struct {
	Operator    LayerOperator
	CacheCount  uint8
	TensorCount uint8
	Caches      [maxLayerCacheBindings]RuntimeCacheBinding
	Tensors     [maxLayerTensorBindings]RuntimeTensorBinding
}

// LayerProgram: ordered fixed-capacity layer instructions.
type LayerProgram struct {
	Count        uint8
	Instructions [maxLayerInstructions]LayerOperatorInstruction
}

func (p LayerProgram) valid() bool {
	return int(p.Count) <= len(p.Instructions)
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
	StateSpace        StateSpacePlan
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
	stateSpace := s.stateSpacePlan(layer, recurrent)
	composition := LayerCompositionStandard
	if stateSpace.kind == stateSpaceNemotronH {
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
	if stateSpace.kind == stateSpaceNemotronH {
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
		if stateSpace.kind == stateSpaceNone || profile.Attention == AttentionQwenGDN {
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
		StateSpace:        stateSpace,
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
	spec         Spec
	profile      ArchitectureProfile
	layers       []LayerPlan
	draftLayers  []LayerPlan
	cacheSchemas []LayerCacheSchema
	cacheLayers  uint32
	cachedGraph  CachedGraphPolicy
	terminal     TerminalPlan
	draft        DraftPlan
	cacheProject CacheProjectionProgram
	projections  [projectionRoleCount]ProjectionProgram
	sequenceOut  SequenceOutputProgram
	forward      ForwardProgram
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
		spec: spec, profile: profile, layers: make([]LayerPlan, layers), cacheLayers: cacheLayers,
		terminal:     TerminalPlan{Normalization: profile.OutputNorm},
		draft:        profile.DraftPlan(spec.NextNPredictLayers),
		cacheProject: compileCacheProjectionProgram(spec, profile),
		projections:  compileProjectionPrograms(spec, profile),
		sequenceOut:  compileSequenceOutputProgram(spec, profile),
		forward:      resolveForwardProgram(profile.Forward, spec.NonCausalAttention),
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
	} else if plan.draft.SingleCatalog && plan.draft.SessionEligible() &&
		(plan.draft.Kind != DraftCohere2MTP || weights.Cohere2MTP != nil) {
		executable, layer := singleDraftExecutableSpec(spec, plan.draft.Kind)
		plan.draftLayers = []LayerPlan{executable.PlanLayer(layer, false)}
	}
	if err := validateModelPlan(spec, weights, plan); err != nil {
		return ModelPlan{}, err
	}
	plan.cachedGraph = cachedGraphPolicy(profile, plan.layers)
	return plan, nil
}

func validateModelPlan(spec Spec, weights Weights, plan ModelPlan) error {
	if !plan.forward.valid() || plan.forward != resolveForwardProgram(plan.profile.Forward, spec.NonCausalAttention) {
		return fmt.Errorf("model plan architecture %s has invalid forward program", spec.Architecture)
	}
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
	} else if plan.draft.SingleCatalog && plan.draft.SessionEligible() &&
		(plan.draft.Kind != DraftCohere2MTP || weights.Cohere2MTP != nil) {
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
		if !layer.Program.valid() || layer.Program.Count == 0 {
			return fmt.Errorf(
				"model plan draft layer %d operator program has %d instructions; capacity is %d",
				offset, layer.Program.Count, len(layer.Program.Instructions),
			)
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
		if !layer.Program.valid() || layer.Program.Count == 0 && !plan.profile.Has(ArchitectureAltUp) {
			return fmt.Errorf(
				"model plan layer %d operator program has %d instructions; capacity is %d",
				index, layer.Program.Count, len(layer.Program.Instructions),
			)
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
		if layer.StateSpace != spec.stateSpacePlan(uint32(index), layer.Recurrent) {
			return fmt.Errorf("model plan layer %d state-space policy is inconsistent", index)
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
		if layer.StateSpace.kind != stateSpaceNone || layer.Attention != AttentionStandard ||
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

// Forward: compiled top-level execution contract.
func (p ModelPlan) Forward() ForwardProgram { return p.forward }

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

// CompiledLayerProgram: sealed executable layer contract.
type CompiledLayerProgram struct {
	spec  Spec
	plan  LayerPlan
	role  layerProgramRole
	draft DraftPlan
}

type layerProgramRole uint8

const (
	programLayer layerProgramRole = iota
	programEncoder
	programDecoder
)

// Layer returns the immutable compiled layer facts.
func (p CompiledLayerProgram) Layer() LayerPlan { return p.plan }

// Spec returns the program-owned model facts.
func (p CompiledLayerProgram) Spec() Spec { return p.spec }

// LayerProgram binds one trunk layer to its validated model spec.
func (p ModelPlan) LayerProgram(layer int) (CompiledLayerProgram, error) {
	if p.profile.Name == "" || p.spec.Architecture != p.profile.Name {
		return CompiledLayerProgram{}, fmt.Errorf(
			"model plan profile %q does not match layer spec %q", p.profile.Name, p.spec.Architecture,
		)
	}
	compiled, err := p.Layer(layer)
	if err != nil {
		return CompiledLayerProgram{}, err
	}
	return CompiledLayerProgram{spec: p.spec, plan: compiled}, nil
}

func (p ModelPlan) sequenceProgram(layer int, role layerProgramRole) (CompiledLayerProgram, error) {
	encoder := p.profile.EncoderGraph.Kind
	limit := p.spec.BlockCount
	if role == programEncoder && encoder != encoderGraphT5 && encoder != encoderGraphT5Encoder {
		return CompiledLayerProgram{}, fmt.Errorf("model plan has no T5 encoder program for %q", p.spec.Architecture)
	}
	if role == programDecoder {
		if p.profile.Family != ArchitectureFamilyEncoderDecoder || p.profile.Forward.Session != ForwardSessionEncoderDecoder {
			return CompiledLayerProgram{}, fmt.Errorf("model plan has no decoder program for %q", p.spec.Architecture)
		}
		limit = p.spec.DecoderBlockCount
	}
	if layer < 0 || uint32(layer) >= limit {
		return CompiledLayerProgram{}, fmt.Errorf("model plan sequence layer %d is outside [0,%d)", layer, limit)
	}
	program, err := p.LayerProgram(layer)
	program.role = role
	return program, err
}

// EncoderProgram returns one compiled encoder layer.
func (p ModelPlan) EncoderProgram(layer int) (CompiledLayerProgram, error) {
	return p.sequenceProgram(layer, programEncoder)
}

// DecoderProgram returns one compiled causal/cross-attention decoder layer.
func (p ModelPlan) DecoderProgram(layer int) (CompiledLayerProgram, error) {
	return p.sequenceProgram(layer, programDecoder)
}

// DraftProgram: bounds-checked executable draft contract.
func (p ModelPlan) DraftProgram(offset uint32) (CompiledLayerProgram, error) {
	if p.profile.Name == "" || p.spec.Architecture != p.profile.Name {
		return CompiledLayerProgram{}, fmt.Errorf(
			"model plan profile %q does not match draft spec %q", p.profile.Name, p.spec.Architecture,
		)
	}
	layer, err := p.DraftLayer(offset)
	if err != nil {
		return CompiledLayerProgram{}, err
	}
	spec := p.spec
	if p.draft.SingleCatalog {
		spec, _ = singleDraftExecutableSpec(spec, p.draft.Kind)
	} else if p.draft.Kind == DraftNextNMTP {
		spec.BlockCount += p.draft.Heads
	}
	return CompiledLayerProgram{spec: spec, plan: layer, draft: p.draft}, nil
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

// Projection: compiled auxiliary projection program; absent roles fail on Build.
func (p ModelPlan) Projection(role ProjectionRole) ProjectionProgram {
	if role >= projectionRoleCount {
		return ProjectionProgram{}
	}
	return p.projections[role]
}

// CacheProjection: compiled auxiliary K/V projection program.
func (p ModelPlan) CacheProjection() CacheProjectionProgram { return p.cacheProject }

// SequenceOutput: compiled terminal sequence program.
func (p ModelPlan) SequenceOutput() SequenceOutputProgram { return p.sequenceOut }

// Draft: compiled speculative policy.
func (p ModelPlan) Draft() DraftPlan { return p.draft }

func compileLayerProgram(plan LayerPlan, profile ArchitectureProfile) LayerProgram {
	recurrent := plan.Recurrent
	composition := plan.Composition
	if profile.Attention == AttentionLFM2 && recurrent {
		return residualMixerProgram(
			recurrentLayerStage(), LayerOperatorFeedForwardStandardSwiGLU, false,
		)
	}
	if profile.DenseGraph == DenseGraphRWKV6Qwen2 {
		return residualMixerProgram(
			recurrentLayerStage(), LayerOperatorFeedForwardStandardSwiGLU,
			plan.ResidualStages.residualScale > 0,
		)
	}
	if profile.Attention == AttentionQwenGDN {
		mixer := attentionLayerStage(LayerOperatorAttentionGatedProjection)
		if recurrent {
			mixer = recurrentLayerStage()
		}
		return residualMixerProgram(mixer, LayerOperatorFeedForwardRoutedSwiGLU, false)
	}
	if plan.StateSpace.kind == stateSpaceFalconH1 {
		return residualMixerProgram(
			hybridLayerStage(), LayerOperatorFeedForwardStandardSwiGLU, false,
		)
	}
	if plan.StateSpace.kind == stateSpaceGraniteHybrid {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(),
			layerStage(LayerOperatorScale), layerStage(LayerOperatorResidual),
			layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(LayerOperatorFeedForwardStandardSwiGLU),
			layerStage(LayerOperatorScale), layerStage(LayerOperatorResidual),
		)
	}
	if plan.StateSpace.kind == stateSpaceJamba {
		return residualMixerProgram(
			recurrentLayerStage(), LayerOperatorFeedForwardStandardSwiGLU, false,
		)
	}
	if plan.StateSpace.kind == stateSpacePLaMo2 {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(),
			layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
			layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(LayerOperatorFeedForwardFusedGLU),
			layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidual),
		)
	}
	if plan.StateSpace.kind == stateSpaceNemotronH {
		switch composition {
		case LayerCompositionRecurrentOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(),
				layerStage(LayerOperatorResidual),
			)
		case LayerCompositionAttentionOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionCausalProjection),
				layerStage(LayerOperatorResidual),
			)
		case LayerCompositionFeedForwardOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), cacheSentinelLayerStage(),
				feedForwardLayerStage(LayerOperatorFeedForwardRoutedSquaredReLU), layerStage(LayerOperatorResidual),
			)
		default:
			return LayerProgram{}
		}
	}
	if plan.StateSpace.kind == stateSpaceMamba || plan.StateSpace.kind == stateSpaceMamba2 {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm),
			recurrentLayerStage(),
			layerStage(LayerOperatorResidual),
		)
	}
	switch {
	default:
		if profile.Forward.Session == ForwardSessionPairedProjection {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionSharedCacheQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(LayerOperatorFeedForwardGatedGELU),
				layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidualScale),
			)
		}
		if profile.Forward.Session == ForwardSessionFeatureDraft {
			return newLayerProgram(
				pairedInputLayerStage(), attentionLayerStage(LayerOperatorAttentionPairedCausalProjection),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorFeedForwardNorm),
				feedForwardLayerStage(LayerOperatorFeedForwardStandardSwiGLU), layerStage(LayerOperatorResidual),
			)
		}
		if profile.DenseGraph == DenseGraphRWKV6 {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(),
				layerStage(LayerOperatorResidual), tokenShiftLayerStage(LayerOperatorGatedTokenShiftSquaredReLU),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorPeriodicScale),
			)
		}
		if profile.DenseGraph == DenseGraphRWKV7 {
			stages := []LayerOperatorInstruction{
				layerStage(LayerOperatorAttentionNorm), recurrentLayerStage(),
				layerStage(LayerOperatorResidual),
			}
			if profile.Normalization == NormalizationLayer {
				stages = append(stages, tokenShiftLayerStage(LayerOperatorTokenShiftSquaredReLU))
			} else {
				stages = append(stages, layerStage(LayerOperatorFeedForwardNorm),
					feedForwardLayerStage(LayerOperatorFeedForwardStandardSwiGLU))
			}
			return newLayerProgram(append(stages, layerStage(LayerOperatorResidual))...)
		}
		if profile.DenseGraph == DenseGraphGemma4 {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionSharedKVQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				feedForwardLayerStage(LayerOperatorFeedForwardParallelGatedGELU), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorOutputAdapter),
			)
		}
		if profile.DenseGraph == DenseGraphTalkie {
			return newLayerProgram(
				layerStage(LayerOperatorRMSNorm), attentionLayerStage(LayerOperatorAttentionCausalPostQKNorm),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorRMSNorm),
				feedForwardLayerStage(LayerOperatorFeedForwardStandardSwiGLU), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorScaledSkip),
			)
		}
		if profile.DenseGraph == DenseGraphGemmaEmbedding {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionBidirectionalQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(LayerOperatorFeedForwardGatedGELU),
				layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidual),
			)
		}
		if profile.DenseGraph == DenseGraphModernBERT {
			return newLayerProgram(
				attentionLayerStage(LayerOperatorAttentionBidirectionalFusedQKV), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), feedForwardLayerStage(LayerOperatorFeedForwardFusedGLU),
				layerStage(LayerOperatorResidual),
			)
		}
		if profile.DenseGraph == DenseGraphBERT {
			stages := []LayerOperatorInstruction{
				attentionLayerStage(LayerOperatorAttentionBidirectionalEncoder),
				layerStage(LayerOperatorAttentionResidualNorm),
			}
			if profile.EncoderGraph.Kind == encoderGraphJinaV2 {
				stages = append(stages, layerStage(LayerOperatorInputResidualNorm))
			}
			return newLayerProgram(append(stages,
				feedForwardLayerStage(LayerOperatorFeedForwardEncoder),
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
					attentionLayerStageWithoutCache(LayerOperatorAttentionOutputProjection),
					layerStage(LayerOperatorResidual))
			}
			return newLayerProgram(append(stages,
				layerStage(LayerOperatorFeedForwardNorm),
				feedForwardLayerStage(LayerOperatorFeedForwardStandardSwiGLU),
				layerStage(LayerOperatorResidual),
			)...)
		}
		if profile.DenseGraph == DenseGraphStandard && !plan.DeciSparse {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionInputNorm),
				attentionLayerStage(LayerOperatorAttentionPlannedProjection),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorFeedForwardInputNorm),
				feedForwardLayerStage(LayerOperatorFeedForwardPlanned),
				layerStage(LayerOperatorFeedForwardOutput), layerStage(LayerOperatorResidual),
			)
		}
		return LayerProgram{}
	case profile.Validation.MLA == MLAValidationKimiLinear:
		if !recurrent {
			return latentLayerProgram(
				profile,
				[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
			)
		}
		return residualMixerProgram(
			recurrentLayerStage(),
			LayerOperatorFeedForwardStandardSwiGLU, false,
		)
	case plan.Attention == AttentionMLA:
		return latentLayerProgram(
			profile,
			[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
		)
	case plan.Attention == AttentionDSA:
		return latentLayerProgram(
			profile,
			[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue, RuntimeCacheIndexerKey},
			[]RuntimeTensorBinding{RuntimeTensorPerLayerInput},
		)
	case profile.Validation.MLA == MLAValidationDeepSeek4:
		return newLayerProgram(
			leafLayerStage(
				LayerOperatorHyperAttention,
				[]RuntimeCacheBinding{RuntimeCachePrimaryKey},
				[]RuntimeTensorBinding{RuntimeTensorCurrentPositions},
			),
			layerStage(LayerOperatorHyperFeedForward),
		)
	}
}

func residualMixerProgram(mixer LayerOperatorInstruction, feedForward LayerOperator, scale bool) LayerProgram {
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
	mix := LayerOperatorFeedForwardStandardSwiGLU
	if profile.FeedForward == FeedForwardSquaredReLU {
		mix = LayerOperatorFeedForwardSquaredReLU
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

func recurrentLayerStage() LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorRecurrentMix)
	instruction.CacheCount = 2
	instruction.Caches[0] = RuntimeCachePrimaryKey
	instruction.Caches[1] = RuntimeCachePrimaryValue
	return instruction
}

func attentionLayerStage(operator LayerOperator) LayerOperatorInstruction {
	instruction := layerStage(operator)
	instruction.CacheCount = 2
	instruction.Caches[0] = RuntimeCachePrimaryKey
	instruction.Caches[1] = RuntimeCachePrimaryValue
	return instruction
}

func attentionLayerStageWithoutCache(operator LayerOperator) LayerOperatorInstruction {
	return layerStage(operator)
}

func hybridLayerStage() LayerOperatorInstruction {
	instruction := layerStage(LayerOperatorHybridMix)
	instruction.CacheCount = 4
	instruction.Caches[0] = RuntimeCachePrimaryKey
	instruction.Caches[1] = RuntimeCachePrimaryValue
	instruction.Caches[2] = RuntimeCacheConvolution
	instruction.Caches[3] = RuntimeCacheSSM
	return instruction
}

func feedForwardLayerStage(operator LayerOperator) LayerOperatorInstruction {
	return layerStage(operator)
}

func tokenShiftLayerStage(operator LayerOperator) LayerOperatorInstruction {
	instruction := layerStage(operator)
	instruction.CacheCount = 1
	instruction.Caches[0] = RuntimeCachePrimaryKey
	return instruction
}

func pairedInputLayerStage() LayerOperatorInstruction {
	return leafLayerStage(
		LayerOperatorPairedInputNorm, nil, []RuntimeTensorBinding{RuntimeTensorPerLayerInput},
	)
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
		return LayerProgram{Count: uint8(len(stages))}
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
