package model

import (
	"errors"
	"fmt"
	"slices"

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
	RuntimeCacheCrossKey
	RuntimeCacheCrossValue
	runtimeCacheBindingCount
)

// RuntimeTensorBinding: indexed auxiliary tensor operand.
type RuntimeTensorBinding uint8

const (
	RuntimeTensorNone RuntimeTensorBinding = iota
	RuntimeTensorPerLayerInput
	RuntimeTensorCurrentPositions
	RuntimeTensorEncoder
	runtimeTensorBindingCount
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
	LayerOperatorCrossAttentionNorm
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
	LayerOperatorAttentionRelativeBidirectional
	LayerOperatorAttentionRelativeCausal
	LayerOperatorAttentionCross
	LayerOperatorHybridMix
	LayerOperatorRecurrentMix
	LayerOperatorFeedForwardNorm
	LayerOperatorFeedForwardPolicy
	LayerOperatorFeedForwardRoutedSquaredReLU
	LayerOperatorFeedForwardRoutedSwiGLU
	LayerOperatorFeedForwardParallelGatedGELU
	LayerOperatorFeedForwardEncoder
	LayerOperatorFeedForwardRelative
	LayerOperatorFeedForwardPostNorm
	LayerOperatorCacheSentinel
	LayerOperatorScale
	LayerOperatorResidual
	LayerOperatorRMSNorm
	LayerOperatorScaledSkip
	LayerOperatorOutputAdapter
	LayerOperatorGatedTokenShiftSquaredReLU
	LayerOperatorTokenShiftSquaredReLU
	LayerOperatorAttentionResidualNorm
	LayerOperatorInputResidualNorm
	LayerOperatorFeedForwardResidualNorm
	LayerOperatorPairedInputNorm
	LayerOperatorResidualScale
	LayerOperatorFeedForwardInputNorm
	LayerOperatorFeedForwardOutput
	LayerOperatorActivationProjection
	LayerOperatorActivatedOutput
)

// LayerOperatorInstruction: compiled operator and operand indexes.
type LayerOperatorInstruction struct {
	Operator    LayerOperator
	Scalar      float32
	CacheCount  uint8
	TensorCount uint8
	Caches      [runtimeCacheBindingCount]RuntimeCacheBinding
	Tensors     [runtimeTensorBindingCount]RuntimeTensorBinding
}

// LayerProgram: ordered recipe instructions.
type LayerProgram struct {
	Instructions []LayerOperatorInstruction
}

func (p LayerProgram) valid() bool {
	return len(p.Instructions) != tensor.FirstOffset
}

func (p LayerProgram) equal(other LayerProgram) bool {
	return slices.Equal(p.Instructions, other.Instructions)
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

func (p LayerProgram) Instruction(index uint32) (LayerOperatorInstruction, bool) {
	if uint64(index) >= uint64(len(p.Instructions)) {
		return LayerOperatorInstruction{}, false
	}
	return p.Instructions[index], true
}

// CachePolicy: compiled layer-state layout.
type CachePolicy uint8

const (
	CacheAttention CachePolicy = iota
	CacheSentinel
	CacheSelectiveScan
	CacheGroupedSelectiveScan
	CacheDoubleTokenShiftRecurrence
	CacheSingleTokenShiftRecurrence
	CacheVariableTokenShiftRecurrence
	CacheKeyedDelta
	CacheGatedDelta
	CacheShortConvolution
	CacheHybridAttentionScan
	CacheCrossAttention
	CacheCompressedAttention
)

func (p CachePolicy) PrimaryMode() CacheStateMode {
	if p >= CacheSelectiveScan && p <= CacheShortConvolution {
		return CacheStateFixed
	}
	return CacheStateToken
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

// LayerTopologyPolicy: compiled operator-sequence topology.
type LayerTopologyPolicy uint8

const (
	LayerTopologyPlannedDecoder LayerTopologyPolicy = iota
	LayerTopologyBidirectionalEncoder
	LayerTopologyBidirectionalFusedQKV
	LayerTopologyBidirectionalQKNorm
	LayerTopologyCausalPostQKNormSkip
	LayerTopologySharedKVAdapter
	LayerTopologySplitProjection
	LayerTopologyDynamicWKV6
	LayerTopologyAffineWKV6
	LayerTopologyDynamicWKV7
	LayerTopologyKeyedDeltaHybrid
	LayerTopologyCompressedHyper
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
	DeepstackSourceBase DeepstackSource = -1 - iota
	DeepstackSourceNone
)

// LayerPlan: derived layer execution contract.
type LayerPlan struct {
	Layer               uint32
	Program             LayerProgram
	Composition         LayerCompositionPolicy
	Cache               CachePolicy
	Attention           AttentionPolicy
	EncoderOperator     EncoderOperatorPolicy
	LatentAttention     latentAttentionPolicy
	LatentYaRNQuery     bool
	Position            PositionPolicy
	Residual            ResidualPolicy
	FeedForward         FeedForwardPolicy
	CacheMode           CacheStateMode
	CacheWrite          CacheWritePolicy
	Recurrent           bool
	Sliding             bool
	UsesRoPE            bool
	MultiAxis           bool
	HasKV               bool
	SharedKV            bool
	KVSource            uint32
	DeepstackBefore     DeepstackSource
	DeepstackAfter      DeepstackSource
	AuxiliaryInput      AuxiliaryFlow
	AuxiliaryOutput     AuxiliaryFlow
	Temperature         AttentionTemperaturePolicy
	AttentionBlocks     AttentionBlockPolicy
	EmbeddingSkip       bool
	PerLayerInput       bool
	Normalization       NormalizationPlan
	Rotary              RotaryPlan
	AttentionGraph      AttentionGraphPlan
	Experts             MoEGraphPlan
	ExpertComposition   ExpertCompositionPlan
	DenseWeights        DenseWeightPlan
	rotaryCatalog       rotaryCatalogPlan
	ExplicitEncoder     bool
	AllowNonCausalCache bool
	DeciSparse          bool
	QKPreprocess        QKPreprocessPlan
	QueryScale          QueryScalePlan
	AttentionOutput     AttentionOutputPlan
	ResidualStages      ResidualStagePlan
	Mixer               RecurrentMixerPolicy
	RecurrentRuntime    RecurrentRuntimePolicy
	PeriodicScale       float32
}

// ExactAttentionReplay reports whether captured Q/K tensors fully determine
// this layer's attention result without additional position-dependent policy.
func (p LayerPlan) ExactAttentionReplay() bool {
	attention := p.AttentionGraph
	return p.HasKV && attention.Causal && !attention.UseSinks && !attention.ChunkedWindow &&
		attention.Window == 0 && attention.Softcap == 0 && attention.MaxALiBiBias == 0
}

func (s Spec) planLayer(profile ArchitectureProfile, layer uint32, recurrent bool) (LayerPlan, error) {
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
	normalization := profile.Runtime.normalizationPlan(s, profile)
	cache := cachePolicy(s, profile, layer, recurrent)
	mixer := compileRecurrentMixer(profile, recurrent)
	composition := LayerCompositionStandard
	if mixer == recurrentMixerSparseGroupedSelectiveScan {
		switch {
		case recurrent:
			composition = LayerCompositionRecurrentOnly
		case s.LayerHasFeedForward(layer):
			composition = LayerCompositionFeedForwardOnly
		default:
			composition = LayerCompositionAttentionOnly
		}
	}
	deciSparse := profile.DeciSparse &&
		(!s.LayerHasFeedForward(layer) || !s.LayerHasAttention(layer) ||
			!s.LayerHasKVHeads(layer))
	if deciSparse {
		switch {
		case !s.LayerHasFeedForward(layer):
			composition = LayerCompositionIdentity
		case !s.LayerHasAttention(layer):
			composition = LayerCompositionFeedForwardOnly
		}
	}
	residualStages := s.residualStagePlan(profile, normalization)
	var periodicScale float32
	if (profile.LayerTopology == LayerTopologyDynamicWKV6 || profile.LayerTopology == LayerTopologyAffineWKV6) &&
		s.IsPeriodicRescaleLayer(layer) {
		periodicScale = profile.Runtime.Recurrent.PeriodicResidualScale
	}
	if profile.LayerTopology == LayerTopologyAffineWKV6 {
		residualStages.residualScale = periodicScale
	}
	experts := s.moeGraphPlan(profile, layer)
	if mixer == recurrentMixerSparseGroupedSelectiveScan {
		experts.Routing = tensor.MoERoutingSigmoid
		experts.Activation = tensor.MoEActivationReLUSquared
		experts.SelectionBias = true
	}
	if profile.Attention == AttentionGatedDelta {
		experts.NormalizeTopKProb = true
	}
	if profile.Attention == AttentionShortConvolution && recurrent {
		experts.NormalizeTopKProb = true
		experts.SelectionBias = true
	}
	cacheWrite := CacheWriteFixed
	if cache.PrimaryMode().TokenAligned() {
		cacheWrite = CacheWriteConcatOnly
		if mixer == recurrentMixerNone || profile.Attention == AttentionGatedDelta {
			cacheWrite = CacheWriteConcatOrAppend
		}
	}
	plan := LayerPlan{
		Layer:               layer,
		Attention:           profile.Attention,
		EncoderOperator:     profile.EncoderOperator,
		LatentAttention:     profile.LatentAttention,
		LatentYaRNQuery:     profile.Has(ArchitectureLatentYaRNQuery),
		Position:            profile.Position,
		Residual:            profile.Residual,
		FeedForward:         profile.FeedForward,
		Composition:         composition,
		Cache:               cache,
		CacheMode:           cache.PrimaryMode(),
		CacheWrite:          cacheWrite,
		Recurrent:           recurrent,
		Sliding:             s.IsSlidingLayer(layer),
		UsesRoPE:            s.UsesRoPE(layer),
		MultiAxis:           profile.Has(ArchitectureMultiAxisPositions),
		HasKV:               hasKV,
		SharedKV:            sharedKV,
		KVSource:            kvSource,
		DeepstackBefore:     deepstackBefore,
		DeepstackAfter:      deepstackAfter,
		AuxiliaryInput:      auxiliaryInput,
		AuxiliaryOutput:     auxiliaryOutput,
		Temperature:         temperature,
		AttentionBlocks:     profile.AttentionBlocks,
		EmbeddingSkip:       profile.Has(ArchitectureEmbeddingSkip),
		PerLayerInput:       s.hasPerLayerEmbeddings(profile),
		Normalization:       normalization,
		Rotary:              s.rotaryPlan(profile, layer),
		AttentionGraph:      s.attentionGraphPlan(profile, layer),
		Experts:             experts,
		ExpertComposition:   s.expertCompositionPlan(profile),
		DenseWeights:        s.denseWeightPlan(profile, layer),
		ExplicitEncoder:     profile.Forward.Session == ForwardSessionEncoderDecoder,
		AllowNonCausalCache: profile.Forward.Session == ForwardSessionPairedFeatures,
		DeciSparse:          deciSparse,
		QKPreprocess:        s.qkPreprocessPlan(profile, layer),
		QueryScale:          s.queryScalePlan(profile, layer),
		AttentionOutput:     s.attentionOutputPlan(profile, normalization),
		ResidualStages:      residualStages,
		Mixer:               mixer,
		RecurrentRuntime:    profile.Runtime.Recurrent,
		PeriodicScale:       periodicScale,
	}
	plan.rotaryCatalog = s.rotaryCatalogPlan(profile, plan)
	program, err := compileLayerProgram(plan, profile)
	if err != nil {
		return LayerPlan{}, fmt.Errorf("model plan layer %d: %w", layer, err)
	}
	plan.Program = program
	return plan, nil
}

// SupportsCapacityCache: all token caches admit bounded append.
func (p ModelPlan) SupportsCapacityCache() bool {
	if len(p.layers) == tensor.FirstOffset {
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

// SupportsMultiAxisPositionsWithProfile: bound-profile MRoPE contract.
func (s Spec) SupportsMultiAxisPositionsWithProfile(profile ArchitectureProfile) bool {
	if !profile.Has(ArchitectureMultiAxisPositions) {
		return false
	}
	sections := false
	for _, count := range s.RopeSections {
		sections = sections || count > int32(tensor.FirstOffset)
	}
	if !sections {
		return false
	}
	return profile.Rotary.MultiAxis != multiAxisRotaryWithSections ||
		hasLeadingRopeSections(s.RopeSections)
}

func hasLeadingRopeSections(sections [tensor.MaxDimensions]int32) bool {
	return sections[tensor.FirstOffset] > int32(tensor.FirstOffset) &&
		sections[tensor.SingletonExtent] > int32(tensor.FirstOffset)
}

func attentionTemperatureConfigured(spec Spec) bool {
	return spec.AttentionTempScale != tensor.FirstOffset ||
		spec.AttentionTempFloor != tensor.FirstOffset ||
		spec.AttentionTempOffset != tensor.FirstOffset
}

func validAttentionTemperature(spec Spec) bool {
	return positiveFinite(spec.AttentionTempScale) &&
		spec.AttentionTempFloor > tensor.FirstOffset && finite(spec.AttentionTempOffset)
}

func deepstackSources(s Spec, profile ArchitectureProfile, layer uint32) (DeepstackSource, DeepstackSource) {
	switch profile.Deepstack {
	case DeepstackMappedBefore:
		if layer == tensor.FirstOffset || int(layer) >= len(s.DeepstackMapping) {
			return DeepstackSourceNone, DeepstackSourceNone
		}
		source := s.DeepstackMapping[layer]
		switch {
		case source < tensor.FirstOffset:
			return DeepstackSourceNone, DeepstackSourceNone
		case source == tensor.FirstOffset:
			return DeepstackSourceBase, DeepstackSourceNone
		default:
			return DeepstackSource(source - tensor.SingletonExtent), DeepstackSourceNone
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
	case AuxiliaryRecurrentValue:
		if layer == tensor.FirstOffset {
			return AuxiliaryNone, AuxiliaryRecurrentValue
		}
		return AuxiliaryRecurrentValue, AuxiliaryNone
	case AuxiliarySparseTopK:
		if s.LayerHasFullIndexer(layer) {
			return AuxiliaryNone, AuxiliarySparseTopK
		}
		return AuxiliarySparseTopK, AuxiliarySparseTopK
	default:
		return AuxiliaryNone, AuxiliaryNone
	}
}

// ModelPlan: immutable model and per-layer execution contract.
type ModelPlan struct {
	spec          Spec
	profile       ArchitectureProfile
	layers        []LayerPlan
	draftLayers   []LayerPlan
	cacheSchemas  []LayerCacheSchema
	cacheLayers   uint32
	cachedGraph   CachedGraphPolicy
	normalization NormalizationPlan
	terminal      TerminalPlan
	draft         DraftPlan
	cacheProject  CacheProjectionProgram
	projections   [projectionRoleCount]ProjectionProgram
	sequenceOut   SequenceOutputProgram
	forward       ForwardProgram
	input         ProjectedInputProgram
}

// ProjectedInputProgram: compiled projected-input and specialized-output admission.
type ProjectedInputProgram struct {
	Overrides          EmbeddingOverridePolicy
	AttentionBlocks    AttentionBlockPolicy
	DeepstackStreams   uint32
	MultiAxis          bool
	PerLayerEmbeddings bool
	ClassifierHead     bool
}

func compileProjectedInputProgram(spec Spec, profile ArchitectureProfile) ProjectedInputProgram {
	var deepstackStreams uint32
	if profile.Deepstack != DeepstackNone {
		deepstackStreams = spec.DeepstackLayerCount
	}
	return ProjectedInputProgram{
		Overrides: profile.Overrides, AttentionBlocks: profile.AttentionBlocks,
		DeepstackStreams:   deepstackStreams,
		MultiAxis:          spec.SupportsMultiAxisPositionsWithProfile(profile),
		PerLayerEmbeddings: spec.hasPerLayerEmbeddings(profile),
		ClassifierHead:     profile.Has(ArchitectureClassifierHead),
	}
}

func (s Spec) hasPerLayerEmbeddings(profile ArchitectureProfile) bool {
	return profile.Has(ArchitecturePerLayerEmbeddings) && s.EmbeddingPerLayer > tensor.FirstOffset
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
	profile = spec.Profile()
	layers := spec.BlockCount
	if profile.Forward.Session == ForwardSessionEncoderDecoder && spec.DecoderBlockCount > layers {
		layers = spec.DecoderBlockCount
	}
	cacheLayers := spec.BlockCount
	if profile.Forward.Session == ForwardSessionEncoderDecoder {
		cacheLayers = spec.DecoderBlockCount
	}
	forward := compileForwardProgram(spec, profile)
	plan := ModelPlan{
		spec: spec, profile: profile, layers: make([]LayerPlan, layers), cacheLayers: cacheLayers,
		normalization: profile.Runtime.normalizationPlan(spec, profile),
		terminal:      TerminalPlan{Normalization: profile.OutputNorm},
		draft:         profile.DraftPlan(spec.NextNPredictLayers),
		cacheProject:  compileCacheProjectionProgram(spec, profile),
		projections:   compileProjectionPrograms(spec, profile),
		sequenceOut:   compileSequenceOutputProgram(spec, profile),
		forward:       forward,
		input:         compileProjectedInputProgram(spec, profile),
	}
	if weights.Output != nil {
		plan.terminal.OutputHead = OutputHeadDedicated
	}
	for layer := range layers {
		recurrent := int(layer) < len(weights.Layers) && weights.Layers[layer].Recurrent
		var err error
		plan.layers[layer], err = spec.planLayer(profile, layer, recurrent)
		if err != nil {
			return ModelPlan{}, err
		}
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
		if plan.draft.Kind == DraftAppendedSingle {
			executable.BlockCount += plan.draft.Heads
		}
		for offset := range plan.draftLayers {
			var err error
			plan.draftLayers[offset], err = executable.planLayer(
				executable.Profile(), spec.BlockCount+uint32(offset), false,
			)
			if err != nil {
				return ModelPlan{}, err
			}
		}
	} else if plan.draft.SingleCatalog && plan.draft.SessionEligible() &&
		(plan.draft.Kind != DraftOptionalSingleCatalog || weights.OptionalCatalogDraft != nil) {
		executable, layer := draftExecutableSpec(spec, plan.draft)
		draftLayer, err := executable.planLayer(executable.Profile(), layer, false)
		if err != nil {
			return ModelPlan{}, err
		}
		plan.draftLayers = []LayerPlan{draftLayer}
	}
	if err := validateModelPlan(spec, weights, plan); err != nil {
		return ModelPlan{}, err
	}
	plan.cachedGraph = cachedGraphPolicy(
		profile.RecurrentMixer != recurrentMixerNone, plan.forward, plan.layers,
	)
	return plan, nil
}

// Normalization returns the immutable model normalization contract.
func (p ModelPlan) Normalization() NormalizationPlan { return p.normalization }

// Spec returns the immutable bound model metadata.
func (p ModelPlan) Spec() Spec { return p.spec }

func validateModelPlan(spec Spec, weights Weights, plan ModelPlan) error {
	if plan.normalization != plan.profile.Runtime.normalizationPlan(spec, plan.profile) {
		return fmt.Errorf("model plan architecture %s has invalid normalization", spec.Architecture)
	}
	if !plan.forward.valid() || !plan.forward.equal(compileForwardProgram(spec, plan.profile)) {
		return fmt.Errorf("model plan architecture %s has invalid forward program", spec.Architecture)
	}
	if plan.input != compileProjectedInputProgram(spec, plan.profile) {
		return fmt.Errorf("model plan architecture %s has invalid projected-input program", spec.Architecture)
	}
	if plan.terminal.OutputHead > OutputHeadDedicated ||
		plan.terminal.Normalization != plan.profile.OutputNorm ||
		(plan.terminal.OutputHead == OutputHeadDedicated) != (weights.Output != nil) {
		return fmt.Errorf("model plan architecture %s has invalid terminal policy", spec.Architecture)
	}
	if plan.draft != plan.profile.DraftPlan(spec.NextNPredictLayers) {
		return fmt.Errorf("model plan architecture %s has invalid draft policy", spec.Architecture)
	}
	var wantDraftLayers int
	if plan.draft.AppendedBlocks {
		wantDraftLayers = int(plan.draft.Heads)
	} else if plan.draft.SingleCatalog && plan.draft.SessionEligible() &&
		(plan.draft.Kind != DraftOptionalSingleCatalog || weights.OptionalCatalogDraft != nil) {
		wantDraftLayers = tensor.SingletonExtent
	}
	if len(plan.draftLayers) != wantDraftLayers {
		return fmt.Errorf("model plan architecture %s has invalid draft layers", spec.Architecture)
	}
	if len(plan.cacheSchemas) != tensor.FirstOffset && len(plan.cacheSchemas) != len(plan.layers) {
		return fmt.Errorf("model plan architecture %s has invalid cache schemas", spec.Architecture)
	}
	for offset, layer := range plan.draftLayers {
		wantLayer := spec.BlockCount + uint32(offset)
		if plan.draft.Kind == DraftSingleCatalog {
			wantLayer = tensor.FirstOffset
		}
		if layer.Layer != wantLayer {
			return fmt.Errorf("model plan draft layer %d identity is inconsistent", offset)
		}
		if !layer.Program.valid() {
			return fmt.Errorf(
				"model plan draft layer %d operator program is empty", offset,
			)
		}
		expected, err := compileLayerProgram(layer, plan.profile)
		if err != nil {
			return fmt.Errorf("model plan draft layer %d: %w", offset, err)
		}
		if !layer.Program.equal(expected) {
			return fmt.Errorf("model plan draft layer %d operator program is inconsistent", offset)
		}
	}
	if spec.SharedKVLayers > tensor.FirstOffset && (!plan.profile.Has(ArchitectureSharedKV) ||
		spec.SharedKVLayers >= spec.BlockCount) {
		return fmt.Errorf("model plan architecture %s has invalid shared-KV layer count %d", spec.Architecture, spec.SharedKVLayers)
	}
	producedAuxiliary := make(map[AuxiliaryFlow]bool)
	norm := plan.normalization
	for index, layer := range plan.layers {
		if layer.Layer != uint32(index) {
			return fmt.Errorf("model plan layer %d identity is inconsistent", index)
		}
		if !layer.Program.valid() {
			return fmt.Errorf(
				"model plan layer %d operator program is empty", index,
			)
		}
		expected, err := compileLayerProgram(layer, plan.profile)
		if err != nil {
			return fmt.Errorf("model plan layer %d: %w", index, err)
		}
		if !layer.Program.equal(expected) {
			return fmt.Errorf("model plan layer %d operator program is inconsistent", index)
		}
		if layer.SharedKV {
			if layer.HasKV || layer.KVSource >= layer.Layer || int(layer.KVSource) >= len(plan.layers) ||
				!plan.layers[layer.KVSource].HasKV {
				return fmt.Errorf("model plan layer %d shared-KV source %d is invalid", index, layer.KVSource)
			}
		}
		for _, source := range []DeepstackSource{layer.DeepstackBefore, layer.DeepstackAfter} {
			if source >= DeepstackSource(tensor.FirstOffset) && uint32(source) >= spec.DeepstackLayerCount {
				return fmt.Errorf("model plan layer %d deepstack source %d exceeds %d streams", index, source, spec.DeepstackLayerCount)
			}
		}
		if layer.AuxiliaryInput != AuxiliaryNone && len(spec.IndexerFullLayers) > tensor.FirstOffset &&
			!producedAuxiliary[layer.AuxiliaryInput] {
			return fmt.Errorf("model plan layer %d consumes auxiliary flow %d before production", index, layer.AuxiliaryInput)
		}
		if layer.AuxiliaryOutput != AuxiliaryNone {
			producedAuxiliary[layer.AuxiliaryOutput] = true
		}
		if layer.Cache == CacheCompressedAttention && len(spec.CompressRatios) <= index {
			return fmt.Errorf("model plan layer %d has no compression ratio", index)
		}
		if layer.Normalization != norm {
			return fmt.Errorf("model plan layer %d normalization drifted from model policy", index)
		}
		if layer.Mixer != compileRecurrentMixer(plan.profile, layer.Recurrent) {
			return fmt.Errorf("model plan layer %d state-space policy is inconsistent", index)
		}
	}
	if spec.DeepstackLayerCount > tensor.FirstOffset {
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
	if attentionTemperatureConfigured(spec) {
		if plan.profile.Temperature == AttentionTemperatureNone || !validAttentionTemperature(spec) {
			return fmt.Errorf("model plan architecture %s has an invalid attention temperature contract", spec.Architecture)
		}
	}
	if norm.PostNormLayout == PostNormLayoutOutputLayer &&
		(norm.Operation != NormalizationLayer || norm.PreAttention || !norm.PostAttention || !norm.Bias) {
		return fmt.Errorf("model plan architecture %s has an invalid output-layer normalization layout", spec.Architecture)
	}
	draft := plan.profile.DraftPlan(spec.NextNPredictLayers)
	if spec.NextNPredictLayers > tensor.FirstOffset && (draft.Kind == DraftNone || !draft.HasHead(firstDraftHead)) {
		return fmt.Errorf("model plan architecture %s has no draft policy for %d heads", spec.Architecture, spec.NextNPredictLayers)
	}
	return nil
}

func cachedGraphPolicy(
	recurrent bool,
	forward ForwardProgram,
	layers []LayerPlan,
) CachedGraphPolicy {
	if len(layers) == tensor.FirstOffset || forward.Operation != ForwardOperationCached ||
		recurrent ||
		forward.AlternateStates() {
		return CachedGraphLayered
	}
	for _, layer := range layers {
		if layer.Mixer != recurrentMixerNone || layer.Attention != AttentionStandard ||
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
func (p ModelPlan) Layer(layer uint32) (LayerPlan, error) {
	if uint64(layer) >= uint64(len(p.layers)) {
		return LayerPlan{}, fmt.Errorf("model plan layer %d is outside [0,%d)", layer, len(p.layers))
	}
	return p.layers[layer], nil
}

// Profile: compiled architecture policy.
func (p ModelPlan) Profile() ArchitectureProfile { return p.profile.clone() }

// SameExecutionProfile reports whether two sealed plans share one policy contract.
func (p ModelPlan) SameExecutionProfile(other ModelPlan) bool { return p.profile.equal(other.profile) }

// Compiled reports whether the plan has a resolved execution identity.
func (p ModelPlan) Compiled() bool { return p.profile.Name != "" }

// Forward: compiled top-level execution contract.
func (p ModelPlan) Forward() ForwardProgram { return p.forward }

// ProjectedInput returns compiled projected-input admission.
func (p ModelPlan) ProjectedInput() ProjectedInputProgram { return p.input }

// LayerCount: compiled trunk layer count.
func (p ModelPlan) LayerCount() int { return len(p.layers) }

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
	draft DraftPlan
}

// Layer returns the immutable compiled layer facts.
func (p CompiledLayerProgram) Layer() LayerPlan { return p.plan }

// Spec returns the program-owned model facts.
func (p CompiledLayerProgram) Spec() Spec { return p.spec }

// LayerProgram binds one trunk layer to its validated model spec.
func (p ModelPlan) LayerProgram(layer uint32) (CompiledLayerProgram, error) {
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

func (p ModelPlan) sequenceProgram(layer uint32, decoder bool) (CompiledLayerProgram, error) {
	encoder := p.profile.EncoderOperator
	limit := p.spec.BlockCount
	if !decoder && encoder != encoderOperatorRelativeEncoderDecoder && encoder != encoderOperatorRelativeEncoder {
		return CompiledLayerProgram{}, fmt.Errorf("model plan has no relative-attention encoder program for %q", p.spec.Architecture)
	}
	if decoder {
		if p.profile.Forward.Session != ForwardSessionEncoderDecoder {
			return CompiledLayerProgram{}, fmt.Errorf("model plan has no decoder program for %q", p.spec.Architecture)
		}
		limit = p.spec.DecoderBlockCount
	}
	if layer >= limit {
		return CompiledLayerProgram{}, fmt.Errorf("model plan sequence layer %d is outside [0,%d)", layer, limit)
	}
	program, err := p.LayerProgram(layer)
	if err != nil {
		return CompiledLayerProgram{}, err
	}
	if decoder {
		program.plan.Program, err = relativeDecoderProgram()
	} else {
		program.plan.Program, err = relativeEncoderProgram()
	}
	return program, err
}

// EncoderProgram returns one compiled encoder layer.
func (p ModelPlan) EncoderProgram(layer uint32) (CompiledLayerProgram, error) {
	return p.sequenceProgram(layer, false)
}

// DecoderProgram returns one compiled causal/cross-attention decoder layer.
func (p ModelPlan) DecoderProgram(layer uint32) (CompiledLayerProgram, error) {
	return p.sequenceProgram(layer, true)
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
		spec, _ = draftExecutableSpec(spec, p.draft)
	} else if p.draft.Kind == DraftAppendedSingle {
		spec.BlockCount += p.draft.Heads
	}
	return CompiledLayerProgram{spec: spec, plan: layer, draft: p.draft}, nil
}

// CacheSchema: materialized compiled layer-cache contract.
func (p ModelPlan) CacheSchema(layer uint32, tokens uint32) (LayerCacheSchema, error) {
	if uint64(layer) >= uint64(len(p.cacheSchemas)) {
		return LayerCacheSchema{}, fmt.Errorf("model plan cache schema %d is unavailable", layer)
	}
	return p.cacheSchemas[layer].WithTokenCount(tokens), nil
}

// HasCacheSchemas: physical cache layout compiled.
func (p ModelPlan) HasCacheSchemas() bool {
	return len(p.layers) != tensor.FirstOffset && len(p.cacheSchemas) == len(p.layers)
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

// AudioWaveform: compiled semantic-token waveform plan.
func (p ModelPlan) AudioWaveform() AudioWaveformPlan { return p.forward.Waveform }

// Draft: compiled speculative policy.
func (p ModelPlan) Draft() DraftPlan { return p.draft }

func compileLayerProgram(plan LayerPlan, profile ArchitectureProfile) (LayerProgram, error) {
	recurrent := plan.Recurrent
	composition := plan.Composition
	if profile.LayerTopology == LayerTopologySplitProjection {
		return newLayerProgram(
			layerStage(LayerOperatorActivationProjection),
			layerStage(LayerOperatorActivatedOutput),
		)
	}
	if profile.Attention == AttentionShortConvolution && recurrent {
		return residualMixerProgram(
			attentionLayerStage(LayerOperatorRecurrentMix), LayerOperatorFeedForwardPolicy,
		)
	}
	if profile.LayerTopology == LayerTopologyAffineWKV6 {
		return residualMixerProgram(
			attentionLayerStage(LayerOperatorRecurrentMix), LayerOperatorFeedForwardPolicy,
			plan.ResidualStages.residualScale,
		)
	}
	if profile.Attention == AttentionGatedDelta {
		mixer := attentionLayerStage(LayerOperatorAttentionGatedProjection)
		if recurrent {
			mixer = attentionLayerStage(LayerOperatorRecurrentMix)
		}
		return residualMixerProgram(mixer, LayerOperatorFeedForwardRoutedSwiGLU)
	}
	if plan.Mixer == recurrentMixerAttentionGroupedSelectiveScan {
		return residualMixerProgram(
			hybridLayerStage(), LayerOperatorFeedForwardPolicy,
		)
	}
	if plan.Mixer == recurrentMixerScaledGroupedSelectiveScan {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorRecurrentMix),
			scaleLayerStage(plan.ResidualStages.residualScale), layerStage(LayerOperatorResidual),
			layerStage(LayerOperatorFeedForwardNorm), layerStage(LayerOperatorFeedForwardPolicy),
			scaleLayerStage(plan.ResidualStages.residualScale), layerStage(LayerOperatorResidual),
		)
	}
	if plan.Mixer == recurrentMixerWeightedSelectiveScan {
		return residualMixerProgram(
			attentionLayerStage(LayerOperatorRecurrentMix), LayerOperatorFeedForwardPolicy,
		)
	}
	if plan.Mixer == recurrentMixerNormalizedSelectiveScan {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorRecurrentMix),
			layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
			layerStage(LayerOperatorFeedForwardNorm), layerStage(LayerOperatorFeedForwardPolicy),
			layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidual),
		)
	}
	if plan.Mixer == recurrentMixerSparseGroupedSelectiveScan {
		switch composition {
		case LayerCompositionRecurrentOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorRecurrentMix),
				layerStage(LayerOperatorResidual),
			)
		case LayerCompositionAttentionOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionCausalProjection),
				layerStage(LayerOperatorResidual),
			)
		case LayerCompositionFeedForwardOnly:
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorCacheSentinel),
				layerStage(LayerOperatorFeedForwardRoutedSquaredReLU), layerStage(LayerOperatorResidual),
			)
		default:
			return LayerProgram{}, errors.New("layer program has unsupported sparse composition")
		}
	}
	if plan.Mixer == recurrentMixerSelectiveScan || plan.Mixer == recurrentMixerGroupedSelectiveScan {
		return newLayerProgram(
			layerStage(LayerOperatorAttentionNorm),
			attentionLayerStage(LayerOperatorRecurrentMix),
			layerStage(LayerOperatorResidual),
		)
	}
	switch {
	default:
		if profile.Forward.Session == ForwardSessionPairedProjection {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionSharedCacheQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), layerStage(LayerOperatorFeedForwardPolicy),
				layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidualScale),
			)
		}
		if profile.Forward.Session == ForwardSessionFeatureDraft {
			return newLayerProgram(
				pairedInputLayerStage(), attentionLayerStage(LayerOperatorAttentionPairedCausalProjection),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorFeedForwardNorm),
				layerStage(LayerOperatorFeedForwardPolicy), layerStage(LayerOperatorResidual),
			)
		}
		if profile.LayerTopology == LayerTopologyDynamicWKV6 {
			stages := []LayerOperatorInstruction{
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorRecurrentMix),
				layerStage(LayerOperatorResidual), tokenShiftLayerStage(LayerOperatorGatedTokenShiftSquaredReLU),
				layerStage(LayerOperatorResidual),
			}
			if positiveFinite(plan.PeriodicScale) {
				stages = append(stages, scaleLayerStage(plan.PeriodicScale))
			}
			return newLayerProgram(stages...)
		}
		if profile.LayerTopology == LayerTopologyDynamicWKV7 {
			stages := []LayerOperatorInstruction{
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorRecurrentMix),
				layerStage(LayerOperatorResidual),
			}
			if profile.Normalization == NormalizationLayer {
				stages = append(stages, tokenShiftLayerStage(LayerOperatorTokenShiftSquaredReLU))
			} else {
				stages = append(stages, layerStage(LayerOperatorFeedForwardNorm),
					layerStage(LayerOperatorFeedForwardPolicy))
			}
			return newLayerProgram(append(stages, layerStage(LayerOperatorResidual))...)
		}
		if profile.LayerTopology == LayerTopologySharedKVAdapter {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionSharedKVQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardParallelGatedGELU), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorOutputAdapter),
			)
		}
		if profile.LayerTopology == LayerTopologyCausalPostQKNormSkip {
			return newLayerProgram(
				layerStage(LayerOperatorRMSNorm), attentionLayerStage(LayerOperatorAttentionCausalPostQKNorm),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorRMSNorm),
				layerStage(LayerOperatorFeedForwardPolicy), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorScaledSkip),
			)
		}
		if profile.LayerTopology == LayerTopologyBidirectionalQKNorm {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionNorm), attentionLayerStage(LayerOperatorAttentionBidirectionalQKNorm),
				layerStage(LayerOperatorAttentionPostNorm), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), layerStage(LayerOperatorFeedForwardPolicy),
				layerStage(LayerOperatorFeedForwardPostNorm), layerStage(LayerOperatorResidual),
			)
		}
		if profile.LayerTopology == LayerTopologyBidirectionalFusedQKV {
			return newLayerProgram(
				attentionLayerStage(LayerOperatorAttentionBidirectionalFusedQKV), layerStage(LayerOperatorResidual),
				layerStage(LayerOperatorFeedForwardNorm), layerStage(LayerOperatorFeedForwardPolicy),
				layerStage(LayerOperatorResidual),
			)
		}
		if profile.LayerTopology == LayerTopologyBidirectionalEncoder {
			stages := []LayerOperatorInstruction{
				attentionLayerStage(LayerOperatorAttentionBidirectionalEncoder),
				layerStage(LayerOperatorAttentionResidualNorm),
			}
			if profile.EncoderOperator.usesALiBiQKNorm() {
				stages = append(stages, layerStage(LayerOperatorInputResidualNorm))
			}
			return newLayerProgram(append(stages,
				layerStage(LayerOperatorFeedForwardEncoder),
				layerStage(LayerOperatorFeedForwardResidualNorm),
			)...)
		}
		if plan.DeciSparse {
			stages := []LayerOperatorInstruction{attentionLayerStage(LayerOperatorCacheSentinel)}
			if plan.Composition == LayerCompositionIdentity {
				return newLayerProgram(stages...)
			}
			if plan.Composition != LayerCompositionFeedForwardOnly {
				stages = append(stages, layerStage(LayerOperatorAttentionNorm),
					layerStage(LayerOperatorAttentionOutputProjection),
					layerStage(LayerOperatorResidual))
			}
			return newLayerProgram(append(stages,
				layerStage(LayerOperatorFeedForwardNorm),
				layerStage(LayerOperatorFeedForwardPolicy),
				layerStage(LayerOperatorResidual),
			)...)
		}
		if profile.LayerTopology == LayerTopologyPlannedDecoder && !plan.DeciSparse {
			return newLayerProgram(
				layerStage(LayerOperatorAttentionInputNorm),
				attentionLayerStage(LayerOperatorAttentionPlannedProjection),
				layerStage(LayerOperatorResidual), layerStage(LayerOperatorFeedForwardInputNorm),
				layerStage(LayerOperatorFeedForwardPolicy),
				layerStage(LayerOperatorFeedForwardOutput), layerStage(LayerOperatorResidual),
			)
		}
		return LayerProgram{}, errors.New("layer program has no compiled topology")
	case profile.LayerTopology == LayerTopologyKeyedDeltaHybrid:
		if !recurrent {
			return latentLayerProgram(
				profile, plan.ResidualStages.residualScale,
				[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
			)
		}
		return residualMixerProgram(
			attentionLayerStage(LayerOperatorRecurrentMix),
			LayerOperatorFeedForwardPolicy,
		)
	case plan.Attention == AttentionLatent:
		return latentLayerProgram(
			profile, plan.ResidualStages.residualScale,
			[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
		)
	case plan.Attention == AttentionSparseLatent:
		return latentLayerProgram(
			profile, plan.ResidualStages.residualScale,
			[]RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue, RuntimeCacheIndexerKey},
			[]RuntimeTensorBinding{RuntimeTensorPerLayerInput},
		)
	case profile.LayerTopology == LayerTopologyCompressedHyper:
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

func (p LayerPlan) splitProjection() bool {
	prefix, prefixOK := p.Program.Instruction(tensor.FirstOffset)
	suffix, suffixOK := p.Program.Instruction(tensor.SingletonExtent)
	return prefixOK && suffixOK && prefix.Operator == LayerOperatorActivationProjection &&
		suffix.Operator == LayerOperatorActivatedOutput
}

func residualMixerProgram(mixer LayerOperatorInstruction, feedForward LayerOperator, scale ...float32) (LayerProgram, error) {
	stages := []LayerOperatorInstruction{
		layerStage(LayerOperatorAttentionNorm), mixer, layerStage(LayerOperatorResidual),
		layerStage(LayerOperatorFeedForwardNorm), layerStage(feedForward), layerStage(LayerOperatorResidual),
	}
	if len(scale) != 0 {
		stages = append(stages, scaleLayerStage(scale[tensor.FirstOffset]))
	}
	return newLayerProgram(stages...)
}

func latentLayerProgram(
	profile ArchitectureProfile,
	scale float32,
	caches []RuntimeCacheBinding,
	tensors []RuntimeTensorBinding,
) (LayerProgram, error) {
	stages := []LayerOperatorInstruction{
		layerStage(LayerOperatorAttentionNorm),
		leafLayerStage(LayerOperatorLatentAttention, caches, tensors),
		layerStage(LayerOperatorResidual), layerStage(LayerOperatorFeedForwardNorm),
		layerStage(LayerOperatorFeedForwardPolicy),
	}
	if profile.LatentAttention == latentAttentionNeoXResidualScale {
		stages = append(stages, scaleLayerStage(scale))
	}
	return newLayerProgram(append(stages, layerStage(LayerOperatorResidual))...)
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

func scaleLayerStage(scale float32) LayerOperatorInstruction {
	return LayerOperatorInstruction{Operator: LayerOperatorScale, Scalar: scale}
}

func attentionLayerStage(operator LayerOperator) LayerOperatorInstruction {
	return leafLayerStage(
		operator, []RuntimeCacheBinding{RuntimeCachePrimaryKey, RuntimeCachePrimaryValue}, nil,
	)
}

func hybridLayerStage() LayerOperatorInstruction {
	return leafLayerStage(
		LayerOperatorHybridMix,
		[]RuntimeCacheBinding{
			RuntimeCachePrimaryKey, RuntimeCachePrimaryValue, RuntimeCacheConvolution, RuntimeCacheSSM,
		}, nil,
	)
}

func tokenShiftLayerStage(operator LayerOperator) LayerOperatorInstruction {
	return leafLayerStage(operator, []RuntimeCacheBinding{RuntimeCachePrimaryKey}, nil)
}

func pairedInputLayerStage() LayerOperatorInstruction {
	return leafLayerStage(
		LayerOperatorPairedInputNorm, nil, []RuntimeTensorBinding{RuntimeTensorPerLayerInput},
	)
}

func relativeEncoderProgram() (LayerProgram, error) {
	return newLayerProgram(
		layerStage(LayerOperatorAttentionNorm),
		layerStage(LayerOperatorAttentionRelativeBidirectional),
		layerStage(LayerOperatorResidual),
		layerStage(LayerOperatorFeedForwardRelative),
		layerStage(LayerOperatorResidual),
	)
}

func relativeDecoderProgram() (LayerProgram, error) {
	return newLayerProgram(
		layerStage(LayerOperatorAttentionNorm),
		attentionLayerStage(LayerOperatorAttentionRelativeCausal),
		layerStage(LayerOperatorResidual),
		layerStage(LayerOperatorCrossAttentionNorm),
		leafLayerStage(
			LayerOperatorAttentionCross,
			[]RuntimeCacheBinding{RuntimeCacheCrossKey, RuntimeCacheCrossValue},
			[]RuntimeTensorBinding{RuntimeTensorEncoder},
		),
		layerStage(LayerOperatorResidual),
		layerStage(LayerOperatorFeedForwardRelative),
		layerStage(LayerOperatorResidual),
	)
}

func newLayerProgram(stages ...LayerOperatorInstruction) (LayerProgram, error) {
	if len(stages) == tensor.FirstOffset {
		return LayerProgram{}, errors.New("layer program is empty")
	}
	return LayerProgram{Instructions: stages}, nil
}

func cachePolicy(spec Spec, profile ArchitectureProfile, layer uint32, recurrent bool) CachePolicy {
	if recurrent && profile.RecurrentCache != CacheAttention {
		return profile.RecurrentCache
	}
	if profile.Cache != CacheAttention {
		return profile.Cache
	}
	if profile.CacheFallback == CacheFallbackFeedForward && spec.LayerHasFeedForward(layer) ||
		profile.CacheFallback == CacheFallbackMissingKV && !spec.LayerHasKVHeads(layer) {
		return CacheSentinel
	}
	return CacheAttention
}
