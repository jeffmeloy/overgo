package model

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/tensor"
)

// runtimeCacheBinding: indexed cache operand.
type runtimeCacheBinding uint8

const (
	runtimeCacheNone runtimeCacheBinding = iota
	runtimeCachePrimaryKey
	runtimeCachePrimaryValue
	runtimeCacheConvolution
	runtimeCacheSSM
	runtimeCacheIndexerKey
	runtimeCacheCrossKey
	runtimeCacheCrossValue
	runtimeCacheBindingCount
)

// runtimeTensorBinding: indexed auxiliary tensor operand.
type runtimeTensorBinding uint8

const (
	runtimeTensorNone runtimeTensorBinding = iota
	runtimeTensorPerLayerInput
	runtimeTensorCurrentPositions
	runtimeTensorEncoder
	runtimeTensorBindingCount
)

// layerOperator: semantic execution stage.
type layerOperator uint8

const (
	layerOperatorNone layerOperator = iota
	layerOperatorAttentionInputNorm
	layerOperatorLatentAttention
	layerOperatorHyperAttention
	layerOperatorHyperFeedForward
	layerOperatorAttentionNorm
	layerOperatorCrossAttentionNorm
	layerOperatorAttentionPostNorm
	layerOperatorAttentionCausalProjection
	layerOperatorAttentionGatedProjection
	layerOperatorAttentionOutputProjection
	layerOperatorAttentionBidirectionalFusedQKV
	layerOperatorAttentionBidirectionalQKNorm
	layerOperatorAttentionCausalPostQKNorm
	layerOperatorAttentionSharedKVQKNorm
	layerOperatorAttentionBidirectionalEncoder
	layerOperatorAttentionPairedCausalProjection
	layerOperatorAttentionSharedCacheQKNorm
	layerOperatorAttentionPlannedProjection
	layerOperatorAttentionRelativeBidirectional
	layerOperatorAttentionRelativeCausal
	layerOperatorAttentionCross
	layerOperatorHybridMix
	layerOperatorRecurrentMix
	layerOperatorFeedForwardNorm
	// layerOperatorFeedForwardPolicy runs the feed-forward declared by the
	// layer plan: routed expert composition when a router is present,
	// the dense feed-forward runtime otherwise.
	layerOperatorFeedForwardPolicy
	layerOperatorFeedForwardRoutedSquaredReLU
	layerOperatorFeedForwardRoutedSwiGLU
	layerOperatorFeedForwardParallelGatedGELU
	layerOperatorFeedForwardEncoder
	layerOperatorFeedForwardRelative
	layerOperatorFeedForwardPostNorm
	layerOperatorCacheSentinel
	layerOperatorScale
	layerOperatorResidual
	layerOperatorRMSNorm
	layerOperatorScaledSkip
	layerOperatorOutputAdapter
	layerOperatorGatedTokenShiftSquaredReLU
	layerOperatorTokenShiftSquaredReLU
	layerOperatorAttentionResidualNorm
	layerOperatorInputResidualNorm
	layerOperatorFeedForwardResidualNorm
	layerOperatorPairedInputNorm
	layerOperatorResidualScale
	layerOperatorFeedForwardInputNorm
	layerOperatorFeedForwardOutput
	layerOperatorActivationProjection
	layerOperatorActivatedOutput
	layerOperatorGatedHyperAttention
	layerOperatorGatedHyperFeedForward
	layerOperatorGatedHyperCombine
	layerOperatorGatedHyperHead
)

// layerOperatorInstruction: compiled operator and operand indexes.
type layerOperatorInstruction struct {
	Operator    layerOperator
	Scalar      float32
	CacheCount  uint8
	TensorCount uint8
	Caches      [runtimeCacheBindingCount]runtimeCacheBinding
	Tensors     [runtimeTensorBindingCount]runtimeTensorBinding
}

// layerProgram: ordered recipe instructions.
type layerProgram struct {
	Instructions []layerOperatorInstruction
}

func (p layerProgram) valid() bool {
	return len(p.Instructions) != tensor.FirstOffset
}

func (p layerProgram) equal(other layerProgram) bool {
	return slices.Equal(p.Instructions, other.Instructions)
}

// layerCompositionPolicy: semantic block-stage composition.
type layerCompositionPolicy uint8

const (
	layerCompositionStandard layerCompositionPolicy = iota
	layerCompositionRecurrentOnly
	layerCompositionAttentionOnly
	layerCompositionFeedForwardOnly
	layerCompositionIdentity
)

func (p layerProgram) Instruction(index uint32) (layerOperatorInstruction, bool) {
	if uint64(index) >= uint64(len(p.Instructions)) {
		return layerOperatorInstruction{}, false
	}
	return p.Instructions[index], true
}

// CachePolicy: compiled layer-state layout.
type CachePolicy uint8

const (
	cacheAttention CachePolicy = iota
	cacheSentinel
	cacheSelectiveScan
	cacheGroupedSelectiveScan
	cacheDoubleTokenShiftRecurrence
	cacheSingleTokenShiftRecurrence
	cacheVariableTokenShiftRecurrence
	cacheKeyedDelta
	cacheGatedDelta
	cacheShortConvolution
	cacheHybridAttentionScan
	CacheCrossAttention
	CacheCompressedAttention
)

func (p CachePolicy) PrimaryMode() CacheStateMode {
	if p >= cacheSelectiveScan && p <= cacheShortConvolution {
		return CacheStateFixed
	}
	return CacheStateToken
}

// CacheFallbackPolicy: non-primary per-layer cache selection.
type CacheFallbackPolicy uint8

const (
	cacheFallbackNone CacheFallbackPolicy = iota
	cacheFallbackFeedForward
	cacheFallbackMissingKV
)

// cacheWritePolicy: compiled cache graph capabilities.
type cacheWritePolicy uint8

const (
	cacheWriteFixed cacheWritePolicy = iota
	cacheWriteConcatOnly
	cacheWriteConcatOrAppend
)

// CachedGraphPolicy: cached decoder graph composition.
type CachedGraphPolicy uint8

const (
	cachedGraphLayered CachedGraphPolicy = iota
	CachedGraphDense
)

// LayerTopologyPolicy: compiled operator-sequence topology.
type LayerTopologyPolicy uint8

const (
	layerTopologyPlannedDecoder LayerTopologyPolicy = iota
	layerTopologyBidirectionalEncoder
	layerTopologyBidirectionalFusedQKV
	layerTopologyBidirectionalQKNorm
	LayerTopologyCausalPostQKNormSkip
	layerTopologySharedKVAdapter
	layerTopologySplitProjection
	layerTopologyDynamicWKV6
	layerTopologyAffineWKV6
	layerTopologyDynamicWKV7
	layerTopologyKeyedDeltaHybrid
	layerTopologyCompressedHyper
	layerTopologyGatedHyper
)

// outputHeadPolicy: compiled terminal projection source.
type outputHeadPolicy uint8

const (
	outputHeadTokenEmbedding outputHeadPolicy = iota
	OutputHeadDedicated
)

// terminalPlan: compiled final normalization and projection.
type terminalPlan struct {
	Normalization OutputNormPolicy
	OutputHead    outputHeadPolicy
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
	Program             layerProgram
	Composition         layerCompositionPolicy
	Cache               CachePolicy
	Attention           AttentionPolicy
	EncoderOperator     EncoderOperatorPolicy
	LatentAttention     latentAttentionPolicy
	LatentYaRNQuery     bool
	Position            PositionPolicy
	Residual            ResidualPolicy
	FeedForward         FeedForwardPolicy
	CacheMode           CacheStateMode
	CacheWrite          cacheWritePolicy
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
	Normalization       normalizationPlan
	Rotary              rotaryPlan
	AttentionGraph      AttentionGraphPlan
	Experts             moEGraphPlan
	ExpertComposition   expertCompositionPlan
	DenseWeights        denseWeightPlan
	rotaryCatalog       rotaryCatalogPlan
	ExplicitEncoder     bool
	AllowNonCausalCache bool
	DeciSparse          bool
	QKPreprocess        qkPreprocessPlan
	QueryScale          queryScalePlan
	AttentionOutput     attentionOutputPlan
	ResidualStages      residualStagePlan
	Mixer               RecurrentMixerPolicy
	RecurrentRuntime    recurrentRuntimePolicy
	PeriodicScale       float32
}

// ExactAttentionReplay reports whether captured Q/K tensors fully determine
// this layer's attention result without additional position-dependent policy.
func (p LayerPlan) ExactAttentionReplay() bool {
	attention := p.AttentionGraph
	return p.HasKV && attention.Causal && !attention.UseSinks && !attention.ChunkedWindow &&
		attention.Window == 0 && attention.Softcap == 0 && attention.MaxALiBiBias == 0 && attention.BlockSize == 0
}

func (s Spec) planLayer(profile ArchitectureProfile, layer uint32, recurrent bool) (LayerPlan, error) {
	recurrent = recurrent || s.IsRecurrentLayer(layer)
	hasKV := s.LayerHasKV(layer)
	sharedKV := profile.Has(architectureSharedKV) && !hasKV
	var kvSource uint32
	if sharedKV {
		kvSource = s.LayerSharedKVSource(layer)
	}
	deepstackBefore, deepstackAfter := deepstackSources(s, profile, layer)
	auxiliaryInput, auxiliaryOutput := auxiliaryFlow(s, profile, layer)
	temperature := profile.Temperature
	if temperature == attentionTemperatureNoRoPE && s.UsesRoPE(layer) {
		temperature = AttentionTemperatureNone
	}
	normalization := profile.Runtime.normalizationPlan(s, profile)
	cache := cachePolicy(s, profile, layer, recurrent)
	mixer := compileRecurrentMixer(profile, recurrent)
	composition := layerCompositionStandard
	if mixer == recurrentMixerSparseGroupedSelectiveScan {
		switch {
		case recurrent:
			composition = layerCompositionRecurrentOnly
		case s.LayerHasFeedForward(layer):
			composition = layerCompositionFeedForwardOnly
		default:
			composition = layerCompositionAttentionOnly
		}
	}
	deciSparse := profile.DeciSparse &&
		(!s.LayerHasFeedForward(layer) || !s.LayerHasAttention(layer) ||
			!s.LayerHasKVHeads(layer))
	if deciSparse {
		switch {
		case !s.LayerHasFeedForward(layer):
			composition = layerCompositionIdentity
		case !s.LayerHasAttention(layer):
			composition = layerCompositionFeedForwardOnly
		}
	}
	residualStages := s.residualStagePlan(profile, normalization)
	var periodicScale float32
	if (profile.LayerTopology == layerTopologyDynamicWKV6 || profile.LayerTopology == layerTopologyAffineWKV6) &&
		s.IsPeriodicRescaleLayer(layer) {
		periodicScale = profile.Runtime.Recurrent.PeriodicResidualScale
	}
	if profile.LayerTopology == layerTopologyAffineWKV6 {
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
	if profile.Attention == attentionShortConvolution && recurrent {
		experts.NormalizeTopKProb = true
		experts.SelectionBias = true
	}
	cacheWrite := cacheWriteFixed
	if cache.PrimaryMode().TokenAligned() {
		cacheWrite = cacheWriteConcatOnly
		if mixer == recurrentMixerNone || profile.Attention == AttentionGatedDelta {
			cacheWrite = cacheWriteConcatOrAppend
		}
	}
	plan := LayerPlan{
		Layer:               layer,
		Attention:           profile.Attention,
		EncoderOperator:     profile.EncoderOperator,
		LatentAttention:     profile.LatentAttention,
		LatentYaRNQuery:     profile.Has(architectureLatentYaRNQuery),
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
		MultiAxis:           profile.Has(architectureMultiAxisPositions),
		HasKV:               hasKV,
		SharedKV:            sharedKV,
		KVSource:            kvSource,
		DeepstackBefore:     deepstackBefore,
		DeepstackAfter:      deepstackAfter,
		AuxiliaryInput:      auxiliaryInput,
		AuxiliaryOutput:     auxiliaryOutput,
		Temperature:         temperature,
		AttentionBlocks:     profile.AttentionBlocks,
		EmbeddingSkip:       profile.Has(architectureEmbeddingSkip),
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

// SupportsCapacityCache: all token caches admit bounded append. A shared-KV
// layer appends nothing and reads its source layer's page in the same graph,
// so it joins the capacity cache when that source does and the plan decodes
// through the persistent device cache that builds the graph.
func (p ModelPlan) SupportsCapacityCache() bool {
	if len(p.layers) == tensor.FirstOffset {
		return false
	}
	for _, layer := range p.layers {
		if layer.AttentionGraph.BlockSize != 0 {
			return false
		}
		if !layer.CacheMode.TokenAligned() {
			continue
		}
		if layer.CacheWrite != cacheWriteConcatOrAppend {
			return false
		}
		if layer.SharedKV && (!p.forward.PersistentDeviceCache() || int(layer.KVSource) >= len(p.layers) ||
			!p.layers[layer.KVSource].CacheMode.TokenAligned() ||
			p.layers[layer.KVSource].CacheWrite != cacheWriteConcatOrAppend) {
			return false
		}
	}
	return true
}

// SupportsMultiAxisPositionsWithProfile: bound-profile MRoPE contract.
func (s Spec) SupportsMultiAxisPositionsWithProfile(profile ArchitectureProfile) bool {
	if !profile.Has(architectureMultiAxisPositions) {
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
	case deepstackMappedBefore:
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
	case deepstackSequentialAfter:
		if layer < s.DeepstackLayerCount {
			return DeepstackSourceNone, DeepstackSource(layer)
		}
	}
	return DeepstackSourceNone, DeepstackSourceNone
}

func auxiliaryFlow(s Spec, profile ArchitectureProfile, layer uint32) (AuxiliaryFlow, AuxiliaryFlow) {
	switch profile.Auxiliary {
	case auxiliaryRecurrentValue:
		if layer == tensor.FirstOffset {
			return AuxiliaryNone, auxiliaryRecurrentValue
		}
		return auxiliaryRecurrentValue, AuxiliaryNone
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
	normalization normalizationPlan
	terminal      terminalPlan
	draft         DraftPlan
	cacheProject  cacheProjectionProgram
	projections   [projectionRoleCount]projectionProgram
	sequenceOut   sequenceOutputProgram
	forward       ForwardProgram
	input         projectedInputProgram
}

// projectedInputProgram: compiled projected-input and specialized-output admission.
type projectedInputProgram struct {
	Overrides          EmbeddingOverridePolicy
	AttentionBlocks    AttentionBlockPolicy
	DeepstackStreams   uint32
	MultiAxis          bool
	PerLayerEmbeddings bool
	ClassifierHead     bool
}

func compileProjectedInputProgram(spec Spec, profile ArchitectureProfile) projectedInputProgram {
	var deepstackStreams uint32
	if profile.Deepstack != deepstackNone {
		deepstackStreams = spec.DeepstackLayerCount
	}
	return projectedInputProgram{
		Overrides: profile.Overrides, AttentionBlocks: profile.AttentionBlocks,
		DeepstackStreams:   deepstackStreams,
		MultiAxis:          spec.SupportsMultiAxisPositionsWithProfile(profile),
		PerLayerEmbeddings: spec.hasPerLayerEmbeddings(profile),
		ClassifierHead:     profile.Has(architectureClassifierHead),
	}
}

func (s Spec) hasPerLayerEmbeddings(profile ArchitectureProfile) bool {
	return profile.Has(architecturePerLayerEmbeddings) && s.EmbeddingPerLayer > tensor.FirstOffset
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
		terminal:      terminalPlan{Normalization: profile.OutputNorm},
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
		(plan.draft.Kind != draftOptionalSingleCatalog || weights.OptionalCatalogDraft != nil) {
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
func (p ModelPlan) Normalization() normalizationPlan { return p.normalization }

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
		(plan.draft.Kind != draftOptionalSingleCatalog || weights.OptionalCatalogDraft != nil) {
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
		if plan.draft.Kind == draftSingleCatalog {
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
	if spec.SharedKVLayers > tensor.FirstOffset && (!plan.profile.Has(architectureSharedKV) ||
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
		case deepstackMappedBefore:
			if len(spec.DeepstackMapping) < len(plan.layers) {
				return fmt.Errorf("model plan deepstack map has %d entries for %d layers", len(spec.DeepstackMapping), len(plan.layers))
			}
		case deepstackSequentialAfter:
			if int(spec.DeepstackLayerCount) > len(plan.layers) {
				return fmt.Errorf("model plan has %d deepstack streams for %d layers", spec.DeepstackLayerCount, len(plan.layers))
			}
		default:
			return fmt.Errorf("model plan architecture %s has no deepstack policy", spec.Architecture)
		}
	}
	if plan.profile.AttentionBlocks != AttentionBlocksNone && !plan.profile.Has(architectureMultimodal) {
		return fmt.Errorf("model plan architecture %s has attention blocks without multimodal support", spec.Architecture)
	}
	if attentionTemperatureConfigured(spec) {
		if plan.profile.Temperature == AttentionTemperatureNone || !validAttentionTemperature(spec) {
			return fmt.Errorf("model plan architecture %s has an invalid attention temperature contract", spec.Architecture)
		}
	}
	if norm.PostNormLayout == postNormLayoutOutputLayer &&
		(norm.Operation != normalizationLayer || norm.PreAttention || !norm.PostAttention || !norm.Bias) {
		return fmt.Errorf("model plan architecture %s has an invalid output-layer normalization layout", spec.Architecture)
	}
	draft := plan.profile.DraftPlan(spec.NextNPredictLayers)
	if spec.NextNPredictLayers > tensor.FirstOffset && (draft.Kind == draftNone || !draft.HasHead(firstDraftHead)) {
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
		return cachedGraphLayered
	}
	for _, layer := range layers {
		if layer.Mixer != recurrentMixerNone || layer.Attention != attentionStandard ||
			layer.Cache != cacheAttention && layer.Cache != cacheSentinel {
			return cachedGraphLayered
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
func (p ModelPlan) ProjectedInput() projectedInputProgram { return p.input }

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
func (p ModelPlan) Terminal() terminalPlan { return p.terminal }

// Projection: compiled auxiliary projection program; absent roles fail on Build.
func (p ModelPlan) Projection(role ProjectionRole) projectionProgram {
	if role >= projectionRoleCount {
		return projectionProgram{}
	}
	return p.projections[role]
}

// CacheProjection: compiled auxiliary K/V projection program.
func (p ModelPlan) CacheProjection() cacheProjectionProgram { return p.cacheProject }

// SequenceOutput: compiled terminal sequence program.
func (p ModelPlan) SequenceOutput() sequenceOutputProgram { return p.sequenceOut }

// AudioWaveform: compiled semantic-token waveform plan.
func (p ModelPlan) AudioWaveform() AudioWaveformPlan { return p.forward.Waveform }

// Draft: compiled speculative policy.
func (p ModelPlan) Draft() DraftPlan { return p.draft }

func compileLayerProgram(plan LayerPlan, profile ArchitectureProfile) (layerProgram, error) {
	recurrent := plan.Recurrent
	composition := plan.Composition
	if profile.LayerTopology == layerTopologySplitProjection {
		return newLayerProgram(
			layerStage(layerOperatorActivationProjection),
			layerStage(layerOperatorActivatedOutput),
		)
	}
	if profile.Attention == attentionShortConvolution && recurrent {
		return residualMixerProgram(
			attentionLayerStage(layerOperatorRecurrentMix), layerOperatorFeedForwardPolicy,
		)
	}
	if profile.LayerTopology == layerTopologyAffineWKV6 {
		return residualMixerProgram(
			attentionLayerStage(layerOperatorRecurrentMix), layerOperatorFeedForwardPolicy,
			plan.ResidualStages.residualScale,
		)
	}
	if profile.Attention == AttentionGatedDelta {
		mixer := attentionLayerStage(layerOperatorAttentionGatedProjection)
		if recurrent {
			mixer = attentionLayerStage(layerOperatorRecurrentMix)
		}
		if profile.LayerTopology == layerTopologyGatedHyper {
			if !recurrent {
				if plan.AttentionGraph.BlockSize == 0 {
					return layerProgram{}, errors.New("block sparse attention compression ratio is missing")
				}
				mixer = leafLayerStage(layerOperatorAttentionGatedProjection,
					[]runtimeCacheBinding{runtimeCachePrimaryKey, runtimeCachePrimaryValue, runtimeCacheIndexerKey},
					[]runtimeTensorBinding{runtimeTensorCurrentPositions})
			}
			stages := []layerOperatorInstruction{
				layerStage(layerOperatorGatedHyperAttention), mixer,
				layerStage(layerOperatorGatedHyperCombine), layerStage(layerOperatorGatedHyperFeedForward),
				layerStage(layerOperatorFeedForwardRoutedSwiGLU), layerStage(layerOperatorGatedHyperCombine),
				layerStage(layerOperatorGatedHyperHead),
			}
			return newLayerProgram(stages...)
		}
		return residualMixerProgram(mixer, layerOperatorFeedForwardRoutedSwiGLU)
	}
	if plan.Mixer == recurrentMixerAttentionGroupedSelectiveScan {
		return residualMixerProgram(
			hybridLayerStage(), layerOperatorFeedForwardPolicy,
		)
	}
	if plan.Mixer == recurrentMixerScaledGroupedSelectiveScan {
		return newLayerProgram(
			layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorRecurrentMix),
			scaleLayerStage(plan.ResidualStages.residualScale), layerStage(layerOperatorResidual),
			layerStage(layerOperatorFeedForwardNorm), layerStage(layerOperatorFeedForwardPolicy),
			scaleLayerStage(plan.ResidualStages.residualScale), layerStage(layerOperatorResidual),
		)
	}
	if plan.Mixer == recurrentMixerWeightedSelectiveScan {
		return residualMixerProgram(
			attentionLayerStage(layerOperatorRecurrentMix), layerOperatorFeedForwardPolicy,
		)
	}
	if plan.Mixer == recurrentMixerNormalizedSelectiveScan {
		return newLayerProgram(
			layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorRecurrentMix),
			layerStage(layerOperatorAttentionPostNorm), layerStage(layerOperatorResidual),
			layerStage(layerOperatorFeedForwardNorm), layerStage(layerOperatorFeedForwardPolicy),
			layerStage(layerOperatorFeedForwardPostNorm), layerStage(layerOperatorResidual),
		)
	}
	if plan.Mixer == recurrentMixerSparseGroupedSelectiveScan {
		switch composition {
		case layerCompositionRecurrentOnly:
			return newLayerProgram(
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorRecurrentMix),
				layerStage(layerOperatorResidual),
			)
		case layerCompositionAttentionOnly:
			return newLayerProgram(
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorAttentionCausalProjection),
				layerStage(layerOperatorResidual),
			)
		case layerCompositionFeedForwardOnly:
			return newLayerProgram(
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorCacheSentinel),
				layerStage(layerOperatorFeedForwardRoutedSquaredReLU), layerStage(layerOperatorResidual),
			)
		default:
			return layerProgram{}, errors.New("layer program has unsupported sparse composition")
		}
	}
	if plan.Mixer == recurrentMixerSelectiveScan || plan.Mixer == recurrentMixerGroupedSelectiveScan {
		return newLayerProgram(
			layerStage(layerOperatorAttentionNorm),
			attentionLayerStage(layerOperatorRecurrentMix),
			layerStage(layerOperatorResidual),
		)
	}
	switch {
	default:
		if profile.Forward.Session == ForwardSessionPairedProjection {
			return newLayerProgram(
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorAttentionSharedCacheQKNorm),
				layerStage(layerOperatorAttentionPostNorm), layerStage(layerOperatorResidual),
				layerStage(layerOperatorFeedForwardNorm), layerStage(layerOperatorFeedForwardPolicy),
				layerStage(layerOperatorFeedForwardPostNorm), layerStage(layerOperatorResidualScale),
			)
		}
		if profile.Forward.Session == ForwardSessionFeatureDraft {
			return newLayerProgram(
				pairedInputLayerStage(), attentionLayerStage(layerOperatorAttentionPairedCausalProjection),
				layerStage(layerOperatorResidual), layerStage(layerOperatorFeedForwardNorm),
				layerStage(layerOperatorFeedForwardPolicy), layerStage(layerOperatorResidual),
			)
		}
		if profile.LayerTopology == layerTopologyDynamicWKV6 {
			stages := []layerOperatorInstruction{
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorRecurrentMix),
				layerStage(layerOperatorResidual), tokenShiftLayerStage(layerOperatorGatedTokenShiftSquaredReLU),
				layerStage(layerOperatorResidual),
			}
			if positiveFinite(plan.PeriodicScale) {
				stages = append(stages, scaleLayerStage(plan.PeriodicScale))
			}
			return newLayerProgram(stages...)
		}
		if profile.LayerTopology == layerTopologyDynamicWKV7 {
			stages := []layerOperatorInstruction{
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorRecurrentMix),
				layerStage(layerOperatorResidual),
			}
			if profile.Normalization == normalizationLayer {
				stages = append(stages, tokenShiftLayerStage(layerOperatorTokenShiftSquaredReLU))
			} else {
				stages = append(stages, layerStage(layerOperatorFeedForwardNorm),
					layerStage(layerOperatorFeedForwardPolicy))
			}
			return newLayerProgram(append(stages, layerStage(layerOperatorResidual))...)
		}
		if profile.LayerTopology == layerTopologySharedKVAdapter {
			return newLayerProgram(
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorAttentionSharedKVQKNorm),
				layerStage(layerOperatorAttentionPostNorm), layerStage(layerOperatorResidual),
				layerStage(layerOperatorFeedForwardParallelGatedGELU), layerStage(layerOperatorResidual),
				layerStage(layerOperatorOutputAdapter),
			)
		}
		if profile.LayerTopology == LayerTopologyCausalPostQKNormSkip {
			return newLayerProgram(
				layerStage(layerOperatorRMSNorm), attentionLayerStage(layerOperatorAttentionCausalPostQKNorm),
				layerStage(layerOperatorResidual), layerStage(layerOperatorRMSNorm),
				layerStage(layerOperatorFeedForwardPolicy), layerStage(layerOperatorResidual),
				layerStage(layerOperatorScaledSkip),
			)
		}
		if profile.LayerTopology == layerTopologyBidirectionalQKNorm {
			return newLayerProgram(
				layerStage(layerOperatorAttentionNorm), attentionLayerStage(layerOperatorAttentionBidirectionalQKNorm),
				layerStage(layerOperatorAttentionPostNorm), layerStage(layerOperatorResidual),
				layerStage(layerOperatorFeedForwardNorm), layerStage(layerOperatorFeedForwardPolicy),
				layerStage(layerOperatorFeedForwardPostNorm), layerStage(layerOperatorResidual),
			)
		}
		if profile.LayerTopology == layerTopologyBidirectionalFusedQKV {
			return newLayerProgram(
				attentionLayerStage(layerOperatorAttentionBidirectionalFusedQKV), layerStage(layerOperatorResidual),
				layerStage(layerOperatorFeedForwardNorm), layerStage(layerOperatorFeedForwardPolicy),
				layerStage(layerOperatorResidual),
			)
		}
		if profile.LayerTopology == layerTopologyBidirectionalEncoder {
			stages := []layerOperatorInstruction{
				attentionLayerStage(layerOperatorAttentionBidirectionalEncoder),
				layerStage(layerOperatorAttentionResidualNorm),
			}
			if profile.EncoderOperator.usesALiBiQKNorm() {
				stages = append(stages, layerStage(layerOperatorInputResidualNorm))
			}
			return newLayerProgram(append(stages,
				layerStage(layerOperatorFeedForwardEncoder),
				layerStage(layerOperatorFeedForwardResidualNorm),
			)...)
		}
		if plan.DeciSparse {
			stages := []layerOperatorInstruction{attentionLayerStage(layerOperatorCacheSentinel)}
			if plan.Composition == layerCompositionIdentity {
				return newLayerProgram(stages...)
			}
			if plan.Composition != layerCompositionFeedForwardOnly {
				stages = append(stages, layerStage(layerOperatorAttentionNorm),
					layerStage(layerOperatorAttentionOutputProjection),
					layerStage(layerOperatorResidual))
			}
			return newLayerProgram(append(stages,
				layerStage(layerOperatorFeedForwardNorm),
				layerStage(layerOperatorFeedForwardPolicy),
				layerStage(layerOperatorResidual),
			)...)
		}
		if profile.LayerTopology == layerTopologyPlannedDecoder && !plan.DeciSparse {
			return newLayerProgram(
				layerStage(layerOperatorAttentionInputNorm),
				attentionLayerStage(layerOperatorAttentionPlannedProjection),
				layerStage(layerOperatorResidual), layerStage(layerOperatorFeedForwardInputNorm),
				layerStage(layerOperatorFeedForwardPolicy),
				layerStage(layerOperatorFeedForwardOutput), layerStage(layerOperatorResidual),
			)
		}
		return layerProgram{}, errors.New("layer program has no compiled topology")
	case profile.LayerTopology == layerTopologyKeyedDeltaHybrid:
		if !recurrent {
			return latentLayerProgram(
				profile, plan.ResidualStages.residualScale,
				[]runtimeCacheBinding{runtimeCachePrimaryKey, runtimeCachePrimaryValue}, nil,
			)
		}
		return residualMixerProgram(
			attentionLayerStage(layerOperatorRecurrentMix),
			layerOperatorFeedForwardPolicy,
		)
	case plan.Attention == attentionLatent:
		return latentLayerProgram(
			profile, plan.ResidualStages.residualScale,
			[]runtimeCacheBinding{runtimeCachePrimaryKey, runtimeCachePrimaryValue}, nil,
		)
	case plan.Attention == AttentionSparseLatent:
		return latentLayerProgram(
			profile, plan.ResidualStages.residualScale,
			[]runtimeCacheBinding{runtimeCachePrimaryKey, runtimeCachePrimaryValue, runtimeCacheIndexerKey},
			[]runtimeTensorBinding{runtimeTensorPerLayerInput},
		)
	case profile.LayerTopology == layerTopologyCompressedHyper:
		return newLayerProgram(
			leafLayerStage(
				layerOperatorHyperAttention,
				[]runtimeCacheBinding{runtimeCachePrimaryKey},
				[]runtimeTensorBinding{runtimeTensorCurrentPositions},
			),
			layerStage(layerOperatorHyperFeedForward),
		)
	}
}

func (p LayerPlan) splitProjection() bool {
	prefix, prefixOK := p.Program.Instruction(tensor.FirstOffset)
	suffix, suffixOK := p.Program.Instruction(tensor.SingletonExtent)
	return prefixOK && suffixOK && prefix.Operator == layerOperatorActivationProjection &&
		suffix.Operator == layerOperatorActivatedOutput
}

func residualMixerProgram(mixer layerOperatorInstruction, feedForward layerOperator, scale ...float32) (layerProgram, error) {
	stages := []layerOperatorInstruction{
		layerStage(layerOperatorAttentionNorm), mixer, layerStage(layerOperatorResidual),
		layerStage(layerOperatorFeedForwardNorm), layerStage(feedForward), layerStage(layerOperatorResidual),
	}
	if len(scale) != 0 && positiveFinite(scale[tensor.FirstOffset]) {
		stages = append(stages, scaleLayerStage(scale[tensor.FirstOffset]))
	}
	return newLayerProgram(stages...)
}

func latentLayerProgram(
	profile ArchitectureProfile,
	scale float32,
	caches []runtimeCacheBinding,
	tensors []runtimeTensorBinding,
) (layerProgram, error) {
	stages := []layerOperatorInstruction{
		layerStage(layerOperatorAttentionNorm),
		leafLayerStage(layerOperatorLatentAttention, caches, tensors),
		layerStage(layerOperatorResidual), layerStage(layerOperatorFeedForwardNorm),
		layerStage(layerOperatorFeedForwardPolicy),
	}
	if profile.LatentAttention == latentAttentionNeoXResidualScale {
		stages = append(stages, scaleLayerStage(scale))
	}
	return newLayerProgram(append(stages, layerStage(layerOperatorResidual))...)
}

func leafLayerStage(
	operator layerOperator,
	caches []runtimeCacheBinding,
	tensors []runtimeTensorBinding,
) layerOperatorInstruction {
	instruction := layerStage(operator)
	instruction.CacheCount = uint8(len(caches))
	instruction.TensorCount = uint8(len(tensors))
	copy(instruction.Caches[:], caches)
	copy(instruction.Tensors[:], tensors)
	return instruction
}

func layerStage(operator layerOperator) layerOperatorInstruction {
	return layerOperatorInstruction{Operator: operator}
}

func scaleLayerStage(scale float32) layerOperatorInstruction {
	return layerOperatorInstruction{Operator: layerOperatorScale, Scalar: scale}
}

func attentionLayerStage(operator layerOperator) layerOperatorInstruction {
	return leafLayerStage(
		operator, []runtimeCacheBinding{runtimeCachePrimaryKey, runtimeCachePrimaryValue}, nil,
	)
}

func hybridLayerStage() layerOperatorInstruction {
	return leafLayerStage(
		layerOperatorHybridMix,
		[]runtimeCacheBinding{
			runtimeCachePrimaryKey, runtimeCachePrimaryValue, runtimeCacheConvolution, runtimeCacheSSM,
		}, nil,
	)
}

func tokenShiftLayerStage(operator layerOperator) layerOperatorInstruction {
	return leafLayerStage(operator, []runtimeCacheBinding{runtimeCachePrimaryKey}, nil)
}

func pairedInputLayerStage() layerOperatorInstruction {
	return leafLayerStage(
		layerOperatorPairedInputNorm, nil, []runtimeTensorBinding{runtimeTensorPerLayerInput},
	)
}

func relativeEncoderProgram() (layerProgram, error) {
	return newLayerProgram(
		layerStage(layerOperatorAttentionNorm),
		layerStage(layerOperatorAttentionRelativeBidirectional),
		layerStage(layerOperatorResidual),
		layerStage(layerOperatorFeedForwardRelative),
		layerStage(layerOperatorResidual),
	)
}

func relativeDecoderProgram() (layerProgram, error) {
	return newLayerProgram(
		layerStage(layerOperatorAttentionNorm),
		attentionLayerStage(layerOperatorAttentionRelativeCausal),
		layerStage(layerOperatorResidual),
		layerStage(layerOperatorCrossAttentionNorm),
		leafLayerStage(
			layerOperatorAttentionCross,
			[]runtimeCacheBinding{runtimeCacheCrossKey, runtimeCacheCrossValue},
			[]runtimeTensorBinding{runtimeTensorEncoder},
		),
		layerStage(layerOperatorResidual),
		layerStage(layerOperatorFeedForwardRelative),
		layerStage(layerOperatorResidual),
	)
}

func newLayerProgram(stages ...layerOperatorInstruction) (layerProgram, error) {
	if len(stages) == tensor.FirstOffset {
		return layerProgram{}, errors.New("layer program is empty")
	}
	return layerProgram{Instructions: stages}, nil
}

func cachePolicy(spec Spec, profile ArchitectureProfile, layer uint32, recurrent bool) CachePolicy {
	if recurrent && profile.RecurrentCache != cacheAttention {
		return profile.RecurrentCache
	}
	if profile.Cache != cacheAttention {
		return profile.Cache
	}
	if profile.CacheFallback == cacheFallbackFeedForward && spec.LayerHasFeedForward(layer) ||
		profile.CacheFallback == cacheFallbackMissingKV && !spec.LayerHasKVHeads(layer) {
		return cacheSentinel
	}
	return cacheAttention
}
