package model

import "llamacpp2go/internal/tensor"

type rotaryGraphKind uint8

const (
	rotaryGraphNone rotaryGraphKind = iota
	rotaryGraphSingle
	rotaryGraphMulti
)

type multiAxisRotaryPolicy uint8

const (
	multiAxisRotaryNone multiAxisRotaryPolicy = iota
	multiAxisRotaryAlways
	multiAxisRotaryWithSections
)

type rotaryPolicyKind uint8

const (
	rotaryPolicyDefault rotaryPolicyKind = iota
	rotaryPolicyLaguna
	rotaryPolicyGrokMellum
	rotaryPolicyLlamaYaRN
	rotaryPolicyNormal
	rotaryPolicyGemma
)

// RotaryUsagePolicy: layer-level RoPE schedule.
type RotaryUsagePolicy uint8

const (
	RotaryUsageStandard RotaryUsagePolicy = iota
	RotaryUsageSlidingMetadata
	RotaryUsageDensePrefixOrSliding
	RotaryUsagePeriodicZeroBased
	RotaryUsageSlidingOnly
)

// RotaryPolicy: architecture-owned rotary selection.
type RotaryPolicy struct {
	Kind              rotaryPolicyKind
	MultiAxis         multiAxisRotaryPolicy
	Usage             RotaryUsagePolicy
	SlidingFrequency  bool
	SlidingScaleReset bool
	Gemma3            bool
	FactorPairs       bool
}

// AttentionGraphPolicy: architecture-owned attention controls.
type AttentionGraphPolicy struct {
	UseSinks      bool
	ChunkedWindow bool
	QwenGDN       qwenGDNPolicy
}

type qwenGDNPolicy uint8

const (
	qwenGDNNone qwenGDNPolicy = iota
	qwenGDNStandard
	qwenGDNRepeatInterleave
)

func applyRoPEPairWithOptions(
	builder *tensor.Builder,
	query, key *tensor.Tensor,
	options tensor.RoPEOptions,
) (*tensor.Tensor, *tensor.Tensor) {
	return builder.RoPEWithOptions(query, options), builder.RoPEWithOptions(key, options)
}

// RotaryPlan: compiled per-layer rotary graph.
type RotaryPlan struct {
	kind             rotaryGraphKind
	layout           tensor.RoPELayout
	sections         [4]int32
	rotaryDimensions uint32
	frequencyBase    float32
	frequencyScale   float32
	yarn             bool
	originalContext  uint32
	extFactor        float32
	attentionFactor  float32
	betaFast         float32
	betaSlow         float32
	factorPairs      uint32
	outputScale      float32
}

// Apply: shared query/key rotary graph.
func (p RotaryPlan) Apply(
	builder *tensor.Builder,
	query, key *tensor.Tensor,
	positions []uint32,
	multiPositions *[4][]uint32,
	frequencyFactors *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor) {
	frequencyFactors = p.resolveFactors(builder, frequencyFactors)
	query = p.applyOne(builder, query, positions, multiPositions, frequencyFactors)
	key = p.applyOne(builder, key, positions, multiPositions, frequencyFactors)
	return query, key
}

// ApplyOne: single-tensor rotary graph.
func (p RotaryPlan) ApplyOne(
	builder *tensor.Builder,
	input *tensor.Tensor,
	positions []uint32,
	multiPositions *[4][]uint32,
	frequencyFactors *tensor.Tensor,
) *tensor.Tensor {
	return p.applyOne(
		builder, input, positions, multiPositions,
		p.resolveFactors(builder, frequencyFactors),
	)
}

func (p RotaryPlan) resolveFactors(
	builder *tensor.Builder,
	frequencyFactors *tensor.Tensor,
) *tensor.Tensor {
	if p.factorPairs > 0 && frequencyFactors != nil &&
		frequencyFactors.Shape.Dims[0] > uint64(p.factorPairs) {
		return builder.FlatSlice(frequencyFactors, 0, uint64(p.factorPairs))
	}
	return frequencyFactors
}

func (p RotaryPlan) applyOne(
	builder *tensor.Builder,
	input *tensor.Tensor,
	positions []uint32,
	multiPositions *[4][]uint32,
	frequencyFactors *tensor.Tensor,
) *tensor.Tensor {
	switch p.kind {
	case rotaryGraphNone:
		return input
	case rotaryGraphMulti:
		resolved := [4][]uint32{}
		if multiPositions == nil {
			for axis := range resolved {
				resolved[axis] = positions
			}
		} else {
			resolved = *multiPositions
		}
		input = builder.RoPEMultiScaled(
			input, resolved, p.sections, p.rotaryDimensions,
			p.frequencyBase, p.frequencyScale,
		)
	case rotaryGraphSingle:
		options := tensor.RoPEOptions{
			Layout: p.layout, Positions: positions, FrequencyFactors: frequencyFactors,
			RotaryDimensions: p.rotaryDimensions, FrequencyBase: p.frequencyBase,
			FrequencyScale: p.frequencyScale, YaRN: p.yarn,
			OriginalContext: p.originalContext, ExtFactor: p.extFactor,
			AttentionFactor: p.attentionFactor, BetaFast: p.betaFast, BetaSlow: p.betaSlow,
		}
		input = builder.RoPEWithOptions(input, options)
	}
	if p.outputScale > 0 && p.outputScale != 1 {
		input = builder.Scale(input, p.outputScale)
	}
	return input
}

func (s Spec) rotaryPlan(profile ArchitectureProfile, layer uint32) RotaryPlan {
	if !s.UsesRoPE(layer) {
		return RotaryPlan{}
	}
	rotaryDimensions := s.KeyLength
	if s.RopeDimensionCount > 0 {
		rotaryDimensions = s.LayerRopeDimensionCount(layer)
	}
	plan := RotaryPlan{
		kind: rotaryGraphSingle, layout: tensor.RoPELayoutNeoX,
		rotaryDimensions: rotaryDimensions, frequencyBase: s.RopeFrequencyBase,
		frequencyScale: 1,
	}
	policy := profile.Rotary
	multiAxis := policy.MultiAxis == multiAxisRotaryAlways ||
		policy.MultiAxis == multiAxisRotaryWithSections &&
			s.RopeSections[0] > 0 && s.RopeSections[1] > 0
	if multiAxis {
		plan.kind = rotaryGraphMulti
		plan.sections = s.RopeSections
		if s.RopeScalingType == "linear" {
			plan.frequencyScale = 1 / s.RopeScalingFactor
		}
		return plan
	}
	setYaRN := func(layout tensor.RoPELayout) {
		plan.layout = layout
		plan.yarn = true
		plan.originalContext = s.OriginalContextLength
		plan.frequencyScale = 1 / s.RopeScalingFactor
		plan.extFactor = s.YaRNExtFactor
		plan.attentionFactor = s.YaRNAttentionFactor
		plan.betaFast = s.YaRNBetaFast
		plan.betaSlow = s.YaRNBetaSlow
	}
	applyNormal := func() {
		plan.layout = tensor.RoPELayoutNormal
		if s.RopeScalingType == "linear" {
			plan.frequencyScale = 1 / s.RopeScalingFactor
		}
		if policy.SlidingFrequency && s.IsSlidingLayer(layer) {
			plan.frequencyBase = s.RopeFrequencySWA
		}
	}
	applyDefault := func() {
		if s.RopeScalingType == "linear" {
			plan.frequencyScale = 1 / s.RopeScalingFactor
		}
		if policy.SlidingScaleReset && s.IsSlidingLayer(layer) {
			plan.frequencyScale = 1
		}
		if policy.SlidingFrequency && s.IsSlidingLayer(layer) {
			plan.frequencyBase = s.RopeFrequencySWA
		}
	}
	switch policy.Kind {
	case rotaryPolicyLaguna:
		if !s.IsSlidingLayer(layer) {
			setYaRN(tensor.RoPELayoutNeoX)
			break
		}
		plan.rotaryDimensions = s.RopeDimensionSWA
		plan.frequencyBase = s.RopeFrequencySWA
	case rotaryPolicyGrokMellum:
		if s.RopeScalingType == "yarn" && !s.IsSlidingLayer(layer) {
			setYaRN(tensor.RoPELayoutNeoX)
		} else {
			applyDefault()
		}
	case rotaryPolicyLlamaYaRN:
		if s.RopeScalingType == "yarn" {
			setYaRN(tensor.RoPELayoutNormal)
		} else {
			applyNormal()
		}
	case rotaryPolicyNormal:
		applyNormal()
	case rotaryPolicyGemma:
		if policy.Gemma3 || s.RopeScalingType == "linear" {
			plan.frequencyScale = 1 / s.RopeScalingFactor
		}
		if s.IsSlidingLayer(layer) {
			plan.frequencyBase = s.RopeFrequencySWA
			if policy.Gemma3 {
				plan.frequencyScale = 1
			}
		}
	default:
		applyDefault()
	}
	if policy.FactorPairs {
		plan.factorPairs = rotaryDimensions / 2
	}
	if profile.Has(ArchitectureLongRoPE) && s.RopeScalingType != "yarn" &&
		s.RopeAttentionFactor > 0 && s.RopeAttentionFactor != 1 {
		plan.outputScale = s.RopeAttentionFactor
	}
	return plan
}

// AttentionGraphPlan: compiled per-layer attention graph.
type AttentionGraphPlan struct {
	Causal        bool
	UseSinks      bool
	ChunkedWindow bool
	Window        uint32
	Softcap       float32
	MaxALiBiBias  float32
}

// Build: materializes compiled attention controls.
func (p AttentionGraphPlan) Build(
	builder *tensor.Builder,
	query, key, value *tensor.Tensor,
	sinks, blockIDs *tensor.Tensor,
	scale float32,
	queryStart uint32,
) *tensor.Tensor {
	if !p.UseSinks {
		sinks = nil
	}
	if p.Window == 0 {
		blockIDs = nil
	}
	return builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
		Sinks: sinks, BlockIDs: blockIDs, Scale: scale, Softcap: p.Softcap,
		MaxALiBiBias: p.MaxALiBiBias, Causal: p.Causal,
		ChunkedWindow: p.ChunkedWindow, QueryStart: queryStart, Window: p.Window,
	})
}

func (s Spec) attentionGraphPlan(layer uint32) AttentionGraphPlan {
	policy := s.Profile().AttentionGraph
	plan := AttentionGraphPlan{Causal: !s.NonCausalAttention}
	if s.IsSlidingLayer(layer) {
		plan.Window = s.SlidingWindow
		plan.ChunkedWindow = policy.ChunkedWindow
		plan.Softcap = s.AttentionSoftcap
	} else {
		plan.MaxALiBiBias = s.MaxALiBiBias
		plan.Softcap = s.AttentionSoftcap
	}
	plan.UseSinks = policy.UseSinks
	return plan
}

// MoEGraphPlan: compiled routed-expert controls.
type MoEGraphPlan struct {
	TopK               uint32
	NormalizeTopKProb  bool
	Scale              float32
	Routing            tensor.MoERouting
	Activation         tensor.MoEActivation
	FusedGateUp        bool
	SelectionBias      bool
	ExpertIndexDivisor uint32
	SwiGLUClamp        float32
}

// MoEGraphInputs: routed-expert tensor bindings.
type MoEGraphInputs struct {
	RouterInput     *tensor.Tensor
	Gate            *tensor.Tensor
	Up              *tensor.Tensor
	Down            *tensor.Tensor
	SelectionBias   *tensor.Tensor
	ExpertScale     *tensor.Tensor
	SelectedExperts *tensor.Tensor
	Biases          *tensor.MoEBiases
}

// Build: materializes compiled routed-expert controls.
func (p MoEGraphPlan) Build(
	builder *tensor.Builder,
	input, router *tensor.Tensor,
	inputs MoEGraphInputs,
) *tensor.Tensor {
	return builder.MoEWithOptions(input, router, inputs.Up, inputs.Down, tensor.MoEOptions{
		RouterInput: inputs.RouterInput, Gate: inputs.Gate,
		SelectionBias: inputs.SelectionBias, ExpertScale: inputs.ExpertScale,
		SelectedExperts: inputs.SelectedExperts, Biases: inputs.Biases,
		TopK: p.TopK, NormalizeTopKProb: p.NormalizeTopKProb, Scale: p.Scale,
		Routing: p.Routing, Activation: p.Activation, FusedGateUp: p.FusedGateUp,
		ExpertIndexDivisor: p.ExpertIndexDivisor, SwiGLUClamp: p.SwiGLUClamp,
	})
}

// BuildLayer: standard layer-weight bindings.
func (p MoEGraphPlan) BuildLayer(
	builder *tensor.Builder,
	input, routerInput *tensor.Tensor,
	weights LayerGraphWeights,
) *tensor.Tensor {
	up := weights.FeedForwardUpExperts
	gate := weights.FeedForwardGateExperts
	if weights.FeedForwardGateUpExperts != nil {
		up = weights.FeedForwardGateUpExperts
		gate = nil
		p.FusedGateUp = true
	}
	var selectionBias *tensor.Tensor
	if p.SelectionBias {
		selectionBias = weights.FeedForwardExpertBias
	}
	var biases *tensor.MoEBiases
	if p.Activation == tensor.MoEActivationSwiGLUOAI {
		biases = &tensor.MoEBiases{
			Router: weights.FeedForwardRouterBias,
			Gate:   weights.FeedForwardGateBias,
			Up:     weights.FeedForwardUpBias,
			Down:   weights.FeedForwardDownBias,
		}
	}
	return p.Build(builder, input, weights.FeedForwardRouter, MoEGraphInputs{
		RouterInput: routerInput, Gate: gate, Up: up,
		Down: weights.FeedForwardDownExperts, SelectionBias: selectionBias,
		ExpertScale: weights.FeedForwardDownExpertsScale, Biases: biases,
	})
}

func (s Spec) moeGraphPlan(layer uint32) MoEGraphPlan {
	policy := s.Profile().Experts
	routing := tensor.MoERoutingSoftmax
	if s.ExpertGatingFunc == expertGatingSigmoid {
		routing = tensor.MoERoutingSigmoid
	}
	switch policy.Routing {
	case expertRouteSigmoid:
		routing = tensor.MoERoutingSigmoid
	case expertRouteSelectedSoftmax:
		routing = tensor.MoERoutingSelectedSoftmax
	}
	normalize := true
	switch policy.Normalization {
	case expertNormalizeMetadata:
		normalize = s.ExpertWeightsNorm
	case expertNormalizeNever:
		normalize = false
	}
	activation := policy.Activation
	if activation == 0 {
		activation = tensor.MoEActivationSiLU
	}
	clamp := float32(0)
	if policy.ClampSwiGLU {
		clamp = s.LayerExpertSwiGLUClamp(layer)
	}
	return MoEGraphPlan{
		TopK: s.ExpertUsedCount, NormalizeTopKProb: normalize,
		Scale: s.ExpertWeightsScale, Routing: routing, Activation: activation,
		SelectionBias: policy.SelectionBias, ExpertIndexDivisor: 1, SwiGLUClamp: clamp,
	}
}
