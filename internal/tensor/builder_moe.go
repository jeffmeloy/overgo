package tensor

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor/dtype"
)

// MaxMoETopK: routed experts supported per token.
const MaxMoETopK uint32 = 16

type moeOptions struct {
	routerInput, gate          *Tensor
	selectionBias, expertScale *Tensor
	selectedExperts            *Tensor
	biases                     *MoEBiases
	topK                       uint32
	normalizeTopKProb          bool
	scale                      float32
	routing                    MoERouting
	activation                 MoEActivation
	fusedGateUp                bool
	expertIndexDivisor         uint32
	swigluClamp                float32
}

// MoEBiases: OpenAI routed-expert biases.
type MoEBiases struct {
	Router *Tensor
	Gate   *Tensor
	Up     *Tensor
	Down   *Tensor
}

// MoEOptions: routed-expert graph controls.
type MoEOptions struct {
	RouterInput        *Tensor
	Gate               *Tensor
	SelectionBias      *Tensor
	ExpertScale        *Tensor
	SelectedExperts    *Tensor
	Biases             *MoEBiases
	TopK               uint32
	NormalizeTopKProb  bool
	Scale              float32
	Routing            MoERouting
	Activation         MoEActivation
	FusedGateUp        bool
	ExpertIndexDivisor uint32
	SwiGLUClamp        float32
}

// MoEWithOptions: typed routed-expert construction.
func (b *Builder) MoEWithOptions(
	input, router, up, down *Tensor,
	options MoEOptions,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		routerInput: options.RouterInput, gate: options.Gate,
		selectionBias: options.SelectionBias, expertScale: options.ExpertScale,
		selectedExperts: options.SelectedExperts, biases: options.Biases,
		topK: options.TopK, normalizeTopKProb: options.NormalizeTopKProb,
		scale: options.Scale, routing: options.Routing, activation: options.Activation,
		fusedGateUp: options.FusedGateUp, expertIndexDivisor: options.ExpertIndexDivisor,
		swigluClamp: options.SwiGLUClamp,
	})
}

// MoE: softmax top-k SwiGLU; GGUF expert layouts.
func (b *Builder) MoE(
	input, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, topK: topK, normalizeTopKProb: normalizeTopKProb, scale: scale,
		routing: MoERoutingSoftmax, activation: MoEActivationSiLU, expertIndexDivisor: 1,
	})
}

// MoEGroupedWithRouterInput: split router input; grouped expert-bank indices.
func (b *Builder) MoEGroupedWithRouterInput(
	input, routerInput, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
	expertIndexDivisor uint32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		routerInput: routerInput, gate: gate, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: MoERoutingSoftmax, activation: MoEActivationSiLU,
		expertIndexDivisor: expertIndexDivisor,
	})
}

func (b *Builder) MoEUngated(
	input, router, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		topK: topK, normalizeTopKProb: normalizeTopKProb, scale: scale,
		routing: MoERoutingSoftmax, activation: MoEActivationSiLU, expertIndexDivisor: 1,
	})
}

// MoEUngatedWithSelectionBias: softmax selection bias; ungated experts.
func (b *Builder) MoEUngatedWithSelectionBias(
	input, router, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		selectionBias: selectionBias, topK: topK, normalizeTopKProb: normalizeTopKProb, scale: scale,
		routing: MoERoutingSoftmax, activation: MoEActivationSiLU, expertIndexDivisor: 1,
	})
}

// MoESoftmaxWithSelectionBias: biased selection; unbiased route weights.
func (b *Builder) MoESoftmaxWithSelectionBias(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, selectionBias: selectionBias, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: MoERoutingSoftmax, activation: MoEActivationSiLU, expertIndexDivisor: 1,
	})
}

// MoESoftmaxLimitedWithSelectionBias: biased selection; limited SwiGLU.
func (b *Builder) MoESoftmaxLimitedWithSelectionBias(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale, swigluClamp float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, selectionBias: selectionBias, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: MoERoutingSoftmax, activation: MoEActivationSiLU,
		expertIndexDivisor: 1, swigluClamp: swigluClamp,
	})
}

// MoESoftmaxFusedGateUp: softmax top-k; fused expert gate/up storage.
func (b *Builder) MoESoftmaxFusedGateUp(
	input, router, gateUp, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, gateUp, down, moeOptions{
		selectionBias: selectionBias, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: MoERoutingSoftmax, activation: MoEActivationSiLU,
		fusedGateUp: true, expertIndexDivisor: 1,
	})
}

// MoESigmoid: sigmoid routes; optional selection bias.
func (b *Builder) MoESigmoid(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, selectionBias: selectionBias, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: MoERoutingSigmoid, activation: MoEActivationSiLU, expertIndexDivisor: 1,
	})
}

// MoESigmoidLimited: sigmoid top-k; limited SwiGLU.
func (b *Builder) MoESigmoidLimited(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale, swigluClamp float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, selectionBias: selectionBias, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: MoERoutingSigmoid, activation: MoEActivationSiLU,
		expertIndexDivisor: 1, swigluClamp: swigluClamp,
	})
}

// MoESigmoidFusedGateUp: sigmoid top-k; fused expert gate/up storage.
func (b *Builder) MoESigmoidFusedGateUp(
	input, router, gateUp, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, gateUp, down, moeOptions{
		selectionBias: selectionBias, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: MoERoutingSigmoid, activation: MoEActivationSiLU,
		fusedGateUp: true, expertIndexDivisor: 1,
	})
}

// MoEReLUWithRouterInput: split router input; gated ReLU experts.
func (b *Builder) MoEReLUWithRouterInput(
	input, routerInput, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
	routing MoERouting,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		routerInput: routerInput, gate: gate, topK: topK, normalizeTopKProb: normalizeTopKProb,
		scale: scale, routing: routing, activation: MoEActivationReLU, expertIndexDivisor: 1,
	})
}

// MoEReLUSquaredWithRouterInput: split router input; ungated squared-ReLU experts.
func (b *Builder) MoEReLUSquaredWithRouterInput(
	input, routerInput, router, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
	routing MoERouting,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		routerInput: routerInput, selectionBias: selectionBias, topK: topK,
		normalizeTopKProb: normalizeTopKProb, scale: scale, routing: routing,
		activation: MoEActivationReLUSquared, expertIndexDivisor: 1,
	})
}

// MoEGELU: softmax top-k GELU/GEGLU experts.
func (b *Builder) MoEGELU(
	input, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, topK: topK, normalizeTopKProb: normalizeTopKProb, scale: scale,
		routing: MoERoutingSoftmax, activation: MoEActivationGELU, expertIndexDivisor: 1,
	})
}

// MoEGELUWithRouterInput: split router input; GELU/GEGLU experts.
func (b *Builder) MoEGELUWithRouterInput(
	input, routerInput, router, gate, up, down, expertScale *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		routerInput: routerInput, gate: gate, expertScale: expertScale, topK: topK,
		normalizeTopKProb: normalizeTopKProb, scale: scale, routing: MoERoutingSoftmax,
		activation: MoEActivationGELU, expertIndexDivisor: 1,
	})
}

// MoEGELUFusedGateUpWithRouterInput: split router input; fused GEGLU experts.
func (b *Builder) MoEGELUFusedGateUpWithRouterInput(
	input, routerInput, router, gateUp, down, expertScale *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, gateUp, down, moeOptions{
		routerInput: routerInput, expertScale: expertScale, topK: topK,
		normalizeTopKProb: normalizeTopKProb, scale: scale, routing: MoERoutingSoftmax,
		activation: MoEActivationGELU, fusedGateUp: true, expertIndexDivisor: 1,
	})
}

// MoEOpenAI: selected-logit softmax; biased OpenAI SwiGLU experts.
func (b *Builder) MoEOpenAI(
	input, router, routerBias, gate, gateBias, up, upBias, down, downBias *Tensor,
	topK uint32,
	scale float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, topK: topK, scale: scale, routing: MoERoutingSelectedSoftmax,
		activation: MoEActivationSwiGLUOAI, expertIndexDivisor: 1,
		biases: &MoEBiases{Router: routerBias, Gate: gateBias, Up: upBias, Down: downBias},
	})
}

// MoESqrtSoftplusLimited: DeepSeek 4 routing; optional fixed expert IDs.
func (b *Builder) MoESqrtSoftplusLimited(
	input, router, gate, up, down, selectionBias, selectedExperts *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale, swigluClamp float32,
) *Tensor {
	return b.buildMoE(input, router, up, down, moeOptions{
		gate: gate, selectionBias: selectionBias, selectedExperts: selectedExperts,
		topK: topK, normalizeTopKProb: normalizeTopKProb, scale: scale,
		routing: MoERoutingSqrtSoftplus, activation: MoEActivationSiLU,
		expertIndexDivisor: 1, swigluClamp: swigluClamp,
	})
}

func (b *Builder) buildMoE(
	input, router, up, down *Tensor,
	options moeOptions,
) *Tensor {
	routerInput := options.routerInput
	if routerInput == nil {
		routerInput = input
	}
	gate := options.gate
	selectionBias := options.selectionBias
	expertScale := options.expertScale
	selectedExperts := options.selectedExperts
	topK := options.topK
	normalizeTopKProb := options.normalizeTopKProb
	scale := options.scale
	routing := options.routing
	activation := options.activation
	fusedGateUp := options.fusedGateUp
	expertIndexDivisor := options.expertIndexDivisor
	swigluClamp := options.swigluClamp
	biases := options.biases
	if b.err != nil {
		return nil
	}
	if input == nil || routerInput == nil || router == nil || up == nil || down == nil {
		b.setError(errors.New("MoE input is nil"))
		return nil
	}
	router = b.mergeLoRAWeight(router)
	gate = b.mergeLoRAWeight(gate)
	up = b.mergeLoRAWeight(up)
	down = b.mergeLoRAWeight(down)
	if b.err != nil {
		return nil
	}
	if input.Type != dtype.F32 || routerInput.Type != dtype.F32 || router.Type != dtype.F32 {
		b.setError(errors.New("MoE inputs and router must be F32"))
		return nil
	}
	expertType := up.Type
	if up.Type != down.Type || gate != nil && gate.Type != up.Type ||
		(expertType != dtype.F32 && !nativeQuantizedType(expertType)) {
		b.setError(errors.New("MoE experts must share F32 or native quantized storage"))
		return nil
	}
	if input.Shape.Rank != 2 || routerInput.Shape.Rank != 2 || router.Shape.Rank != 2 ||
		gate != nil && gate.Shape.Rank != 3 || up.Shape.Rank != 3 || down.Shape.Rank != 3 {
		b.setError(errors.New("MoE input ranks are invalid"))
		return nil
	}
	hidden := input.Shape.Dims[0]
	routerHidden := routerInput.Shape.Dims[0]
	experts := router.Shape.Dims[1]
	if expertIndexDivisor == 0 || experts%uint64(expertIndexDivisor) != 0 {
		b.setError(errors.New("MoE expert index divisor is invalid"))
		return nil
	}
	bankExperts := experts / uint64(expertIndexDivisor)
	intermediate := up.Shape.Dims[1]
	if fusedGateUp {
		if gate != nil || intermediate%2 != 0 {
			b.setError(errors.New("MoE fused gate/up dimensions are invalid"))
			return nil
		}
		intermediate /= 2
	}
	if hidden == 0 || experts == 0 || experts > math.MaxUint32 || topK == 0 ||
		uint64(topK) > experts || uint64(topK) > bankExperts || topK > MaxMoETopK || scale == 0 ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) ||
		routerHidden == 0 || router.Shape.Dims[0] != routerHidden || up.Shape.Dims[0] != hidden ||
		routerInput.Shape.Dims[1] != input.Shape.Dims[1] ||
		up.Shape.Dims[2] != bankExperts ||
		down.Shape.Dims[0] != intermediate || down.Shape.Dims[1] != hidden ||
		down.Shape.Dims[2] != bankExperts {
		b.setError(errors.New("MoE dimensions or routing attributes are invalid"))
		return nil
	}
	if gate != nil && (gate.Shape.Dims[0] != hidden || gate.Shape.Dims[1] != intermediate ||
		gate.Shape.Dims[2] != bankExperts) {
		b.setError(errors.New("MoE gate dimensions are invalid"))
		return nil
	}
	if routing != MoERoutingSoftmax && routing != MoERoutingSigmoid &&
		routing != MoERoutingSelectedSoftmax && routing != MoERoutingSqrtSoftplus {
		b.setError(errors.New("MoE routing function is invalid"))
		return nil
	}
	if activation != MoEActivationSiLU && activation != MoEActivationReLU && activation != MoEActivationGELU &&
		activation != MoEActivationReLUSquared &&
		activation != MoEActivationSwiGLUOAI {
		b.setError(errors.New("MoE activation is invalid"))
		return nil
	}
	if activation == MoEActivationSwiGLUOAI && (routing != MoERoutingSelectedSoftmax || gate == nil || biases == nil) ||
		activation != MoEActivationSwiGLUOAI && biases != nil {
		b.setError(errors.New("OpenAI MoE routing, activation, gate, or biases are inconsistent"))
		return nil
	}
	if swigluClamp < 0 || math.IsNaN(float64(swigluClamp)) || math.IsInf(float64(swigluClamp), 0) ||
		swigluClamp > 0 && (activation != MoEActivationSiLU || gate == nil && !fusedGateUp) {
		b.setError(errors.New("MoE SwiGLU clamp is invalid"))
		return nil
	}
	if nativeQuantizedType(expertType) {
		traits, _ := expertType.Traits()
		if hidden%traits.BlockSize != 0 || intermediate%traits.BlockSize != 0 {
			b.setError(fmt.Errorf("%s MoE hidden and intermediate widths must be block aligned", expertType))
			return nil
		}
	}
	inputs := []*Tensor{input, routerInput, router}
	if gate != nil {
		inputs = append(inputs, gate)
	}
	inputs = append(inputs, up, down)
	if selectionBias != nil {
		if selectionBias.Type != dtype.F32 || selectionBias.Shape.Rank != 1 ||
			selectionBias.Shape.Dims[0] != experts {
			b.setError(errors.New("MoE selection bias must be rank-1 F32 with one value per expert"))
			return nil
		}
		inputs = append(inputs, selectionBias)
	}
	if expertScale != nil {
		if expertScale.Type != dtype.F32 || expertScale.Shape.Rank != 1 ||
			expertScale.Shape.Dims[0] != bankExperts {
			b.setError(errors.New("MoE expert scale must be rank-1 F32 with one value per expert bank"))
			return nil
		}
		inputs = append(inputs, expertScale)
	}
	if biases != nil {
		if biases.Router == nil || biases.Gate == nil || biases.Up == nil || biases.Down == nil ||
			biases.Router.Type != dtype.F32 || biases.Gate.Type != dtype.F32 ||
			biases.Up.Type != dtype.F32 || biases.Down.Type != dtype.F32 ||
			biases.Router.Shape != MustShape(experts) ||
			biases.Gate.Shape != MustShape(intermediate, bankExperts) ||
			biases.Up.Shape != MustShape(intermediate, bankExperts) ||
			biases.Down.Shape != MustShape(hidden, bankExperts) {
			b.setError(errors.New("MoE bias shapes are invalid"))
			return nil
		}
		inputs = append(inputs, biases.Router, biases.Gate, biases.Up, biases.Down)
	}
	if selectedExperts != nil {
		if selectedExperts.Type != dtype.F32 || selectedExperts.Shape != MustShape(uint64(topK), input.Shape.Dims[1]) {
			b.setError(errors.New("MoE selected experts must be F32 [top-k,tokens]"))
			return nil
		}
		inputs = append(inputs, selectedExperts)
	}
	return b.add("", dtype.F32, input.Shape, OpMoE,
		inputs, MoEAttributes{
			Experts: uint32(experts), ExpertIndexDivisor: expertIndexDivisor, TopK: topK,
			NormalizeTopKProb: normalizeTopKProb, Scale: scale, Routing: routing,
			Activation: activation, Gated: gate != nil || fusedGateUp, FusedGateUp: fusedGateUp,
			HasSelectionBias: selectionBias != nil, HasExpertScale: expertScale != nil,
			HasRouterBias: biases != nil, HasExpertBiases: biases != nil,
			HasSelectedExperts: selectedExperts != nil,
			SwiGLUClamp:        swigluClamp,
		})
}
