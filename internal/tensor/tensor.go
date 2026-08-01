package tensor

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor/dtype"
)

// Op: identifies typed graph operation
type Op uint16

const (
	OpInput Op = iota
	OpAdd
	OpMultiply
	OpScale
	OpRMSNorm
	OpSoftmax
	OpSiLU
	OpMulMat
	OpGetRows
	OpRoPENeoX
	OpReshape
	OpAttention
	OpConcat
	OpRoPENormal
	OpSigmoid
	OpSoftplus
	OpL2Norm
	OpSSMConv
	OpGatedDeltaNet
	OpTranspose2D
	OpGroupSlice
	OpFlatSlice
	OpRoPEMulti
	OpGELU
	OpLayerNorm
	OpReLUSquared
	OpXIELU
	OpMoE
	OpRepeatHeads
	OpClamp
	OpGroupedMulMat
)

var opNames = [...]string{
	"input",
	"add",
	"multiply",
	"scale",
	"rms_norm",
	"softmax",
	"silu",
	"mul_mat",
	"get_rows",
	"rope_neox",
	"reshape",
	"attention",
	"concat",
	"rope_normal",
	"sigmoid",
	"softplus",
	"l2_norm",
	"ssm_conv",
	"gated_delta_net",
	"transpose_2d",
	"group_slice",
	"flat_slice",
	"rope_multi",
	"gelu",
	"layer_norm",
	"relu_squared",
	"xielu",
	"moe",
	"repeat_heads",
	"clamp",
	"grouped_mul_mat",
}

func (o Op) String() string {
	if int(o) >= len(opNames) {
		return fmt.Sprintf("op_%d", o)
	}
	return opNames[o]
}

type ScaleAttributes struct {
	Value float32
}

type ClampAttributes struct {
	Minimum float32
	Maximum float32
}

type RMSNormAttributes struct {
	Epsilon float32
}

type LayerNormAttributes struct {
	Epsilon float32
}

type L2NormAttributes struct {
	Epsilon float32
}

type XIELUAttributes struct {
	AlphaN  float32
	AlphaP  float32
	Beta    float32
	Epsilon float32
}

type GetRowsAttributes struct {
	Rows []uint32
}

type RoPEAttributes struct {
	Positions        []uint32
	RotaryDimensions uint32
	FrequencyBase    float32
	FrequencyScale   float32
	OriginalContext  uint32
	ExtFactor        float32
	AttentionFactor  float32
	BetaFast         float32
	BetaSlow         float32
}

type RoPENeoXAttributes = RoPEAttributes

type RoPEMultiAttributes struct {
	Positions        [4][]uint32
	Sections         [4]int32
	RotaryDimensions uint32
	FrequencyBase    float32
	FrequencyScale   float32
}

type AttentionAttributes struct {
	Scale           float32
	Softcap         float32
	MaxALiBiBias    float32
	Causal          bool
	HasSinks        bool
	SymmetricWindow bool
	ChunkedWindow   bool
	QueryStart      uint32
	Window          uint32
	RelativeBuckets uint32
}

type MoEAttributes struct {
	Experts            uint32
	ExpertIndexDivisor uint32
	TopK               uint32
	NormalizeTopKProb  bool
	Scale              float32
	Routing            MoERouting
	Activation         MoEActivation
	Gated              bool
	FusedGateUp        bool
	HasSelectionBias   bool
	HasExpertScale     bool
	HasRouterBias      bool
	HasExpertBiases    bool
	SwiGLUClamp        float32
}

type GatedDeltaNetAttributes struct {
	RepeatInterleave bool
}

type MoERouting uint32

const (
	MoERoutingSoftmax         MoERouting = 1
	MoERoutingSigmoid         MoERouting = 2
	MoERoutingSelectedSoftmax MoERouting = 3
)

type MoEActivation uint32

const (
	MoEActivationSiLU      MoEActivation = 1
	MoEActivationReLU      MoEActivation = 2
	MoEActivationGELU      MoEActivation = 3
	MoEActivationSwiGLUOAI MoEActivation = 4
)

type moeBiases struct {
	router *Tensor
	gate   *Tensor
	up     *Tensor
	down   *Tensor
}

type RepeatHeadsAttributes struct {
	Heads uint32
}

type ConcatAttributes struct {
	Axis uint32
}

type GroupSliceAttributes struct {
	Offset uint64
	Width  uint64
	Groups uint64
	Stride uint64
}

type FlatSliceAttributes struct {
	Offset uint64
}

// Tensor: immutable graph node descriptor
type Tensor struct {
	ID     uint64
	Name   string
	Type   dtype.Type
	Shape  Shape
	Stride [MaxDimensions]uint64
	Op     Op
	Inputs []*Tensor
	Attrs  any
}

// Builder: constructs and validates tensor graph
type Builder struct {
	nextID uint64
	nodes  []*Tensor
	err    error
}

func NewBuilder() *Builder {
	return &Builder{nextID: 1}
}

func (b *Builder) Err() error {
	return b.err
}

func (b *Builder) Nodes() []*Tensor {
	return append([]*Tensor(nil), b.nodes...)
}

func (b *Builder) Input(name string, dataType dtype.Type, shape Shape) *Tensor {
	if b.err != nil {
		return nil
	}
	if name == "" {
		b.err = errors.New("input tensor name is empty")
		return nil
	}
	return b.add(name, dataType, shape, OpInput, nil, nil)
}

func (b *Builder) Add(left, right *Tensor) *Tensor {
	return b.binary(OpAdd, left, right)
}

func (b *Builder) Multiply(left, right *Tensor) *Tensor {
	return b.binary(OpMultiply, left, right)
}

func (b *Builder) Scale(input *Tensor, value float32) *Tensor {
	return b.unary(OpScale, input, ScaleAttributes{Value: value})
}

func (b *Builder) Clamp(input *Tensor, minimum, maximum float32) *Tensor {
	if math.IsNaN(float64(minimum)) || math.IsNaN(float64(maximum)) ||
		math.IsInf(float64(minimum), 0) || math.IsInf(float64(maximum), 0) || minimum > maximum {
		b.setError(errors.New("clamp bounds are invalid"))
		return nil
	}
	return b.unary(OpClamp, input, ClampAttributes{Minimum: minimum, Maximum: maximum})
}

func (b *Builder) RMSNorm(input *Tensor, epsilon float32) *Tensor {
	if epsilon <= 0 {
		b.setError(errors.New("RMSNorm epsilon must be positive"))
		return nil
	}
	return b.unary(OpRMSNorm, input, RMSNormAttributes{Epsilon: epsilon})
}

// WeightedRMSNorm: applies RMSNorm and learned per-channel weight
func (b *Builder) WeightedRMSNorm(input, weight *Tensor, epsilon float32) *Tensor {
	return b.Multiply(b.RMSNorm(input, epsilon), weight)
}

func (b *Builder) LayerNorm(input *Tensor, epsilon float32) *Tensor {
	if epsilon <= 0 {
		b.setError(errors.New("LayerNorm epsilon must be positive"))
		return nil
	}
	return b.unary(OpLayerNorm, input, LayerNormAttributes{Epsilon: epsilon})
}

// AffineLayerNorm: applies LayerNorm and learned per-channel weight and bias
func (b *Builder) AffineLayerNorm(input, weight, bias *Tensor, epsilon float32) *Tensor {
	return b.Add(b.Multiply(b.LayerNorm(input, epsilon), weight), bias)
}

func (b *Builder) Softmax(input *Tensor) *Tensor {
	return b.unary(OpSoftmax, input, nil)
}

func (b *Builder) SiLU(input *Tensor) *Tensor {
	return b.unary(OpSiLU, input, nil)
}

func (b *Builder) GELU(input *Tensor) *Tensor {
	return b.unary(OpGELU, input, nil)
}

func (b *Builder) XIELU(input *Tensor, alphaN, alphaP, beta, epsilon float32) *Tensor {
	parameters := []float32{alphaN, alphaP, beta, epsilon}
	for _, parameter := range parameters {
		if math.IsNaN(float64(parameter)) || math.IsInf(float64(parameter), 0) {
			b.setError(errors.New("xIELU parameters must be finite"))
			return nil
		}
	}
	return b.unary(OpXIELU, input, XIELUAttributes{
		AlphaN: alphaN, AlphaP: alphaP, Beta: beta, Epsilon: epsilon,
	})
}

func (b *Builder) ReLUSquared(input *Tensor) *Tensor {
	return b.unary(OpReLUSquared, input, nil)
}

func (b *Builder) Sigmoid(input *Tensor) *Tensor {
	return b.unary(OpSigmoid, input, nil)
}

func (b *Builder) Softplus(input *Tensor) *Tensor {
	return b.unary(OpSoftplus, input, nil)
}

func (b *Builder) L2Norm(input *Tensor, epsilon float32) *Tensor {
	if epsilon < 0 || math.IsNaN(float64(epsilon)) {
		b.setError(errors.New("L2Norm epsilon must be non-negative"))
		return nil
	}
	return b.unary(OpL2Norm, input, L2NormAttributes{Epsilon: epsilon})
}

// SSMConv: applies channel-wise sliding convolution used by recurrent SSM
// blocks; Input is [kernel-1+tokens, channels, sequences] and weights are
// [kernel, channels]; output is [channels, tokens, sequences]
func (b *Builder) SSMConv(input, weights *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || weights == nil {
		b.setError(errors.New("SSMConv input is nil"))
		return nil
	}
	if (input.Shape.Rank != 2 && input.Shape.Rank != 3) || weights.Shape.Rank != 2 {
		b.setError(errors.New("SSMConv requires rank-2/3 input and rank-2 weights"))
		return nil
	}
	if input.Type != dtype.F32 || weights.Type != dtype.F32 {
		b.setError(errors.New("SSMConv currently requires F32 inputs"))
		return nil
	}
	kernelSize := weights.Shape.Dims[0]
	if kernelSize == 0 || input.Shape.Dims[0] < kernelSize {
		b.setError(errors.New("SSMConv kernel exceeds input window"))
		return nil
	}
	if input.Shape.Dims[1] != weights.Shape.Dims[1] {
		b.setError(errors.New("SSMConv channel counts differ"))
		return nil
	}
	tokens := input.Shape.Dims[0] - kernelSize + 1
	var shape Shape
	var err error
	if input.Shape.Rank == 2 {
		shape, err = NewShape(input.Shape.Dims[1], tokens)
	} else {
		shape, err = NewShape(input.Shape.Dims[1], tokens, input.Shape.Dims[2])
	}
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpSSMConv, []*Tensor{input, weights}, nil)
}

// GatedDeltaNet: applies llama.cpp's fused K=1 recurrent delta-net update
// Q/K/V are [state, heads, tokens, sequences], scalar or vector gate has
// [1|state, valueHeads, tokens, sequences], beta is [1,valueHeads,tokens,
// sequences], and state is [state,state,valueHeads,sequences]; output
// packs attention values followed by newest state snapshot
func (b *Builder) GatedDeltaNet(q, k, v, gate, beta, state *Tensor) *Tensor {
	return b.gatedDeltaNet(q, k, v, gate, beta, state, false)
}

func (b *Builder) GatedDeltaNetRepeatInterleave(q, k, v, gate, beta, state *Tensor) *Tensor {
	return b.gatedDeltaNet(q, k, v, gate, beta, state, true)
}

func (b *Builder) gatedDeltaNet(
	q, k, v, gate, beta, state *Tensor,
	repeatInterleave bool,
) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{q, k, v, gate, beta, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 || input.Shape.Rank != 4 {
			b.setError(errors.New("GatedDeltaNet requires six rank-4 F32 inputs"))
			return nil
		}
	}
	size := v.Shape.Dims[0]
	heads := v.Shape.Dims[1]
	tokens := v.Shape.Dims[2]
	sequences := v.Shape.Dims[3]
	if q.Shape.Dims[0] != size || k.Shape.Dims[0] != size ||
		q.Shape.Dims[2] != tokens || k.Shape.Dims[2] != tokens ||
		q.Shape.Dims[3] != sequences || k.Shape.Dims[3] != sequences ||
		heads%q.Shape.Dims[1] != 0 || heads%k.Shape.Dims[1] != 0 {
		b.setError(errors.New("GatedDeltaNet Q/K/V shapes are incompatible"))
		return nil
	}
	if (gate.Shape.Dims[0] != 1 && gate.Shape.Dims[0] != size) ||
		gate.Shape.Dims[1] != heads || gate.Shape.Dims[2] != tokens ||
		gate.Shape.Dims[3] != sequences ||
		beta.Shape.Dims[0] != 1 || beta.Shape.Dims[1] != heads ||
		beta.Shape.Dims[2] != tokens || beta.Shape.Dims[3] != sequences ||
		state.Shape.Dims[0] != size || state.Shape.Dims[1] != size ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences {
		b.setError(errors.New("GatedDeltaNet gate/beta/state shapes are incompatible"))
		return nil
	}
	shape, err := NewShape(size*heads, tokens*sequences+size*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add(
		"", dtype.F32, shape, OpGatedDeltaNet, inputs,
		GatedDeltaNetAttributes{RepeatInterleave: repeatInterleave},
	)
}

// MoE: softmax top-k SwiGLU; GGUF expert layouts.
func (b *Builder) MoE(
	input, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, gate, up, down, nil, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationSiLU, false, 1, 0, nil)
}

// MoEGroupedWithRouterInput: split router input; grouped expert-bank indices.
func (b *Builder) MoEGroupedWithRouterInput(
	input, routerInput, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
	expertIndexDivisor uint32,
) *Tensor {
	return b.moe(input, routerInput, router, gate, up, down, nil, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationSiLU, false, expertIndexDivisor, 0, nil)
}

func (b *Builder) MoEUngated(
	input, router, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, nil, up, down, nil, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationSiLU, false, 1, 0, nil)
}

// MoEUngatedWithSelectionBias: softmax selection bias; ungated experts.
func (b *Builder) MoEUngatedWithSelectionBias(
	input, router, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, nil, up, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationSiLU, false, 1, 0, nil)
}

// MoESoftmaxWithSelectionBias: biased selection; unbiased route weights.
func (b *Builder) MoESoftmaxWithSelectionBias(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, gate, up, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationSiLU, false, 1, 0, nil)
}

// MoESoftmaxLimitedWithSelectionBias: biased selection; limited SwiGLU.
func (b *Builder) MoESoftmaxLimitedWithSelectionBias(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale, swigluClamp float32,
) *Tensor {
	return b.moe(input, input, router, gate, up, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationSiLU, false, 1, swigluClamp, nil)
}

// MoESoftmaxFusedGateUp: softmax top-k; fused expert gate/up storage.
func (b *Builder) MoESoftmaxFusedGateUp(
	input, router, gateUp, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, nil, gateUp, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationSiLU, true, 1, 0, nil)
}

// MoESigmoid: sigmoid routes; optional selection bias.
func (b *Builder) MoESigmoid(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, gate, up, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		MoERoutingSigmoid, MoEActivationSiLU, false, 1, 0, nil)
}

// MoESigmoidLimited: sigmoid top-k; limited SwiGLU.
func (b *Builder) MoESigmoidLimited(
	input, router, gate, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale, swigluClamp float32,
) *Tensor {
	return b.moe(input, input, router, gate, up, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		MoERoutingSigmoid, MoEActivationSiLU, false, 1, swigluClamp, nil)
}

// MoESigmoidFusedGateUp: sigmoid top-k; fused expert gate/up storage.
func (b *Builder) MoESigmoidFusedGateUp(
	input, router, gateUp, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, nil, gateUp, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		MoERoutingSigmoid, MoEActivationSiLU, true, 1, 0, nil)
}

// MoEReLUWithRouterInput: split router input; gated ReLU experts.
func (b *Builder) MoEReLUWithRouterInput(
	input, routerInput, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
	routing MoERouting,
) *Tensor {
	return b.moe(input, routerInput, router, gate, up, down, nil, nil, topK, normalizeTopKProb, scale,
		routing, MoEActivationReLU, false, 1, 0, nil)
}

// MoEGELU: softmax top-k GELU/GEGLU experts.
func (b *Builder) MoEGELU(
	input, router, gate, up, down *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, gate, up, down, nil, nil, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationGELU, false, 1, 0, nil)
}

// MoEGELUWithRouterInput: split router input; GELU/GEGLU experts.
func (b *Builder) MoEGELUWithRouterInput(
	input, routerInput, router, gate, up, down, expertScale *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, routerInput, router, gate, up, down, nil, expertScale, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationGELU, false, 1, 0, nil)
}

// MoEGELUFusedGateUpWithRouterInput: split router input; fused GEGLU experts.
func (b *Builder) MoEGELUFusedGateUpWithRouterInput(
	input, routerInput, router, gateUp, down, expertScale *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
) *Tensor {
	return b.moe(input, routerInput, router, nil, gateUp, down, nil, expertScale, topK, normalizeTopKProb, scale,
		MoERoutingSoftmax, MoEActivationGELU, true, 1, 0, nil)
}

// MoEOpenAI: selected-logit softmax; biased OpenAI SwiGLU experts.
func (b *Builder) MoEOpenAI(
	input, router, routerBias, gate, gateBias, up, upBias, down, downBias *Tensor,
	topK uint32,
	scale float32,
) *Tensor {
	return b.moe(input, input, router, gate, up, down, nil, nil, topK, false, scale,
		MoERoutingSelectedSoftmax, MoEActivationSwiGLUOAI, false, 1, 0,
		&moeBiases{router: routerBias, gate: gateBias, up: upBias, down: downBias})
}

func (b *Builder) moe(
	input, routerInput, router, gate, up, down, selectionBias, expertScale *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
	routing MoERouting,
	activation MoEActivation,
	fusedGateUp bool,
	expertIndexDivisor uint32,
	swigluClamp float32,
	biases *moeBiases,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || routerInput == nil || router == nil || up == nil || down == nil {
		b.setError(errors.New("MoE input is nil"))
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
		uint64(topK) > experts || uint64(topK) > bankExperts || topK > 16 || scale == 0 ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) ||
		router.Shape.Dims[0] != hidden || up.Shape.Dims[0] != hidden ||
		routerInput.Shape.Dims[0] != hidden || routerInput.Shape.Dims[1] != input.Shape.Dims[1] ||
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
	if routing != MoERoutingSoftmax && routing != MoERoutingSigmoid && routing != MoERoutingSelectedSoftmax {
		b.setError(errors.New("MoE routing function is invalid"))
		return nil
	}
	if activation != MoEActivationSiLU && activation != MoEActivationReLU && activation != MoEActivationGELU &&
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
		if biases.router == nil || biases.gate == nil || biases.up == nil || biases.down == nil ||
			biases.router.Type != dtype.F32 || biases.gate.Type != dtype.F32 ||
			biases.up.Type != dtype.F32 || biases.down.Type != dtype.F32 ||
			biases.router.Shape != MustShape(experts) ||
			biases.gate.Shape != MustShape(intermediate, bankExperts) ||
			biases.up.Shape != MustShape(intermediate, bankExperts) ||
			biases.down.Shape != MustShape(hidden, bankExperts) {
			b.setError(errors.New("MoE bias shapes are invalid"))
			return nil
		}
		inputs = append(inputs, biases.router, biases.gate, biases.up, biases.down)
	}
	return b.add("", dtype.F32, input.Shape, OpMoE,
		inputs, MoEAttributes{
			Experts: uint32(experts), ExpertIndexDivisor: expertIndexDivisor, TopK: topK,
			NormalizeTopKProb: normalizeTopKProb, Scale: scale, Routing: routing,
			Activation: activation, Gated: gate != nil || fusedGateUp, FusedGateUp: fusedGateUp,
			HasSelectionBias: selectionBias != nil, HasExpertScale: expertScale != nil,
			HasRouterBias: biases != nil, HasExpertBiases: biases != nil,
			SwiGLUClamp: swigluClamp,
		})
}

// RepeatHeads: broadcasts single head in [width,1,tokens] tensor across
// requested head count without changing token ordering
func (b *Builder) RepeatHeads(input *Tensor, heads uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank != 3 ||
		input.Shape.Dims[1] != 1 || heads == 0 {
		b.setError(errors.New("RepeatHeads requires rank-3 F32 [width,1,tokens] input and positive heads"))
		return nil
	}
	shape, err := NewShape(input.Shape.Dims[0], uint64(heads), input.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpRepeatHeads, []*Tensor{input}, RepeatHeadsAttributes{Heads: heads})
}

// SwiGLU: computes SiLU(gate) * up
func (b *Builder) SwiGLU(gate, up *Tensor) *Tensor {
	return b.Multiply(b.SiLU(gate), up)
}

// GEGLU: computes GELU(gate) * up
func (b *Builder) GEGLU(gate, up *Tensor) *Tensor {
	return b.Multiply(b.GELU(gate), up)
}

// MulMat: follows ggml semantics; Left has shape [K,M], right has shape [K,N],
// and result has shape [M,N]
func (b *Builder) MulMat(left, right *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if left == nil || right == nil {
		b.setError(errors.New("mul_mat input is nil"))
		return nil
	}
	if left.Shape.Rank != 2 || right.Shape.Rank != 2 {
		b.setError(errors.New("initial mul_mat implementation requires rank-2 inputs"))
		return nil
	}
	if left.Shape.Dims[0] != right.Shape.Dims[0] {
		b.setError(fmt.Errorf(
			"mul_mat inner dimensions differ: %d and %d",
			left.Shape.Dims[0],
			right.Shape.Dims[0],
		))
		return nil
	}
	outputType := left.Type
	if nativeQuantizedType(left.Type) && right.Type == dtype.F32 {
		outputType = dtype.F32
	} else if left.Type != right.Type {
		b.setError(fmt.Errorf("mul_mat types are unsupported: %s and %s", left.Type, right.Type))
		return nil
	}
	shape, err := NewShape(left.Shape.Dims[1], right.Shape.Dims[1])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", outputType, shape, OpMulMat, []*Tensor{left, right}, nil)
}

// GroupedMulMat: per-group ggml matmul; [K,M,G] x [K,G,N] -> [M,G,N]
func (b *Builder) GroupedMulMat(left, right *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if left == nil || right == nil || left.Shape.Rank != 3 || right.Shape.Rank != 3 {
		b.setError(errors.New("grouped_mul_mat requires rank-3 inputs"))
		return nil
	}
	if left.Shape.Dims[0] != right.Shape.Dims[0] || left.Shape.Dims[2] != right.Shape.Dims[1] {
		b.setError(errors.New("grouped_mul_mat inner or group dimensions differ"))
		return nil
	}
	outputType := left.Type
	if nativeQuantizedType(left.Type) && right.Type == dtype.F32 {
		outputType = dtype.F32
	} else if left.Type != right.Type {
		b.setError(fmt.Errorf("grouped_mul_mat types are unsupported: %s and %s", left.Type, right.Type))
		return nil
	}
	shape, err := NewShape(left.Shape.Dims[1], left.Shape.Dims[2], right.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", outputType, shape, OpGroupedMulMat, []*Tensor{left, right}, nil)
}

// GetRows gathers vocabulary rows from rank-2 table in ggml layout;
// table shape is [embedding, rows] and result is [embedding, len(rows)]
func (b *Builder) GetRows(table *Tensor, rows []uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if table == nil {
		b.setError(errors.New("get_rows input is nil"))
		return nil
	}
	if table.Shape.Rank != 2 {
		b.setError(errors.New("get_rows table must have rank 2"))
		return nil
	}
	if len(rows) == 0 {
		b.setError(errors.New("get_rows row list is empty"))
		return nil
	}
	for index, row := range rows {
		if uint64(row) >= table.Shape.Dims[1] {
			b.setError(fmt.Errorf("get_rows row %d at index %d exceeds table size %d", row, index, table.Shape.Dims[1]))
			return nil
		}
	}
	shape, err := NewShape(table.Shape.Dims[0], uint64(len(rows)))
	if err != nil {
		b.setError(err)
		return nil
	}
	attributes := GetRowsAttributes{Rows: append([]uint32(nil), rows...)}
	outputType := table.Type
	if nativeQuantizedType(table.Type) {
		outputType = dtype.F32
	}
	return b.add("", outputType, shape, OpGetRows, []*Tensor{table}, attributes)
}

func nativeQuantizedType(value dtype.Type) bool {
	switch value {
	case dtype.Q4_0, dtype.Q4_1, dtype.Q5_0, dtype.Q5_1,
		dtype.Q8_0, dtype.Q8_1, dtype.Q2K, dtype.Q3K, dtype.Q4K, dtype.Q5K, dtype.Q6K, dtype.Q8K,
		dtype.IQ2XXS, dtype.IQ2XS, dtype.IQ2S, dtype.IQ3XXS, dtype.IQ3S, dtype.IQ1S, dtype.IQ1M,
		dtype.IQ4NL, dtype.IQ4XS, dtype.MXFP4, dtype.NVFP4,
		dtype.Q1_0, dtype.Q2_0, dtype.TQ1_0, dtype.TQ2_0:
		return true
	default:
		return false
	}
}

// RoPENeoX: applies split-half rotary layout used by Qwen3; Input shape is
// [head width, heads, tokens] (optionally with batch dimension)
func (b *Builder) RoPENeoX(input *Tensor, positions []uint32, rotaryDimensions uint32, frequencyBase float32) *Tensor {
	return b.rope(OpRoPENeoX, "rope_neox", input, positions, rotaryDimensions, frequencyBase, 1, nil)
}

func (b *Builder) RoPENeoXScaled(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
) *Tensor {
	return b.rope(
		OpRoPENeoX, "rope_neox", input, positions,
		rotaryDimensions, frequencyBase, frequencyScale, nil,
	)
}

// RoPENeoXYaRN: applies YaRN interpolation/extrapolation and magnitude scaling
// to split-half rotary layout
func (b *Builder) RoPENeoXYaRN(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
) *Tensor {
	return b.ropeYaRN(
		OpRoPENeoX, "rope_neox", input, positions, rotaryDimensions, originalContext,
		frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow,
	)
}

func (b *Builder) RoPENeoXScaledWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.rope(
		OpRoPENeoX, "rope_neox", input, positions,
		rotaryDimensions, frequencyBase, frequencyScale, frequencyFactors,
	)
}

// RoPENeoXWithFactors: applies one frequency divisor per rotary pair
func (b *Builder) RoPENeoXWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.rope(
		OpRoPENeoX, "rope_neox", input, positions,
		rotaryDimensions, frequencyBase, 1, frequencyFactors,
	)
}

// RoPENormal: applies rotary embeddings to consecutive channel pairs, as used
// by Llama architecture family
func (b *Builder) RoPENormal(input *Tensor, positions []uint32, rotaryDimensions uint32, frequencyBase float32) *Tensor {
	return b.rope(OpRoPENormal, "rope_normal", input, positions, rotaryDimensions, frequencyBase, 1, nil)
}

func (b *Builder) RoPENormalScaled(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
) *Tensor {
	return b.rope(
		OpRoPENormal, "rope_normal", input, positions,
		rotaryDimensions, frequencyBase, frequencyScale, nil,
	)
}

// RoPENormalYaRN: applies YaRN interpolation/extrapolation and magnitude
// scaling to consecutive rotary pairs
func (b *Builder) RoPENormalYaRN(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
) *Tensor {
	return b.ropeYaRN(
		OpRoPENormal, "rope_normal", input, positions, rotaryDimensions, originalContext,
		frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow,
	)
}

func (b *Builder) ropeYaRN(
	operation Op,
	name string,
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
) *Tensor {
	if originalContext == 0 || attentionFactor <= 0 || betaFast <= 0 || betaSlow <= 0 || extFactor < 0 {
		b.setError(errors.New("YaRN RoPE parameters are invalid"))
		return nil
	}
	result := b.rope(
		operation, name, input, positions, rotaryDimensions, frequencyBase, frequencyScale, nil,
	)
	if result == nil {
		return nil
	}
	attributes := result.Attrs.(RoPEAttributes)
	attributes.OriginalContext = originalContext
	attributes.ExtFactor = extFactor
	attributes.AttentionFactor = attentionFactor
	attributes.BetaFast = betaFast
	attributes.BetaSlow = betaSlow
	result.Attrs = attributes
	return result
}

func (b *Builder) RoPENormalScaledWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.rope(
		OpRoPENormal, "rope_normal", input, positions,
		rotaryDimensions, frequencyBase, frequencyScale, frequencyFactors,
	)
}

// RoPENormalWithFactors: applies one frequency divisor per rotary pair
func (b *Builder) RoPENormalWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.rope(
		OpRoPENormal, "rope_normal", input, positions,
		rotaryDimensions, frequencyBase, 1, frequencyFactors,
	)
}

// RoPEMulti: applies llama.cpp's split-half multi-axis rotary layout; Positions
// temporal, height, width, and extra axes; sections count rotary pairs
func (b *Builder) RoPEMulti(
	input *Tensor,
	positions [4][]uint32,
	sections [4]int32,
	rotaryDimensions uint32,
	frequencyBase float32,
) *Tensor {
	return b.RoPEMultiScaled(input, positions, sections, rotaryDimensions, frequencyBase, 1)
}

func (b *Builder) RoPEMultiScaled(
	input *Tensor,
	positions [4][]uint32,
	sections [4]int32,
	rotaryDimensions uint32,
	frequencyBase, frequencyScale float32,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Shape.Rank < 3 {
		b.setError(errors.New("rope_multi input must have rank 3 or 4"))
		return nil
	}
	if rotaryDimensions == 0 || rotaryDimensions%2 != 0 ||
		uint64(rotaryDimensions) > input.Shape.Dims[0] {
		b.setError(errors.New("rope_multi rotary dimensions are invalid"))
		return nil
	}
	var sectionPairs int64
	for axis := range sections {
		if sections[axis] < 0 {
			b.setError(errors.New("rope_multi section count is negative"))
			return nil
		}
		sectionPairs += int64(sections[axis])
		if len(positions[axis]) != int(input.Shape.Dims[2]) {
			b.setError(errors.New("rope_multi position count differs from token count"))
			return nil
		}
	}
	if sectionPairs == 0 || sectionPairs > int64(rotaryDimensions/2) {
		b.setError(errors.New("rope_multi sections exceed rotary pair count"))
		return nil
	}
	if frequencyBase <= 0 || frequencyScale <= 0 ||
		math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		b.setError(errors.New("rope_multi frequency parameters must be positive and finite"))
		return nil
	}
	attributes := RoPEMultiAttributes{
		Sections:         sections,
		RotaryDimensions: rotaryDimensions,
		FrequencyBase:    frequencyBase,
		FrequencyScale:   frequencyScale,
	}
	for axis := range positions {
		attributes.Positions[axis] = append([]uint32(nil), positions[axis]...)
	}
	return b.add("", input.Type, input.Shape, OpRoPEMulti, []*Tensor{input}, attributes)
}

func (b *Builder) rope(
	operation Op,
	name string,
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
	frequencyFactors *Tensor,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil {
		b.setError(fmt.Errorf("%s input is nil", name))
		return nil
	}
	if input.Shape.Rank < 3 {
		b.setError(fmt.Errorf("%s input must have rank 3 or 4", name))
		return nil
	}
	if rotaryDimensions == 0 || rotaryDimensions%2 != 0 || uint64(rotaryDimensions) > input.Shape.Dims[0] {
		b.setError(fmt.Errorf("%s rotary dimensions %d are invalid for head width %d", name, rotaryDimensions, input.Shape.Dims[0]))
		return nil
	}
	if len(positions) != int(input.Shape.Dims[2]) {
		b.setError(fmt.Errorf("%s has %d positions, need %d", name, len(positions), input.Shape.Dims[2]))
		return nil
	}
	if frequencyBase <= 0 {
		b.setError(fmt.Errorf("%s frequency base must be positive", name))
		return nil
	}
	if frequencyScale <= 0 {
		b.setError(fmt.Errorf("%s frequency scale must be positive", name))
		return nil
	}
	if frequencyFactors != nil {
		if frequencyFactors.Type != dtype.F32 ||
			frequencyFactors.Shape.Rank != 1 ||
			frequencyFactors.Shape.Dims[0] != uint64(rotaryDimensions/2) {
			b.setError(fmt.Errorf(
				"%s frequency factors must be F32 with shape [%d]",
				name,
				rotaryDimensions/2,
			))
			return nil
		}
	}
	attributes := RoPEAttributes{
		Positions:        append([]uint32(nil), positions...),
		RotaryDimensions: rotaryDimensions,
		FrequencyBase:    frequencyBase,
		FrequencyScale:   frequencyScale,
		AttentionFactor:  1,
	}
	inputs := []*Tensor{input}
	if frequencyFactors != nil {
		inputs = append(inputs, frequencyFactors)
	}
	return b.add("", input.Type, input.Shape, operation, inputs, attributes)
}

// Reshape: changes only logical dimensions and preserves contiguous order
func (b *Builder) Reshape(input *Tensor, dimensions ...uint64) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil {
		b.setError(errors.New("reshape input is nil"))
		return nil
	}
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	inputElements, err := input.Shape.Elements()
	if err != nil {
		b.setError(err)
		return nil
	}
	outputElements, err := shape.Elements()
	if err != nil {
		b.setError(err)
		return nil
	}
	if inputElements != outputElements {
		b.setError(fmt.Errorf("reshape changes element count from %d to %d", inputElements, outputElements))
		return nil
	}
	return b.add("", input.Type, shape, OpReshape, []*Tensor{input}, nil)
}

// Transpose2D materializes transpose of contiguous rank-2 tensor
func (b *Builder) Transpose2D(input *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Shape.Rank != 2 {
		b.setError(errors.New("Transpose2D requires a rank-2 input"))
		return nil
	}
	shape, err := NewShape(input.Shape.Dims[1], input.Shape.Dims[0])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", input.Type, shape, OpTranspose2D, []*Tensor{input}, nil)
}

// GroupSlice: extracts equally-strided groups from dimension zero; input
// rank must be 2 or 3 and result prepends [width, groups] to input's
// remaining dimensions
func (b *Builder) GroupSlice(
	input *Tensor,
	offset, width, groups, stride uint64,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || (input.Shape.Rank != 2 && input.Shape.Rank != 3) {
		b.setError(errors.New("GroupSlice requires a rank-2/3 input"))
		return nil
	}
	if width == 0 || groups == 0 || stride < width ||
		groups-1 > (math.MaxUint64-offset-width)/stride ||
		offset+(groups-1)*stride+width > input.Shape.Dims[0] {
		b.setError(errors.New("GroupSlice range exceeds input dimension zero"))
		return nil
	}
	dimensions := []uint64{width, groups, input.Shape.Dims[1]}
	if input.Shape.Rank == 3 {
		dimensions = append(dimensions, input.Shape.Dims[2])
	}
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add(
		"",
		input.Type,
		shape,
		OpGroupSlice,
		[]*Tensor{input},
		GroupSliceAttributes{Offset: offset, Width: width, Groups: groups, Stride: stride},
	)
}

// FlatSlice: copies contiguous element range and gives it requested
// logical shape
func (b *Builder) FlatSlice(input *Tensor, offset uint64, dimensions ...uint64) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil {
		b.setError(errors.New("FlatSlice input is nil"))
		return nil
	}
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	inputElements, inputErr := input.Shape.Elements()
	outputElements, outputErr := shape.Elements()
	if inputErr != nil || outputErr != nil {
		b.setError(errors.Join(inputErr, outputErr))
		return nil
	}
	if offset > inputElements || outputElements > inputElements-offset {
		b.setError(errors.New("FlatSlice range exceeds input storage"))
		return nil
	}
	return b.add(
		"",
		input.Type,
		shape,
		OpFlatSlice,
		[]*Tensor{input},
		FlatSliceAttributes{Offset: offset},
	)
}

// Attention: computes grouped-query scaled dot-product attention; Q has shape
// [key width, query heads, tokens], K is [key width, KV heads, tokens], and V
// [value width, KV heads, tokens]
func (b *Builder) Attention(query, key, value *Tensor, scale float32, causal bool) *Tensor {
	return b.AttentionWithOffset(query, key, value, scale, causal, 0)
}

// AttentionWithOffset: permits query to represent only suffix beginning at
// queryStart in longer cached key/value sequence
func (b *Builder) AttentionWithOffset(
	query, key, value *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.attentionWithWindow(query, key, value, nil, nil, scale, 0, 0, causal, false, queryStart, 0)
}

func (b *Builder) AttentionWithSinksWithOffset(
	query, key, value, sinks *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.attentionWithWindow(query, key, value, nil, sinks, scale, 0, 0, causal, false, queryStart, 0)
}

// AttentionALiBiWithOffset: applies llama.cpp-compatible head slopes to
// linear relative-position mask; maxBias controls steepest slope
func (b *Builder) AttentionALiBiWithOffset(
	query, key, value *Tensor,
	scale, maxBias float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.attentionWithWindow(
		query, key, value, nil, nil, scale, 0, maxBias, causal, false, queryStart, 0,
	)
}

// AttentionSoftcappedWithOffset: applies cap*tanh(score/cap) before softmax
func (b *Builder) AttentionSoftcappedWithOffset(
	query, key, value *Tensor,
	scale float32,
	softcap float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	return b.attentionWithWindow(
		query, key, value, nil, nil, scale, softcap, 0, causal, false, queryStart, 0,
	)
}

// AttentionWithRelativeBias: computes full bidirectional attention and adds
// T5-style bucketed relative-position bias; Bias has shape [heads, buckets]
func (b *Builder) AttentionWithRelativeBias(
	query, key, value, bias *Tensor,
	scale float32,
) *Tensor {
	return b.attentionWithWindow(query, key, value, bias, nil, scale, 0, 0, false, false, 0, 0)
}

func (b *Builder) AttentionWindowWithOffset(
	query, key, value *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.attentionWithWindow(query, key, value, nil, nil, scale, 0, 0, causal, false, queryStart, window)
}

func (b *Builder) AttentionWindowWithSinksWithOffset(
	query, key, value, sinks *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.attentionWithWindow(query, key, value, nil, sinks, scale, 0, 0, causal, false, queryStart, window)
}

func (b *Builder) AttentionChunkedWindowWithOffset(
	query, key, value *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention chunk must be positive"))
		return nil
	}
	result := b.attentionWithWindow(
		query, key, value, nil, nil, scale, 0, 0, causal, false, queryStart, window,
	)
	if result != nil {
		attributes := result.Attrs.(AttentionAttributes)
		attributes.ChunkedWindow = true
		result.Attrs = attributes
	}
	return result
}

func (b *Builder) AttentionWindowSoftcappedWithOffset(
	query, key, value *Tensor,
	scale float32,
	softcap float32,
	causal bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.attentionWithWindow(
		query, key, value, nil, nil, scale, softcap, 0, causal, false, queryStart, window,
	)
}

func (b *Builder) AttentionSymmetricWindow(
	query, key, value *Tensor,
	scale float32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.attentionWithWindow(query, key, value, nil, nil, scale, 0, 0, false, true, 0, window)
}

func (b *Builder) attentionWithWindow(
	query, key, value, bias, sinks *Tensor,
	scale float32,
	softcap float32,
	maxALiBiBias float32,
	causal bool,
	symmetricWindow bool,
	queryStart uint32,
	window uint32,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || key == nil || value == nil {
		b.setError(errors.New("attention input is nil"))
		return nil
	}
	if query.Type != key.Type || query.Type != value.Type {
		b.setError(errors.New("attention input types differ"))
		return nil
	}
	if query.Shape.Rank != 3 || key.Shape.Rank != 3 || value.Shape.Rank != 3 {
		b.setError(errors.New("attention inputs must have rank 3"))
		return nil
	}
	if query.Shape.Dims[0] != key.Shape.Dims[0] {
		b.setError(errors.New("attention query/key widths differ"))
		return nil
	}
	if key.Shape.Dims[1] != value.Shape.Dims[1] {
		b.setError(errors.New("attention key/value head counts differ"))
		return nil
	}
	if key.Shape.Dims[2] != value.Shape.Dims[2] {
		b.setError(errors.New("attention key/value token counts differ"))
		return nil
	}
	if uint64(queryStart) > key.Shape.Dims[2] ||
		query.Shape.Dims[2] > key.Shape.Dims[2]-uint64(queryStart) {
		b.setError(fmt.Errorf(
			"attention query range [%d,%d) exceeds key/value token count %d",
			queryStart,
			uint64(queryStart)+query.Shape.Dims[2],
			key.Shape.Dims[2],
		))
		return nil
	}
	if query.Shape.Dims[1]%key.Shape.Dims[1] != 0 {
		b.setError(errors.New("attention query head count is not divisible by KV head count"))
		return nil
	}
	if scale <= 0 {
		b.setError(errors.New("attention scale must be positive"))
		return nil
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) ||
		math.IsInf(float64(softcap), 0) {
		b.setError(errors.New("attention softcap must be finite and non-negative"))
		return nil
	}
	if maxALiBiBias < 0 || math.IsNaN(float64(maxALiBiBias)) ||
		math.IsInf(float64(maxALiBiBias), 0) {
		b.setError(errors.New("attention ALiBi bias must be finite and non-negative"))
		return nil
	}
	if symmetricWindow && (causal || queryStart != 0 || query.Shape.Dims[2] != key.Shape.Dims[2]) {
		b.setError(errors.New("symmetric-window attention requires a full bidirectional sequence"))
		return nil
	}
	var relativeBuckets uint32
	if bias != nil {
		if maxALiBiBias > 0 {
			b.setError(errors.New("attention cannot combine learned relative bias and ALiBi"))
			return nil
		}
		if causal || queryStart != 0 || window != 0 {
			b.setError(errors.New("relative-bias attention must be full and bidirectional"))
			return nil
		}
		if bias.Type != query.Type || bias.Shape.Rank != 2 ||
			bias.Shape.Dims[0] != query.Shape.Dims[1] ||
			bias.Shape.Dims[1] < 4 || bias.Shape.Dims[1]%2 != 0 ||
			bias.Shape.Dims[1] > math.MaxUint32 {
			b.setError(errors.New("attention relative bias must have shape [query heads, even buckets >= 4]"))
			return nil
		}
		relativeBuckets = uint32(bias.Shape.Dims[1])
	}
	if sinks != nil {
		if bias != nil {
			b.setError(errors.New("attention cannot combine learned relative bias and sinks"))
			return nil
		}
		if sinks.Type != query.Type || sinks.Shape.Rank != 1 ||
			sinks.Shape.Dims[0] != query.Shape.Dims[1] {
			b.setError(errors.New("attention sinks must have shape [query heads]"))
			return nil
		}
	}
	shape, err := NewShape(value.Shape.Dims[0], query.Shape.Dims[1], query.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	inputs := []*Tensor{query, key, value}
	if bias != nil {
		inputs = append(inputs, bias)
	} else if sinks != nil {
		inputs = append(inputs, sinks)
	}
	return b.add(
		"",
		query.Type,
		shape,
		OpAttention,
		inputs,
		AttentionAttributes{
			Scale: scale, Softcap: softcap, MaxALiBiBias: maxALiBiBias, Causal: causal,
			HasSinks:        sinks != nil,
			SymmetricWindow: symmetricWindow,
			QueryStart:      queryStart, Window: window,
			RelativeBuckets: relativeBuckets,
		},
	)
}

// Concat: joins rank-2/3 tensors along dimension zero, or rank-3 tensors along
// token dimension
func (b *Builder) Concat(left, right *Tensor, axis uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if left == nil || right == nil {
		b.setError(errors.New("concat input is nil"))
		return nil
	}
	if left.Type != right.Type {
		b.setError(errors.New("concat input types differ"))
		return nil
	}
	if left.Shape.Rank != right.Shape.Rank ||
		(left.Shape.Rank != 2 && left.Shape.Rank != 3) ||
		(axis != 0 && (axis != 2 || left.Shape.Rank != 3)) {
		b.setError(errors.New("concat supports rank-2/3 axis 0 and rank-3 axis 2"))
		return nil
	}
	dimensions := left.Shape.Slice()
	for dimension := range left.Shape.Rank {
		if uint32(dimension) != axis && left.Shape.Dims[dimension] != right.Shape.Dims[dimension] {
			b.setError(errors.New("concat non-joined dimensions differ"))
			return nil
		}
	}
	if left.Shape.Dims[axis] > math.MaxUint64-right.Shape.Dims[axis] {
		b.setError(errors.New("concat joined dimension overflows"))
		return nil
	}
	dimensions[axis] += right.Shape.Dims[axis]
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", left.Type, shape, OpConcat, []*Tensor{left, right}, ConcatAttributes{Axis: axis})
}

func (b *Builder) unary(op Op, input *Tensor, attrs any) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil {
		b.setError(fmt.Errorf("%s input is nil", op))
		return nil
	}
	return b.add("", input.Type, input.Shape, op, []*Tensor{input}, attrs)
}

func (b *Builder) binary(op Op, left, right *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if left == nil || right == nil {
		b.setError(fmt.Errorf("%s input is nil", op))
		return nil
	}
	if left.Type != right.Type {
		b.setError(fmt.Errorf("%s input types differ: %s and %s", op, left.Type, right.Type))
		return nil
	}
	shape, err := broadcastShape(left.Shape, right.Shape)
	if err != nil {
		b.setError(fmt.Errorf("%s: %w", op, err))
		return nil
	}
	return b.add("", left.Type, shape, op, []*Tensor{left, right}, nil)
}

func broadcastShape(left, right Shape) (Shape, error) {
	rank := left.Rank
	if right.Rank > rank {
		rank = right.Rank
	}
	dimensions := make([]uint64, rank)
	for axis := uint8(0); axis < rank; axis++ {
		leftDimension := left.Dims[axis]
		rightDimension := right.Dims[axis]
		if leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1 {
			return Shape{}, fmt.Errorf(
				"input shapes %v and %v cannot broadcast at dimension %d",
				left.Slice(),
				right.Slice(),
				axis,
			)
		}
		if leftDimension > rightDimension {
			dimensions[axis] = leftDimension
		} else {
			dimensions[axis] = rightDimension
		}
	}
	return NewShape(dimensions...)
}

func (b *Builder) add(
	name string,
	dataType dtype.Type,
	shape Shape,
	op Op,
	inputs []*Tensor,
	attrs any,
) *Tensor {
	stride, err := contiguousStride(dataType, shape)
	if err != nil {
		b.setError(err)
		return nil
	}
	tensor := &Tensor{
		ID:     b.nextID,
		Name:   name,
		Type:   dataType,
		Shape:  shape,
		Stride: stride,
		Op:     op,
		Inputs: inputs,
		Attrs:  attrs,
	}
	b.nextID++
	b.nodes = append(b.nodes, tensor)
	return tensor
}

func (b *Builder) setError(err error) {
	if b.err == nil {
		b.err = err
	}
}

func contiguousStride(dataType dtype.Type, shape Shape) ([MaxDimensions]uint64, error) {
	traits, ok := dataType.Traits()
	if !ok {
		return [MaxDimensions]uint64{}, fmt.Errorf("unknown tensor type %d", dataType)
	}
	if shape.Dims[0]%traits.BlockSize != 0 {
		return [MaxDimensions]uint64{}, fmt.Errorf(
			"row width %d is not divisible by %s block size %d",
			shape.Dims[0],
			traits.Name,
			traits.BlockSize,
		)
	}
	stride := [MaxDimensions]uint64{}
	stride[0] = traits.TypeSize
	rowBlocks := shape.Dims[0] / traits.BlockSize
	if rowBlocks > math.MaxUint64/stride[0] {
		return stride, errors.New("tensor row stride overflows uint64")
	}
	stride[1] = rowBlocks * stride[0]
	for index := 2; index < MaxDimensions; index++ {
		if shape.Dims[index-1] > math.MaxUint64/stride[index-1] {
			return stride, errors.New("tensor stride overflows uint64")
		}
		stride[index] = stride[index-1] * shape.Dims[index-1]
	}
	return stride, nil
}
