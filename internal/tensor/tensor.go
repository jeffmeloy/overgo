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
	OpSSMScan
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
	OpTanh
	OpExp
	OpGatedLinearAttention
	OpRWKV6
	OpSumRows
	OpRWKV7
	OpFWHT
	OpTopK
	OpGatherLast
	OpSparseAttention
	OpIndexerScore
	OpReLU
	OpConv1DSame
	OpGroupNorm
	OpDeepSeek4HCInit
	OpDeepSeek4HCPre
	OpDeepSeek4HCPost
	OpDeepSeek4HCHead
	OpDeepSeek4Attention
	OpLoRAMerge
	OpDivide
	OpBF16Round
	OpGELUErf
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
	"ssm_scan",
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
	"tanh",
	"exp",
	"gated_linear_attention",
	"rwkv6",
	"sum_rows",
	"rwkv7",
	"fwht",
	"top_k",
	"gather_last",
	"sparse_attention",
	"indexer_score",
	"relu",
	"conv_1d_same",
	"group_norm",
	"deepseek4_hc_init",
	"deepseek4_hc_pre",
	"deepseek4_hc_post",
	"deepseek4_hc_head",
	"deepseek4_attention",
	"lora_merge",
	"divide",
	"bf16_round",
	"gelu_erf",
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
	Scale                 float32
	Softcap               float32
	MaxALiBiBias          float32
	Causal                bool
	HasSinks              bool
	HasBlockMask          bool
	SymmetricWindow       bool
	ChunkedWindow         bool
	QueryStart            uint32
	Window                uint32
	RelativeBuckets       uint32
	RelativeBidirectional bool
}

type Conv1DAttributes struct {
	Depthwise bool
}

type GroupNormAttributes struct {
	Groups  uint32
	Epsilon float32
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
	HasSelectedExperts bool
	SwiGLUClamp        float32
}

type DeepSeek4HCAttributes struct {
	HyperConnections   uint32
	SinkhornIterations uint32
	NormEpsilon        float32
	Epsilon            float32
}

type DeepSeek4AttentionAttributes struct {
	Positions        []uint32
	Ratio            uint32
	Window           uint32
	Heads            uint32
	IndexerHeads     uint32
	IndexerTopK      uint32
	RotaryDimensions uint32
	FrequencyBase    float32
	FrequencyScale   float32
	OriginalContext  uint32
	ExtFactor        float32
	AttentionFactor  float32
	BetaFast         float32
	BetaSlow         float32
	NormEpsilon      float32
}

type GatedDeltaNetAttributes struct {
	RepeatInterleave bool
}

type GatedLinearAttentionAttributes struct {
	Scale float32
}

type MoERouting uint32

const (
	MoERoutingSoftmax         MoERouting = 1
	MoERoutingSigmoid         MoERouting = 2
	MoERoutingSelectedSoftmax MoERouting = 3
	MoERoutingSqrtSoftplus    MoERouting = 4
)

type MoEActivation uint32

const (
	MoEActivationSiLU        MoEActivation = 1
	MoEActivationReLU        MoEActivation = 2
	MoEActivationGELU        MoEActivation = 3
	MoEActivationSwiGLUOAI   MoEActivation = 4
	MoEActivationReLUSquared MoEActivation = 5
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

type TopKAttributes struct {
	K uint32
}

type SparseAttentionAttributes struct {
	Scale      float32
	Causal     bool
	QueryStart uint32
}

type IndexerScoreAttributes struct {
	Scale      float32
	QueryStart uint32
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

// EmbeddedInputAttributes: immutable graph-owned F32 feed.
type EmbeddedInputAttributes struct {
	Data []float32
}

// LoRADefinition: one named adapter projection.
type LoRADefinition struct {
	AName, BName   string
	AShape, BShape Shape
	AData, BData   []float32
	Scale          float32
	Embedding      bool
}

// LoRAMergeAttributes: one adapter scale.
type LoRAMergeAttributes struct {
	Scale float32
}

// Builder: graph constructor and validator.
type Builder struct {
	nextID     uint64
	nodes      []*Tensor
	err        error
	loras      map[string][]LoRADefinition
	loraInputs map[string]*Tensor
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

// SetLoRA: installs exact base-tensor adapter bindings.
func (b *Builder) SetLoRA(bindings map[string][]LoRADefinition) {
	if b.err != nil || len(bindings) == 0 {
		return
	}
	b.loras = make(map[string][]LoRADefinition, len(bindings))
	b.loraInputs = make(map[string]*Tensor)
	for base, definitions := range bindings {
		if base == "" {
			b.setError(errors.New("LoRA base tensor name is empty"))
			return
		}
		for _, definition := range definitions {
			if definition.AName == "" || definition.BName == "" ||
				definition.AName == definition.BName ||
				math.IsNaN(float64(definition.Scale)) || math.IsInf(float64(definition.Scale), 0) {
				b.setError(fmt.Errorf("LoRA binding for %q is invalid", base))
				return
			}
			for name, value := range map[string]struct {
				shape Shape
				data  []float32
			}{
				definition.AName: {definition.AShape, definition.AData},
				definition.BName: {definition.BShape, definition.BData},
			} {
				elements, err := value.shape.Elements()
				if err != nil || elements != uint64(len(value.data)) {
					b.setError(fmt.Errorf("LoRA tensor %q data shape is invalid", name))
					return
				}
			}
			b.loras[base] = append(b.loras[base], definition)
		}
	}
}

func (b *Builder) loraInput(name string, shape Shape, data []float32) *Tensor {
	if input := b.loraInputs[name]; input != nil {
		if !input.Shape.Equal(shape) {
			b.setError(fmt.Errorf("LoRA input %q has conflicting shapes", name))
			return nil
		}
		return input
	}
	input := b.add(name, dtype.F32, shape, OpInput, nil, EmbeddedInputAttributes{Data: data})
	b.loraInputs[name] = input
	return input
}

func (b *Builder) mergeLoRAWeight(base *Tensor) *Tensor {
	if b.err != nil || base == nil || base.Name == "" {
		return base
	}
	result := base
	for _, definition := range b.loras[base.Name] {
		if definition.Scale == 0 || definition.Embedding {
			continue
		}
		if result.Type != dtype.F32 || result.Shape.Rank < 2 || result.Shape.Rank > 3 {
			b.setError(fmt.Errorf("LoRA weight merge for %q requires rank-2/3 F32 base", base.Name))
			return nil
		}
		a := b.loraInput(definition.AName, definition.AShape, definition.AData)
		c := b.loraInput(definition.BName, definition.BShape, definition.BData)
		if a.Shape.Rank != base.Shape.Rank || c.Shape.Rank != base.Shape.Rank ||
			a.Shape.Dims[0] != base.Shape.Dims[0] || c.Shape.Dims[1] != base.Shape.Dims[1] ||
			a.Shape.Dims[1] != c.Shape.Dims[0] {
			b.setError(fmt.Errorf("LoRA weight merge for %q has incompatible pair", base.Name))
			return nil
		}
		if base.Shape.Rank == 3 &&
			(a.Shape.Dims[2] != base.Shape.Dims[2] || c.Shape.Dims[2] != base.Shape.Dims[2]) {
			b.setError(fmt.Errorf("LoRA weight merge for %q has incompatible groups", base.Name))
			return nil
		}
		if math.IsNaN(float64(definition.Scale)) || math.IsInf(float64(definition.Scale), 0) {
			b.setError(fmt.Errorf("LoRA weight merge for %q has invalid scale", base.Name))
			return nil
		}
		result = b.add("", dtype.F32, base.Shape, OpLoRAMerge,
			[]*Tensor{result, a, c}, LoRAMergeAttributes{Scale: definition.Scale})
	}
	return result
}

func (b *Builder) Add(left, right *Tensor) *Tensor {
	return b.binary(OpAdd, left, right)
}

func (b *Builder) Multiply(left, right *Tensor) *Tensor {
	return b.binary(OpMultiply, left, right)
}

func (b *Builder) Divide(left, right *Tensor) *Tensor {
	return b.binary(OpDivide, left, right)
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

func (b *Builder) BF16Round(input *Tensor) *Tensor {
	return b.unary(OpBF16Round, input, nil)
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

func (b *Builder) GELUErf(input *Tensor) *Tensor {
	return b.unary(OpGELUErf, input, nil)
}

func (b *Builder) ReLU(input *Tensor) *Tensor {
	return b.unary(OpReLU, input, nil)
}

// Conv1DSame: odd-kernel stride-one convolution.
func (b *Builder) Conv1DSame(input, weight, bias *Tensor, depthwise bool) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || weight == nil || bias == nil {
		b.setError(errors.New("same Conv1D input is nil"))
		return nil
	}
	if input.Type != weight.Type || input.Type != bias.Type ||
		input.Shape.Rank != 2 || weight.Shape.Rank != 3 ||
		weight.Shape.Dims[0] == 0 || weight.Shape.Dims[0]%2 == 0 {
		b.setError(errors.New("same Conv1D tensor shape is invalid"))
		return nil
	}
	inputChannels := input.Shape.Dims[0]
	outputChannels := weight.Shape.Dims[2]
	if (depthwise && (weight.Shape.Dims[1] != 1 || outputChannels != inputChannels)) ||
		(!depthwise && weight.Shape.Dims[1] != inputChannels) {
		b.setError(errors.New("same Conv1D channel shape is incompatible"))
		return nil
	}
	biasOK := bias.Shape.Rank == 1 && bias.Shape.Dims[0] == outputChannels
	biasOK = biasOK || bias.Shape.Rank == 2 && bias.Shape.Dims[0] == 1 && bias.Shape.Dims[1] == outputChannels
	if !biasOK {
		b.setError(errors.New("same Conv1D bias shape is incompatible"))
		return nil
	}
	shape, err := NewShape(outputChannels, input.Shape.Dims[1])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", input.Type, shape, OpConv1DSame, []*Tensor{input, weight, bias}, Conv1DAttributes{Depthwise: depthwise})
}

// GroupNorm: channel groups across one sequence.
func (b *Builder) GroupNorm(input, weight, bias *Tensor, groups uint32, epsilon float32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || weight == nil || bias == nil {
		b.setError(errors.New("group norm input is nil"))
		return nil
	}
	if epsilon <= 0 || groups == 0 || input.Shape.Rank != 2 ||
		input.Shape.Dims[0]%uint64(groups) != 0 ||
		input.Type != weight.Type || input.Type != bias.Type {
		b.setError(errors.New("group norm configuration is invalid"))
		return nil
	}
	channels := input.Shape.Dims[0]
	weightOK := weight.Shape.Rank == 1 && weight.Shape.Dims[0] == channels
	weightOK = weightOK || weight.Shape.Rank == 2 && weight.Shape.Dims[0] == 1 && weight.Shape.Dims[1] == channels
	biasOK := bias.Shape.Rank == 1 && bias.Shape.Dims[0] == channels
	biasOK = biasOK || bias.Shape.Rank == 2 && bias.Shape.Dims[0] == 1 && bias.Shape.Dims[1] == channels
	if !weightOK || !biasOK {
		b.setError(errors.New("group norm affine shape is incompatible"))
		return nil
	}
	return b.add("", input.Type, input.Shape, OpGroupNorm, []*Tensor{input, weight, bias}, GroupNormAttributes{Groups: groups, Epsilon: epsilon})
}

// DeepSeek4HCInit: replicate embedding across hyper streams.
func (b *Builder) DeepSeek4HCInit(input *Tensor, hyperConnections uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank != 2 || hyperConnections == 0 {
		b.setError(errors.New("DeepSeek 4 HC init input is invalid"))
		return nil
	}
	shape, err := NewShape(input.Shape.Dims[0], uint64(hyperConnections), input.Shape.Dims[1])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpDeepSeek4HCInit, []*Tensor{input}, DeepSeek4HCAttributes{HyperConnections: hyperConnections})
}

// DeepSeek4HCPre: hyper-stream branch input.
func (b *Builder) DeepSeek4HCPre(
	input, fn, scale, base *Tensor,
	hyperConnections, sinkhornIterations uint32,
	normEpsilon, epsilon float32,
) *Tensor {
	if !b.validateDeepSeek4HC(input, fn, scale, base, hyperConnections, sinkhornIterations, normEpsilon, epsilon, false) {
		return nil
	}
	shape, _ := NewShape(input.Shape.Dims[0], input.Shape.Dims[2])
	return b.add("", dtype.F32, shape, OpDeepSeek4HCPre, []*Tensor{input, fn, scale, base}, DeepSeek4HCAttributes{
		HyperConnections: hyperConnections, SinkhornIterations: sinkhornIterations, NormEpsilon: normEpsilon, Epsilon: epsilon,
	})
}

// DeepSeek4HCPost: branch merge into hyper streams.
func (b *Builder) DeepSeek4HCPost(
	branch, residual, fn, scale, base *Tensor,
	hyperConnections, sinkhornIterations uint32,
	normEpsilon, epsilon float32,
) *Tensor {
	if !b.validateDeepSeek4HC(residual, fn, scale, base, hyperConnections, sinkhornIterations, normEpsilon, epsilon, false) {
		return nil
	}
	if branch == nil || branch.Type != dtype.F32 || branch.Shape.Rank != 2 ||
		branch.Shape.Dims[0] != residual.Shape.Dims[0] || branch.Shape.Dims[1] != residual.Shape.Dims[2] {
		b.setError(errors.New("DeepSeek 4 HC post branch shape is invalid"))
		return nil
	}
	return b.add("", dtype.F32, residual.Shape, OpDeepSeek4HCPost, []*Tensor{branch, residual, fn, scale, base}, DeepSeek4HCAttributes{
		HyperConnections: hyperConnections, SinkhornIterations: sinkhornIterations, NormEpsilon: normEpsilon, Epsilon: epsilon,
	})
}

// DeepSeek4HCHead: collapse final hyper streams.
func (b *Builder) DeepSeek4HCHead(input, fn, scale, base *Tensor, hyperConnections uint32, normEpsilon, epsilon float32) *Tensor {
	if !b.validateDeepSeek4HC(input, fn, scale, base, hyperConnections, 1, normEpsilon, epsilon, true) {
		return nil
	}
	shape, _ := NewShape(input.Shape.Dims[0], input.Shape.Dims[2])
	return b.add("", dtype.F32, shape, OpDeepSeek4HCHead, []*Tensor{input, fn, scale, base}, DeepSeek4HCAttributes{
		HyperConnections: hyperConnections, SinkhornIterations: 1, NormEpsilon: normEpsilon, Epsilon: epsilon,
	})
}

func (b *Builder) validateDeepSeek4HC(
	input, fn, scale, base *Tensor,
	hyperConnections, sinkhornIterations uint32,
	normEpsilon, epsilon float32,
	head bool,
) bool {
	if b.err != nil {
		return false
	}
	if input == nil || fn == nil || scale == nil || base == nil ||
		input.Type != dtype.F32 || fn.Type != dtype.F32 || scale.Type != dtype.F32 || base.Type != dtype.F32 ||
		input.Shape.Rank != 3 || fn.Shape.Rank != 2 || scale.Shape.Rank != 1 || base.Shape.Rank != 1 ||
		hyperConnections == 0 || sinkhornIterations == 0 || normEpsilon <= 0 || epsilon <= 0 ||
		math.IsNaN(float64(normEpsilon)) || math.IsInf(float64(normEpsilon), 0) ||
		math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) ||
		input.Shape.Dims[1] != uint64(hyperConnections) || fn.Shape.Dims[0] != input.Shape.Dims[0]*uint64(hyperConnections) {
		b.setError(errors.New("DeepSeek 4 HC metadata or input shape is invalid"))
		return false
	}
	mix := uint64(hyperConnections)
	scaleWidth := uint64(1)
	if !head {
		mix *= uint64(2 + hyperConnections)
		scaleWidth = 3
	}
	if fn.Shape.Dims[1] != mix || scale.Shape.Dims[0] != scaleWidth || base.Shape.Dims[0] != mix {
		b.setError(errors.New("DeepSeek 4 HC weight shape is invalid"))
		return false
	}
	return true
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

func (b *Builder) Tanh(input *Tensor) *Tensor {
	return b.unary(OpTanh, input, nil)
}

func (b *Builder) Exp(input *Tensor) *Tensor {
	return b.unary(OpExp, input, nil)
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

// SSMScan: selective state update; packed output then final state.
func (b *Builder) SSMScan(state, x, dt, a, beta, c *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{state, x, dt, a, beta, c}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("SSMScan requires six F32 inputs"))
			return nil
		}
	}
	if state.Shape.Rank != 4 || x.Shape.Rank != 4 || dt.Shape.Rank != 3 ||
		a.Shape.Rank != 2 || beta.Shape.Rank != 4 || c.Shape.Rank != 4 {
		b.setError(errors.New("SSMScan input ranks are invalid"))
		return nil
	}
	stateWidth := state.Shape.Dims[0]
	dimension := state.Shape.Dims[1]
	heads := state.Shape.Dims[2]
	sequences := state.Shape.Dims[3]
	tokens := x.Shape.Dims[2]
	groups := beta.Shape.Dims[1]
	if stateWidth == 0 || dimension == 0 || heads == 0 || sequences == 0 || tokens == 0 || groups == 0 ||
		x.Shape.Dims[0] != dimension || x.Shape.Dims[1] != heads || x.Shape.Dims[3] != sequences ||
		dt.Shape.Dims[0] != heads || dt.Shape.Dims[1] != tokens || dt.Shape.Dims[2] != sequences ||
		(a.Shape.Dims[0] != 1 && a.Shape.Dims[0] != stateWidth) || a.Shape.Dims[1] != heads || heads%groups != 0 ||
		beta.Shape.Dims[0] != stateWidth || beta.Shape.Dims[2] != tokens || beta.Shape.Dims[3] != sequences ||
		!beta.Shape.Equal(c.Shape) {
		b.setError(errors.New("SSMScan input shapes are incompatible"))
		return nil
	}
	attentionElements := dimension * heads * tokens * sequences
	stateElements := stateWidth * dimension * heads * sequences
	shape, err := NewShape(attentionElements + stateElements)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpSSMScan, inputs, nil)
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

// GatedLinearAttention: QRWKV linear attention; packed output and state.
func (b *Builder) GatedLinearAttention(key, value, receptance, decay, state *Tensor, scale float32) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{key, value, receptance, decay, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("GatedLinearAttention requires five F32 inputs"))
			return nil
		}
	}
	if key.Shape.Rank != 4 || value.Shape.Rank != 4 || receptance.Shape.Rank != 4 ||
		decay.Shape.Rank != 4 || state.Shape.Rank != 4 || !key.Shape.Equal(value.Shape) {
		b.setError(errors.New("GatedLinearAttention input shapes are incompatible"))
		return nil
	}
	width, heads := receptance.Shape.Dims[0], receptance.Shape.Dims[1]
	tokens, sequences := receptance.Shape.Dims[2], receptance.Shape.Dims[3]
	if width == 0 || heads == 0 || tokens == 0 || sequences == 0 ||
		key.Shape.Dims[0] != width || key.Shape.Dims[1] == 0 || heads%key.Shape.Dims[1] != 0 ||
		key.Shape.Dims[2] != tokens || key.Shape.Dims[3] != sequences ||
		!receptance.Shape.Equal(decay.Shape) ||
		state.Shape.Dims[0] != width || state.Shape.Dims[1] != width ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		b.setError(errors.New("GatedLinearAttention state or scale is invalid"))
		return nil
	}
	shape, err := NewShape(width*heads, tokens*sequences+width*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpGatedLinearAttention, inputs, GatedLinearAttentionAttributes{Scale: scale})
}

// RWKV6: classic WKV6 recurrence; packed output and state.
func (b *Builder) RWKV6(key, value, receptance, first, decay, state *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{key, value, receptance, first, decay, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("RWKV6 requires six F32 inputs"))
			return nil
		}
	}
	if key.Shape.Rank != 4 || !key.Shape.Equal(value.Shape) || !key.Shape.Equal(receptance.Shape) ||
		!key.Shape.Equal(decay.Shape) || first.Shape.Rank != 2 || state.Shape.Rank != 4 {
		b.setError(errors.New("RWKV6 input shapes are incompatible"))
		return nil
	}
	width, heads := key.Shape.Dims[0], key.Shape.Dims[1]
	tokens, sequences := key.Shape.Dims[2], key.Shape.Dims[3]
	if width == 0 || heads == 0 || tokens == 0 || sequences == 0 ||
		first.Shape.Dims[0] != width || first.Shape.Dims[1] != heads ||
		state.Shape.Dims[0] != width || state.Shape.Dims[1] != width ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences {
		b.setError(errors.New("RWKV6 state or time-first shape is invalid"))
		return nil
	}
	shape, err := NewShape(width*heads, tokens*sequences+width*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpRWKV6, inputs, nil)
}

// SumRows: first-dimension reduction.
func (b *Builder) SumRows(input *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank == 0 || input.Shape.Dims[0] == 0 {
		b.setError(errors.New("SumRows requires non-empty F32 input"))
		return nil
	}
	dimensions := input.Shape.Slice()
	dimensions[0] = 1
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpSumRows, []*Tensor{input}, nil)
}

// FWHT: orthonormal Walsh-Hadamard transform over dimension zero.
func (b *Builder) FWHT(input *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank == 0 {
		b.setError(errors.New("FWHT requires non-empty F32 input"))
		return nil
	}
	width := input.Shape.Dims[0]
	if width == 0 || width&(width-1) != 0 {
		b.setError(errors.New("FWHT dimension zero must be a power of two"))
		return nil
	}
	return b.add("", dtype.F32, input.Shape, OpFWHT, []*Tensor{input}, nil)
}

// TopK: descending dimension-zero indices; lower index wins ties.
func (b *Builder) TopK(input *Tensor, k uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank == 0 {
		b.setError(errors.New("TopK requires non-empty F32 input"))
		return nil
	}
	if k == 0 || uint64(k) > input.Shape.Dims[0] {
		b.setError(errors.New("TopK count exceeds dimension zero"))
		return nil
	}
	if input.Shape.Dims[0] > 1<<24 {
		b.setError(errors.New("TopK dimension zero exceeds exact F32 index range"))
		return nil
	}
	dimensions := input.Shape.Slice()
	dimensions[0] = uint64(k)
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpTopK, []*Tensor{input}, TopKAttributes{K: k})
}

// GatherLast: dynamic gather from final dimension; exact F32 indices.
func (b *Builder) GatherLast(input, indices *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || indices == nil || input.Type != dtype.F32 || indices.Type != dtype.F32 ||
		input.Shape.Rank == 0 || indices.Shape.Rank == 0 {
		b.setError(errors.New("GatherLast requires non-empty F32 inputs"))
		return nil
	}
	if int(input.Shape.Rank)-1+int(indices.Shape.Rank) > MaxDimensions {
		b.setError(errors.New("GatherLast output rank exceeds limit"))
		return nil
	}
	dimensions := append([]uint64(nil), input.Shape.Slice()[:input.Shape.Rank-1]...)
	dimensions = append(dimensions, indices.Shape.Slice()...)
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpGatherLast, []*Tensor{input, indices}, nil)
}

// RWKV7: vector-valued decay recurrence; packed output and state.
func (b *Builder) RWKV7(receptance, decay, key, value, a, bVector, state *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{receptance, decay, key, value, a, bVector, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("RWKV7 requires seven F32 inputs"))
			return nil
		}
	}
	if receptance.Shape.Rank != 4 || !receptance.Shape.Equal(decay.Shape) ||
		!receptance.Shape.Equal(key.Shape) || !receptance.Shape.Equal(value.Shape) ||
		!receptance.Shape.Equal(a.Shape) || !receptance.Shape.Equal(bVector.Shape) || state.Shape.Rank != 4 {
		b.setError(errors.New("RWKV7 input shapes are incompatible"))
		return nil
	}
	width, heads := key.Shape.Dims[0], key.Shape.Dims[1]
	tokens, sequences := key.Shape.Dims[2], key.Shape.Dims[3]
	if width == 0 || heads == 0 || tokens == 0 || sequences == 0 ||
		state.Shape.Dims[0] != width || state.Shape.Dims[1] != width ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences {
		b.setError(errors.New("RWKV7 state shape is invalid"))
		return nil
	}
	shape, err := NewShape(width*heads, tokens*sequences+width*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpRWKV7, inputs, nil)
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

// MoEReLUSquaredWithRouterInput: split router input; ungated squared-ReLU experts.
func (b *Builder) MoEReLUSquaredWithRouterInput(
	input, routerInput, router, up, down, selectionBias *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale float32,
	routing MoERouting,
) *Tensor {
	return b.moe(input, routerInput, router, nil, up, down, selectionBias, nil, topK, normalizeTopKProb, scale,
		routing, MoEActivationReLUSquared, false, 1, 0, nil)
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

// MoESqrtSoftplusLimited: DeepSeek 4 routing; optional fixed expert IDs.
func (b *Builder) MoESqrtSoftplusLimited(
	input, router, gate, up, down, selectionBias, selectedExperts *Tensor,
	topK uint32,
	normalizeTopKProb bool,
	scale, swigluClamp float32,
) *Tensor {
	return b.moeWithSelected(
		input, input, router, gate, up, down, selectionBias, nil,
		topK, normalizeTopKProb, scale, MoERoutingSqrtSoftplus,
		MoEActivationSiLU, false, 1, swigluClamp, nil, selectedExperts,
	)
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
	return b.moeWithSelected(
		input, routerInput, router, gate, up, down, selectionBias, expertScale,
		topK, normalizeTopKProb, scale, routing, activation, fusedGateUp,
		expertIndexDivisor, swigluClamp, biases, nil,
	)
}

func (b *Builder) moeWithSelected(
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
	selectedExperts *Tensor,
) *Tensor {
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
		uint64(topK) > experts || uint64(topK) > bankExperts || topK > 16 || scale == 0 ||
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

// ReGLU: ReLU(gate) * up
func (b *Builder) ReGLU(gate, up *Tensor) *Tensor {
	return b.Multiply(b.ReLU(gate), up)
}

// MulMat: follows ggml semantics; Left has shape [K,M], right has shape [K,N],
// and result has shape [M,N]
func (b *Builder) MulMat(left, right *Tensor) *Tensor {
	result := b.mulMat(left, right)
	if result == nil || left == nil || left.Name == "" {
		return result
	}
	for _, definition := range b.loras[left.Name] {
		if definition.Scale == 0 || definition.Embedding {
			continue
		}
		a := b.loraInput(definition.AName, definition.AShape, definition.AData)
		c := b.loraInput(definition.BName, definition.BShape, definition.BData)
		delta := b.mulMat(c, b.mulMat(a, right))
		result = b.Add(result, b.Scale(delta, definition.Scale))
	}
	return result
}

func (b *Builder) mulMat(left, right *Tensor) *Tensor {
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
	result := b.groupedMulMat(left, right)
	if result == nil || left == nil || left.Name == "" {
		return result
	}
	for _, definition := range b.loras[left.Name] {
		if definition.Scale == 0 || definition.Embedding {
			continue
		}
		a := b.loraInput(definition.AName, definition.AShape, definition.AData)
		c := b.loraInput(definition.BName, definition.BShape, definition.BData)
		delta := b.groupedMulMat(c, b.groupedMulMat(a, right))
		result = b.Add(result, b.Scale(delta, definition.Scale))
	}
	return result
}

func (b *Builder) groupedMulMat(left, right *Tensor) *Tensor {
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
	result := b.getRows(table, rows)
	if result == nil || table == nil || table.Name == "" {
		return result
	}
	for _, definition := range b.loras[table.Name] {
		if definition.Scale == 0 || !definition.Embedding {
			continue
		}
		a := b.loraInput(definition.AName, definition.AShape, definition.AData)
		c := b.loraInput(definition.BName, definition.BShape, definition.BData)
		delta := b.mulMat(c, b.getRows(a, rows))
		result = b.Add(result, b.Scale(delta, definition.Scale))
	}
	return result
}

func (b *Builder) getRows(table *Tensor, rows []uint32) *Tensor {
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
	return b.ropeYaRNWithFactors(
		OpRoPENormal, "rope_normal", input, positions, rotaryDimensions, originalContext,
		frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow, nil,
	)
}

// RoPENormalYaRNWithFactors: YaRN plus pair divisors.
func (b *Builder) RoPENormalYaRNWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.ropeYaRNWithFactors(
		OpRoPENormal, "rope_normal", input, positions, rotaryDimensions, originalContext,
		frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow,
		frequencyFactors,
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
	return b.ropeYaRNWithFactors(
		operation, name, input, positions, rotaryDimensions, originalContext,
		frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow, nil,
	)
}

func (b *Builder) ropeYaRNWithFactors(
	operation Op,
	name string,
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
	frequencyFactors *Tensor,
) *Tensor {
	if originalContext == 0 || attentionFactor <= 0 || betaFast <= 0 || betaSlow <= 0 || extFactor < 0 {
		b.setError(errors.New("YaRN RoPE parameters are invalid"))
		return nil
	}
	result := b.rope(
		operation, name, input, positions, rotaryDimensions, frequencyBase, frequencyScale, frequencyFactors,
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

// RoPEMulti: adjacent-pair multi-axis rotation; axes T/H/W/extra.
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
	if sectionPairs == 0 {
		b.setError(errors.New("rope_multi sections are empty"))
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

// DeepSeek4Attention: raw plus reconstructed compressed attention.
func (b *Builder) DeepSeek4Attention(
	query, cacheKV, cachePositions, sinks, compressorKV, compressorScore, compressorNorm,
	indexerQuery, indexerWeights, indexerKV, indexerScore, indexerNorm *Tensor,
	attributes DeepSeek4AttentionAttributes,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || cacheKV == nil || cachePositions == nil || sinks == nil ||
		query.Type != dtype.F32 || cacheKV.Type != dtype.F32 || cachePositions.Type != dtype.F32 || sinks.Type != dtype.F32 ||
		query.Shape.Rank != 3 || cacheKV.Shape.Rank != 3 || sinks.Shape.Rank != 1 ||
		cacheKV.Shape.Dims[0] != query.Shape.Dims[0] || cacheKV.Shape.Dims[1] != 1 ||
		cachePositions.Shape != MustShape(1, 1, cacheKV.Shape.Dims[2]) ||
		sinks.Shape.Dims[0] != query.Shape.Dims[1] ||
		len(attributes.Positions) != int(query.Shape.Dims[2]) ||
		attributes.Heads != uint32(query.Shape.Dims[1]) || attributes.Window == 0 ||
		attributes.RotaryDimensions == 0 || attributes.RotaryDimensions%2 != 0 ||
		uint64(attributes.RotaryDimensions) > query.Shape.Dims[0] ||
		attributes.FrequencyBase <= 0 || attributes.FrequencyScale <= 0 || attributes.NormEpsilon <= 0 ||
		(attributes.Ratio != 0 && attributes.Ratio != 4 && attributes.Ratio != 128) {
		b.setError(errors.New("DeepSeek 4 attention metadata or base inputs are invalid"))
		return nil
	}
	inputs := []*Tensor{query, cacheKV, cachePositions, sinks}
	tokens := cacheKV.Shape.Dims[2]
	if attributes.Ratio != 0 {
		coefficient := uint64(1)
		if attributes.Ratio == 4 {
			coefficient = 2
		}
		if compressorKV == nil || compressorScore == nil || compressorNorm == nil ||
			compressorKV.Type != dtype.F32 || compressorScore.Type != dtype.F32 || compressorNorm.Type != dtype.F32 ||
			compressorKV.Shape != MustShape(coefficient*query.Shape.Dims[0], 1, tokens) ||
			!compressorScore.Shape.Equal(compressorKV.Shape) ||
			compressorNorm.Shape != MustShape(query.Shape.Dims[0]) {
			b.setError(errors.New("DeepSeek 4 compressor inputs are invalid"))
			return nil
		}
		inputs = append(inputs, compressorKV, compressorScore, compressorNorm)
	}
	if attributes.Ratio == 4 {
		if attributes.IndexerHeads == 0 || attributes.IndexerTopK == 0 ||
			indexerQuery == nil || indexerWeights == nil || indexerKV == nil || indexerScore == nil || indexerNorm == nil ||
			indexerQuery.Type != dtype.F32 || indexerWeights.Type != dtype.F32 || indexerKV.Type != dtype.F32 ||
			indexerScore.Type != dtype.F32 || indexerNorm.Type != dtype.F32 ||
			indexerQuery.Shape.Rank != 3 || indexerQuery.Shape.Dims[1] != uint64(attributes.IndexerHeads) ||
			indexerQuery.Shape.Dims[2] != query.Shape.Dims[2] ||
			indexerWeights.Shape != MustShape(uint64(attributes.IndexerHeads), query.Shape.Dims[2]) ||
			indexerKV.Shape != MustShape(2*indexerQuery.Shape.Dims[0], 1, tokens) ||
			!indexerScore.Shape.Equal(indexerKV.Shape) || indexerNorm.Shape != MustShape(indexerQuery.Shape.Dims[0]) {
			b.setError(errors.New("DeepSeek 4 indexer inputs are invalid"))
			return nil
		}
		inputs = append(inputs, indexerQuery, indexerWeights, indexerKV, indexerScore, indexerNorm)
	}
	attributes.Positions = append([]uint32(nil), attributes.Positions...)
	return b.add("", dtype.F32, query.Shape, OpDeepSeek4Attention, inputs, attributes)
}

// SparseAttention: top-k indexed grouped-query attention.
func (b *Builder) SparseAttention(
	query, key, value, indices *Tensor,
	scale float32,
) *Tensor {
	return b.SparseAttentionWithOffset(query, key, value, indices, scale, false, 0)
}

// SparseAttentionWithOffset: optional causal suffix attention.
func (b *Builder) SparseAttentionWithOffset(
	query, key, value, indices *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || key == nil || value == nil || indices == nil {
		b.setError(errors.New("SparseAttention input is nil"))
		return nil
	}
	if query.Type != dtype.F32 || key.Type != dtype.F32 || value.Type != dtype.F32 || indices.Type != dtype.F32 {
		b.setError(errors.New("SparseAttention requires F32 inputs"))
		return nil
	}
	if query.Shape.Rank != 3 || key.Shape.Rank != 3 || value.Shape.Rank != 3 || indices.Shape.Rank != 2 {
		b.setError(errors.New("SparseAttention requires rank-3 Q/K/V and rank-2 indices"))
		return nil
	}
	if query.Shape.Dims[0] != key.Shape.Dims[0] ||
		key.Shape.Dims[1] != value.Shape.Dims[1] ||
		key.Shape.Dims[2] != value.Shape.Dims[2] ||
		query.Shape.Dims[1]%key.Shape.Dims[1] != 0 ||
		indices.Shape.Dims[0] == 0 || indices.Shape.Dims[0] > key.Shape.Dims[2] ||
		indices.Shape.Dims[1] != query.Shape.Dims[2] {
		b.setError(errors.New("SparseAttention input shapes are incompatible"))
		return nil
	}
	if uint64(queryStart) > key.Shape.Dims[2] ||
		query.Shape.Dims[2] > key.Shape.Dims[2]-uint64(queryStart) {
		b.setError(errors.New("SparseAttention query range exceeds KV tokens"))
		return nil
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		b.setError(errors.New("SparseAttention scale must be finite"))
		return nil
	}
	shape, err := NewShape(value.Shape.Dims[0], query.Shape.Dims[1], query.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add(
		"", dtype.F32, shape, OpSparseAttention,
		[]*Tensor{query, key, value, indices},
		SparseAttentionAttributes{Scale: scale, Causal: causal, QueryStart: queryStart},
	)
}

// IndexerScore: causal DSA head-reduced scores.
func (b *Builder) IndexerScore(
	query, key, weights *Tensor,
	scale float32,
	queryStart uint32,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || key == nil || weights == nil || query.Type != dtype.F32 ||
		key.Type != dtype.F32 || weights.Type != dtype.F32 {
		b.setError(errors.New("IndexerScore requires F32 inputs"))
		return nil
	}
	if query.Shape.Rank != 3 || key.Shape.Rank != 3 || weights.Shape.Rank != 2 ||
		query.Shape.Dims[0] != key.Shape.Dims[0] || key.Shape.Dims[1] != 1 ||
		weights.Shape.Dims[0] != query.Shape.Dims[1] ||
		weights.Shape.Dims[1] != query.Shape.Dims[2] ||
		uint64(queryStart) > key.Shape.Dims[2] ||
		query.Shape.Dims[2] > key.Shape.Dims[2]-uint64(queryStart) {
		b.setError(errors.New("IndexerScore input shapes are incompatible"))
		return nil
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		b.setError(errors.New("IndexerScore scale must be finite"))
		return nil
	}
	shape, err := NewShape(key.Shape.Dims[2], query.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpIndexerScore, []*Tensor{query, key, weights},
		IndexerScoreAttributes{Scale: scale, QueryStart: queryStart})
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
	return b.attentionWithRelativeBias(query, key, value, bias, scale, false, 0, true)
}

// AttentionWithRelativeBiasAndOffset: causal T5 decoder attention.
func (b *Builder) AttentionWithRelativeBiasAndOffset(
	query, key, value, bias *Tensor,
	scale float32,
	queryStart uint32,
) *Tensor {
	return b.attentionWithRelativeBias(query, key, value, bias, scale, true, queryStart, false)
}

func (b *Builder) attentionWithRelativeBias(
	query, key, value, bias *Tensor,
	scale float32,
	causal bool,
	queryStart uint32,
	bidirectional bool,
) *Tensor {
	result := b.attentionWithWindow(query, key, value, bias, nil, scale, 0, 0, causal, false, queryStart, 0)
	if result != nil {
		attributes := result.Attrs.(AttentionAttributes)
		attributes.RelativeBidirectional = bidirectional
		result.Attrs = attributes
	}
	return result
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

// AttentionWindowWithBlockMaskWithOffset: causal window plus bidirectional
// nonnegative block IDs. Negative IDs remain causal.
func (b *Builder) AttentionWindowWithBlockMaskWithOffset(
	query, key, value, blockIDs *Tensor,
	scale float32,
	queryStart uint32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	result := b.attentionWithWindow(
		query, key, value, nil, nil, scale, 0, 0, true, false, queryStart, window,
	)
	if result == nil {
		return nil
	}
	if blockIDs == nil || blockIDs.Type != query.Type || blockIDs.Shape.Rank != 1 ||
		blockIDs.Shape.Dims[0] != key.Shape.Dims[2] {
		b.setError(errors.New("attention block IDs must have shape [key tokens] and match input type"))
		return nil
	}
	result.Inputs = append(result.Inputs, blockIDs)
	attributes := result.Attrs.(AttentionAttributes)
	attributes.HasBlockMask = true
	result.Attrs = attributes
	return result
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

func (b *Builder) AttentionSymmetricWindowWithSinks(
	query, key, value, sinks *Tensor,
	scale float32,
	window uint32,
) *Tensor {
	if window == 0 {
		b.setError(errors.New("attention window must be positive"))
		return nil
	}
	return b.attentionWithWindow(query, key, value, nil, sinks, scale, 0, 0, false, true, 0, window)
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
		if window != 0 {
			b.setError(errors.New("relative-bias attention cannot use a window"))
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

// Concat: joins rank-2/3 tensors along dimension zero or the outer dimension.
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
		(axis != 0 && ((left.Shape.Rank == 2 && axis != 1) || (left.Shape.Rank == 3 && axis != 2))) {
		b.setError(errors.New("concat supports dimension zero or the outer dimension"))
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
