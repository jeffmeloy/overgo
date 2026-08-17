package tensor

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor/dtype"
)

const maxExactFloat32Integer = 1 << 24

// MaxTopKPairs: bounded device candidate width.
const MaxTopKPairs = uint32(64)

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
	OpWKV6
	OpSumRows
	OpWKV7
	OpFWHT
	OpTopK
	OpGatherLast
	OpSparseAttention
	OpIndexerScore
	OpReLU
	OpConv1DSame
	OpGroupNorm
	OpHyperConnectionInit
	OpHyperConnectionPre
	OpHyperConnectionPost
	OpHyperConnectionHead
	OpCompressedAttention
	OpLoRAMerge
	OpDivide
	OpBF16Round
	OpGELUErf
	OpConv2D
	OpWindowPartition2D
	OpWindowUnpartition2D
	OpSAMAttention
	OpTopKPairs
	OpTopKPartials
	OpCacheAppend
	OpMADNorm
	OpAtan
	OpPixelShuffle2D
	OpCount
)

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

type MADNormAttributes struct {
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

// MulMatCompute selects backend arithmetic. Zero preserves exact F32 operands.
type MulMatCompute uint8

const (
	MulMatComputeExact MulMatCompute = iota
	MulMatComputeBF16TensorCore
)

type MulMatAttributes struct {
	Compute MulMatCompute
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
	KeyValueTokens        uint32
	Window                uint32
	RelativeBuckets       uint32
	RelativeBidirectional bool
	// HasKeyBias: an additive per-key score bias [key tokens] is the final input,
	// added to QK^T before softmax (0.0 attended, large-negative to drop a pad
	// key). Ports adaptive runtime_causal_gqa_masked_bf16's per-key pad mask as an
	// additive bias so masked keys underflow to 0 probability; non-breaking (the
	// input and flag are absent for every existing caller).
	HasKeyBias bool
	// NaiveF32: route dense F32 attention through the per-query online kernel
	// (skip the blas-chunked scores path) so the reduction matches a reference
	// per-query two-pass attn_fwd; opt-in for F32 parity-critical towers.
	NaiveF32 bool
}

type Conv1DAttributes struct {
	Depthwise bool
}

type Conv2DAttributes struct {
	StrideX, StrideY                     uint32
	PadLeft, PadRight, PadTop, PadBottom uint32
	Depthwise, HasBias                   bool
}

type Window2DAttributes struct {
	Width, Height, Window uint32
}

type PixelShuffle2DAttributes struct {
	Scale uint32
}

type SAMAttentionAttributes struct {
	Scale, RelativeScale float32
	SpatialSize          uint32
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

type HyperConnectionAttributes struct {
	HyperConnections   uint32
	SinkhornIterations uint32
	NormEpsilon        float32
	Epsilon            float32
}

// CompressionRatio: serialized layer compression policy.
type CompressionRatio uint32

const (
	CompressionNone    CompressionRatio = 0
	CompressionOverlap CompressionRatio = 4
	CompressionWide    CompressionRatio = 128
)

func (r CompressionRatio) Valid() bool {
	return r == CompressionNone ||
		r == CompressionOverlap ||
		r == CompressionWide
}

func (r CompressionRatio) Enabled() bool {
	return r != CompressionNone
}

func (r CompressionRatio) UsesIndexer() bool {
	return r == CompressionOverlap
}

func (r CompressionRatio) KVWidthMultiplier() uint64 {
	if r.UsesIndexer() {
		return 2
	}
	return 1
}

type CompressedAttentionAttributes struct {
	Positions        []uint32
	Ratio            CompressionRatio
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

// CacheAppendAttributes: bounded logical append into capacity storage.
type CacheAppendAttributes struct {
	Axis   uint32
	Offset uint32
}

type TopKAttributes struct {
	K     uint32
	Chunk uint32
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
	Attrs  Attributes
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
	nextID        uint64
	nodes         []*Tensor
	err           error
	loras         map[string][]LoRADefinition
	loraInputs    map[string]*Tensor
	cacheAppend   *CacheAppendPlan
	mulMatCompute MulMatCompute
}

// CacheAppendPlan: logical range inside fixed-capacity cache storage.
type CacheAppendPlan struct {
	ActiveTokens         uint32
	SourceCapacityTokens uint32
	CapacityTokens       uint32
}

// CacheWriteMode: compiled cache topology.
type CacheWriteMode uint8

const (
	CacheWriteConcat CacheWriteMode = iota
	CacheWriteAppend
)

func NewBuilder() *Builder {
	return &Builder{nextID: 1}
}

func (b *Builder) Err() error {
	return b.err
}

func (b *Builder) Nodes() []*Tensor {
	return append([]*Tensor(nil), b.nodes...)
}

// SetMulMatCompute sets graph-default projection arithmetic.
func (b *Builder) SetMulMatCompute(compute MulMatCompute) {
	if b.err != nil {
		return
	}
	if len(b.nodes) != 0 {
		b.setError(errors.New("mul_mat compute policy must be set before graph construction"))
		return
	}
	if compute != MulMatComputeExact && compute != MulMatComputeBF16TensorCore {
		b.setError(fmt.Errorf("mul_mat compute policy %d is invalid", compute))
		return
	}
	b.mulMatCompute = compute
}

// SetCacheAppendPlan: fixed-capacity cache construction.
func (b *Builder) SetCacheAppendPlan(plan CacheAppendPlan) {
	if b.err != nil {
		return
	}
	if plan.SourceCapacityTokens == 0 {
		plan.SourceCapacityTokens = plan.CapacityTokens
	}
	if plan.CapacityTokens == 0 || plan.ActiveTokens >= plan.CapacityTokens ||
		plan.SourceCapacityTokens > plan.CapacityTokens ||
		plan.ActiveTokens > plan.SourceCapacityTokens {
		b.setError(errors.New("cache append plan is invalid"))
		return
	}
	b.cacheAppend = &plan
}

// CacheTokenOffset: configured logical offset or fallback.
func (b *Builder) CacheTokenOffset(fallback uint32) uint32 {
	if b != nil && b.cacheAppend != nil {
		return b.cacheAppend.ActiveTokens
	}
	return fallback
}

// CacheCapacity: configured cache storage width.
func (b *Builder) CacheCapacity() (uint32, bool) {
	if b == nil || b.cacheAppend == nil {
		return 0, false
	}
	return b.cacheAppend.CapacityTokens, true
}

// CacheSourceCapacity: configured source storage width.
func (b *Builder) CacheSourceCapacity() (uint32, bool) {
	if b == nil || b.cacheAppend == nil {
		return 0, false
	}
	return b.cacheAppend.SourceCapacityTokens, true
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

// Concat: joins tensors along one dimension.
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
	if left.Shape.Rank != right.Shape.Rank || left.Shape.Rank == 0 ||
		axis >= uint32(left.Shape.Rank) {
		b.setError(errors.New("concat axis exceeds input rank"))
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

// WriteCache: explicit cache-write topology.
func (b *Builder) WriteCache(left, right *Tensor, axis uint32, mode CacheWriteMode) *Tensor {
	switch mode {
	case CacheWriteConcat:
		return b.Concat(left, right, axis)
	case CacheWriteAppend:
		return b.AppendCache(left, right, axis)
	default:
		if b != nil {
			b.setError(errors.New("cache write mode is invalid"))
		}
		return nil
	}
}

// AppendCache: bounded fixed-capacity append.
func (b *Builder) AppendCache(left, right *Tensor, axis uint32) *Tensor {
	if b == nil {
		return nil
	}
	if b.cacheAppend == nil {
		b.setError(errors.New("cache append plan is unavailable"))
		return nil
	}
	if b.err != nil {
		return nil
	}
	if left == nil || right == nil || left.Type != right.Type ||
		left.Shape.Rank == 0 || left.Shape.Rank != right.Shape.Rank ||
		axis+1 != uint32(left.Shape.Rank) ||
		left.Shape.Dims[axis] != uint64(b.cacheAppend.SourceCapacityTokens) {
		b.setError(errors.New("cache append input shape is invalid"))
		return nil
	}
	for dimension := range left.Shape.Rank {
		if uint32(dimension) != axis && left.Shape.Dims[dimension] != right.Shape.Dims[dimension] {
			b.setError(errors.New("cache append non-token dimensions differ"))
			return nil
		}
	}
	added := right.Shape.Dims[axis]
	if added > math.MaxUint32 ||
		uint64(b.cacheAppend.ActiveTokens)+added > uint64(b.cacheAppend.CapacityTokens) {
		b.setError(errors.New("cache append exceeds capacity"))
		return nil
	}
	shape := left.Shape
	shape.Dims[axis] = uint64(b.cacheAppend.CapacityTokens)
	return b.add(
		"", left.Type, shape, OpCacheAppend, []*Tensor{left, right},
		CacheAppendAttributes{Axis: axis, Offset: b.cacheAppend.ActiveTokens},
	)
}

func (b *Builder) unary(op Op, input *Tensor, attrs Attributes) *Tensor {
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
	attrs Attributes,
) *Tensor {
	if err := ValidateOperationAttributes(op, attrs); err != nil {
		b.setError(err)
		return nil
	}
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
