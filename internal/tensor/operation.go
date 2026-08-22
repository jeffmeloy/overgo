package tensor

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor/dtype"
)

// OperationClass: planner and diagnostic grouping.
type OperationClass uint8

const (
	OperationInput OperationClass = iota
	OperationElementwise
	OperationLinear
	OperationLayout
	OperationPosition
	OperationAttention
	OperationState
)

// OperationBackend: declared execution coverage.
type OperationBackend uint8

const (
	BackendReference OperationBackend = 1 << iota
	BackendCUDA
)

// CUDAProgram is the compiled CUDA launcher family for one operation.
type CUDAProgram uint8

const (
	// CUDAProgramNone rejects CUDA dispatch.
	CUDAProgramNone CUDAProgram = iota
	// CUDAProgramReference selects reference launchers.
	CUDAProgramReference
	// CUDAProgramMathVision selects math and vision launchers.
	CUDAProgramMathVision
	// CUDAProgramRecurrentSelection selects recurrent launchers.
	CUDAProgramRecurrentSelection
	// CUDAProgramMoE selects mixture-of-experts launchers.
	CUDAProgramMoE
	// CUDAProgramLinearLayout selects linear and layout launchers.
	CUDAProgramLinearLayout
	// CUDAProgramRoPE selects rotary-position launchers.
	CUDAProgramRoPE
	// CUDAProgramAttentionLayout selects attention launchers.
	CUDAProgramAttentionLayout
)

// OperationDescriptor: single operation identity and backend contract.
type OperationDescriptor struct {
	Op       Op
	Name     string
	Class    OperationClass
	Backends OperationBackend
	CUDA     CUDAProgram
}

// OperationStorage: output storage relationship.
type OperationStorage uint8

const (
	StorageDistinct OperationStorage = iota
	StorageViewWhole
	StorageViewFlat
)

// StorageView: resolved input-backed output range.
type StorageView struct {
	Input         int
	ElementOffset uint64
}

// OutputAliasContract: one permitted in-place target relationship.
type OutputAliasContract struct {
	Input            int
	InitializedBytes uint64
	WriteOffsetBytes uint64
	WriteBytes       uint64
}

// OutputTargetContract: compiled retained-target storage requirements.
type OutputTargetContract struct {
	Bytes     uint64
	Alignment uint64
	Alias     *OutputAliasContract
}

var operationStorage = [...]OperationStorage{
	OpReshape:   StorageViewWhole,
	OpFlatSlice: StorageViewFlat,
}

const allExecutionBackends = BackendReference | BackendCUDA

var operationDescriptors = [...]OperationDescriptor{
	{OpInput, "input", OperationInput, 0, CUDAProgramNone},
	{OpAdd, "add", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpMultiply, "multiply", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpScale, "scale", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpRMSNorm, "rms_norm", OperationElementwise, allExecutionBackends, CUDAProgramLinearLayout},
	{OpSoftmax, "softmax", OperationElementwise, allExecutionBackends, CUDAProgramLinearLayout},
	{OpSiLU, "silu", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpMulMat, "mul_mat", OperationLinear, allExecutionBackends, CUDAProgramLinearLayout},
	{OpGetRows, "get_rows", OperationLinear, allExecutionBackends, CUDAProgramLinearLayout},
	{OpRoPENeoX, "rope_neox", OperationPosition, allExecutionBackends, CUDAProgramRoPE},
	{OpReshape, "reshape", OperationLayout, allExecutionBackends, CUDAProgramAttentionLayout},
	{OpAttention, "attention", OperationAttention, allExecutionBackends, CUDAProgramAttentionLayout},
	{OpConcat, "concat", OperationLayout, allExecutionBackends, CUDAProgramAttentionLayout},
	{OpRoPENormal, "rope_normal", OperationPosition, allExecutionBackends, CUDAProgramRoPE},
	{OpSigmoid, "sigmoid", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpSoftplus, "softplus", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpL2Norm, "l2_norm", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpSSMConv, "ssm_conv", OperationState, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpSSMScan, "ssm_scan", OperationState, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpGatedDeltaNet, "gated_delta_net", OperationState, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpTranspose2D, "transpose_2d", OperationLayout, allExecutionBackends, CUDAProgramLinearLayout},
	{OpGroupSlice, "group_slice", OperationLayout, allExecutionBackends, CUDAProgramLinearLayout},
	{OpFlatSlice, "flat_slice", OperationLayout, allExecutionBackends, CUDAProgramLinearLayout},
	{OpRoPEMulti, "rope_multi", OperationPosition, allExecutionBackends, CUDAProgramRoPE},
	{OpGELU, "gelu", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpLayerNorm, "layer_norm", OperationElementwise, allExecutionBackends, CUDAProgramLinearLayout},
	{OpReLUSquared, "relu_squared", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpXIELU, "xielu", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpMoE, "moe", OperationLinear, allExecutionBackends, CUDAProgramMoE},
	{OpRepeatHeads, "repeat_heads", OperationLayout, allExecutionBackends, CUDAProgramLinearLayout},
	{OpClamp, "clamp", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpGroupedMulMat, "grouped_mul_mat", OperationLinear, allExecutionBackends, CUDAProgramLinearLayout},
	{OpTanh, "tanh", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpExp, "exp", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpGatedLinearAttention, "gated_linear_attention", OperationState, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpWKV6, "wkv6", OperationState, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpSumRows, "sum_rows", OperationLinear, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpWKV7, "wkv7", OperationState, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpFWHT, "fwht", OperationLinear, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpTopK, "top_k", OperationLinear, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpGatherLast, "gather_last", OperationLayout, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpSparseAttention, "sparse_attention", OperationAttention, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpIndexerScore, "indexer_score", OperationAttention, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpReLU, "relu", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpConv1DSame, "conv_1d_same", OperationLinear, allExecutionBackends, CUDAProgramMathVision},
	{OpGroupNorm, "group_norm", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpHyperConnectionInit, "hyper_connection_init", OperationState, allExecutionBackends, CUDAProgramReference},
	{OpHyperConnectionPre, "hyper_connection_pre", OperationState, allExecutionBackends, CUDAProgramReference},
	{OpHyperConnectionPost, "hyper_connection_post", OperationState, allExecutionBackends, CUDAProgramReference},
	{OpHyperConnectionHead, "hyper_connection_head", OperationState, allExecutionBackends, CUDAProgramReference},
	{OpCompressedAttention, "compressed_attention", OperationAttention, allExecutionBackends, CUDAProgramReference},
	{OpLoRAMerge, "lora_merge", OperationLinear, allExecutionBackends, CUDAProgramMathVision},
	{OpDivide, "divide", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpBF16Round, "bf16_round", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpGELUErf, "gelu_erf", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpConv2D, "conv_2d", OperationLinear, allExecutionBackends, CUDAProgramMathVision},
	{OpWindowPartition2D, "window_partition_2d", OperationLayout, allExecutionBackends, CUDAProgramMathVision},
	{OpWindowUnpartition2D, "window_unpartition_2d", OperationLayout, allExecutionBackends, CUDAProgramMathVision},
	{OpSAMAttention, "sam_attention", OperationAttention, allExecutionBackends, CUDAProgramMathVision},
	{OpTopKPairs, "top_k_pairs", OperationLinear, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpTopKPartials, "top_k_partials", OperationLinear, allExecutionBackends, CUDAProgramRecurrentSelection},
	{OpCacheAppend, "cache_append", OperationLayout, allExecutionBackends, CUDAProgramAttentionLayout},
	{OpMADNorm, "mad_norm", OperationElementwise, allExecutionBackends, CUDAProgramLinearLayout},
	{OpAtan, "atan", OperationElementwise, allExecutionBackends, CUDAProgramMathVision},
	{OpPixelShuffle2D, "pixel_shuffle_2d", OperationLayout, allExecutionBackends, CUDAProgramMathVision},
}

// DescribeOperation: typed operation lookup.
func DescribeOperation(op Op) (OperationDescriptor, bool) {
	if int(op) >= len(operationDescriptors) {
		return OperationDescriptor{}, false
	}
	descriptor := operationDescriptors[op]
	return descriptor, descriptor.Op == op && descriptor.Name != ""
}

// DescribeOperationStorage: authoritative output-storage contract.
func DescribeOperationStorage(op Op) OperationStorage {
	if int(op) >= len(operationStorage) {
		return StorageDistinct
	}
	return operationStorage[op]
}

// ResolveStorageView: validates and resolves an operation's input-backed range.
func ResolveStorageView(node *Tensor) (StorageView, bool, error) {
	if node == nil {
		return StorageView{}, false, errors.New("storage view tensor is nil")
	}
	storage := DescribeOperationStorage(node.Op)
	if storage == StorageDistinct {
		return StorageView{}, false, nil
	}
	if len(node.Inputs) != 1 || node.Inputs[0] == nil {
		return StorageView{}, false, fmt.Errorf("%s storage view requires one input", node.Op)
	}
	view := StorageView{Input: 0}
	if storage == StorageViewFlat {
		attributes, ok := node.Attrs.(FlatSliceAttributes)
		if !ok {
			return StorageView{}, false, errors.New("flat_slice storage attributes are invalid")
		}
		view.ElementOffset = attributes.Offset
	}
	return view, true, nil
}

// ByteOffset: converts a view element offset into physical storage bytes.
func (v StorageView) ByteOffset(storage dtype.Type) (uint64, error) {
	traits, ok := storage.Traits()
	if !ok || traits.BlockSize == 0 || v.ElementOffset%traits.BlockSize != 0 {
		return 0, fmt.Errorf("%s view offset %d is not storage aligned", storage, v.ElementOffset)
	}
	blocks := v.ElementOffset / traits.BlockSize
	if blocks > math.MaxUint64/traits.TypeSize {
		return 0, errors.New("storage view byte offset overflows")
	}
	return blocks * traits.TypeSize, nil
}

// CompileOutputTargetContract: validates retained-output extent and legal overlap.
func CompileOutputTargetContract(node *Tensor) (OutputTargetContract, error) {
	if node == nil {
		return OutputTargetContract{}, errors.New("output target tensor is nil")
	}
	bytes, err := node.Shape.Bytes(node.Type)
	if err != nil {
		return OutputTargetContract{}, err
	}
	traits, ok := node.Type.Traits()
	if !ok || traits.TypeSize == 0 {
		return OutputTargetContract{}, fmt.Errorf("output target type %s is unsupported", node.Type)
	}
	alignment := traits.TypeSize
	if traits.BlockSize != 1 || alignment&(alignment-1) != 0 {
		alignment = 1
	}
	contract := OutputTargetContract{Bytes: bytes, Alignment: alignment}
	if node.Op == OpCacheAppend {
		if len(node.Inputs) != 2 {
			return OutputTargetContract{}, errors.New("cache append input contract is invalid")
		}
		attributes, ok := node.Attrs.(CacheAppendAttributes)
		if !ok || attributes.Axis+1 != uint32(node.Shape.Rank) {
			return OutputTargetContract{}, errors.New("cache append attributes are invalid")
		}
		inputBytes, sizeErr := node.Inputs[0].Shape.Bytes(node.Inputs[0].Type)
		writeBytes, writeErr := node.Inputs[1].Shape.Bytes(node.Inputs[1].Type)
		inner := uint64(1)
		for axis := uint32(0); axis < attributes.Axis; axis++ {
			if inner > math.MaxUint64/node.Shape.Dims[axis] {
				return OutputTargetContract{}, errors.New("cache append offset overflows")
			}
			inner *= node.Shape.Dims[axis]
		}
		if inner != 0 && uint64(attributes.Offset) > math.MaxUint64/inner {
			return OutputTargetContract{}, errors.New("cache append offset overflows")
		}
		writeOffsetElements := uint64(attributes.Offset) * inner
		view := StorageView{ElementOffset: writeOffsetElements}
		writeOffset, offsetErr := view.ByteOffset(node.Type)
		if sizeErr != nil || writeErr != nil || offsetErr != nil || inputBytes > bytes ||
			writeOffset > bytes || writeBytes > bytes-writeOffset {
			return OutputTargetContract{}, errors.New("cache append output target extent is invalid")
		}
		contract.Alias = &OutputAliasContract{
			Input: 0, InitializedBytes: inputBytes,
			WriteOffsetBytes: writeOffset, WriteBytes: writeBytes,
		}
		return contract, nil
	}
	if node.Op != OpConcat || len(node.Inputs) != 2 {
		return contract, nil
	}
	attributes, ok := node.Attrs.(ConcatAttributes)
	if !ok || attributes.Axis+1 != uint32(node.Shape.Rank) {
		return contract, nil
	}
	leftBytes, err := node.Inputs[0].Shape.Bytes(node.Inputs[0].Type)
	if err != nil {
		return OutputTargetContract{}, err
	}
	rightBytes, err := node.Inputs[1].Shape.Bytes(node.Inputs[1].Type)
	if err != nil || leftBytes > bytes || rightBytes != bytes-leftBytes {
		return OutputTargetContract{}, errors.New("concat output target extent is invalid")
	}
	contract.Alias = &OutputAliasContract{
		Input: 0, InitializedBytes: leftBytes,
		WriteOffsetBytes: leftBytes, WriteBytes: rightBytes,
	}
	return contract, nil
}

func (o Op) String() string {
	if descriptor, ok := DescribeOperation(o); ok {
		return descriptor.Name
	}
	return fmt.Sprintf("op_%d", o)
}
