package tensor

import (
	"errors"
	"fmt"
	"math"
	"slices"

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

// OperationDescriptor: single operation identity and backend contract.
type OperationDescriptor struct {
	Op       Op
	Name     string
	Class    OperationClass
	Backends OperationBackend
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
	{OpInput, "input", OperationInput, 0},
	{OpAdd, "add", OperationElementwise, allExecutionBackends},
	{OpMultiply, "multiply", OperationElementwise, allExecutionBackends},
	{OpScale, "scale", OperationElementwise, allExecutionBackends},
	{OpRMSNorm, "rms_norm", OperationElementwise, allExecutionBackends},
	{OpSoftmax, "softmax", OperationElementwise, allExecutionBackends},
	{OpSiLU, "silu", OperationElementwise, allExecutionBackends},
	{OpMulMat, "mul_mat", OperationLinear, allExecutionBackends},
	{OpGetRows, "get_rows", OperationLinear, allExecutionBackends},
	{OpRoPENeoX, "rope_neox", OperationPosition, allExecutionBackends},
	{OpReshape, "reshape", OperationLayout, allExecutionBackends},
	{OpAttention, "attention", OperationAttention, allExecutionBackends},
	{OpConcat, "concat", OperationLayout, allExecutionBackends},
	{OpRoPENormal, "rope_normal", OperationPosition, allExecutionBackends},
	{OpSigmoid, "sigmoid", OperationElementwise, allExecutionBackends},
	{OpSoftplus, "softplus", OperationElementwise, allExecutionBackends},
	{OpL2Norm, "l2_norm", OperationElementwise, allExecutionBackends},
	{OpSSMConv, "ssm_conv", OperationState, allExecutionBackends},
	{OpSSMScan, "ssm_scan", OperationState, allExecutionBackends},
	{OpGatedDeltaNet, "gated_delta_net", OperationState, allExecutionBackends},
	{OpTranspose2D, "transpose_2d", OperationLayout, allExecutionBackends},
	{OpGroupSlice, "group_slice", OperationLayout, allExecutionBackends},
	{OpFlatSlice, "flat_slice", OperationLayout, allExecutionBackends},
	{OpRoPEMulti, "rope_multi", OperationPosition, allExecutionBackends},
	{OpGELU, "gelu", OperationElementwise, allExecutionBackends},
	{OpLayerNorm, "layer_norm", OperationElementwise, allExecutionBackends},
	{OpReLUSquared, "relu_squared", OperationElementwise, allExecutionBackends},
	{OpXIELU, "xielu", OperationElementwise, allExecutionBackends},
	{OpMoE, "moe", OperationLinear, allExecutionBackends},
	{OpRepeatHeads, "repeat_heads", OperationLayout, allExecutionBackends},
	{OpClamp, "clamp", OperationElementwise, allExecutionBackends},
	{OpGroupedMulMat, "grouped_mul_mat", OperationLinear, allExecutionBackends},
	{OpTanh, "tanh", OperationElementwise, allExecutionBackends},
	{OpExp, "exp", OperationElementwise, allExecutionBackends},
	{OpGatedLinearAttention, "gated_linear_attention", OperationState, allExecutionBackends},
	{OpWKV6, "wkv6", OperationState, allExecutionBackends},
	{OpSumRows, "sum_rows", OperationLinear, allExecutionBackends},
	{OpWKV7, "wkv7", OperationState, allExecutionBackends},
	{OpFWHT, "fwht", OperationLinear, allExecutionBackends},
	{OpTopK, "top_k", OperationLinear, allExecutionBackends},
	{OpGatherLast, "gather_last", OperationLayout, allExecutionBackends},
	{OpSparseAttention, "sparse_attention", OperationAttention, allExecutionBackends},
	{OpIndexerScore, "indexer_score", OperationAttention, allExecutionBackends},
	{OpReLU, "relu", OperationElementwise, allExecutionBackends},
	{OpConv1DSame, "conv_1d_same", OperationLinear, allExecutionBackends},
	{OpGroupNorm, "group_norm", OperationElementwise, allExecutionBackends},
	{OpHyperConnectionInit, "hyper_connection_init", OperationState, allExecutionBackends},
	{OpHyperConnectionPre, "hyper_connection_pre", OperationState, allExecutionBackends},
	{OpHyperConnectionPost, "hyper_connection_post", OperationState, allExecutionBackends},
	{OpHyperConnectionHead, "hyper_connection_head", OperationState, allExecutionBackends},
	{OpCompressedAttention, "compressed_attention", OperationAttention, allExecutionBackends},
	{OpLoRAMerge, "lora_merge", OperationLinear, allExecutionBackends},
	{OpDivide, "divide", OperationElementwise, allExecutionBackends},
	{OpBF16Round, "bf16_round", OperationElementwise, allExecutionBackends},
	{OpGELUErf, "gelu_erf", OperationElementwise, allExecutionBackends},
	{OpConv2D, "conv_2d", OperationLinear, allExecutionBackends},
	{OpWindowPartition2D, "window_partition_2d", OperationLayout, allExecutionBackends},
	{OpWindowUnpartition2D, "window_unpartition_2d", OperationLayout, allExecutionBackends},
	{OpSAMAttention, "sam_attention", OperationAttention, allExecutionBackends},
	{OpTopKPairs, "top_k_pairs", OperationLinear, allExecutionBackends},
	{OpTopKPartials, "top_k_partials", OperationLinear, allExecutionBackends},
	{OpCacheAppend, "cache_append", OperationLayout, allExecutionBackends},
	{OpMADNorm, "mad_norm", OperationElementwise, allExecutionBackends},
}

// DescribeOperation: typed operation lookup.
func DescribeOperation(op Op) (OperationDescriptor, bool) {
	if int(op) >= len(operationDescriptors) {
		return OperationDescriptor{}, false
	}
	descriptor := operationDescriptors[op]
	return descriptor, descriptor.Op == op && descriptor.Name != ""
}

// Operations: ordered immutable operation catalog copy.
func Operations() []OperationDescriptor {
	return slices.Clone(operationDescriptors[:])
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
