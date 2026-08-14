package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
)

func launchNode(
	state *device.State,
	functions functionSet,
	blas *blasState,
	q8Input *q8InputState,
	node *tensor.Tensor,
	attributes tensor.Attributes,
	pointers launchPointerFrame,
	attributePointers devicePointerTable,
) error {
	descriptor, ok := tensor.DescribeOperation(node.Op)
	if !ok || descriptor.Backends&tensor.BackendCUDA == 0 {
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
	switch node.Op {
	case tensor.OpHyperConnectionInit, tensor.OpHyperConnectionPre, tensor.OpHyperConnectionPost,
		tensor.OpHyperConnectionHead, tensor.OpCompressedAttention:
		return launchReferenceFamily(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpLoRAMerge, tensor.OpAdd, tensor.OpMultiply, tensor.OpDivide, tensor.OpScale,
		tensor.OpClamp, tensor.OpBF16Round, tensor.OpSiLU, tensor.OpGELU, tensor.OpGELUErf,
		tensor.OpReLU, tensor.OpReLUSquared, tensor.OpSigmoid, tensor.OpSoftplus, tensor.OpTanh,
		tensor.OpExp, tensor.OpXIELU, tensor.OpConv1DSame, tensor.OpConv2D,
		tensor.OpWindowPartition2D, tensor.OpWindowUnpartition2D, tensor.OpSAMAttention,
		tensor.OpGroupNorm, tensor.OpL2Norm:
		return launchMathVision(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpSSMConv, tensor.OpSSMScan, tensor.OpGatedDeltaNet,
		tensor.OpGatedLinearAttention, tensor.OpWKV6, tensor.OpSumRows, tensor.OpFWHT,
		tensor.OpTopK, tensor.OpTopKPairs, tensor.OpTopKPartials, tensor.OpGatherLast, tensor.OpSparseAttention, tensor.OpIndexerScore,
		tensor.OpWKV7:
		return launchRecurrentSelection(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpMoE:
		return launchMoE(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpRepeatHeads, tensor.OpTranspose2D, tensor.OpGroupSlice, tensor.OpFlatSlice,
		tensor.OpRMSNorm, tensor.OpMADNorm, tensor.OpLayerNorm, tensor.OpSoftmax, tensor.OpMulMat,
		tensor.OpGroupedMulMat, tensor.OpGetRows:
		return launchLinearLayout(
			state, functions, blas, q8Input, node, attributes, pointers, attributePointers,
		)
	case tensor.OpRoPENeoX, tensor.OpRoPENormal, tensor.OpRoPEMulti:
		return launchRoPE(state, functions, blas, node, attributes, pointers, attributePointers)
	case tensor.OpReshape, tensor.OpAttention, tensor.OpConcat, tensor.OpCacheAppend:
		return launchAttentionLayout(
			state, functions, blas, node, attributes, pointers, attributePointers,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
