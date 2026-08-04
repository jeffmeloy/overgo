package executor

import (
	"fmt"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/tensor"
)

func launchNode(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	descriptor, ok := tensor.DescribeOperation(node.Op)
	if !ok || descriptor.Backends&tensor.BackendCUDA == 0 {
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
	switch node.Op {
	case tensor.OpDeepSeek4HCInit, tensor.OpDeepSeek4HCPre, tensor.OpDeepSeek4HCPost,
		tensor.OpDeepSeek4HCHead, tensor.OpDeepSeek4Attention:
		return launchReferenceFamily(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpLoRAMerge, tensor.OpAdd, tensor.OpMultiply, tensor.OpDivide, tensor.OpScale,
		tensor.OpClamp, tensor.OpBF16Round, tensor.OpSiLU, tensor.OpGELU, tensor.OpGELUErf,
		tensor.OpReLU, tensor.OpReLUSquared, tensor.OpSigmoid, tensor.OpSoftplus, tensor.OpTanh,
		tensor.OpExp, tensor.OpXIELU, tensor.OpConv1DSame, tensor.OpConv2D,
		tensor.OpWindowPartition2D, tensor.OpWindowUnpartition2D, tensor.OpSAMAttention,
		tensor.OpGroupNorm, tensor.OpL2Norm:
		return launchMathVision(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpSSMConv, tensor.OpSSMScan, tensor.OpGatedDeltaNet,
		tensor.OpGatedLinearAttention, tensor.OpRWKV6, tensor.OpSumRows, tensor.OpFWHT,
		tensor.OpTopK, tensor.OpGatherLast, tensor.OpSparseAttention, tensor.OpIndexerScore,
		tensor.OpRWKV7:
		return launchRecurrentSelection(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpMoE:
		return launchMoE(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpRepeatHeads, tensor.OpTranspose2D, tensor.OpGroupSlice, tensor.OpFlatSlice,
		tensor.OpRMSNorm, tensor.OpLayerNorm, tensor.OpSoftmax, tensor.OpMulMat,
		tensor.OpGroupedMulMat, tensor.OpGetRows:
		return launchLinearLayout(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpRoPENeoX, tensor.OpRoPENormal, tensor.OpRoPEMulti:
		return launchRoPE(state, functions, blas, node, pointers, attributePointers)
	case tensor.OpReshape, tensor.OpAttention, tensor.OpConcat:
		return launchAttentionLayout(state, functions, blas, node, pointers, attributePointers)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
