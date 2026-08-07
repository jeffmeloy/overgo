package tensor

import (
	"fmt"
	"slices"
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
	{OpRWKV6, "rwkv6", OperationState, allExecutionBackends},
	{OpSumRows, "sum_rows", OperationLinear, allExecutionBackends},
	{OpRWKV7, "rwkv7", OperationState, allExecutionBackends},
	{OpFWHT, "fwht", OperationLinear, allExecutionBackends},
	{OpTopK, "top_k", OperationLinear, allExecutionBackends},
	{OpGatherLast, "gather_last", OperationLayout, allExecutionBackends},
	{OpSparseAttention, "sparse_attention", OperationAttention, allExecutionBackends},
	{OpIndexerScore, "indexer_score", OperationAttention, allExecutionBackends},
	{OpReLU, "relu", OperationElementwise, allExecutionBackends},
	{OpConv1DSame, "conv_1d_same", OperationLinear, allExecutionBackends},
	{OpGroupNorm, "group_norm", OperationElementwise, allExecutionBackends},
	{OpDeepSeek4HCInit, "deepseek4_hc_init", OperationState, allExecutionBackends},
	{OpDeepSeek4HCPre, "deepseek4_hc_pre", OperationState, allExecutionBackends},
	{OpDeepSeek4HCPost, "deepseek4_hc_post", OperationState, allExecutionBackends},
	{OpDeepSeek4HCHead, "deepseek4_hc_head", OperationState, allExecutionBackends},
	{OpDeepSeek4Attention, "deepseek4_attention", OperationAttention, allExecutionBackends},
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

func (o Op) String() string {
	if descriptor, ok := DescribeOperation(o); ok {
		return descriptor.Name
	}
	return fmt.Sprintf("op_%d", o)
}
