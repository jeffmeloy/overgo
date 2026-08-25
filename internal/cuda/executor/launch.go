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
	program tensor.CUDAProgram,
) error {
	switch program {
	case tensor.CUDAProgramReference:
		return launchReferenceFamily(state, functions, blas, node, pointers)
	case tensor.CUDAProgramMathVision:
		return launchMathVision(state, functions, blas, node, pointers)
	case tensor.CUDAProgramRecurrentSelection:
		return launchRecurrentSelection(state, functions, blas, node, pointers)
	case tensor.CUDAProgramMoE:
		return launchMoE(state, functions, blas, node, pointers)
	case tensor.CUDAProgramLinearLayout:
		return launchLinearLayout(state, functions, blas, q8Input, node, attributes, pointers)
	case tensor.CUDAProgramRoPE:
		return launchRoPE(state, functions, blas, node, attributes, pointers)
	case tensor.CUDAProgramAttentionLayout:
		return launchAttentionLayout(state, functions, blas, node, attributes, pointers)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
