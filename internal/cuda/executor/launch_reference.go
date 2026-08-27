package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
)

func launchReferenceFamily(
	state *device.State,
	functions functionSet,
	blas *blasState,
	_ *q8InputState,
	node *tensor.Tensor,
	_ tensor.Attributes,
	pointers launchPointerFrame,
) error {
	switch node.Op {
	case tensor.OpHyperConnectionInit, tensor.OpHyperConnectionPre, tensor.OpHyperConnectionPost,
		tensor.OpHyperConnectionHead, tensor.OpCompressedAttention:
		return launchReferenceNode(state, node, pointers)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
