package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

func launchReferenceFamily(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	switch node.Op {
	case tensor.OpDeepSeek4HCInit, tensor.OpDeepSeek4HCPre, tensor.OpDeepSeek4HCPost,
		tensor.OpDeepSeek4HCHead, tensor.OpDeepSeek4Attention:
		return launchReferenceNode(state, node, pointers)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
