package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
)

type nodeLauncher func(
	*device.State,
	functionSet,
	*blasState,
	*q8InputState,
	*tensor.Tensor,
	tensor.Attributes,
	launchPointerFrame,
) error

var nodeLaunchers = [...]nodeLauncher{
	tensor.CUDAProgramReference:          launchReferenceFamily,
	tensor.CUDAProgramMathVision:         launchMathVision,
	tensor.CUDAProgramRecurrentSelection: launchRecurrentSelection,
	tensor.CUDAProgramMoE:                launchMoE,
	tensor.CUDAProgramLinearLayout:       launchLinearLayout,
	tensor.CUDAProgramRoPE:               launchRoPE,
	tensor.CUDAProgramAttentionLayout:    launchAttentionLayout,
}
