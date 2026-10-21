package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// launchGatedDeltaConv runs gated_delta_conv_f32: a block a head of state-width channels, which
// tensor.GatedDeltaFused must cover.
func launchGatedDeltaConv(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	attributes := node.Attrs.(tensor.GatedDeltaConvAttributes)
	qkv, weights := node.Inputs[tensor.GatedDeltaConvQKV], node.Inputs[tensor.GatedDeltaConvTaps]
	layout := tensor.GatedDeltaConvLayout{
		StateWidth: uint64(attributes.StateWidth), KeyHeads: uint64(attributes.KeyHeads), ValueHeads: uint64(attributes.ValueHeads),
		Kernel: weights.Shape.Dims[0], Tokens: qkv.Shape.Dims[1],
	}
	if !tensor.GatedDeltaFused(layout.StateWidth, layout.Kernel, layout.Tokens, weights.Type, dtype.F32, dtype.F32) {
		return errors.New("gated-delta convolution is outside its kernel")
	}
	channels, kernelTaps, tokens := uint32(layout.Channels()), uint32(layout.Kernel), uint32(layout.Tokens)
	history, input := pointers.input(tensor.GatedDeltaConvState), pointers.input(tensor.GatedDeltaConvQKV)
	taps, output := pointers.input(tensor.GatedDeltaConvTaps), pointers.output()
	return launchGridABI(
		state, functions[kernelGatedDeltaConvF32],
		kernel.Grid1D(int(channels/attributes.StateWidth)), kernel.Grid1D(int(attributes.StateWidth)),
		&history, &input, &taps, &output, &channels, &attributes.StateWidth, &attributes.KeyHeads, &attributes.ValueHeads,
		&kernelTaps, &tokens, &attributes.Epsilon,
	)
}

// launchGatedDeltaGates runs gated_delta_gates_f32: a block an alpha row, then a beta row, of mul_mat_f32's threads,
// so each product sums as mul_mat_f32 sums it.
func launchGatedDeltaGates(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	input, alpha := node.Inputs[tensor.GatedDeltaGatesInput], node.Inputs[tensor.GatedDeltaGatesAlpha]
	embedding, heads, tokens := uint32(input.Shape.Dims[0]), uint32(alpha.Shape.Dims[1]), uint32(input.Shape.Dims[1])
	if tokens > tensor.GatedDeltaFusedTokens {
		return errors.New("gated-delta gates are outside their kernel")
	}
	values, alphaWeights := pointers.input(tensor.GatedDeltaGatesInput), pointers.input(tensor.GatedDeltaGatesAlpha)
	betaWeights, timeStep := pointers.input(tensor.GatedDeltaGatesBeta), pointers.input(tensor.GatedDeltaGatesTimeStep)
	decay, output := pointers.input(tensor.GatedDeltaGatesDecay), pointers.output()
	// A block an alpha row, then a beta row.
	return launchGridABI(
		state, functions[kernelGatedDeltaGatesF32], kernel.Grid1D(int(tensor.PairedExtent*heads)), kernel.DefaultBlock1D(),
		&values, &alphaWeights, &betaWeights, &timeStep, &decay, &output, &embedding, &heads, &tokens,
	)
}
