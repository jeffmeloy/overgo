package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// Strata's gated-delta verify launches (kernels/cuda/strata_gdn.cuh, ported from Strata at 296b59d2, MIT:
// src/kernels/cuda/verify_kernels.cu), at the geometry the kernels are compiled for.

// The kernels' compiled geometry: a head's 128 state columns, the recurrence's 4 row groups a block (RG), the history
// commit's 256 threads a block and the state commit's 32 columns a block, each fixed by the kernel's own bounds.
const (
	strataGatedDeltaWidth         = 128
	strataGatedDeltaRowGroups     = 4
	strataGatedDeltaCommitThreads = 256
	strataGatedDeltaCommitColumns = 32
)

// strataGatedDeltaWindow is a window's buffers: its qkv rows and convolved rows [token][channel], its gate and beta
// [token][value head], its z and outputs [token][value head][128].
type strataGatedDeltaWindow struct {
	qkv, rows, gate, beta, z, outputs driver.DevicePtr
}

// launchStrataGatedDeltaConv launches gdn_conv_l2_multi over tokens from first, the history kept when commit.
func launchStrataGatedDeltaConv(state *device.State, functions functionSet, history, weights driver.DevicePtr,
	window strataGatedDeltaWindow, channels, keyHeads int32, epsilon float32, first, tokens int32, commit bool) error {
	qkHeads := tensor.PairedExtent * keyHeads
	var keep int32
	if commit {
		keep = 1
	}
	return launchGridABI(state, functions[kernelStrataGdnConvL2Multi],
		driver.Dim3{X: uint32(channels) / strataGatedDeltaWidth, Y: uint32(tokens), Z: 1},
		driver.Dim3{X: strataGatedDeltaWidth, Y: 1, Z: 1},
		&history, &window.qkv, &weights, &window.rows, &channels, &qkHeads, &epsilon, &first, &keep)
}

// launchStrataGatedDeltaStep launches gdn_step_norm_multi: its every-output form without a kept count, its kept form
// (outputs from outFirst, the state written after the kept count) with one.
func launchStrataGatedDeltaStep(state *device.State, functions functionSet, stateBuffer, gamma, kept driver.DevicePtr,
	window strataGatedDeltaWindow, channels, keyHeads, valueHeads int32, epsilon float32, tokens, outFirst int32) error {
	grid := driver.Dim3{X: uint32(valueHeads), Y: 1, Z: 1}
	block := driver.Dim3{X: strataGatedDeltaWidth, Y: strataGatedDeltaRowGroups, Z: 1}
	if kept == 0 {
		return launchGridABI(state, functions[kernelStrataGdnStepNormMultiAll], grid, block,
			&stateBuffer, &window.rows, &channels, &window.gate, &window.beta, &window.z, &gamma, &epsilon, &window.outputs,
			&keyHeads, &valueHeads, &tokens)
	}
	return launchGridABI(state, functions[kernelStrataGdnStepNormMultiKept], grid, block,
		&stateBuffer, &window.rows, &channels, &window.gate, &window.beta, &window.z, &gamma, &epsilon, &window.outputs,
		&keyHeads, &valueHeads, &tokens, &kept, &outFirst)
}

// launchStrataGatedDeltaCommit launches gdn_conv_commit and gdn_step_commit over the kept count, in place.
func launchStrataGatedDeltaCommit(state *device.State, functions functionSet, history, stateBuffer, kept driver.DevicePtr,
	window strataGatedDeltaWindow, channels, keyHeads, valueHeads int32) error {
	if err := launchGridABI(state, functions[kernelStrataGdnConvCommit],
		driver.Dim3{X: (uint32(channels) + strataGatedDeltaCommitThreads - 1) / strataGatedDeltaCommitThreads, Y: 1, Z: 1},
		driver.Dim3{X: strataGatedDeltaCommitThreads, Y: 1, Z: 1},
		&history, &window.qkv, &channels, &kept); err != nil {
		return err
	}
	return launchGridABI(state, functions[kernelStrataGdnStepCommit],
		driver.Dim3{X: uint32(valueHeads), Y: strataGatedDeltaWidth / strataGatedDeltaCommitColumns, Z: 1},
		driver.Dim3{X: strataGatedDeltaCommitColumns, Y: strataGatedDeltaRowGroups, Z: 1},
		&stateBuffer, &window.rows, &channels, &window.gate, &window.beta, &keyHeads, &valueHeads, &kept)
}

// launchGatedDeltaWindowConv launches a window's conv (tensor.Builder.GatedDeltaWindowConv), committing its one token
// when its attributes say so.
func launchGatedDeltaWindowConv(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	attributes := node.Attrs.(tensor.GatedDeltaWindowAttributes)
	channels, tokens := int32(node.Shape.Dims[0]), int32(node.Shape.Dims[1])
	window := strataGatedDeltaWindow{qkv: pointers.input(tensor.GatedDeltaWindowConvQKV), rows: pointers.output()}
	return launchStrataGatedDeltaConv(state, functions, pointers.input(tensor.GatedDeltaWindowConvHistory),
		pointers.input(tensor.GatedDeltaWindowConvWeights), window, channels, int32(attributes.KeyHeads), attributes.Epsilon,
		0, tokens, attributes.Commit)
}

// launchGatedDeltaWindowStep launches a window's recurrence (tensor.Builder.GatedDeltaWindowStep): every output from
// the state it leaves, or with a kept count the outputs and the state after the count.
func launchGatedDeltaWindowStep(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	attributes := node.Attrs.(tensor.GatedDeltaWindowAttributes)
	rows := node.Inputs[tensor.GatedDeltaWindowStepRows].Shape
	window := strataGatedDeltaWindow{
		rows: pointers.input(tensor.GatedDeltaWindowStepRows), gate: pointers.input(tensor.GatedDeltaWindowStepGate),
		beta: pointers.input(tensor.GatedDeltaWindowStepBeta), z: pointers.input(tensor.GatedDeltaWindowStepZ),
		outputs: pointers.output(),
	}
	var kept driver.DevicePtr
	if len(node.Inputs) > tensor.GatedDeltaWindowStepOperands {
		kept = pointers.input(tensor.GatedDeltaWindowStepOperands)
	}
	return launchStrataGatedDeltaStep(state, functions, pointers.input(tensor.GatedDeltaWindowStepState),
		pointers.input(tensor.GatedDeltaWindowStepGamma), kept, window, int32(rows.Dims[0]), int32(attributes.KeyHeads),
		int32(attributes.ValueHeads), attributes.Epsilon, int32(rows.Dims[1]), 0)
}

// launchGatedDeltaCommit launches a gated-delta layer's commit (tensor.Builder.GatedDeltaCommit) and copies its kept
// count's word to the output.
func launchGatedDeltaCommit(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	attributes := node.Attrs.(tensor.GatedDeltaWindowAttributes)
	window := strataGatedDeltaWindow{
		qkv: pointers.input(tensor.GatedDeltaCommitQKV), rows: pointers.input(tensor.GatedDeltaCommitRows),
		gate: pointers.input(tensor.GatedDeltaCommitGate), beta: pointers.input(tensor.GatedDeltaCommitBeta),
	}
	kept, output := pointers.input(tensor.GatedDeltaCommitKept), pointers.output()
	if err := launchStrataGatedDeltaCommit(state, functions, pointers.input(tensor.GatedDeltaCommitHistory),
		pointers.input(tensor.GatedDeltaCommitState), kept, window, int32(node.Inputs[tensor.GatedDeltaCommitRows].Shape.Dims[0]),
		int32(attributes.KeyHeads), int32(attributes.ValueHeads)); err != nil {
		return err
	}
	return launchCommitWord(state, functions, kept, output)
}

// strataCopyIndexedBlocks is copy_indexed's most blocks, its grid-stride loop covering the rest.
const strataCopyIndexedBlocks = 64

// launchCopyIndexed launches copy_indexed (tensor.Builder.CopyIndexed) over the target's values and copies its index's
// word to the output.
func launchCopyIndexed(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	attributes := node.Attrs.(tensor.CopyIndexedAttributes)
	values, err := node.Inputs[tensor.CopyIndexedTarget].Shape.Elements()
	if err != nil {
		return err
	}
	target, source := pointers.input(tensor.CopyIndexedTarget), pointers.input(tensor.CopyIndexedSource)
	index := pointers.input(tensor.CopyIndexedIndex)
	stride, count := int64(attributes.Stride), int64(values)
	blocks := min((uint32(values)+strataGatedDeltaCommitThreads-1)/strataGatedDeltaCommitThreads, strataCopyIndexedBlocks)
	if err := launchGridABI(state, functions[kernelStrataCopyIndexed],
		driver.Dim3{X: blocks, Y: 1, Z: 1}, driver.Dim3{X: strataGatedDeltaCommitThreads, Y: 1, Z: 1},
		&target, &source, &stride, &index, &count); err != nil {
		return err
	}
	return launchCommitWord(state, functions, index, pointers.output())
}

// launchCommitWord copies a commit's kept count, a word, to its output: a copy reads and writes it without arithmetic.
func launchCommitWord(state *device.State, functions functionSet, kept, output driver.DevicePtr) error {
	count := uint32(1)
	return launch1DABI(state, functions[kernelCopyF32], count, &kept, &output, &count)
}
