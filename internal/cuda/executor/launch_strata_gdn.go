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

// strataGatedDeltaCommitLayers is the most layers one launch of the layers' commit kernels takes, their tables' entries
// (strata_gdn.cuh STRATA_GDN_COMMIT_LAYERS).
const strataGatedDeltaCommitLayers = 64

// strataGatedDeltaConvCommitTable is gdn_conv_commit_layers' table, by value: each layer's history and qkv rows.
type strataGatedDeltaConvCommitTable struct {
	history, qkv [strataGatedDeltaCommitLayers]driver.DevicePtr
}

// strataGatedDeltaStepCommitTable is gdn_step_commit_layers' table, by value: each layer's state, convolved rows, gate
// and beta.
type strataGatedDeltaStepCommitTable struct {
	state, rows, gate, beta [strataGatedDeltaCommitLayers]driver.DevicePtr
}

// launchStrataGatedDeltaCommitLayers launches gdn_conv_commit_layers and gdn_step_commit_layers over the kept count,
// in place, for histories, states and windows a layer each, a launch of each for every strataGatedDeltaCommitLayers.
func launchStrataGatedDeltaCommitLayers(state *device.State, functions functionSet, histories, states []driver.DevicePtr,
	kept driver.DevicePtr, windows []strataGatedDeltaWindow, channels, keyHeads, valueHeads int32) error {
	for first := 0; first < len(windows); first += strataGatedDeltaCommitLayers {
		end := min(first+strataGatedDeltaCommitLayers, len(windows))
		var conv strataGatedDeltaConvCommitTable
		var step strataGatedDeltaStepCommitTable
		for layer := first; layer < end; layer++ {
			at := layer - first
			conv.history[at], conv.qkv[at] = histories[layer], windows[layer].qkv
			step.state[at], step.rows[at], step.gate[at], step.beta[at] = states[layer], windows[layer].rows, windows[layer].gate, windows[layer].beta
		}
		layers := uint32(end - first)
		if err := launchGridABI(state, functions[kernelStrataGdnConvCommitLayers],
			driver.Dim3{X: (uint32(channels) + strataGatedDeltaCommitThreads - 1) / strataGatedDeltaCommitThreads, Y: layers, Z: 1},
			driver.Dim3{X: strataGatedDeltaCommitThreads, Y: 1, Z: 1},
			&conv, &channels, &kept); err != nil {
			return err
		}
		if err := launchGridABI(state, functions[kernelStrataGdnStepCommitLayers],
			driver.Dim3{X: uint32(valueHeads), Y: strataGatedDeltaWidth / strataGatedDeltaCommitColumns, Z: layers},
			driver.Dim3{X: strataGatedDeltaCommitColumns, Y: strataGatedDeltaRowGroups, Z: 1},
			&step, &channels, &keyHeads, &valueHeads, &kept); err != nil {
			return err
		}
	}
	return nil
}

// launchGatedDeltaCommit launches a gated-delta commit (tensor.Builder.GatedDeltaCommitLayers) - a layer's by its own
// kernels, several layers' by the layers' kernels - and copies its kept count's word to the output.
func launchGatedDeltaCommit(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	attributes := node.Attrs.(tensor.GatedDeltaWindowAttributes)
	layers := attributes.CommitLayers()
	channels := int32(node.Inputs[tensor.GatedDeltaCommitRows].Shape.Dims[0])
	histories, states := make([]driver.DevicePtr, layers), make([]driver.DevicePtr, layers)
	windows := make([]strataGatedDeltaWindow, layers)
	for layer := range layers {
		at := layer * tensor.GatedDeltaCommitKept
		histories[layer], states[layer] = pointers.input(at+tensor.GatedDeltaCommitHistory), pointers.input(at+tensor.GatedDeltaCommitState)
		windows[layer] = strataGatedDeltaWindow{
			qkv: pointers.input(at + tensor.GatedDeltaCommitQKV), rows: pointers.input(at + tensor.GatedDeltaCommitRows),
			gate: pointers.input(at + tensor.GatedDeltaCommitGate), beta: pointers.input(at + tensor.GatedDeltaCommitBeta),
		}
	}
	kept, output := pointers.input(layers*tensor.GatedDeltaCommitKept), pointers.output()
	var err error
	if layers == tensor.SingletonExtent {
		err = launchStrataGatedDeltaCommit(state, functions, histories[0], states[0], kept, windows[0], channels,
			int32(attributes.KeyHeads), int32(attributes.ValueHeads))
	} else {
		err = launchStrataGatedDeltaCommitLayers(state, functions, histories, states, kept, windows, channels,
			int32(attributes.KeyHeads), int32(attributes.ValueHeads))
	}
	if err != nil {
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
