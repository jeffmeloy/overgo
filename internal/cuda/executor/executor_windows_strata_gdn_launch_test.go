//go:build windows

package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's gated-delta verify launches (kernels/cuda/strata_gdn.cuh, from verify_kernels.cu at Strata's geometry) that
// the parity program calls and no served step does yet; the window step serves them.

// The recurrence's geometry the kernels are compiled for: a head's 128 state columns, 4 row groups a block (RG), the
// history commit's 256 threads a block and the state commit's 32 columns a block.
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
	qkHeads := 2 * keyHeads
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
