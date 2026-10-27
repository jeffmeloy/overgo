package executor

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's lexical-injection (PLE) post-ops, weighted RMS norm and history advance
// (kernels/cuda/strata_native_ple.cuh and strata_round_elementwise.cuh, ported from Strata at 296b59d2, MIT:
// src/kernels/cuda/native_ple_postops.cu, native_gr_norm.cu and ple.cu): native_gr_rms_norm_weighted_multi,
// native_ple_postops and ple_history_advance, their host sides translated.

// The PLE's compiled geometry: Flash-Next's n_embd, its four streams and the post-ops' launches.
const (
	strataPLEEmbedding   = 2560 // N
	strataPLEStreams     = 4    // H
	strataPLEWidth       = strataPLEEmbedding * strataPLEStreams
	strataPLEGateThreads = 512  // gate_kernel's block: SUM_ROWS' 512 threads for a row
	strataPLEThreads     = 256  // broadcast, conv_residual and history_advance's block
	strataPLENormWide    = 1024 // weighted_rms_norm's block from 1,024 columns
	strataPLENormNarrow  = 256  // and below
	strataPLEEpsilon     = float32(1e-6)
)

// strataPLEWeights are the PLE post-ops' weights (PleWeights' norms and conv): the key, query and conv norms [H][N] and
// the conv taps [D][4] in fp16.
type strataPLEWeights struct {
	NormKey, NormQuery, NormConv, ConvF16 driver.DevicePtr
}

// strataPLEBuffers are NativePlePostopsBuffers: the normed key, the normed query, the gates [H], the gated value [D],
// the normed gated value, the conv's activations and the result (which may be hidden).
type strataPLEBuffers struct {
	Key, Query, Gate, Gated, Normalized, Conv, Result driver.DevicePtr
}

// launchStrataNativeWeightedRMSNorm is native_gr_rms_norm_weighted_multi: tokens groups of rows rows of cols values,
// each row normed by its root mean square plus epsilon and its own gamma row (gamma [rows][cols]).
func launchStrataNativeWeightedRMSNorm(state *device.State, functions functionSet, input, gamma, output driver.DevicePtr,
	cols, rows, tokens int32, epsilon float32) error {
	if cols <= 0 || rows <= 0 || tokens <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) || epsilon < 0 {
		return fmt.Errorf("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon")
	}
	kernel, block := kernelStrataNativeWeightedRmsNorm1024, uint32(strataPLENormWide)
	if cols < strataPLENormWide {
		kernel, block = kernelStrataNativeWeightedRmsNorm256, strataPLENormNarrow
	}
	return launchGridABI(state, functions[kernel], driver.Dim3{X: uint32(rows), Y: uint32(tokens), Z: 1},
		driver.Dim3{X: block, Y: 1, Z: 1}, &input, &gamma, &output, &cols, &epsilon)
}

// launchStrataNativePLEPostops is native_ple_postops: the projected key and the hidden rows normed, the gates from their
// streams' dot products, the value broadcast by them, normed, convolved with the history's taps and added to hidden
// with the gated value into the result.
func launchStrataNativePLEPostops(state *device.State, functions functionSet, projectedKey, hidden, value, history driver.DevicePtr,
	weights strataPLEWeights, buffers strataPLEBuffers) error {
	if err := launchStrataNativeWeightedRMSNorm(state, functions, projectedKey, weights.NormKey, buffers.Key,
		strataPLEEmbedding, strataPLEStreams, 1, strataPLEEpsilon); err != nil {
		return err
	}
	if err := launchStrataNativeWeightedRMSNorm(state, functions, hidden, weights.NormQuery, buffers.Query,
		strataPLEEmbedding, strataPLEStreams, 1, strataPLEEpsilon); err != nil {
		return err
	}
	scale := 1 / float32(math.Sqrt(strataPLEEmbedding))
	if err := launchGridABI(state, functions[kernelStrataNativePleGate], driver.Dim3{X: strataPLEStreams, Y: 1, Z: 1},
		driver.Dim3{X: strataPLEGateThreads, Y: 1, Z: 1}, &buffers.Key, &buffers.Query, &buffers.Gate, &scale); err != nil {
		return err
	}
	blocks := driver.Dim3{X: strataPLEWidth / strataPLEThreads, Y: 1, Z: 1}
	threads := driver.Dim3{X: strataPLEThreads, Y: 1, Z: 1}
	if err := launchGridABI(state, functions[kernelStrataNativePleBroadcast], blocks, threads,
		&value, &buffers.Gate, &buffers.Gated); err != nil {
		return err
	}
	if err := launchStrataNativeWeightedRMSNorm(state, functions, buffers.Gated, weights.NormConv, buffers.Normalized,
		strataPLEEmbedding, strataPLEStreams, 1, strataPLEEpsilon); err != nil {
		return err
	}
	return launchGridABI(state, functions[kernelStrataNativePleConvResidual], blocks, threads,
		&history, &buffers.Normalized, &weights.ConvF16, &hidden, &buffers.Gated, &buffers.Conv, &buffers.Result)
}

// launchStrataPLEHistoryAdvance is ple_history_advance: each channel's history shifted up a row, the normalized row last.
func launchStrataPLEHistoryAdvance(state *device.State, functions functionSet, history, normalized driver.DevicePtr) error {
	return launchGridABI(state, functions[kernelStrataPleHistoryAdvance],
		driver.Dim3{X: (strataPLEWidth + strataPLEThreads - 1) / strataPLEThreads, Y: 1, Z: 1},
		driver.Dim3{X: strataPLEThreads, Y: 1, Z: 1}, &history, &normalized)
}
