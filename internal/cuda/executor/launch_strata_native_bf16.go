package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's native bf16 projections over f32 activations (kernels/cuda/strata_native_bf16.cuh, ported from Strata at
// 296b59d2, MIT: src/kernels/cuda/native_bf16.cu): bf16_gemv_fp32_mmvf and bf16_gemv_fp32_mmvf_multi, their host
// sides translated at the block size mmvf_block_size gives the round's 2,560-value inputs.

// strataBF16Block is the block size the ported instances take: mmvf_block_size's for an input of 2,560 values, the
// fewest iterations of two values a thread among 32 to 256 threads in steps of 32.
const strataBF16Block = 256

// strataBF16MaxRows is bf16_gemv_fp32_mmvf_multi's most activation rows.
const strataBF16MaxRows = 8

// strataBF16BlockStep is mmvf_block_size's candidate step, and its smallest block: a warp.
const strataBF16BlockStep = 32

// strataBF16MultiExact are bf16_f32_mmvf_multi_kernel's exact instances by rows (one to six); seven and eight take the
// inexact eight-row instance.
var strataBF16MultiExact = [...]kernelFunctionID{
	kernelStrataBf16F32MmvfMultiB256T1, kernelStrataBf16F32MmvfMultiB256T2, kernelStrataBf16F32MmvfMultiB256T3,
	kernelStrataBf16F32MmvfMultiB256T4, kernelStrataBf16F32MmvfMultiB256T5, kernelStrataBf16F32MmvfMultiB256T6,
}

// strataBF16BlockSize is mmvf_block_size: the block, from a warp to strataBF16Block threads in warps, that walks n_in in
// the fewest iterations of two values a thread, the smallest on a tie.
func strataBF16BlockSize(nIn int64) int64 {
	best := int64(strataBF16BlockStep)
	bestIterations := (nIn + tensorPairValues*strataBF16BlockStep - 1) / (tensorPairValues * strataBF16BlockStep)
	for candidate := int64(tensorPairValues * strataBF16BlockStep); candidate <= strataBF16Block; candidate += strataBF16BlockStep {
		if iterations := (nIn + tensorPairValues*candidate - 1) / (tensorPairValues * candidate); iterations < bestIterations {
			best, bestIterations = candidate, iterations
		}
	}
	return best
}

// tensorPairValues is the two values (a weight pair) a thread takes an iteration.
const tensorPairValues = 2

// launchStrataBF16GEMV is bf16_gemv_fp32_mmvf: y (n_out) = w (n_out bf16 rows of n_in) times x.
func launchStrataBF16GEMV(state *device.State, functions functionSet, x, w, y driver.DevicePtr, nIn, nOut int64) error {
	if nIn <= 0 || nIn%tensorPairValues != 0 || nOut <= 0 || x == 0 || w == 0 || y == 0 {
		return fmt.Errorf("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out")
	}
	if block := strataBF16BlockSize(nIn); block != strataBF16Block {
		return fmt.Errorf("bf16_gemv_fp32_mmvf: n_in %d takes a block of %d, whose instance is not ported", nIn, block)
	}
	inputs := int32(nIn)
	return launchGridABI(state, functions[kernelStrataBf16F32MmvfB256], driver.Dim3{X: uint32(nOut), Y: 1, Z: 1},
		driver.Dim3{X: strataBF16Block, Y: 1, Z: 1}, &x, &w, &y, &inputs)
}

// launchStrataBF16GEMVMulti is bf16_gemv_fp32_mmvf_multi: rows activation rows of x (ldx apart) times w into y (ldy
// apart), the weights read once and each output bitwise its one-row call's; one row whose outputs fit ldy is
// bf16_gemv_fp32_mmvf.
func launchStrataBF16GEMVMulti(state *device.State, functions functionSet, x driver.DevicePtr, ldx int64, w, y driver.DevicePtr,
	ldy, nIn, nOut int64, rows int32) error {
	if rows == 1 && ldy >= nOut {
		return launchStrataBF16GEMV(state, functions, x, w, y, nIn, nOut)
	}
	if rows < 1 || rows > strataBF16MaxRows || nIn <= 0 || nIn%tensorPairValues != 0 || nOut <= 0 || ldx%tensorPairValues != 0 ||
		x == 0 || w == 0 || y == 0 {
		return fmt.Errorf("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers")
	}
	if block := strataBF16BlockSize(nIn); block != strataBF16Block {
		return fmt.Errorf("bf16_gemv_fp32_mmvf_multi: n_in %d takes a block of %d, whose instance is not ported", nIn, block)
	}
	kernel := kernelStrataBf16F32MmvfMultiB256T8
	if int(rows) <= len(strataBF16MultiExact) {
		kernel = strataBF16MultiExact[rows-1]
	}
	inputs, tokens := int32(nIn), rows
	return launchGridABI(state, functions[kernel], driver.Dim3{X: uint32(nOut), Y: 1, Z: 1},
		driver.Dim3{X: strataBF16Block, Y: 1, Z: 1}, &x, &ldx, &w, &y, &ldy, &inputs, &tokens)
}
