package executor

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// Int8ConvRotMerge is an int8 ConvRot weight with low-rank adapter pairs merged into it as ComfyUI merges a LoRA into
// one while a module casts it (comfy/ops.py post_cast, with dynamic VRAM): the weight decoded to BF16 as comfy_kitchen's
// CUDA dequantize decodes it, each pair's product added in BF16 as comfy/weight_adapter/lora.py adds it, and the sum
// quantized again as comfy_kitchen's quantize_int8_convrot_weight does with a stochastic rounding.
type Int8ConvRotMerge struct {
	// Stored is the weight as dtype.I8ConvRot holds it: Rows rows of Inner int8 values, then their F32 scales; or, where
	// Held is set, empty, Held being the weight so held on the card already, where a program bound it from its file.
	Stored      []byte
	Held        driver.DevicePtr
	Rows, Inner uint32
	// Pairs are the adapters' pairs over the weight's rows, in the order ComfyUI adds them.
	Pairs []Int8ConvRotPair
	// FirstRow is the weight's first row in the weight ComfyUI quantizes again, a q, k or v projection's in its fused
	// qkv, which places its stochastic rounding's noise, and Seed is that rounding's seed.
	FirstRow, Seed uint64
	// Output is where the merged weight goes, as dtype.I8ConvRot holds it.
	Output driver.DevicePtr
}

// Int8ConvRotPair is one low-rank pair a merge adds: Up, the weight's Rows rows of Rank BF16 values, and Down, Rank rows
// of the weight's Inner BF16 values, whose product, rounded to BF16, is added times Factor (a strength times the pair's
// alpha over its rank, rounded to F32 as torch rounds a scalar it multiplies BF16 by).
type Int8ConvRotPair struct {
	Up, Down []byte
	Rank     uint32
	Factor   float32
}

// MergeInt8ConvRot merges the pairs into the weight on the card, on the worker's turn (state): the stored weight is
// decoded to BF16 values in F32 (dequantize_int8_convrot_rows_f32), each pair's product is formed by cuBLAS as torch.mm
// forms a BF16 product (GEMMEx over BF16 operands into BF16, accumulated in F32, the operands in torch's order),
// widened, multiplied by the factor and rounded to BF16, and added to the weight and rounded to BF16, and the sum is
// quantized to the output with the stochastic rounding (quantize_int8_convrot_rows_stochastic_f32). A weight held on
// the card is read where it lies, and the merges share one working memory, grown to the largest and kept, with no wait
// of their own: each had uploaded its weight from the file and allocated, waited for and freed its own, 252 merges a
// section in 3.4 to 6.2 s (moss-merge-time).
func (e *Executor) MergeInt8ConvRot(state *device.State, merge Int8ConvRotMerge) (err error) {
	rows, inner := uint64(merge.Rows), uint64(merge.Inner)
	elements := rows * inner
	stored := uint64(len(merge.Stored))
	if merge.Held != 0 {
		stored = 0
	}
	if rows == 0 || inner%int8ConvRotGroup != 0 || inner > int8ConvRotMaxInner || elements > 1<<31 ||
		(merge.Held == 0) == (len(merge.Stored) == 0) || merge.Held == 0 && stored != elements+4*rows || merge.Output == 0 {
		return fmt.Errorf("int8 ConvRot merge of %d x %d over %d stored bytes is not a weight the merge takes", rows, inner, len(merge.Stored))
	}
	rank := uint64(0)
	for index, pair := range merge.Pairs {
		if pair.Rank == 0 || uint64(len(pair.Up)) != 2*rows*uint64(pair.Rank) || uint64(len(pair.Down)) != 2*inner*uint64(pair.Rank) {
			return fmt.Errorf("int8 ConvRot merge pair %d is not %d and %d rows of rank %d in BF16", index, rows, inner, pair.Rank)
		}
		rank = max(rank, uint64(pair.Rank))
	}
	resources, err := e.ensureResources(state, len(merge.Pairs) > 0, 0, 0, 0)
	if err != nil {
		return err
	}
	// The working memory, each part aligned: the stored weight, the weight in F32, a product in BF16 and in F32, and a
	// pair's operands.
	sizes := []uint64{stored, 4 * elements, 2 * elements, 4 * elements, 2 * rows * rank, 2 * inner * rank}
	offsets := make([]uint64, len(sizes)+1)
	for index, size := range sizes {
		offsets[index+1] = offsets[index] + (size+bf16LinearAlignment-1)/bf16LinearAlignment*bf16LinearAlignment
	}
	work, err := e.mergeWork(state, offsets[len(sizes)])
	if err != nil {
		return err
	}
	at := func(index int) driver.DevicePtr { return work + driver.DevicePtr(offsets[index]) }
	weight, merged, product, widened, up, down := at(0), at(1), at(2), at(3), at(4), at(5)
	if merge.Held != 0 {
		weight = merge.Held
	} else if err := state.Driver.MemcpyHtoD(weight, merge.Stored); err != nil {
		return err
	}
	functions, rowsArgument, innerArgument := resources.functions, merge.Rows, merge.Inner
	scales := weight + driver.DevicePtr(elements)
	if err := launchGridABI(state, functions[kernelDequantizeInt8ConvrotRowsF32], kernel.Grid1D(int(rows)), kernel.DefaultBlock1D(),
		&weight, &scales, &merged, &innerArgument, &rowsArgument); err != nil {
		return err
	}
	count := uint32(elements)
	for _, pair := range merge.Pairs {
		if err := state.Driver.MemcpyHtoD(up, pair.Up); err != nil {
			return err
		}
		if err := state.Driver.MemcpyHtoD(down, pair.Down); err != nil {
			return err
		}
		// torch.mm(up, down) over row-major operands is cuBLAS's column-major product down x up: m the weight's inner
		// width, n its rows, k the rank.
		blas := resources.blas
		if err := blas.library.GEMMEx(blas.handle, cublas.OperationNone, cublas.OperationNone,
			int32(inner), int32(rows), int32(pair.Rank), 1, down, cublas.DataBF16, int32(inner), up, cublas.DataBF16, int32(pair.Rank),
			0, product, cublas.DataBF16, int32(inner), cublas.ComputeF32, cublas.GemmDefault); err != nil {
			return err
		}
		factor := pair.Factor
		if err := launch1DABI(state, functions[kernelBf16ToF32], count, &product, &widened, &count); err != nil {
			return err
		}
		if err := launch1DABI(state, functions[kernelScaleBf16RoundF32], count, &widened, &widened, &factor, &count); err != nil {
			return err
		}
		if err := launch1DABI(state, functions[kernelAddBf16RoundF32], count, &merged, &widened, &merged, &count); err != nil {
			return err
		}
	}
	output, outputScales := merge.Output, merge.Output+driver.DevicePtr(elements)
	firstRow, seed := merge.FirstRow, merge.Seed
	return launchGridABI(state, functions[kernelQuantizeInt8ConvrotRowsStochasticF32], kernel.Grid1D(int(rows)), kernel.DefaultBlock1D(),
		&merged, &output, &outputScales, &innerArgument, &rowsArgument, &firstRow, &seed)
}

// mergeWork is the merges' working memory of at least bytes, grown where a merge needs more: the stream is waited for
// before the smaller is freed, the one wait a merge may cause. The stream is a blocking one, so a merge's synchronous
// uploads into it follow the last merge's launches on it.
func (e *Executor) mergeWork(state *device.State, bytes uint64) (driver.DevicePtr, error) {
	if e.resources.mergeWorkBytes >= bytes {
		return e.resources.mergeWork, nil
	}
	if e.resources.mergeWork != 0 {
		if err := errors.Join(state.Driver.StreamSynchronize(state.Stream), state.Driver.MemFree(e.resources.mergeWork)); err != nil {
			return 0, err
		}
		e.resources.mergeWork, e.resources.mergeWorkBytes = 0, 0
	}
	work, err := state.Driver.MemAlloc(bytes)
	if err != nil {
		return 0, err
	}
	e.resources.mergeWork, e.resources.mergeWorkBytes = work, bytes
	return work, nil
}
