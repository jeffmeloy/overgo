package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's native MMVQ projections (kernels/cuda/strata_native_mmvq.cuh, ported from Strata at 296b59d2, MIT:
// src/kernels/cuda/native_mmvq.cu): its host side's dispatch, translated - native_mmvq over the formats Flash-Next's
// dense projections and head store (Q8_0 through small_mmvq, Q6_K through native_q6_k_mmvq, each to launch_multi at
// two or more columns) and the K-quants beside Q6_K (Q4_K and Q5_K through native_q4_k_mmvq and native_q5_k_mmvq),
// native_quantize_q8_1, native_swiglu_quantize_q8_1, native_q8_1_interleave and native_mmvq_il
// with its rows tables. Activations are q8_1 blocks, a column after another; outputs a column after another.

// The GGML types native_mmvq takes here, by GGML's numbering.
const (
	strataGGMLQ8_0 = 8
	strataGGMLQ4K  = 12
	strataGGMLQ5K  = 13
	strataGGMLQ6K  = 14
)

// native_mmvq.cu's geometry.
const (
	strataNativeQK           = 256 // values a K-quant block
	strataNativeQ8K          = 32  // values a q8_1 block
	strataNativeWarps        = 4   // WARPS
	strataNativeWarp         = 32  // WARP
	strataNativeQuantThreads = 256 // QUANT_THREADS
	strataNativeMaxColumns   = 8   // MAX_NCOLS
	strataNativeQ81Bytes     = 36  // sizeof(Q81Block): a half2 scale and sum, then 32 codes
	strataNativeQ80Qi        = 8   // Q8_0's Qi (SmallTraits<Q80Block, 8>)
	strataNativeVDR          = 2   // the blocks a thread of the 32-value formats and the K-quants takes an iteration (VDR)
	strataNativeQI           = 32  // a K-quant block's ints of codes (QI)
	strataNativeMultiFirst   = 2   // the fewest columns launch_multi takes
	strataNativeMultiRows    = 2   // launch_multi_n's rows a block (ROWS)
)

// native_q8_1_interleave's and native_mmvq_il's geometry.
const (
	strataNativeILThreads     = 256 // native_q8_1_interleave_kernel's block
	strataNativeILMaxColumns  = 4   // native_mmvq_il's most columns
	strataNativeILPairRoom    = 2   // native_q8_1_il_cp's room for two columns
	strataNativeILBlockInts   = 8   // a q8_1 block's codes in ints
	strataNativeILCopies      = 2   // the block-major and the position-major codes
	strataNativeILWordBytes   = 4   // an int code word, a float scale
	strataNativeILWarpsABlock = 4   // native_mmvq_il_kernel's warps a block
	strataNativeILClasses     = 5   // native_mmvq_il_rows_for's output classes
	strataNativeILRowCounts   = 3   // the rows a warp the kernels take: 1, 2 and 4
	strataNativeILRowsPair    = 2   // two rows a warp
	strataNativeILRowsQuad    = 4   // four rows a warp
	strataNativeILMinMajor    = 8   // il_arch_ok: the interleaved kernels on sm_80 and newer
	strataNativeCCTens        = 10  // a compute capability's major*10+minor
)

// native_mmvq_il_rows_for's output classes' upper bounds: the keys' 512 below 2048, the output projections' 2560 below
// 4096, the gate projection's 6144 below 8192, qkv's 10240 below 12288, and more (the queries' 12288, the head's 248K).
const (
	strataNativeILClassKeys    = 2048
	strataNativeILClassOutputs = 4096
	strataNativeILClassGate    = 8192
	strataNativeILClassQKV     = 12288
)

// The compute capabilities with an interleaved rows table of their own in the source (kIlArch: Pascal GP100 and GP10x,
// Volta, sm_86), whose tables are not ported: such a card takes native_mmvq's kernels.
const (
	strataNativeCCPascal     = 60
	strataNativeCCPascalDP4A = 61
	strataNativeCCVolta      = 70
	strataNativeCCAmpere86   = 86
)

// strataNativeQ81Size is native_q8_1_bytes: ncols columns of n_in values as q8_1 blocks.
func strataNativeQ81Size(nIn, ncols int32) uint64 {
	return uint64(ncols) * uint64(nIn/strataNativeQ8K) * strataNativeQ81Bytes
}

// strataNativeQ81ILRoom is native_q8_1_il_cp: the columns an interleaved copy holds room for.
func strataNativeQ81ILRoom(ncols int32) int32 {
	switch {
	case ncols <= strataNativeILPairRoom:
		return strataNativeILPairRoom
	case ncols <= strataNativeILMaxColumns:
		return strataNativeILMaxColumns
	}
	return strataNativeMaxColumns
}

// strataNativeQ81ILSize is native_q8_1_il_bytes: the interleaved copy of ncols columns of n_in values, its block-major
// and position-major codes and its scales.
func strataNativeQ81ILSize(nIn, ncols int32) uint64 {
	return uint64(nIn/strataNativeQ8K) * uint64(strataNativeQ81ILRoom(ncols)) *
		(strataNativeILCopies*strataNativeILBlockInts + 1) * strataNativeILWordBytes
}

func strataNativeShape(nIn, ncols, blockValues int32) error {
	if nIn <= 0 || nIn%blockValues != 0 {
		return fmt.Errorf("native MMVQ requires n_in > 0 and divisible by its block element count")
	}
	if ncols < 1 || ncols > strataNativeMaxColumns {
		return fmt.Errorf("native MMVQ requires 1 <= ncols <= %d", strataNativeMaxColumns)
	}
	return nil
}

// launchStrataNativeQuantizeQ81 is native_quantize_q8_1: ncols contiguous columns of n_in values quantize as one vector,
// every 32-value block inside one column.
func launchStrataNativeQuantizeQ81(state *device.State, functions functionSet, x, xq driver.DevicePtr, nIn, ncols int32) error {
	if err := strataNativeShape(nIn, ncols, strataNativeQ8K); err != nil {
		return err
	}
	total := nIn * ncols
	return launchGridABI(state, functions[kernelStrataNativeQuantizeQ81],
		driver.Dim3{X: (uint32(total) + strataNativeQuantThreads - 1) / strataNativeQuantThreads, Y: 1, Z: 1},
		driver.Dim3{X: strataNativeQuantThreads, Y: 1, Z: 1}, &x, &xq, &total)
}

// launchStrataNativeSwigluQuantizeQ81 is native_swiglu_quantize_q8_1: SiLU(gate) * up, quantized as
// native_quantize_q8_1 quantizes.
func launchStrataNativeSwigluQuantizeQ81(state *device.State, functions functionSet, gate, up, xq driver.DevicePtr, nIn, ncols int32) error {
	if err := strataNativeShape(nIn, ncols, strataNativeQ8K); err != nil {
		return err
	}
	total := nIn * ncols
	return launchGridABI(state, functions[kernelStrataNativeSwigluQuantizeQ81],
		driver.Dim3{X: (uint32(total) + strataNativeQuantThreads - 1) / strataNativeQuantThreads, Y: 1, Z: 1},
		driver.Dim3{X: strataNativeQuantThreads, Y: 1, Z: 1}, &gate, &up, &xq, &total)
}

// strataNativeMultiQ80, strataNativeMultiQ4K, strataNativeMultiQ5K and strataNativeMultiQ6K are launch_multi's instances
// by columns, from two.
var (
	strataNativeMultiQ4K = [...]kernelFunctionID{
		kernelStrataNativeMmvqMultiQ4KC2, kernelStrataNativeMmvqMultiQ4KC3, kernelStrataNativeMmvqMultiQ4KC4,
		kernelStrataNativeMmvqMultiQ4KC5, kernelStrataNativeMmvqMultiQ4KC6, kernelStrataNativeMmvqMultiQ4KC7,
		kernelStrataNativeMmvqMultiQ4KC8,
	}
	strataNativeMultiQ5K = [...]kernelFunctionID{
		kernelStrataNativeMmvqMultiQ5KC2, kernelStrataNativeMmvqMultiQ5KC3, kernelStrataNativeMmvqMultiQ5KC4,
		kernelStrataNativeMmvqMultiQ5KC5, kernelStrataNativeMmvqMultiQ5KC6, kernelStrataNativeMmvqMultiQ5KC7,
		kernelStrataNativeMmvqMultiQ5KC8,
	}
	strataNativeMultiQ80 = [...]kernelFunctionID{
		kernelStrataNativeMmvqMultiQ80C2, kernelStrataNativeMmvqMultiQ80C3, kernelStrataNativeMmvqMultiQ80C4,
		kernelStrataNativeMmvqMultiQ80C5, kernelStrataNativeMmvqMultiQ80C6, kernelStrataNativeMmvqMultiQ80C7,
		kernelStrataNativeMmvqMultiQ80C8,
	}
	strataNativeMultiQ6K = [...]kernelFunctionID{
		kernelStrataNativeMmvqMultiQ6KC2, kernelStrataNativeMmvqMultiQ6KC3, kernelStrataNativeMmvqMultiQ6KC4,
		kernelStrataNativeMmvqMultiQ6KC5, kernelStrataNativeMmvqMultiQ6KC6, kernelStrataNativeMmvqMultiQ6KC7,
		kernelStrataNativeMmvqMultiQ6KC8,
	}
)

// launchStrataNativeMMVQ is native_mmvq for a GGML type: y (ncols columns of n_out) = weights (n_out rows of n_in) times
// the q8_1 columns xq. One column takes the format's single-column kernel, its small-K form where a row has fewer blocks
// than an iteration covers; more take launch_multi_n's exact layout (four warps, two rows a block: launch_multi_n's one
// row a block is for gfx12 and sm_86 cards only).
func launchStrataNativeMMVQ(state *device.State, functions functionSet, ggmlType int, weights, xq, y driver.DevicePtr,
	nIn, nOut, ncols int32) error {
	var single, smallK kernelFunctionID
	var multi []kernelFunctionID
	var blockValues, smallBelow int32
	switch ggmlType {
	case strataGGMLQ8_0:
		single, smallK, multi = kernelStrataNativeSmallMmvqQ80, kernelStrataNativeSmallMmvqQ80SmallK, strataNativeMultiQ80[:]
		blockValues, smallBelow = strataNativeQ8K, strataNativeVDR*strataNativeWarps*strataNativeWarp/strataNativeQ80Qi
	case strataGGMLQ6K:
		single, smallK, multi = kernelStrataNativeQ6KMmvq, kernelStrataNativeQ6KMmvqSmallK, strataNativeMultiQ6K[:]
		blockValues, smallBelow = strataNativeQK, strataNativeWarps*strataNativeWarp/strataNativeQ8K
	case strataGGMLQ5K:
		single, smallK, multi = kernelStrataNativeQ5KMmvq, kernelStrataNativeQ5KMmvqSmallK, strataNativeMultiQ5K[:]
		blockValues, smallBelow = strataNativeQK, strataNativeVDR*strataNativeWarps*strataNativeWarp/strataNativeQI
	case strataGGMLQ4K:
		single, smallK, multi = kernelStrataNativeQ4KMmvq, kernelStrataNativeQ4KMmvqSmallK, strataNativeMultiQ4K[:]
		blockValues, smallBelow = strataNativeQK, strataNativeVDR*strataNativeWarps*strataNativeWarp/strataNativeQI
	default:
		return fmt.Errorf("unsupported native MMVQ GGML type %d", ggmlType)
	}
	if err := strataNativeShape(nIn, ncols, blockValues); err != nil {
		return err
	}
	if nOut <= 0 {
		return fmt.Errorf("native MMVQ requires n_out > 0")
	}
	threads := driver.Dim3{X: strataNativeWarp, Y: strataNativeWarps, Z: 1}
	if ncols > 1 {
		blocks := (uint32(nOut) + strataNativeMultiRows - 1) / strataNativeMultiRows
		return launchGridABI(state, functions[multi[ncols-strataNativeMultiFirst]], driver.Dim3{X: blocks, Y: 1, Z: 1},
			threads, &weights, &xq, &y, &nIn, &nOut)
	}
	if nIn/blockValues < smallBelow {
		return launchGridABI(state, functions[smallK],
			driver.Dim3{X: (uint32(nOut) + strataNativeWarps - 1) / strataNativeWarps, Y: 1, Z: 1}, threads,
			&weights, &xq, &y, &nIn, &nOut)
	}
	return launchGridABI(state, functions[single], driver.Dim3{X: uint32(nOut), Y: 1, Z: 1}, threads,
		&weights, &xq, &y, &nIn, &nOut)
}

// launchStrataNativeQ81Interleave is native_q8_1_interleave: the interleaved copy of 2 to 4 columns of q8_1 blocks into
// xil (strataNativeQ81ILSize), the same values in the same bytes.
func launchStrataNativeQ81Interleave(state *device.State, functions functionSet, xq, xil driver.DevicePtr, nIn, ncols int32) error {
	if err := strataNativeShape(nIn, ncols, strataNativeQ8K); err != nil {
		return err
	}
	if ncols < strataNativeMultiFirst || ncols > strataNativeILMaxColumns {
		return fmt.Errorf("native_q8_1_interleave requires 2 <= ncols <= %d", strataNativeILMaxColumns)
	}
	blocks, room := nIn/strataNativeQ8K, strataNativeQ81ILRoom(ncols)
	bm, pm, scales := strataNativeQ81ILParts(xil, nIn, ncols)
	total := blocks * strataNativeILBlockInts * ncols
	return launchGridABI(state, functions[kernelStrataNativeQ81Interleave],
		driver.Dim3{X: (uint32(total) + strataNativeILThreads - 1) / strataNativeILThreads, Y: 1, Z: 1},
		driver.Dim3{X: strataNativeILThreads, Y: 1, Z: 1}, &xq, &bm, &pm, &scales, &blocks, &total, &room)
}

// strataNativeQ81ILParts is q8_1_il_parts: the block-major codes, the position-major codes and the scales of an
// interleaved copy.
func strataNativeQ81ILParts(xil driver.DevicePtr, nIn, ncols int32) (bm, pm, scales driver.DevicePtr) {
	words := uint64(nIn/strataNativeQ8K) * strataNativeILBlockInts * uint64(strataNativeQ81ILRoom(ncols))
	return xil, xil + driver.DevicePtr(words*strataNativeILWordBytes),
		xil + driver.DevicePtr(strataNativeILCopies*words*strataNativeILWordBytes)
}

// strataNativeILRows is a rows table's entry: a GGML type's rows a warp at 2 to 4 columns ([ncols - 2]) by the matrix's
// output class; 0 takes native_mmvq's kernels.
type strataNativeILRows struct {
	ggmlType int
	rows     [strataNativeILMaxColumns - 1][strataNativeILClasses]int32
}

// strataNativeILRowsShared is kIlRowsShared: sm_86 and sm_120 measured together, and every sm_80+ card without a table
// of its own (Ada among them). The types with interleaved kernels ported here are Q6_K's.
var strataNativeILRowsShared = [...]strataNativeILRows{
	{ggmlType: strataGGMLQ6K, rows: [strataNativeILMaxColumns - 1][strataNativeILClasses]int32{
		{0, 0, 1, 1, 0}, {0, 0, 1, 1, strataNativeILRowsPair}, {0, 0, 1, 1, 1},
	}},
}

// strataNativeMMVQILRows is native_mmvq_il_rows_for and native_mmvq_il_supported for a card of compute capability
// major.minor: the rows a warp, 0 for native_mmvq's kernels (one column always: Strata's single-column table names no
// card of the shared table).
func strataNativeMMVQILRows(major, minor int, ggmlType int, ncols, nOut int32) int32 {
	if ncols < strataNativeMultiFirst || ncols > strataNativeILMaxColumns || major < strataNativeILMinMajor {
		return 0
	}
	switch major*strataNativeCCTens + minor {
	case strataNativeCCPascal, strataNativeCCPascalDP4A, strataNativeCCVolta, strataNativeCCAmpere86:
		return 0
	}
	class := strataNativeILClasses - 1
	for index, bound := range [...]int32{strataNativeILClassKeys, strataNativeILClassOutputs, strataNativeILClassGate, strataNativeILClassQKV} {
		if nOut < bound {
			class = index
			break
		}
	}
	for _, entry := range strataNativeILRowsShared {
		if entry.ggmlType == ggmlType {
			return entry.rows[ncols-strataNativeMultiFirst][class]
		}
	}
	return 0
}

// strataNativeILQ6K is launch_il's Q6_K instances by columns (2 to 4) and rows a warp (1, 2, 4).
var strataNativeILQ6K = [strataNativeILMaxColumns - 1][strataNativeILRowCounts]kernelFunctionID{
	{kernelStrataNativeMmvqIlQ6KC2R1, kernelStrataNativeMmvqIlQ6KC2R2, kernelStrataNativeMmvqIlQ6KC2R4},
	{kernelStrataNativeMmvqIlQ6KC3R1, kernelStrataNativeMmvqIlQ6KC3R2, kernelStrataNativeMmvqIlQ6KC3R4},
	{kernelStrataNativeMmvqIlQ6KC4R1, kernelStrataNativeMmvqIlQ6KC4R2, kernelStrataNativeMmvqIlQ6KC4R4},
}

// launchStrataNativeMMVQIL is native_mmvq_il: the interleaved kernel where the rows table names a count, from xil (the
// interleaved copy of xq's columns, native_q8_1_interleave), else native_mmvq from xq.
func launchStrataNativeMMVQIL(state *device.State, functions functionSet, major, minor int, ggmlType int,
	weights, xq, xil, y driver.DevicePtr, nIn, nOut, ncols int32) error {
	rows := strataNativeMMVQILRows(major, minor, ggmlType, ncols, nOut)
	if rows == 0 || ggmlType != strataGGMLQ6K {
		return launchStrataNativeMMVQ(state, functions, ggmlType, weights, xq, y, nIn, nOut, ncols)
	}
	if err := strataNativeShape(nIn, ncols, strataNativeQK); err != nil {
		return err
	}
	rowIndex := 0
	switch rows {
	case strataNativeILRowsPair:
		rowIndex = 1
	case strataNativeILRowsQuad:
		rowIndex = strataNativeILRowCounts - 1
	}
	bm, _, scales := strataNativeQ81ILParts(xil, nIn, ncols)
	blocks := ((uint32(nOut)+uint32(rows)-1)/uint32(rows) + strataNativeILWarpsABlock - 1) / strataNativeILWarpsABlock
	return launchGridABI(state, functions[strataNativeILQ6K[ncols-strataNativeMultiFirst][rowIndex]],
		driver.Dim3{X: blocks, Y: 1, Z: 1}, driver.Dim3{X: strataNativeILWarpsABlock * strataNativeWarp, Y: 1, Z: 1},
		&weights, &bm, &scales, &y, &nIn, &nOut)
}
