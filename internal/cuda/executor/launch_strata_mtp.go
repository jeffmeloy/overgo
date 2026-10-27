package executor

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// The launches of Strata's draft layer beyond its verify window's (kernels/cuda/strata_mtp.cuh, strata_fused_gr.cuh's
// default read and strata_native_qsa.cuh's norm and rope, ported from Strata at 296b59d2, MIT): their host sides from
// src/kernels/cuda/s2_expert_grouped.cu (moe_group_resident, moe_grouped_s2), quantize_act.cu (quantize_q8_0_scaled),
// elementwise.cu (copy_from_mapped_multi), verify_kernels.cu (add_streams_broadcast, gather_rows, map_ids, mtp_select,
// copy_row_to_first, row_top_prob_split, window_ids), fused_gr.cu (fused_gr_read_multi's default read, launch_multi at
// kHcStaged, and down_chunk's attributes), native_qsa.cu (native_qsa_rms_norm_weighted) and native_rope.cu
// (native_rope_apply), translated.

// The launches' geometry, which each kernel's indexing fixes.
const (
	// s2_expert_grouped.cu: the S2 blob's rows (H, FF) and the grouped kernels' rows a block (GU_ROWS, D_ROWS), the
	// fused SwiGLU's threads, a warp's values, and the device grouping's one block of 128 (a window's routed entries).
	strataS2Embedding       = 2560
	strataS2FF              = 640
	strataS2GateUpRows      = 32
	strataS2DownRows        = 64
	strataS2Threads         = 256
	strataS2Warp            = 32
	strataS2GroupThreads    = 128
	strataS2Q80Bytes        = 34
	strataS2ScratchAlign    = 16
	strataQuantizeThreads   = 128 // quantize_q8_0_scaled: a warp a block of 32 values, 128 threads
	strataMappedCopiesMost  = 8   // copy_from_mapped_multi: at most 8 copies
	strataMappedCopyBlocks  = 32  // its most blocks a copy
	strataMappedCopyBlock   = 256
	strataMappedCopyUnit    = 4 // a float4's words
	strataStreamsAddThreads = 256
	strataWindowIDsBlocks   = 8 // window_ids: (8, n) blocks of 256
	strataWindowIDsThreads  = 256
	strataRowCopyBlocks     = 16 // copy_row_to_first: 16 blocks of 256
	strataRowCopyThreads    = 256
	strataSelectThreads     = 256 // mtp_select: one block
	strataMapIDsThreads     = 64  // map_ids: one block
	strataTopBlocks         = 8   // row_top_prob_split: kTopBlocks, 1024 / kTopBlocks threads
	strataTopThreads        = 1024 / strataTopBlocks
	strataTopRowBytes       = 144    // TopRow: a counter, three pads and 32 floats
	strataGatherBlocks      = 48 * 8 // gather_rows: 384 blocks of 256
	strataGatherRowThreads  = 256
	strataGatherWord        = 4 // the uint32 form: a Q6_K row's 2,100 bytes divide into words, not 16-byte units
	// fused_gr.cu's staged read: H_TILE floats a staged tile, two tiles a token, its down blocks (40 and the inject
	// block), and the carveout down_chunk prefers; the shared memory a block gets without opting in.
	strataGRStagedTile     = 1280
	strataGRStagedTiles    = 2
	strataGRDownBlocks     = strataGRRank/(strataGRThreads/strataS2Warp) + 1
	strataGRStagedMax      = 6
	strataGRSharedCarveout = 100
	strataDefaultShared    = 48 * 1024
	// native_qsa.cu's norm: 256 threads below 1,024 columns, else 1,024; native_rope.cu's apply: 32 threads a block in
	// place at the fused width, else 128.
	strataQSANormNarrow  = 256
	strataQSANormWide    = 1024
	strataRopeApplyLane  = 32
	strataRopeApplyBlock = 128
)

// strataGRDownStaged and strataGRUpMulti are launch_multi's kHcStaged instances by token count, from one.
var (
	strataGRDownStaged = [strataGRStagedMax]kernelFunctionID{
		kernelStrataGrDownStagedT1, kernelStrataGrDownStagedT2, kernelStrataGrDownStagedT3,
		kernelStrataGrDownStagedT4, kernelStrataGrDownStagedT5, kernelStrataGrDownStagedT6,
	}
	strataGRUpMulti = [strataGRStagedMax]kernelFunctionID{
		kernelStrataGrUpMultiT1, kernelStrataGrUpMultiT2, kernelStrataGrUpMultiT3,
		kernelStrataGrUpMultiT4, kernelStrataGrUpMultiT5, kernelStrataGrUpMultiT6,
	}
)

// strataGRStagedShared is gr_down_staged_kernel's dynamic shared memory for tokens: two tiles a token.
func strataGRStagedShared(tokens int) uint32 {
	return uint32(strataGRStagedTiles * tokens * strataGRStagedTile * f32ByteSize)
}

// configureStrataGRStaged is down_chunk's set_staged_attr for the instances the read launches: each opted into its
// two tiles a token where they pass the default 48 KB, and every one preferring the whole carveout for shared memory.
func configureStrataGRStaged(lib *driver.Library, functions functionSet) error {
	for index, id := range strataGRDownStaged {
		function := functions[id].function
		if bytes := strataGRStagedShared(index + 1); bytes > strataDefaultShared {
			if err := lib.FuncSetMaxDynamicShared(function, bytes); err != nil {
				return err
			}
		}
		if err := lib.FuncSetPreferredSharedCarveout(function, strataGRSharedCarveout); err != nil {
			return err
		}
	}
	return nil
}

// launchStrataFusedGRRead is fused_gr_read_multi over bf16 projections at the variant Strata's check accepts here
// (launch_multi, kHcStaged): the split norm, the staged down projection in one launch of every token (down_chunk's
// staged capacity is past the drafter's six), the up projection; xn is the tokens' scratch.
func launchStrataFusedGRRead(state *device.State, functions functionSet, args []strataFusedGRArgs, xn driver.DevicePtr) error {
	tokens := len(args)
	if tokens < 1 || tokens > strataGRStagedMax || xn == 0 {
		return fmt.Errorf("fused_gr_read_multi: invalid arguments")
	}
	var multi strataGRMulti
	for index, token := range args {
		if token.R == 0 || token.WNorm == 0 || token.WDown == 0 || token.WUp == 0 || token.Lo == 0 || token.Rs == 0 ||
			token.Mixed == 0 || (token.WInject != 0 && token.InjectOut == 0) ||
			(token.Apply != 0 && (token.BoPrev == 0 || token.InjPrev == 0 || token.ROut == 0)) ||
			token.WDown != args[0].WDown || token.WUp != args[0].WUp || token.WInject != args[0].WInject ||
			token.WNorm != args[0].WNorm {
			return fmt.Errorf("fused_gr_read_multi: invalid arguments for token %d", index)
		}
		multi.A[index] = token
		multi.A[index].Q8Mixed, multi.A[index].Q8Count = 0, 0
	}
	multi.Xn, multi.T = xn, int32(tokens)
	block := driver.Dim3{X: strataGRThreads, Y: 1, Z: 1}
	if err := launchGridABI(state, functions[kernelStrataGrNormSplit],
		driver.Dim3{X: uint32(tokens), Y: strataGRStreams, Z: 1}, block, &multi); err != nil {
		return err
	}
	if err := launchGridSharedABI(state, functions[strataGRDownStaged[tokens-1]],
		driver.Dim3{X: strataGRDownBlocks, Y: 1, Z: 1}, block, strataGRStagedShared(tokens), &multi); err != nil {
		return err
	}
	return launchGridABI(state, functions[strataGRUpMulti[tokens-1]],
		driver.Dim3{X: strataGRUpBlocks, Y: 1, Z: 1}, block, &multi)
}

// launchStrataNativeQSANorm is native_qsa_rms_norm_weighted: rows rows of cols values normed by their root mean square
// and gamma, into output (which may be input).
func launchStrataNativeQSANorm(state *device.State, functions functionSet, input, gamma, output driver.DevicePtr,
	cols, rows int32, epsilon float32) error {
	if input == 0 || gamma == 0 || output == 0 || cols <= 0 || rows <= 0 || math.IsNaN(float64(epsilon)) ||
		math.IsInf(float64(epsilon), 0) || epsilon < 0 {
		return fmt.Errorf("native QSA requires finite nonnegative epsilon and positive bounded dimensions")
	}
	kernel, threads := kernelStrataNativeQsaNorm1024, uint32(strataQSANormWide)
	if cols < strataQSANormWide {
		kernel, threads = kernelStrataNativeQsaNorm256, strataQSANormNarrow
	}
	return launchGridABI(state, functions[kernel], driver.Dim3{X: uint32(rows), Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1}, &input, &gamma, &output, &cols, &epsilon)
}

// launchStrataNativeRopeApply is native_rope_apply: the first nRot values of rows rows of headDim rotated at each
// row's position (multiAxis, a multi-axis table, or 0), into out; in place at the fused width a warp a row.
func launchStrataNativeRopeApply(state *device.State, functions functionSet, x, out driver.DevicePtr, rows, headDim, nRot int32,
	scaling strataRopeScaling, positions, multiAxis driver.DevicePtr) error {
	if x == 0 || out == 0 || positions == 0 || rows < 1 || rows > strataNativeNormRopeMaxRows ||
		(headDim != strataNativeNormRopeNarrow && headDim != strataNativeNormRopeWide) || nRot != strataNativeNormRopeRotation {
		return fmt.Errorf("native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base/scaling and explicit stream")
	}
	thetaScale := float32(math.Pow(scaling.freqBase, -strataPairWidth/float64(nRot)))
	args := scaling.kernelArgs(int(nRot))
	threads := uint32(strataRopeApplyBlock)
	grid := driver.Dim3{X: (uint32(headDim)/strataPairWidth + strataRopeApplyBlock - 1) / strataRopeApplyBlock, Y: uint32(rows), Z: 1}
	if x == out {
		threads, grid.X = strataRopeApplyLane, 1
	}
	return launchGridABI(state, functions[kernelStrataNativeRopeApply], grid, driver.Dim3{X: threads, Y: 1, Z: 1},
		&x, &out, &rows, &headDim, &nRot, &thetaScale, &args.freqScale, &args.corrLow, &args.corrHigh, &args.extFactor,
		&args.attnFactor, &positions, &multiAxis)
}

// launchStrataGroupResident is moe_group_resident: the n routed ids (kPerToken a token) grouped by expert, each
// group's blob at base + id * blob, groups in first-appearance order and entries in routing order.
func launchStrataGroupResident(state *device.State, functions functionSet, ids driver.DevicePtr, n, kPerToken int32,
	base driver.DevicePtr, blob int64, groupPointers, groupStart, counts, entryDestination, entryToken driver.DevicePtr) error {
	if n < 1 || n > strataS2GroupThreads {
		return fmt.Errorf("moe_group_resident: n must be 1..128")
	}
	return launchGridABI(state, functions[kernelStrataGroupResident], driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: strataS2GroupThreads, Y: 1, Z: 1},
		&ids, &n, &kPerToken, &base, &blob, &groupPointers, &groupStart, &counts, &entryDestination, &entryToken)
}

// launchStrataQuantizeQ80Scaled is quantize_q8_0_scaled: n values to Q8_0 blocks and their fp32 scales, a warp a block.
func launchStrataQuantizeQ80Scaled(state *device.State, functions functionSet, x, blocks, scales driver.DevicePtr, n int64) error {
	if n <= 0 {
		return nil
	}
	if n%strataS2Warp != 0 || scales == 0 {
		return fmt.Errorf("quantize_q8_0_scaled: n %d is not a multiple of %d or scales is null", n, strataS2Warp)
	}
	count := n / strataS2Warp
	warps := int64(strataQuantizeThreads / strataS2Warp)
	return launchGridABI(state, functions[kernelStrataQuantizeQ80Scaled],
		driver.Dim3{X: uint32((count + warps - 1) / warps), Y: 1, Z: 1}, driver.Dim3{X: strataQuantizeThreads, Y: 1, Z: 1},
		&x, &blocks, &scales, &count)
}

// strataS2Layout is moe_grouped_s2's carve of its scratch: the gate and up rows' bytes, then the intermediate's Q8_0
// blocks', then its scales'.
func strataS2Layout(capEntries int64) (gateUp, blocks, scales int64) {
	align := func(bytes int64) int64 { return (bytes + strataS2ScratchAlign - 1) &^ (strataS2ScratchAlign - 1) }
	gateUp = align(capEntries * strataGateUpParts * strataS2FF * f32ByteSize)
	blocks = align(capEntries * (strataS2FF / strataS2Warp) * strataS2Q80Bytes)
	scales = align(capEntries * (strataS2FF / strataS2Warp) * f32ByteSize)
	return gateUp, blocks, scales
}

// launchStrataMoEGroupedS2 is moe_grouped_s2 on aligned activations with their fp32 scales (new_grouped, the fused
// SwiGLU and scaled Q8_0 quantize): the groups' gate and up rows, the SwiGLU into the intermediate's Q8_0, the down
// rows into out at each entry's destination row.
func launchStrataMoEGroupedS2(state *device.State, functions functionSet, groupPointers, groupStart, groups,
	entryDestination, entryToken driver.DevicePtr, capGroups, capEntries int64, xQ80, xScales, scratch, out driver.DevicePtr) error {
	if capGroups <= 0 || capEntries <= 0 {
		return nil
	}
	if xScales == 0 || xQ80%strataMappedCopyUnit != 0 || scratch%strataMappedCopyUnit != 0 {
		return fmt.Errorf("moe_grouped_s2: the drafter's path needs aligned activations and their scales")
	}
	gateUpBytes, blockBytes, _ := strataS2Layout(capEntries)
	gateUp := scratch
	hiddenQ80 := scratch + driver.DevicePtr(gateUpBytes)
	hiddenScales := hiddenQ80 + driver.DevicePtr(blockBytes)
	block := driver.Dim3{X: strataS2Threads, Y: 1, Z: 1}
	entries := int32(capEntries)
	if err := launchGridABI(state, functions[kernelStrataGuGroupedT],
		driver.Dim3{X: strataGateUpParts * strataS2FF / strataS2GateUpRows, Y: uint32(capGroups), Z: 1}, block,
		&groupPointers, &groupStart, &groups, &entryToken, &xQ80, &xScales, &gateUp, &entries); err != nil {
		return err
	}
	pairs := capEntries * strataS2FF
	count := pairs / strataS2Warp
	warps := int64(strataS2Threads / strataS2Warp)
	if err := launchGridABI(state, functions[kernelStrataSwigluQuantizeQ80Scaled],
		driver.Dim3{X: uint32((count + warps - 1) / warps), Y: 1, Z: 1}, block,
		&gateUp, &pairs, &hiddenQ80, &hiddenScales); err != nil {
		return err
	}
	return launchGridABI(state, functions[kernelStrataDownGroupedT],
		driver.Dim3{X: strataS2Embedding / strataS2DownRows, Y: uint32(capGroups), Z: 1}, block,
		&groupPointers, &groupStart, &groups, &entryDestination, &hiddenQ80, &hiddenScales, &out)
}

// strataMappedCopy is elementwise.hpp's MappedCopy: words 32-bit words from src to dst.
type strataMappedCopy struct {
	Dst, Src driver.DevicePtr
	Words    int64
}

// strataMappedCopies is copy_from_mapped_multi_kernel's argument in its C layout: up to eight copies, by value.
type strataMappedCopies struct {
	C [strataMappedCopiesMost]strataMappedCopy
}

// launchStrataCopyFromMappedMulti is copy_from_mapped_multi: the copies in one launch, a grid row each, the blocks
// sized by the longest.
func launchStrataCopyFromMappedMulti(state *device.State, functions functionSet, copies []strataMappedCopy) error {
	if len(copies) == 0 {
		return nil
	}
	if len(copies) > strataMappedCopiesMost {
		return fmt.Errorf("copy_from_mapped_multi: at most 8 copies")
	}
	var all strataMappedCopies
	most := int64(0)
	for index, copy := range copies {
		all.C[index] = copy
		most = max(most, copy.Words)
	}
	units := (most + strataMappedCopyUnit - 1) / strataMappedCopyUnit
	blocks := min((units+strataMappedCopyBlock-1)/strataMappedCopyBlock, strataMappedCopyBlocks)
	return launchGridABI(state, functions[kernelStrataCopyFromMappedMulti],
		driver.Dim3{X: uint32(max(blocks, 1)), Y: uint32(len(copies)), Z: 1},
		driver.Dim3{X: strataMappedCopyBlock, Y: 1, Z: 1}, &all)
}

// launchStrataAddStreamsBroadcast is add_streams_broadcast: each of tokens' streams rows of h plus the token's e row,
// into R.
func launchStrataAddStreamsBroadcast(state *device.State, functions functionSet, h, e, r driver.DevicePtr, embedding int64,
	streams, tokens int32) error {
	return launchGridABI(state, functions[kernelStrataAddStreamsBroadcast],
		driver.Dim3{X: uint32((embedding*int64(streams) + strataStreamsAddThreads - 1) / strataStreamsAddThreads), Y: uint32(tokens), Z: 1},
		driver.Dim3{X: strataStreamsAddThreads, Y: 1, Z: 1}, &h, &e, &r, &embedding, &streams)
}

// launchStrataWindowIDs is window_ids: each of n step records' last window cells as ids (stride apart), its width
// written to the record's fourth word.
func launchStrataWindowIDs(state *device.State, functions functionSet, steps driver.DevicePtr, n, window int32, ids driver.DevicePtr,
	stride int64) error {
	return launchGridABI(state, functions[kernelStrataWindowIds], driver.Dim3{X: strataWindowIDsBlocks, Y: uint32(n), Z: 1},
		driver.Dim3{X: strataWindowIDsThreads, Y: 1, Z: 1}, &steps, &window, &ids, &stride)
}

// launchStrataCopyRowToFirst is copy_row_to_first: the row *row of each of a, b and c copied over its row 0 (nothing
// for row 0).
func launchStrataCopyRowToFirst(state *device.State, functions functionSet, row, a driver.DevicePtr, aN int64, b driver.DevicePtr,
	bN int64, c driver.DevicePtr, cN int64) error {
	return launchGridABI(state, functions[kernelStrataCopyRowToFirst], driver.Dim3{X: strataRowCopyBlocks, Y: 1, Z: 1},
		driver.Dim3{X: strataRowCopyThreads, Y: 1, Z: 1}, &row, &a, &aN, &b, &bN, &c, &cN)
}

// launchStrataMTPSelect is mtp_select: row *row of the residuals (stride apart) and its token to the next step's
// inputs, and the token and its probability to the mapped outputs' slot j.
func launchStrataMTPSelect(state *device.State, functions functionSet, source driver.DevicePtr, stride int64, ids, row,
	destination, token, out driver.DevicePtr, j int32, probabilities, outProbabilities driver.DevicePtr) error {
	return launchGridABI(state, functions[kernelStrataMtpSelect], driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: strataSelectThreads, Y: 1, Z: 1},
		&source, &stride, &ids, &row, &destination, &token, &out, &j, &probabilities, &outProbabilities)
}

// launchStrataMapIDs is map_ids: each of n ids replaced by its entry of table.
func launchStrataMapIDs(state *device.State, functions functionSet, ids, table driver.DevicePtr, n int32) error {
	return launchGridABI(state, functions[kernelStrataMapIds], driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: strataMapIDsThreads, Y: 1, Z: 1}, &ids, &table, &n)
}

// launchStrataRowTopProbSplit is row_top_prob_split: each row's probability of its id under the row's softmax, over
// scratch (strataTopRowBytes a row).
func launchStrataRowTopProbSplit(state *device.State, functions functionSet, logits driver.DevicePtr, rows, vocab int32,
	ids, probabilities, scratch driver.DevicePtr) error {
	if rows <= 0 {
		return nil
	}
	return launchGridABI(state, functions[kernelStrataRowTopProbSplit], driver.Dim3{X: strataTopBlocks, Y: uint32(rows), Z: 1},
		driver.Dim3{X: strataTopThreads, Y: 1, Z: 1}, &logits, &vocab, &ids, &probabilities, &scratch)
}

// launchStrataGatherRows is gather_rows over rows whose bytes divide into words: the n rows of src the ids name, back
// to back in dst.
func launchStrataGatherRows(state *device.State, functions functionSet, source driver.DevicePtr, rowBytes int64, ids driver.DevicePtr,
	n int64, destination driver.DevicePtr) error {
	if rowBytes%strataGatherWord != 0 {
		return fmt.Errorf("gather_rows: a row of %d bytes is not whole words", rowBytes)
	}
	words := rowBytes / strataGatherWord
	return launchGridABI(state, functions[kernelStrataGatherRowsU32], driver.Dim3{X: strataGatherBlocks, Y: 1, Z: 1},
		driver.Dim3{X: strataGatherRowThreads, Y: 1, Z: 1}, &source, &words, &ids, &n, &destination)
}
