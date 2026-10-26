package executor

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's QSA attention layer, ported literally (kernels/cuda/strata_qsa.cu; Strata at 296b59d2, MIT,
// licenses/Strata-MIT.txt). These are Strata's host launch functions with its grids and blocks: kv_append_step and
// kv_gather_step (src/kernels/cuda/qsa.cu), kv_append_q8_step, kv_append_q8_steps and kv_gather_q8_step
// (src/kernels/cuda/kv_q8.cu). Every per-token count is read by the kernels from the device step state, so the
// launches here are sized by capacities and stay constant across tokens.
const (
	// strataAppendThreads is qsa.cu's THREADS, the fp16 append's block.
	strataAppendThreads = uint32(128)
	// strataGatherThreads is the gathers' block, and strataGatherValues the values each thread moves (a uint2 of
	// fp16 or a char4 of codes).
	strataGatherThreads = uint32(256)
	strataGatherValues  = uint32(4)
	// strataKVQ8Group is kv_q8.hpp's KV_Q8_GROUP, one fp16 scale a 64-value group of a cell's head, and the int8
	// append's block. Provisional for the load-costs step: Strata's constant, tuned on its author's cards.
	strataKVQ8Group = uint32(64)
	// strataKVHalves is the append grid's K and V planes.
	strataKVHalves = uint32(2)
)

// strataKVGeometry is the part of qsa.hpp's QsaShapes the cache kernels read.
type strataKVGeometry struct {
	kvHeads, headDim, pageSize int32
}

func (g strataKVGeometry) validate(int8 bool) error {
	if g.kvHeads <= 0 || g.headDim <= 0 || g.pageSize <= 0 || uint32(g.headDim)%strataGatherValues != 0 {
		return errors.New("the Strata K/V geometry is not set up: heads, a head width of whole 4-value runs, a page")
	}
	if int8 && uint32(g.headDim)%strataKVQ8Group != 0 {
		return errors.New("the Strata int8 K/V takes head widths of whole 64-value groups")
	}
	return nil
}

// strataKVPools are one layer's K/V pools: fp16 values in key and value, or int8 codes in key and value with their
// fp16 scales. A zero host pool set is a layer with no host copy.
type strataKVPools struct {
	key, value, keyScale, valueScale driver.DevicePtr
}

// launchStrataKVAppend writes the cell at step[pos] into the pools through the page table, and into the host copy
// when one is set: kv_append_step (fp16) or kv_append_q8_step (int8).
func launchStrataKVAppend(
	state *device.State, functions functionSet, pools, host strataKVPools, table, step, keys, values driver.DevicePtr,
	geometry strataKVGeometry, int8 bool,
) error {
	if err := geometry.validate(int8); err != nil {
		return err
	}
	heads, width, page := geometry.kvHeads, geometry.headDim, geometry.pageSize
	if !int8 {
		count := uint32(heads * width)
		return launchGridABI(state, functions[kernelStrataKvAppendF16],
			driver.Dim3{X: (count + strataAppendThreads - 1) / strataAppendThreads, Y: 1, Z: 1},
			driver.Dim3{X: strataAppendThreads, Y: 1, Z: 1},
			&pools.key, &pools.value, &table, &step, &keys, &values, &heads, &width, &page, &host.key, &host.value)
	}
	return launchGridABI(state, functions[kernelStrataKvAppendQ8],
		driver.Dim3{X: uint32(heads), Y: uint32(width) / strataKVQ8Group, Z: strataKVHalves},
		driver.Dim3{X: strataKVQ8Group, Y: 1, Z: 1},
		&pools.key, &pools.value, &pools.keyScale, &pools.valueScale, &table, &step, &keys, &values,
		&heads, &width, &page, &host.key, &host.value, &host.keyScale, &host.valueScale)
}

// launchStrataKVAppendSteps writes tokens cells in one launch, token j at step + j * stepStride with its rows at
// keys / values + j * rowStride: kv_append_q8_steps, bitwise the stores of tokens single appends.
func launchStrataKVAppendSteps(
	state *device.State, functions functionSet, pools, host strataKVPools, table, step, keys, values driver.DevicePtr,
	geometry strataKVGeometry, tokens, stepStride, rowStride int32,
) error {
	if err := geometry.validate(true); err != nil {
		return err
	}
	if tokens < 1 {
		return nil
	}
	heads, width, page := geometry.kvHeads, geometry.headDim, geometry.pageSize
	return launchGridABI(state, functions[kernelStrataKvAppendQ8Steps],
		driver.Dim3{X: uint32(heads), Y: uint32(width) / strataKVQ8Group, Z: strataKVHalves * uint32(tokens)},
		driver.Dim3{X: strataKVQ8Group, Y: 1, Z: 1},
		&pools.key, &pools.value, &pools.keyScale, &pools.valueScale, &table, &step, &keys, &values,
		&heads, &width, &page, &host.key, &host.value, &host.keyScale, &host.valueScale, &stepStride, &rowStride)
}

// launchStrataKVGather gathers step[width] selected cells named by ids through the page table into fp16 scratch
// `[id][kv_head][head_dim]`, dequantizing int8 cells: kv_gather_step or kv_gather_q8_step. The grid is sized by
// maxIDs, the capacity; the kernels read the real count from the step state.
func launchStrataKVGather(
	state *device.State, functions functionSet, pools strataKVPools, table, ids, step driver.DevicePtr, maxIDs uint32,
	geometry strataKVGeometry, keyScratch, valueScratch driver.DevicePtr, int8 bool,
) error {
	if err := geometry.validate(int8); err != nil {
		return err
	}
	if maxIDs == 0 {
		return nil
	}
	heads, width, page := geometry.kvHeads, geometry.headDim, geometry.pageSize
	total := maxIDs * uint32(heads) * (uint32(width) / strataGatherValues)
	grid := driver.Dim3{X: (total + strataGatherThreads - 1) / strataGatherThreads, Y: 1, Z: 1}
	block := driver.Dim3{X: strataGatherThreads, Y: 1, Z: 1}
	if !int8 {
		return launchGridABI(state, functions[kernelStrataKvGatherF16], grid, block,
			&pools.key, &pools.value, &table, &ids, &step, &heads, &width, &page, &keyScratch, &valueScratch)
	}
	return launchGridABI(state, functions[kernelStrataKvGatherQ8], grid, block,
		&pools.key, &pools.value, &pools.keyScale, &pools.valueScale, &table, &ids, &step, &heads, &width, &page,
		&keyScratch, &valueScratch)
}

// strataIndexerBuffers are qsa.hpp's QsaIndexerBuffers: the raw tail (idx_block - 1 rows), the spare key, the pooled
// rows (one a completed block plus the spare slot) and the last completed block's rotation position.
type strataIndexerBuffers struct {
	tail, dead, pooled, blockPos driver.DevicePtr
}

// strataIndexerGeometry is the indexer's part of QsaShapes: its key width, its block (r cells) and the rotated dims.
type strataIndexerGeometry struct {
	keyWidth, block, rotated int32
}

// launchStrataIndexerKeyAppend appends the raw indexer key of the cell at positions[0] and, on a block completion,
// pools, norms and rotates the block's key: qsa.cu's indexer_key_append, one block of one thread a key dim with the
// pooling's doubles in shared memory. multiAxis is the multi-axis position table, zero for text.
func launchStrataIndexerKeyAppend(
	state *device.State, functions functionSet, raw, positions driver.DevicePtr, positionBase int32,
	normWeight driver.DevicePtr, epsilon float32, buffers strataIndexerBuffers, geometry strataIndexerGeometry,
	cosines, sines, multiAxis driver.DevicePtr,
) error {
	if geometry.keyWidth <= 0 || geometry.block < 2 || geometry.rotated <= 0 || geometry.rotated%2 != 0 ||
		geometry.rotated > geometry.keyWidth {
		return errors.New("the Strata indexer geometry is not set up: a key width, a block of two cells or more, even rotated dims")
	}
	width, block, rotated := geometry.keyWidth, geometry.block, geometry.rotated
	return launchGridSharedABI(state, functions[kernelStrataIndexerKeyAppend],
		driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: uint32(width), Y: 1, Z: 1}, uint32(width)*f64ScalarBytes,
		&raw, &positions, &positionBase, &normWeight, &epsilon, &buffers.tail, &buffers.dead, &buffers.pooled,
		&buffers.blockPos, &width, &block, &rotated, &cosines, &sines, &multiAxis)
}

// launchStrataGateApply is qsa.cu's qsa_gate_apply_f32: attention times the sigmoid of the second half of each
// head's query-and-gate block, in double, unrounded.
func launchStrataGateApply(state *device.State, functions functionSet, attention, queryAndGate driver.DevicePtr, heads, width int32, output driver.DevicePtr) error {
	return launch1DABI(state, functions[kernelStrataGateApplyF32], uint32(heads*width), &attention, &queryAndGate, &heads, &width, &output)
}

// qsa_select.cu's launch constants: 8 warps a block-scores CTA (SCORE_WARPS), the multi-query kernel for calls of at
// most 8 queries (MQ) on a fixed grid of 256 CTAs, the reference top-k's 256 threads (TOPK_T), the register and wide
// top-k's 1,024 threads (TK_T) holding up to 33 blocks' keys a thread in registers (TK_PER, Strata's CUDA value), and
// the 4,608 blocks below which a counted prompt batch past the register reach takes the reference (kWideMinBlocks).
// Provisional for the load-costs step: Strata's constants, tuned on its author's cards.
const (
	strataScoreWarps      = uint32(8)
	strataScoreThreads    = strataScoreWarps * 32
	strataScoresMultiMax  = int64(8)
	strataScoresMultiGrid = uint32(256)
	strataTopKRefThreads  = uint32(256)
	strataTopKThreads     = uint32(1024)
	strataTopKPerThread   = int64(33)
	strataWideMinBlocks   = int64(4608)
)

// launchStrataBlockScores is qsa_block_scores: the indexer score of every block up to each query's n_bid, the tail
// block's from the spare key with +1e9 when it holds cells. A call of at most 8 queries without an active-block count
// (decode, captured or not) takes the multi-query kernel with its early exit; else one CTA row a query over the
// blocks the active count reaches, or the capacity.
func launchStrataBlockScores(
	state *device.State, functions functionSet, pooled, dead, queries, steps driver.DevicePtr, queryCount int64,
	maxBlocks int64, scores driver.DevicePtr, activeBlocks int64,
) error {
	if queryCount <= 0 {
		return nil
	}
	if queryCount <= strataScoresMultiMax && activeBlocks <= 0 {
		count, earlyExit := int32(queryCount), int32(1)
		return launchGridABI(state, functions[kernelStrataBlockScoresMulti],
			driver.Dim3{X: strataScoresMultiGrid, Y: 1, Z: 1}, driver.Dim3{X: strataScoreThreads, Y: 1, Z: 1},
			&pooled, &dead, &queries, &steps, &count, &maxBlocks, &scores, &earlyExit)
	}
	reach := maxBlocks
	if activeBlocks > 0 && activeBlocks < maxBlocks {
		reach = activeBlocks
	}
	return launchGridABI(state, functions[kernelStrataBlockScores],
		driver.Dim3{X: uint32((reach + int64(strataScoreWarps) - 1) / int64(strataScoreWarps)), Y: uint32(queryCount), Z: 1},
		driver.Dim3{X: strataScoreThreads, Y: 1, Z: 1},
		&pooled, &dead, &queries, &steps, &maxBlocks, &scores)
}

// qsa_select.cu's tensor-core scorer: 16 queries a CTA (TC_QT), tiles of 32 blocks (TC_NB), 4 tiles a CTA (TC_ITER),
// 128 threads, its query and key tiles' padded rows in dynamic shared memory; the tail kernel one warp a query.
const (
	strataScoresTCQueries   = int64(16)
	strataScoresTCBlocks    = int64(32 * 4)
	strataScoresTCThreads   = uint32(128)
	strataScoresTailThreads = uint32(32)
)

// launchStrataBlockScoresTC is qsa_block_scores_tc as it runs from sm_80 on: the blocks below each query's n_bid as
// one 3xTF32 GEMM over the blocks the active count reaches (or the capacity), then the tail blocks by the warp kernel's
// arithmetic. The prompt path's scorer; its sums are FP32-level but not the warp kernel's bits.
func launchStrataBlockScoresTC(
	state *device.State, functions functionSet, pooled, dead, queries, steps driver.DevicePtr, queryCount int64,
	maxBlocks int64, scores driver.DevicePtr, activeBlocks int64,
) error {
	if queryCount <= 0 {
		return nil
	}
	reach := maxBlocks
	if activeBlocks > 0 && activeBlocks < maxBlocks {
		reach = activeBlocks
	}
	if err := launchGridSharedABI(state, functions[kernelStrataBlockScoresTc],
		driver.Dim3{X: uint32((reach + strataScoresTCBlocks - 1) / strataScoresTCBlocks), Y: uint32((queryCount + strataScoresTCQueries - 1) / strataScoresTCQueries), Z: 1},
		driver.Dim3{X: strataScoresTCThreads, Y: 1, Z: 1}, strataScoresTCSharedBytes,
		&pooled, &queries, &steps, &queryCount, &maxBlocks, &reach, &scores); err != nil {
		return err
	}
	return launchGridABI(state, functions[kernelStrataBlockScoresTail],
		driver.Dim3{X: uint32(queryCount), Y: 1, Z: 1}, driver.Dim3{X: strataScoresTailThreads, Y: 1, Z: 1},
		&dead, &queries, &steps, &maxBlocks, &scores)
}

// launchStrataBlockTopK is qsa_block_topk as Strata dispatches it on this card's architecture (sm_89): no cluster
// kernel (sm_90 and later) and no counted register dispatch (Turing only), so the blocks a query can have are the
// capacity's; a counted batch past the register reach and under 4,608 blocks takes the reference, past the reach the
// wide kernel, else the register kernel. Every kernel selects the same ids.
func launchStrataBlockTopK(
	state *device.State, functions functionSet, scores, steps driver.DevicePtr, queryCount, maxBlocks, capacity int64,
	ids driver.DevicePtr, activeBlocks int64,
) error {
	if queryCount <= 0 {
		return nil
	}
	fit := int64(strataTopKThreads) * strataTopKPerThread
	if maxBlocks > fit && activeBlocks > 0 && activeBlocks < strataWideMinBlocks {
		return launchStrataBlockTopKRef(state, functions, scores, steps, queryCount, maxBlocks, capacity, ids)
	}
	kernel := kernelStrataBlockTopkReg
	if maxBlocks > fit {
		kernel = kernelStrataBlockTopkWide
	}
	return launchGridABI(state, functions[kernel],
		driver.Dim3{X: uint32(queryCount), Y: 1, Z: 1}, driver.Dim3{X: strataTopKThreads, Y: 1, Z: 1},
		&scores, &steps, &maxBlocks, &capacity, &ids)
}

// launchStrataBlockTopKRef is qsa_block_topk_ref: the original kernel, one 256-thread CTA a query.
func launchStrataBlockTopKRef(
	state *device.State, functions functionSet, scores, steps driver.DevicePtr, queryCount, maxBlocks, capacity int64,
	ids driver.DevicePtr,
) error {
	return launchGridABI(state, functions[kernelStrataBlockTopkRef],
		driver.Dim3{X: uint32(queryCount), Y: 1, Z: 1}, driver.Dim3{X: strataTopKRefThreads, Y: 1, Z: 1},
		&scores, &steps, &maxBlocks, &capacity, &ids)
}

// qsa_decode_attn.cu's geometry: compiled for a 256-wide head and 12 query heads a KV head (core/geometry.hpp), 64
// cells a chunk, 256 threads a CTA; the merge runs one thread a head dimension.
const (
	strataAttentionWidth   = int64(256)
	strataAttentionGroup   = int64(12)
	strataAttentionChunk   = int64(64)
	strataAttentionThreads = uint32(256)
	// strataAttentionScratchPad is qsa_decode_attn_scratch_floats' 64 floats past the partials.
	strataAttentionScratchPad = int64(64)
)

// strataAttentionPools are qsa_decode_attn.hpp's QsaAttnPools for fp16 or int8 pools, behind the page table.
type strataAttentionPools struct {
	kv    strataKVPools
	table driver.DevicePtr
}

// strataAttentionScratchFloats is qsa_decode_attn_scratch_floats: a query's partial accumulators, maxima and sums.
func strataAttentionScratchFloats(capacity int64, heads int64) int64 {
	chunks := (capacity + strataAttentionChunk - 1) / strataAttentionChunk
	return chunks*heads*(strataAttentionWidth+2) + strataAttentionScratchPad
}

// launchStrataDecodeAttention is qsa_decode_attn_batch: queryCount queries [n_q, n_head, 256], each with its own
// selection ids [n_q, capacity] and step record, attention over the selected cells read in place through the page
// table, written to output [n_q, n_head, 256]; the chunk kernel, then the merge. A single query is the batch of one
// (qsa_decode_attn_step's offsets are then zero).
func launchStrataDecodeAttention(
	state *device.State, functions functionSet, queries driver.DevicePtr, pools strataAttentionPools, ids, steps driver.DevicePtr,
	capacity int64, kvHeads, pageSize int32, scratch, output driver.DevicePtr, queryCount int64, int8 bool,
) error {
	if queryCount <= 0 {
		return nil
	}
	heads := int64(kvHeads) * strataAttentionGroup
	if capacity <= 0 || kvHeads <= 0 || pageSize <= 0 || scratch == 0 || ids == 0 || steps == 0 || pools.table == 0 {
		return errors.New("the Strata decode attention: unsupported geometry or missing buffers")
	}
	chunks := int32((capacity + strataAttentionChunk - 1) / strataAttentionChunk)
	stride := strataAttentionScratchFloats(capacity, heads)
	accumulators := scratch
	maxima := scratch + driver.DevicePtr(int64(chunks)*heads*strataAttentionWidth*f32Bytes)
	sums := maxima + driver.DevicePtr(int64(chunks)*heads*f32Bytes)
	scale := float32(1 / math.Sqrt(float64(strataAttentionWidth)))
	capacity32 := int32(capacity)
	grid := driver.Dim3{X: uint32(chunks), Y: uint32(kvHeads), Z: uint32(queryCount)}
	block := driver.Dim3{X: strataAttentionThreads, Y: 1, Z: 1}
	var err error
	if int8 {
		err = launchGridABI(state, functions[kernelStrataAttnChunkQ8], grid, block,
			&queries, &pools.kv.key, &pools.kv.value, &pools.kv.keyScale, &pools.kv.valueScale, &pools.table, &ids, &steps,
			&kvHeads, &pageSize, &scale, &accumulators, &maxima, &sums, &chunks, &capacity32, &stride)
	} else {
		err = launchGridABI(state, functions[kernelStrataAttnChunkF16], grid, block,
			&queries, &pools.kv.key, &pools.kv.value, &pools.table, &ids, &steps,
			&kvHeads, &pageSize, &scale, &accumulators, &maxima, &sums, &chunks, &capacity32, &stride)
	}
	if err != nil {
		return err
	}
	return launchGridABI(state, functions[kernelStrataAttnMerge],
		driver.Dim3{X: uint32(heads), Y: uint32(queryCount), Z: 1}, driver.Dim3{X: uint32(strataAttentionWidth), Y: 1, Z: 1},
		&accumulators, &maxima, &sums, &chunks, &output, &stride)
}

// qsa_prompt_attn.cu's int8 kernel: 128 threads, its dynamic shared memory Smem2 (two cp.async stages of four warps'
// K and V 32-cell slices, their scales, the q.k partials, the probabilities and the per-row softmax state), and at
// most 65,535 queries a launch (the grid's x).
const (
	strataPromptThreads     = uint32(128)
	strataPromptSharedBytes = uint32(2*4*2*32*64 + 2*4*2*32*4 + 4*16*33*4 + 16*33*4 + 4*4 + 3*16*4)
	strataPromptQueryLaunch = int64(65535)
)

// launchStrataPromptAttention is qsa_prompt_attn_batch on int8 pools as Strata dispatches it from sm_80 on
// (launch_i8): queryCount queries, each with its own selection, one CTA a (query, K/V head) with an online softmax
// over its cells on tensor cores, written to output [n_q, n_head, 256].
func launchStrataPromptAttention(
	state *device.State, functions functionSet, queries driver.DevicePtr, pools strataAttentionPools, ids, steps driver.DevicePtr,
	capacity int64, kvHeads, pageSize int32, output driver.DevicePtr, queryCount int64,
) error {
	if capacity <= 0 || kvHeads <= 0 || pageSize <= 0 || ids == 0 || steps == 0 || pools.table == 0 || pools.kv.keyScale == 0 {
		return errors.New("the Strata prompt attention: unsupported geometry or missing int8 pools")
	}
	heads := int64(kvHeads) * strataAttentionGroup
	scaleLog2 := float32(math.Log2E / math.Sqrt(float64(strataAttentionWidth)))
	capacity32 := int32(capacity)
	for first := int64(0); first < queryCount; first += strataPromptQueryLaunch {
		count := min(queryCount-first, strataPromptQueryLaunch)
		q := queries + driver.DevicePtr(first*heads*strataAttentionWidth*f32Bytes)
		selection := ids + driver.DevicePtr(first*capacity*f32Bytes)
		records := steps + driver.DevicePtr(first*int64(strataStepRecord)*f32Bytes)
		out := output + driver.DevicePtr(first*heads*strataAttentionWidth*f32Bytes)
		if err := launchGridSharedABI(state, functions[kernelStrataPromptAttnI8],
			driver.Dim3{X: uint32(count), Y: uint32(kvHeads), Z: 1}, driver.Dim3{X: strataPromptThreads, Y: 1, Z: 1},
			strataPromptSharedBytes,
			&q, &pools.kv.key, &pools.kv.value, &pools.kv.keyScale, &pools.kv.valueScale, &pools.table, &selection, &records,
			&kvHeads, &pageSize, &scaleLog2, &out, &capacity32); err != nil {
			return err
		}
	}
	return nil
}

// strataStepRecord is qsa.hpp's kStepCount: a step record's int32 entries (pos, n_kv, n_bid, width).
const strataStepRecord = 4

// f32Bytes is a float's bytes, the decode attention's scratch element.
const f32Bytes = int64(4)

// f64ScalarBytes is a double's bytes, the indexer pooling's shared-memory element.
const f64ScalarBytes = uint32(8)

// buildStrataRopeTable is rope.cu's build_rope_table, unscaled: cos and sin of position p and pair i at
// `[p][n_rot / 2]`, each angle p * theta^(-2i / n_rot) in float64 in the reference's order, then rounded to float32.
func buildStrataRopeTable(rotated int, theta float64, positions int) (cosines, sines []float32) {
	half := rotated / 2
	cosines, sines = make([]float32, positions*half), make([]float32, positions*half)
	for position := range positions {
		for pair := range half {
			inverse := math.Pow(theta, -2*float64(pair)/float64(rotated))
			angle := float64(position) * inverse
			cosines[position*half+pair] = float32(math.Cos(angle))
			sines[position*half+pair] = float32(math.Sin(angle))
		}
	}
	return cosines, sines
}
