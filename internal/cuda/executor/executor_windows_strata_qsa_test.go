//go:build windows

package executor

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// TestQSAKernelsMatchStrata holds the ported QSA kernels (kernels/cuda/strata_qsa.cu) to Strata's own parity
// programs (Strata at 296b59d2, MIT, licenses/Strata-MIT.txt), transcribed: their host references, their checks and
// their tolerances with the reasons they state.
func TestQSAKernelsMatchStrata(t *testing.T) {
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	t.Run("kv_q8", func(t *testing.T) { strataKVQ8Parity(t, cuda) })
	t.Run("indexer_key_append", func(t *testing.T) { strataIndexerAppendParity(t, cuda) })
	t.Run("native_indexer_batch", func(t *testing.T) { strataNativeIndexerBatchParity(t, cuda) })
	t.Run("block_topk", func(t *testing.T) { strataBlockTopKParity(t, cuda) })
	t.Run("block_scores", func(t *testing.T) { strataBlockScoresParity(t, cuda) })
	t.Run("block_scores_tc", func(t *testing.T) { strataBlockScoresTCParity(t, cuda) })
	t.Run("prompt_shared_memory", func(t *testing.T) {
		// The prompt launch's dynamic shared memory is the compiler's sizeof the kernel's layout.
		buffers := newStrataDevice(t, cuda)
		defer buffers.release(t)
		size := buffers.zeros(t, 4)
		buffers.run(t, func(state *device.State, functions functionSet) error {
			return launchGridABI(state, functions[kernelStrataPromptAttnI8Smem], driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 1, Y: 1, Z: 1}, &size)
		})
		if got := binary.LittleEndian.Uint32(buffers.read(t, size, 4)); got != strataPromptSharedBytes {
			t.Errorf("the prompt kernel's shared layout is %d bytes, the launch gives %d", got, strataPromptSharedBytes)
		}
	})
	for _, c := range []struct {
		name          string
		quantized     bool
		cells, querys int64
	}{
		{"attention/int8/32768", true, 32768, 2048}, {"attention/fp16/32768", false, 32768, 2048},
		{"attention/int8/1500", true, 1500, 1500}, {"attention/int8/2100", true, 2100, 256},
	} {
		t.Run(c.name, func(t *testing.T) { strataAttentionParity(t, cuda, c.quantized, c.cells, c.querys) })
	}
}

// The attention geometry of qsa_real_shapes: 24 query heads, 2 K/V heads, a 256-wide head, 4-cell pages.
const (
	strataQueryHeads = 24
	strataKVHeads    = 2
	strataHeadWidth  = 256
	strataPageCells  = 4
)

// strataAttentionParity is src/kernels/qsa_prompt_attn_parity.cpp's run(): int8 or fp16 pools with random codes,
// scales and values behind a shuffled page table, queries at positions cells - queries .. cells - 1 with selections
// shaped like the prompt path's (the widest 2,051: a recent window of 512 plus older cells drifting about 3% a query;
// short contexts take every cell), the decode attention run in batches of 32 as prefill.cpp calls it and held to an
// FP64 reference on a sample of the queries. The prompt kernel's two checks join when it is ported.
func strataAttentionParity(t *testing.T, cuda *Executor, quantized bool, cells, queries int64) {
	const (
		recent = 512
		batch  = 32
	)
	pages := (cells + strataPageCells - 1) / strataPageCells
	rows := pages * strataKVHeads * strataPageCells
	draws := rand.New(rand.NewPCG(1234, map[bool]uint64{true: 1, false: 0}[quantized]))
	var keyCodes, valueCodes []int8
	var keyScales, valueScales, keyHalves, valueHalves []uint16
	if quantized {
		keyCodes, valueCodes = make([]int8, rows*strataHeadWidth), make([]int8, rows*strataHeadWidth)
		for index := range keyCodes {
			keyCodes[index], valueCodes[index] = int8(draws.IntN(255)-127), int8(draws.IntN(255)-127)
		}
		groups := rows * strataHeadWidth / int64(strataKVQ8Group)
		keyScales, valueScales = make([]uint16, groups), make([]uint16, groups)
		for index := range keyScales {
			keyScales[index] = strataF16FromF32(0.005 + 0.025*draws.Float32())
			valueScales[index] = strataF16FromF32(0.005 + 0.025*draws.Float32())
		}
	} else {
		keyHalves, valueHalves = make([]uint16, rows*strataHeadWidth), make([]uint16, rows*strataHeadWidth)
		for index := range keyHalves {
			keyHalves[index] = strataF16FromF32(float32(draws.NormFloat64()) * 1.5)
			valueHalves[index] = strataF16FromF32(float32(draws.NormFloat64()))
		}
	}
	table := make([]int32, pages)
	for index, page := range draws.Perm(int(pages)) {
		table[index] = int32(page)
	}
	capacity := strataSelectionWidth(cells)
	ids := make([]int32, queries*capacity)
	steps := make([]int32, queries*strataStepCount)
	q := make([]float32, queries*strataQueryHeads*strataHeadWidth)
	for index := range q {
		q[index] = float32(draws.NormFloat64()) * 2
	}
	var older []int32
	for query := range queries {
		position := cells - queries + query
		count := position + 1
		width := strataSelectionWidth(count)
		selection := ids[query*capacity:][:width]
		steps[query*strataStepCount+3] = int32(width)
		if width == count {
			for cell := range selection {
				selection[cell] = int32(cell)
			}
			continue
		}
		olderCount := width - recent
		if int64(len(older)) != olderCount {
			all := draws.Perm(int(count - recent))
			older = make([]int32, olderCount)
			for index := range older {
				older[index] = int32(all[index])
			}
		} else {
			for range olderCount / 32 {
				cell := int32(draws.Int64N(count - recent))
				if !slices.Contains(older, cell) {
					older[draws.Int64N(olderCount)] = cell
				}
			}
		}
		chosen := append(slices.Clone(older), make([]int32, 0, recent)...)
		for cell := count - recent; cell < count; cell++ {
			chosen = append(chosen, int32(cell))
		}
		slices.Sort(chosen)
		copy(selection, chosen)
	}

	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	var pools strataAttentionPools
	if quantized {
		pools.kv = strataKVPools{
			key: buffers.upload(t, int8Bytes(keyCodes)), value: buffers.upload(t, int8Bytes(valueCodes)),
			keyScale: buffers.upload(t, uint16Bytes(keyScales)), valueScale: buffers.upload(t, uint16Bytes(valueScales)),
		}
	} else {
		pools.kv = strataKVPools{key: buffers.upload(t, uint16Bytes(keyHalves)), value: buffers.upload(t, uint16Bytes(valueHalves))}
	}
	pools.table = buffers.upload(t, int32Bytes(table))
	idBuffer, stepBuffer, queryBuffer := buffers.upload(t, int32Bytes(ids)), buffers.upload(t, int32Bytes(steps)), buffers.upload(t, float32Bytes(q))
	outputBytes := int(queries * strataQueryHeads * strataHeadWidth * 4)
	output := buffers.zeros(t, outputBytes)
	scratch := buffers.zeros(t, int(batch*strataAttentionScratchFloats(capacity, strataQueryHeads)*4))
	buffers.run(t, func(state *device.State, functions functionSet) error {
		for first := int64(0); first < queries; first += batch {
			count := min(batch, queries-first)
			if err := launchStrataDecodeAttention(state, functions,
				queryBuffer+driver.DevicePtr(first*strataQueryHeads*strataHeadWidth*4), pools,
				idBuffer+driver.DevicePtr(first*capacity*4), stepBuffer+driver.DevicePtr(first*strataStepCount*4),
				capacity, strataKVHeads, strataPageCells, scratch,
				output+driver.DevicePtr(first*strataQueryHeads*strataHeadWidth*4), count, quantized); err != nil {
				return err
			}
		}
		return nil
	})
	got := float32s(buffers.read(t, output, outputBytes))
	// The prompt kernel on the same queries, int8 pools: qsa_prompt_attn_batch's launch_i8.
	var prompt []float32
	if quantized {
		promptOutput := buffers.zeros(t, outputBytes)
		buffers.run(t, func(state *device.State, functions functionSet) error {
			return launchStrataPromptAttention(state, functions, queryBuffer, pools, idBuffer, stepBuffer, capacity, strataKVHeads, strataPageCells, promptOutput, queries)
		})
		prompt = float32s(buffers.read(t, promptOutput, outputBytes))
	}
	value := func(key bool, row, d int64) float64 {
		switch {
		case quantized && key:
			return float64(keyCodes[row*strataHeadWidth+d]) * float64(strataF32FromF16(keyScales[row*4+d/64]))
		case quantized:
			return float64(valueCodes[row*strataHeadWidth+d]) * float64(strataF32FromF16(valueScales[row*4+d/64]))
		case key:
			return float64(strataF32FromF16(keyHalves[row*strataHeadWidth+d]))
		default:
			return float64(strataF32FromF16(valueHalves[row*strataHeadWidth+d]))
		}
	}
	errorOld, errorNew, referenceScale := 0.0, 0.0, 0.0
	for query := int64(0); query < queries; query += max(1, queries/16) {
		width := int64(steps[query*strataStepCount+3])
		selection := ids[query*capacity:][:width]
		for head := range int64(strataQueryHeads) {
			kvHead := head / (strataQueryHeads / strataKVHeads)
			scores := make([]float64, width)
			maximum := math.Inf(-1)
			for index, cell := range selection {
				row := (int64(table[int64(cell)/strataPageCells])*strataKVHeads+kvHead)*strataPageCells + int64(cell)%strataPageCells
				dot := 0.0
				for d := range int64(strataHeadWidth) {
					dot += float64(q[(query*strataQueryHeads+head)*strataHeadWidth+d]) * value(true, row, d)
				}
				scores[index] = dot / 16
				maximum = max(maximum, scores[index])
			}
			sum := 0.0
			for index := range scores {
				scores[index] = math.Exp(scores[index] - maximum)
				sum += scores[index]
			}
			for d := range int64(strataHeadWidth) {
				accumulated := 0.0
				for index, cell := range selection {
					row := (int64(table[int64(cell)/strataPageCells])*strataKVHeads+kvHead)*strataPageCells + int64(cell)%strataPageCells
					accumulated += scores[index] * value(false, row, d)
				}
				reference := accumulated / sum
				at := (query*strataQueryHeads+head)*strataHeadWidth + d
				referenceScale = max(referenceScale, math.Abs(reference))
				errorOld = max(errorOld, math.Abs(float64(got[at])-reference))
				if prompt != nil {
					errorNew = max(errorNew, math.Abs(float64(prompt[at])-reference))
				}
			}
		}
	}
	// The decode attention in FP32: a 256-product dot a score, __expf weights, an fmaf sum over up to 2,051 cells
	// and a 33-chunk merge; a few thousand roundings of 2^-24 of the output scale, so 1e-3 of it bounds them.
	if !(errorOld <= 1e-3*referenceScale) {
		t.Errorf("the decode attention stands %.3g from the FP64 reference, past 1e-3 of the output scale %.3g", errorOld, referenceScale)
	}
	t.Logf("decode attention against FP64: %.3g (output scale %.3g, %.2g of it)", errorOld, referenceScale, errorOld/referenceScale)
	if prompt == nil {
		return
	}
	// The parity's two checks: 1. against FP64, the prompt kernel's error no larger than four times the decode
	// kernel's (both FP32 math) or 1e-6 of the output scale; 2. the two outputs within 1e-4 of the output scale.
	difference, scale := 0.0, 0.0
	for index := range got {
		difference = max(difference, math.Abs(float64(got[index]-prompt[index])))
		scale = max(scale, math.Abs(float64(got[index])))
	}
	if !(errorNew <= max(4*errorOld, 1e-6*referenceScale)) {
		t.Errorf("the prompt attention stands %.3g from FP64, past four times the decode kernel's %.3g", errorNew, errorOld)
	}
	if !(difference <= 1e-4*scale) {
		t.Errorf("the prompt and decode attention differ by %.3g, past 1e-4 of the output scale %.3g", difference, scale)
	}
	t.Logf("prompt attention against FP64: %.3g; against the decode kernel %.3g (%.2g of the scale)", errorNew, difference, difference/scale)
}

func int8Bytes(values []int8) []byte {
	data := make([]byte, len(values))
	for index, value := range values {
		data[index] = byte(value)
	}
	return data
}

func uint16Bytes(values []uint16) []byte {
	data := make([]byte, 2*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint16(data[2*index:], value)
	}
	return data
}

// strataBlockScoresParity holds the decode's block scorers to qsa_select.cu's statements: the multi-query kernel
// gives block_scores_kernel's bits (per block and query "the same arithmetic in the same order"; "the scores are the
// same bits" with and without its early exit), and both give ref/qsa.py's score, relu per indexer head summed, the
// tail block scoring the spare key with +1e9 when it holds cells, within FP32 rounding of the dot products' terms.
func strataBlockScoresParity(t *testing.T, cuda *Executor) {
	const (
		maxBlocks = 3000
		queries   = 5
	)
	width := strataIndexerWidth
	draws := rand.New(rand.NewPCG(11, 13))
	pooled := make([]float32, maxBlocks*width)
	for index := range pooled {
		pooled[index] = float32(draws.NormFloat64())
	}
	dead := make([]float32, width)
	for index := range dead {
		dead[index] = float32(draws.NormFloat64())
	}
	q := make([]float32, queries*strataIdxHeads*width)
	for index := range q {
		q[index] = float32(draws.NormFloat64())
	}
	// Consecutive and scattered positions, with and without an incomplete tail.
	steps := make([]int32, queries*strataStepCount)
	for query, position := range []int64{4000, 4001, 4003, 9000, 11998} {
		record := steps[query*strataStepCount:]
		record[0], record[1] = int32(position), int32(position+1)
		record[2], record[3] = int32((position+1)/strataIndexerBlock), int32(strataSelectionWidth(position+1))
	}
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	pooledBuffer, deadBuffer, queryBuffer := buffers.upload(t, float32Bytes(pooled)), buffers.upload(t, float32Bytes(dead)), buffers.upload(t, float32Bytes(q))
	stepBuffer := buffers.upload(t, int32Bytes(steps))
	multi, single := buffers.zeros(t, 4*queries*maxBlocks), buffers.zeros(t, 4*queries*maxBlocks)
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataBlockScores(state, functions, pooledBuffer, deadBuffer, queryBuffer, stepBuffer, queries, maxBlocks, multi, -1); err != nil {
			return err
		}
		return launchStrataBlockScores(state, functions, pooledBuffer, deadBuffer, queryBuffer, stepBuffer, queries, maxBlocks, single, maxBlocks)
	})
	gotMulti, gotSingle := float32s(buffers.read(t, multi, 4*queries*maxBlocks)), float32s(buffers.read(t, single, 4*queries*maxBlocks))
	worst, differing := 0.0, 0
	for query := range queries {
		record := steps[query*strataStepCount:]
		cells, blocks := int64(record[1]), int64(record[2])
		for block := int64(0); block <= blocks && block < maxBlocks; block++ {
			at := int64(query)*maxBlocks + block
			if math.Float32bits(gotMulti[at]) != math.Float32bits(gotSingle[at]) {
				differing++
			}
			key := pooled[block*int64(width):][:width]
			if block == blocks {
				key = dead
			}
			score, terms := 0.0, 0.0
			for head := range strataIdxHeads {
				dot, magnitude := 0.0, 0.0
				for d := range width {
					product := float64(key[d]) * float64(q[(query*strataIdxHeads+head)*width+d])
					dot, magnitude = dot+product, magnitude+math.Abs(product)
				}
				score, terms = score+max(dot, 0), terms+magnitude
			}
			if block == blocks && cells%strataIndexerBlock != 0 {
				score += 1e9
				terms += 1e9
			}
			worst = max(worst, math.Abs(float64(gotSingle[at])-score)/max(terms, 1e-30))
		}
	}
	if differing != 0 {
		t.Errorf("the multi-query scores differ from the per-query kernel's bits in %d blocks", differing)
	}
	// Each dot is 128 FP32 products folded by a 32-lane shuffle tree: a few dozen roundings of 2^-24 of its terms.
	if bound := 64.0 / (1 << 24); !(worst <= bound) {
		t.Errorf("block scores stand %.3e from the float64 reference relative to their terms, past %.3e", worst, bound)
	}
	t.Logf("block scores against the float64 reference: %.3e of their terms", worst)
}

// strataBlockScoresTCParity is src/kernels/qsa_select_bench.cpp's accuracy check, its timings left out: 256 queries
// at a 131,072-cell context, keys and queries sharing a direction so the scores carry structure; the warp kernel and
// the tensor-core scorer called as the prompt path calls them (with the call's active block count), both held to an
// FP64 reference on 1,024 blocks of 32 of the queries, the tensor-core error no larger than four times the warp
// kernel's or 1e-5 of the score scale.
func strataBlockScoresTCParity(t *testing.T, cuda *Executor) {
	const (
		cells   = 131072
		queries = 256
		samples = 32
		blocksA = 1024
		leading = 64
	)
	width := strataIndexerWidth
	maxBlocks := int64(cells/strataIndexerBlock + 1)
	draws := rand.New(rand.NewPCG(7, 0))
	direction := make([]float32, width)
	for index := range direction {
		direction[index] = float32(draws.NormFloat64())
	}
	pooled := make([]float32, maxBlocks*int64(width))
	for block := range maxBlocks {
		amplitude := float32(draws.NormFloat64())
		for d := range width {
			pooled[block*int64(width)+int64(d)] = 0.5*amplitude*direction[d] + float32(draws.NormFloat64())
		}
	}
	dead := make([]float32, width)
	for index := range dead {
		dead[index] = float32(draws.NormFloat64())
	}
	q := make([]float32, queries*strataIdxHeads*width)
	for index := range q {
		q[index] = 0.2*direction[index%width] + 0.1*float32(draws.NormFloat64())
	}
	steps := make([]int32, queries*strataStepCount)
	for query := range int64(queries) {
		position := int64(cells) - queries + query
		record := steps[query*strataStepCount:]
		record[0], record[1] = int32(position), int32(position+1)
		record[2], record[3] = int32((position+1)/strataIndexerBlock), int32(strataSelectionWidth(position+1))
	}
	active := int64(steps[(queries-1)*strataStepCount+2]) + 1
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	pooledBuffer, deadBuffer, queryBuffer := buffers.upload(t, float32Bytes(pooled)), buffers.upload(t, float32Bytes(dead)), buffers.upload(t, float32Bytes(q))
	stepBuffer := buffers.upload(t, int32Bytes(steps))
	warpScores, tcScores := buffers.zeros(t, int(4*queries*maxBlocks)), buffers.zeros(t, int(4*queries*maxBlocks))
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataBlockScores(state, functions, pooledBuffer, deadBuffer, queryBuffer, stepBuffer, queries, maxBlocks, warpScores, active); err != nil {
			return err
		}
		return launchStrataBlockScoresTC(state, functions, pooledBuffer, deadBuffer, queryBuffer, stepBuffer, queries, maxBlocks, tcScores, active)
	})
	warp, tc := float32s(buffers.read(t, warpScores, int(4*queries*maxBlocks))), float32s(buffers.read(t, tcScores, int(4*queries*maxBlocks)))
	sampler := rand.New(rand.NewPCG(11, 0))
	errorWarp, errorTC, scale := 0.0, 0.0, 0.0
	for sample := range int64(samples) {
		query := sample * queries / samples
		blocks := int64(steps[query*strataStepCount+2])
		if blocks <= 0 {
			continue
		}
		for pick := range int64(blocksA) {
			block := min(blocks-1, pick)
			if pick >= leading {
				block = sampler.Int64N(blocks)
			}
			reference := 0.0
			for head := range strataIdxHeads {
				dot := 0.0
				for d := range width {
					dot += float64(q[(query*strataIdxHeads+int64(head))*int64(width)+int64(d)]) * float64(pooled[block*int64(width)+int64(d)])
				}
				reference += max(dot, 0)
			}
			at := query*maxBlocks + block
			scale = max(scale, math.Abs(reference))
			errorWarp = max(errorWarp, math.Abs(reference-float64(warp[at])))
			errorTC = max(errorTC, math.Abs(reference-float64(tc[at])))
		}
	}
	if !(errorTC <= max(4*errorWarp, 1e-5*scale)) {
		t.Errorf("the tensor-core block scores stand %.3g from FP64, past four times the warp kernel's %.3g (scale %.3g)", errorTC, errorWarp, scale)
	}
	// The tail blocks are the warp kernel's arithmetic in both: the same bits.
	for query := range int64(queries) {
		at := query*maxBlocks + int64(steps[query*strataStepCount+2])
		if math.Float32bits(warp[at]) != math.Float32bits(tc[at]) {
			t.Errorf("query %d's tail block scores %v on tensor cores, %v in the warp kernel", query, tc[at], warp[at])
			break
		}
	}
	t.Logf("block scores against FP64: warp %.3g, tensor cores %.3g (scale %.3g)", errorWarp, errorTC, scale)
}

// strataIdxHeads is qsa_real_shapes' idx_n_head.
const strataIdxHeads = 4

// strataTopKBudget is qsa_real_shapes' idx_top_k; a query's ids hold the selection width at strataTopKMaxCells cells,
// top_k + idx_block - 1.
const strataTopKBudget = 2048

// strataSelectionWidth is qsa_selection_width: every cell up to the budget's whole blocks plus an incomplete tail.
func strataSelectionWidth(cells int64) int64 {
	return min(cells, strataTopKBudget+strataIndexerBlock-1)
}

// strataBlockTopKParity is src/kernels/qsa_topk_parity.cpp's self-test: the dispatched top-k against the original
// kernel, the selected cells identical, at nine contexts under their capacities (both sides of the register kernel's
// reach of 33,792 blocks, short contexts under a long capacity, the long contexts themselves), as a decode window
// calls it (8 queries, no block count) and as the prompt path does (64 queries with the call's largest block count),
// each over continuous scores, 64 levels (many ties at the threshold) and all-equal scores (ties to the lowest
// index). The parity's timings are no check and are left out.
func strataBlockTopKParity(t *testing.T, cuda *Executor) {
	capacityIDs := strataSelectionWidth(strataTopKMaxCells)
	for _, c := range [][2]int64{
		{9000, 9000}, {135160, 135160}, {135176, 135176}, {2100, 262144}, {9000, 262144}, {32768, 262144},
		{262144, 262144}, {300001, 524288}, {524288, 524288},
	} {
		for _, call := range []struct {
			queries int64
			counted bool
		}{{8, false}, {64, true}} {
			context, capacity := c[0], c[1]
			maxBlocks := capacity/strataIndexerBlock + 2
			steps := make([]int32, call.queries*strataStepCount)
			for query := range call.queries {
				position := context - call.queries + query
				record := steps[query*strataStepCount:]
				record[0], record[1] = int32(position), int32(position+1)
				record[2], record[3] = int32((position+1)/strataIndexerBlock), int32(strataSelectionWidth(position+1))
			}
			draws := rand.New(rand.NewPCG(uint64(context*31+capacity), 3))
			continuous := make([]float32, call.queries*maxBlocks)
			levels, equal := make([]float32, len(continuous)), make([]float32, len(continuous))
			for query := range call.queries {
				record := steps[query*strataStepCount:]
				row := continuous[query*maxBlocks : (query+1)*maxBlocks]
				for block := range row {
					row[block] = float32(draws.NormFloat64())
				}
				if record[1]%strataIndexerBlock != 0 {
					row[record[2]] += 1e9 // the tail block as the scorers leave it
				}
				for block, score := range row {
					level := score
					if score <= 1e8 {
						level = float32(math.Floor(float64(score*8))) / 8 // about 64 levels
					}
					levels[query*maxBlocks+int64(block)] = level
					equal[query*maxBlocks+int64(block)] = 0.25
				}
			}
			active := int64(-1)
			if call.counted {
				active = int64(steps[(call.queries-1)*strataStepCount+2]) + 1
			}
			buffers := newStrataDevice(t, cuda)
			stepBuffer := buffers.upload(t, int32Bytes(steps))
			scoreBuffer := buffers.zeros(t, 4*len(continuous))
			idsRef, idsNew := buffers.zeros(t, int(4*call.queries*capacityIDs)), buffers.zeros(t, int(4*call.queries*capacityIDs))
			unset := make([]byte, 4*call.queries*capacityIDs)
			for index := range unset {
				unset[index] = 0xff
			}
			for name, scores := range map[string][]float32{"continuous": continuous, "64 levels": levels, "equal scores": equal} {
				buffers.write(t, scoreBuffer, float32Bytes(scores))
				buffers.write(t, idsRef, unset)
				buffers.write(t, idsNew, unset)
				buffers.run(t, func(state *device.State, functions functionSet) error {
					if err := launchStrataBlockTopKRef(state, functions, scoreBuffer, stepBuffer, call.queries, maxBlocks, capacityIDs, idsRef); err != nil {
						return err
					}
					return launchStrataBlockTopK(state, functions, scoreBuffer, stepBuffer, call.queries, maxBlocks, capacityIDs, idsNew, active)
				})
				reference := buffers.read(t, idsRef, int(4*call.queries*capacityIDs))
				dispatched := buffers.read(t, idsNew, int(4*call.queries*capacityIDs))
				same := 0
				for query := range call.queries {
					width := int64(steps[query*strataStepCount+3])
					from, to := 4*query*capacityIDs, 4*(query*capacityIDs+width)
					if string(reference[from:to]) == string(dispatched[from:to]) {
						same++
					}
				}
				if same != int(call.queries) {
					t.Errorf("context %d under a capacity of %d (%d blocks), %d queries, counted %v, %s: identical ids for %d of %d queries",
						context, capacity, maxBlocks, call.queries, call.counted, name, same, call.queries)
				}
			}
			buffers.release(t)
		}
	}
}

// The indexer geometry of qsa_real_shapes and qsa.hpp's constants: a 128-wide key, 4-cell blocks, 64 rotated dims,
// freq_base 1e7 (qsa_freq_base) and rms epsilon 1e-6 (qsa_rms_eps).
const (
	strataIndexerWidth   = 128
	strataIndexerBlock   = 4
	strataIndexerRotated = 64
	strataFreqBase       = 1e7
	strataRMSEpsilon     = float32(1e-6)
)

// strataPoolingReading names qsa_parity.cpp's rival readings of the pooling; the zero value is the source's.
type strataPoolingReading struct {
	rotateAtLastCell, normThenPool, spareFromLast, rotateRaw bool
}

// strataRMSNorm is qsa_parity.cpp's ref_rms_norm in double, in place: each square rounded before its add, as the
// kernel's __dadd_rn(ss, __dmul_rn(v, v)) does (the conversions keep Go from fusing them).
func strataRMSNorm(x []float64, weights []float32, epsilon float32) {
	sum := 0.0
	for _, value := range x {
		sum = float64(sum + float64(value*value))
	}
	inverse := 1 / math.Sqrt(sum/float64(len(x))+float64(epsilon))
	for index := range x {
		x[index] = float64(x[index]*inverse) * float64(weights[index])
	}
}

// strataRopeNeox is ref_rope_neox in double: NEOX pairs over the first rotated dims; terms, when set, receives each
// component's |a c| + |b s|, the denominator a cancelling rotation is compared by.
func strataRopeNeox(x []float64, position float64, rotated int, theta float64, terms []float64) {
	half := rotated / 2
	a, b := append([]float64(nil), x[:half]...), append([]float64(nil), x[half:rotated]...)
	if terms != nil {
		for index := rotated; index < len(x); index++ {
			terms[index] = math.Abs(x[index])
		}
	}
	for pair := range half {
		angle := position * math.Pow(theta, -2*float64(pair)/float64(rotated))
		c, s := math.Cos(angle), math.Sin(angle)
		x[pair] = a[pair]*c - b[pair]*s
		x[half+pair] = a[pair]*s + b[pair]*c
		if terms != nil {
			terms[pair] = math.Abs(a[pair]*c) + math.Abs(b[pair]*s)
			terms[half+pair] = math.Abs(a[pair]*s) + math.Abs(b[pair]*c)
		}
	}
}

// strataPooled is ref_pooled: the completed blocks' keys and the spare key, (n_bid + 1) rows of the key width.
func strataPooled(raw [][]float64, positions []float64, weights []float32, reading strataPoolingReading, terms []float64) []float64 {
	blocks, width := len(raw)/strataIndexerBlock, strataIndexerWidth
	pooled := make([]float64, (blocks+1)*width)
	rows := make([][]float64, len(raw))
	for index := range raw {
		rows[index] = append([]float64(nil), raw[index]...)
		if reading.rotateRaw {
			strataRopeNeox(rows[index], positions[index], strataIndexerRotated, strataFreqBase, nil)
		}
	}
	for block := range blocks {
		mean := make([]float64, width)
		for cell := range strataIndexerBlock {
			row := rows[block*strataIndexerBlock+cell]
			if reading.normThenPool {
				row = append([]float64(nil), row...)
				strataRMSNorm(row, weights, strataRMSEpsilon)
			}
			for d := range width {
				mean[d] += row[d] / strataIndexerBlock
			}
		}
		if !reading.normThenPool {
			strataRMSNorm(mean, weights, strataRMSEpsilon)
		}
		at := positions[block*strataIndexerBlock]
		if reading.rotateAtLastCell {
			at = positions[block*strataIndexerBlock+strataIndexerBlock-1]
		}
		var blockTerms []float64
		if terms != nil {
			blockTerms = terms[block*width : (block+1)*width]
		}
		strataRopeNeox(mean, at, strataIndexerRotated, strataFreqBase, blockTerms)
		copy(pooled[block*width:], mean)
	}
	spare := append([]float64(nil), rows[0]...)
	if reading.spareFromLast {
		spare = append([]float64(nil), rows[len(rows)-1]...)
	}
	strataRMSNorm(spare, weights, strataRMSEpsilon)
	strataRopeNeox(spare, 0, strataIndexerRotated, strataFreqBase, nil)
	copy(pooled[blocks*width:], spare)
	return pooled
}

// strataIndexerAppendParity is qsa_parity.cpp's section 2: ten cells (two complete blocks and a two-cell tail) at
// positions 100..109, appended one at a time, with cell magnitudes 1, 4, 16 and 64 so pool-then-norm and
// norm-then-pool cannot agree by accident; the spare key BIT-EXACT, the pooled blocks within 1e-6 of the rotation's
// terms, the tail not pooled, and each rival reading and the position base required to differ.
func strataIndexerAppendParity(t *testing.T, cuda *Executor) {
	const (
		cells          = 10
		positionBase   = 100
		tablePositions = 2000
	)
	width := strataIndexerWidth
	draws := rand.New(rand.NewPCG(5, 7))
	raw := make([][]float64, cells)
	positions := make([]float64, cells)
	flat := make([]float32, 0, cells*width)
	for cell := range cells {
		raw[cell] = make([]float64, width)
		for d := range width {
			value := float32(draws.NormFloat64() * math.Pow(4, float64(cell%4)))
			flat = append(flat, value)
			raw[cell][d] = float64(value) // the reference takes the f32 promotion of what the device takes
		}
		positions[cell] = float64(positionBase + cell)
	}
	weights := make([]float32, width)
	for index := range weights {
		weights[index] = 1 + float32(0.1*draws.NormFloat64())
	}
	cosines, sines := tensor.QSARopeTable(tablePositions)

	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	blocks := cells / strataIndexerBlock
	indexer := strataIndexerBuffers{
		tail: buffers.zeros(t, 4*(strataIndexerBlock-1)*width), dead: buffers.zeros(t, 4*width),
		pooled: buffers.zeros(t, 4*(blocks+1)*width), blockPos: buffers.zeros(t, 4),
	}
	normWeight, cosTable, sinTable := buffers.upload(t, float32Bytes(weights)), buffers.upload(t, float32Bytes(cosines)), buffers.upload(t, float32Bytes(sines))
	rawRow, position := buffers.zeros(t, 4*width), buffers.zeros(t, 4)
	geometry := strataIndexerGeometry{keyWidth: int32(width), block: strataIndexerBlock, rotated: strataIndexerRotated}
	for cell := range cells {
		buffers.write(t, rawRow, float32Bytes(flat[cell*width:(cell+1)*width]))
		buffers.write(t, position, int32Bytes([]int32{int32(cell)}))
		buffers.run(t, func(state *device.State, functions functionSet) error {
			return launchStrataIndexerKeyAppend(state, functions, rawRow, position, positionBase, normWeight, strataRMSEpsilon, indexer, geometry, cosTable, sinTable, 0)
		})
	}
	got := float32s(buffers.read(t, indexer.pooled, 4*(blocks+1)*width))
	gotSpare := float32s(buffers.read(t, indexer.dead, 4*width))
	gotBlockPos := int32(binary.LittleEndian.Uint32(buffers.read(t, indexer.blockPos, 4)))
	terms := make([]float64, (blocks+1)*width)
	want := strataPooled(raw, positions, weights, strataPoolingReading{}, terms)

	// (a) the spare key, BIT-EXACT: rope at position 0 is the identity in both.
	spareWrong := 0
	for d := range width {
		if gotSpare[d] != float32(want[blocks*width+d]) {
			spareWrong++
		}
	}
	if spareWrong != 0 {
		t.Errorf("the spare slot's key (rms_norm of cell 0) differs in %d of %d values", spareWrong, width)
	}
	// (b) the pooled blocks, relative to the rotation's terms (a*cos - b*sin cancels).
	worst := 0.0
	for index := range blocks * width {
		worst = max(worst, math.Abs(want[index]-float64(got[index]))/max(terms[index], 1e-30))
	}
	if !(worst <= 1e-6) {
		t.Errorf("pooled block keys stand %.3e from the reference relative to their rope terms, past 1e-6", worst)
	}
	t.Logf("pooled block keys against the reference: %.3e of their rope terms", worst)
	if wantBlockPos := int32(positionBase + (blocks-1)*strataIndexerBlock); gotBlockPos != wantBlockPos {
		t.Errorf("the last completed block's rotation position is %d, want %d", gotBlockPos, wantBlockPos)
	}
	// (c) the incomplete tail is NOT pooled into the spare slot.
	tailMean := make([]float64, width)
	for cell := blocks * strataIndexerBlock; cell < cells; cell++ {
		for d := range width {
			tailMean[d] += raw[cell][d] / 2
		}
	}
	strataRMSNorm(tailMean, weights, strataRMSEpsilon)
	if share := strataRelativeL1(tailMean, float32To64(gotSpare)); !(share > 0.5) {
		t.Errorf("the spare slot holds the tail's mean (%.2f%% from it), want cell 0's key", 100*share)
	}
	// Each rival reading, and the position base, is observable: required to differ by more than 5%.
	for name, reading := range map[string]strataPoolingReading{
		"rotating at the block's last cell":   {rotateAtLastCell: true},
		"norm-then-pool for pool-then-norm":   {normThenPool: true},
		"the spare key from the last raw key": {spareFromLast: true},
		"rotating the raw indexer keys":       {rotateRaw: true},
	} {
		if share := strataRelativeL1(strataPooled(raw, positions, weights, reading, nil), want); !(share > 0.05) {
			t.Errorf("%s differs by %.2f%%, too little for the fixture to see it", name, 100*share)
		}
	}
	flatPositions := make([]float64, cells)
	for cell := range cells {
		flatPositions[cell] = float64(cell)
	}
	if share := strataRelativeL1(strataPooled(raw, flatPositions, weights, strataPoolingReading{}, nil), want); !(share > 0.05) {
		t.Errorf("the rotation's position base differs by %.2f%%, too little for the fixture to see it", 100*share)
	}
}

// strataNativeIndexerBatchParity is qsa_parity.cpp's "the native indexer's batched append vs the sequential one": eight
// cells at positions 100..107 (exactly two completed blocks, so block_pos agrees too), appended one at a time, as one
// batch, and as one per-step launch at position strides 1 and 5; every end state - pooled rows, the spare, the tail,
// block_pos - BIT-IDENTICAL to the sequential appends' (compared as bits, so a NaN payload or a -0.0 cannot pass),
// under no scaling, linear 2 and YaRN 2.
func strataNativeIndexerBatchParity(t *testing.T, cuda *Executor) {
	const (
		cells    = 8
		base     = 100
		capacity = 1024
	)
	width := strataIndexerWidth
	draws := rand.New(rand.NewPCG(23, 29))
	raws := make([]float32, cells*width)
	for cell := range cells {
		for d := range width {
			raws[cell*width+d] = float32(draws.NormFloat64() * math.Pow(4, float64(cell%4)))
		}
	}
	gamma := make([]float32, width)
	for index := range gamma {
		gamma[index] = 1 + float32(0.1*draws.NormFloat64())
	}
	pooledBytes, tailBytes := 4*(capacity/strataIndexerBlock+1)*width, 4*(strataIndexerBlock-1)*width
	for _, variant := range []struct {
		name    string
		scaling strataRopeScaling
	}{
		{"none", newStrataRopeScaling(0, 1, 0)}, {"linear 2", newStrataRopeScaling(1, 2, 0)}, {"yarn 2", newStrataRopeScaling(2, 2, 1)},
	} {
		buffers := newStrataDevice(t, cuda)
		gammaBuffer, rawAll := buffers.upload(t, float32Bytes(gamma)), buffers.upload(t, float32Bytes(raws))
		fresh := func() strataIndexerBuffers {
			return strataIndexerBuffers{tail: buffers.zeros(t, tailBytes), dead: buffers.zeros(t, 4*width), pooled: buffers.zeros(t, pooledBytes), blockPos: buffers.zeros(t, 4)}
		}
		state := func(b strataIndexerBuffers) string {
			return string(buffers.read(t, b.pooled, pooledBytes)) + string(buffers.read(t, b.dead, 4*width)) +
				string(buffers.read(t, b.tail, tailBytes)) + string(buffers.read(t, b.blockPos, 4))
		}
		sequential, batch := fresh(), fresh()
		row, position := buffers.zeros(t, 4*width), buffers.zeros(t, 4)
		for cell := range cells {
			buffers.write(t, row, float32Bytes(raws[cell*width:(cell+1)*width]))
			buffers.write(t, position, int32Bytes([]int32{int32(cell)}))
			buffers.run(t, func(s *device.State, functions functionSet) error {
				return launchStrataNativeIndexerSteps(s, functions, row, position, 1, 1, base, gammaBuffer, strataRMSEpsilon, sequential, capacity, variant.scaling, 0)
			})
		}
		buffers.run(t, func(s *device.State, functions functionSet) error {
			return launchStrataNativeIndexerBatch(s, functions, rawAll, cells, 0, base, gammaBuffer, strataRMSEpsilon, batch, variant.scaling, 0)
		})
		want := state(sequential)
		if state(batch) != want {
			t.Errorf("%s: the batched append's end state differs from the sequential appends'", variant.name)
		}
		for _, stride := range []int32{1, 5} {
			steps := fresh()
			positions := make([]int32, cells*int(stride))
			for index := range positions {
				positions[index] = -1
			}
			for cell := range cells {
				positions[cell*int(stride)] = int32(cell)
			}
			positionBuffer := buffers.upload(t, int32Bytes(positions))
			buffers.run(t, func(s *device.State, functions functionSet) error {
				return launchStrataNativeIndexerSteps(s, functions, rawAll, positionBuffer, stride, cells, base, gammaBuffer, strataRMSEpsilon, steps, capacity, variant.scaling, 0)
			})
			if state(steps) != want {
				t.Errorf("%s: the per-step append at stride %d differs from the sequential appends'", variant.name, stride)
			}
		}
		buffers.release(t)
	}
}

// strataRelativeL1 is the parity's sum |a - b| over sum |b|.
func strataRelativeL1(a, b []float64) float64 {
	var difference, magnitude float64
	for index := range b {
		difference += math.Abs(a[index] - b[index])
		magnitude += math.Abs(b[index])
	}
	return difference / magnitude
}

func float32s(data []byte) []float32 {
	values := make([]float32, len(data)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*index:]))
	}
	return values
}

func float32To64(values []float32) []float64 {
	wide := make([]float64, len(values))
	for index, value := range values {
		wide[index] = float64(value)
	}
	return wide
}

// strataKVQ8Parity is src/kernels/kv_q8_parity.cpp: random K/V cells appended at scattered positions through a
// non-identity page table into both the int8 pools and the fp16 pools, random cell sets gathered from both, and:
//  1. int8 codes and fp16 scales BITWISE equal to the host reference (the same fp16 rounding);
//  2. the int8 gather BITWISE equal to the host dequantization of those codes;
//  3. against the fp16 path, the int8 values within half a quantization step of their group plus the fp16
//     rounding of both sides (0.624 steps).
func strataKVQ8Parity(t *testing.T, cuda *Executor) {
	const (
		heads, width, page = 2, 256, 64 // qsa_real_shapes' K/V heads and head width; the parity's page of 64 cells
		pages              = 8
		cells              = pages * page
		empty              = 37 // cells left unwritten
		maxIDs             = 200
		trials             = 20
	)
	groups := width / int(strataKVQ8Group)
	geometry := strataKVGeometry{kvHeads: heads, headDim: width, pageSize: page}
	table := make([]int32, pages)
	for index := range table {
		table[index] = int32((index*5 + 3) % pages) // a non-identity permutation
	}
	draws := rand.New(rand.NewPCG(17, 0))
	positions := draws.Perm(cells)
	fill := cells - empty
	hostKeys, hostValues := make([][]float32, cells), make([][]float32, cells)

	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	tableBuffer := buffers.upload(t, int32Bytes(table))
	q8 := strataKVPools{
		key: buffers.zeros(t, cells*heads*width), value: buffers.zeros(t, cells*heads*width),
		keyScale: buffers.zeros(t, 2*cells*heads*groups), valueScale: buffers.zeros(t, 2*cells*heads*groups),
	}
	f16 := strataKVPools{key: buffers.zeros(t, 2*cells*heads*width), value: buffers.zeros(t, 2*cells*heads*width)}
	keyRow, valueRow := buffers.zeros(t, 4*heads*width), buffers.zeros(t, 4*heads*width)
	step := buffers.zeros(t, 4*strataStepCount)
	for n := range fill {
		position := positions[n]
		keys, values := make([]float32, heads*width), make([]float32, heads*width)
		scale := float32(1)
		switch {
		case n%7 == 0:
			scale = 1e-3 // a small group
		case n%11 == 0:
			scale = 40 // a large group
		}
		for index := range keys {
			keys[index] = float32(draws.NormFloat64()) * scale
			values[index] = float32(draws.NormFloat64()) * scale
		}
		if n%13 == 0 {
			clear(keys[:strataKVQ8Group]) // an all-zero group
		}
		hostKeys[position], hostValues[position] = keys, values
		buffers.write(t, step, int32Bytes([]int32{int32(position), int32(position + 1), 0, 0}))
		buffers.write(t, keyRow, float32Bytes(keys))
		buffers.write(t, valueRow, float32Bytes(values))
		buffers.run(t, func(state *device.State, functions functionSet) error {
			if err := launchStrataKVAppend(state, functions, q8, strataKVPools{}, tableBuffer, step, keyRow, valueRow, geometry, true); err != nil {
				return err
			}
			return launchStrataKVAppend(state, functions, f16, strataKVPools{}, tableBuffer, step, keyRow, valueRow, geometry, false)
		})
	}

	// 1. codes and scales bitwise against the host reference.
	keyCodes, valueCodes := buffers.read(t, q8.key, cells*heads*width), buffers.read(t, q8.value, cells*heads*width)
	keyScales := uint16s(buffers.read(t, q8.keyScale, 2*cells*heads*groups))
	valueScales := uint16s(buffers.read(t, q8.valueScale, 2*cells*heads*groups))
	badCodes := 0
	for n := range fill {
		position := positions[n]
		for half, rows := range [][]float32{hostKeys[position], hostValues[position]} {
			codes, scales := keyCodes, keyScales
			if half == 1 {
				codes, scales = valueCodes, valueScales
			}
			for head := range heads {
				row := (int(table[position/page])*heads+head)*page + position%page
				for group := range groups {
					x := rows[head*width+group*int(strataKVQ8Group):][:strataKVQ8Group]
					var maximum float32
					for _, value := range x {
						maximum = max(maximum, float32(math.Abs(float64(value))))
					}
					bits := strataF16FromF32(maximum / 127)
					stored := strataF32FromF16(bits)
					if scales[row*groups+group] != bits {
						badCodes++
					}
					for index, value := range x {
						code := 0
						if stored > 0 {
							code = min(max(int(math.RoundToEven(float64(value/stored))), -127), 127)
						}
						if int8(codes[row*width+group*int(strataKVQ8Group)+index]) != int8(code) {
							badCodes++
						}
					}
				}
			}
		}
	}
	if badCodes != 0 {
		t.Errorf("%d codes or scales differ from the host reference", badCodes)
	}

	// 2 and 3. gather random selections from both paths.
	ids := buffers.zeros(t, 4*maxIDs)
	gathered := [4]driver.DevicePtr{}
	for index := range gathered {
		gathered[index] = buffers.zeros(t, 2*maxIDs*heads*width)
	}
	worst := 0.0
	badGather := 0
	for range trials {
		count := 1 + draws.IntN(maxIDs)
		selection := make([]int32, count)
		for index := range selection {
			selection[index] = int32(positions[draws.IntN(fill)])
		}
		buffers.write(t, ids, int32Bytes(selection))
		buffers.write(t, step, int32Bytes([]int32{0, 0, 0, int32(count)}))
		buffers.run(t, func(state *device.State, functions functionSet) error {
			if err := launchStrataKVGather(state, functions, q8, tableBuffer, ids, step, maxIDs, geometry, gathered[0], gathered[1], true); err != nil {
				return err
			}
			return launchStrataKVGather(state, functions, f16, tableBuffer, ids, step, maxIDs, geometry, gathered[2], gathered[3], false)
		})
		keys8, values8 := uint16s(buffers.read(t, gathered[0], 2*count*heads*width)), uint16s(buffers.read(t, gathered[1], 2*count*heads*width))
		keys16, values16 := uint16s(buffers.read(t, gathered[2], 2*count*heads*width)), uint16s(buffers.read(t, gathered[3], 2*count*heads*width))
		for id, cell := range selection {
			for head := range heads {
				row := (int(table[int(cell)/page])*heads+head)*page + int(cell)%page
				for d := range width {
					at := (id*heads+head)*width + d
					for half := range 2 {
						codes, scales, got, reference := keyCodes, keyScales, keys8, keys16
						if half == 1 {
							codes, scales, got, reference = valueCodes, valueScales, values8, values16
						}
						stored := strataF32FromF16(scales[row*groups+d/int(strataKVQ8Group)])
						want := strataF16FromF32(float32(int8(codes[row*width+d])) * stored)
						if got[at] != want {
							badGather++
						}
						unit := float64(stored)
						if unit <= 0 {
							unit = 1
						}
						worst = max(worst, math.Abs(float64(strataF32FromF16(got[at])-strataF32FromF16(reference[at])))/unit)
					}
				}
			}
		}
	}
	if badGather != 0 {
		t.Errorf("%d gathered int8 values differ from the host dequantization", badGather)
	}
	// |x - q*sf| <= sf/2 (the stored scale keeps |x|/sf <= 127.06, so no clamping), plus the fp16 rounding of each
	// side: at most 2^-11 relative of a value up to 127*sf, 127/2048 = 0.062 steps each, so 0.5 + 0.124.
	if bound := 0.5 + 2*127.0/2048 + 1e-3; !(worst <= bound) {
		t.Errorf("int8 against fp16 error %.3f steps, past %.3f", worst, bound)
	}
	t.Logf("worst int8-against-fp16 error %.3f quantization steps", worst)
}

// strataStepCount is qsa.hpp's kStepCount: pos, n_kv, n_bid, width.
const strataStepCount = 4

// strataF16FromF32 is f16_bits.hpp's f16_from_f32, the host side of the parity: round-to-nearest-even, finite
// overflow to inf, NaN kept.
func strataF16FromF32(value float32) uint16 {
	x := math.Float32bits(value)
	sign := (x >> 16) & 0x8000
	raw := (x >> 23) & 0xFF
	exponent := int(raw) - 127 + 15
	mantissa := x & 0x7FFFFF
	if raw == 0xFF {
		nan := uint32(0)
		if mantissa != 0 {
			nan = 0x200
		}
		return uint16(sign | 0x7C00 | nan)
	}
	if exponent >= 31 {
		return uint16(sign | 0x7C00)
	}
	if exponent <= 0 {
		if exponent < -10 {
			return uint16(sign)
		}
		mantissa |= 0x800000
		shift := uint32(14 - exponent)
		half := (mantissa >> shift) & 0x3FF
		remainder := mantissa & (1<<shift - 1)
		if remainder > 1<<(shift-1) || remainder == 1<<(shift-1) && half&1 != 0 {
			half++
		}
		return uint16(sign | half)
	}
	half := uint16(sign | uint32(exponent)<<10 | mantissa>>13)
	remainder := mantissa & 0x1FFF
	if remainder > 0x1000 || remainder == 0x1000 && half&1 != 0 {
		half++
	}
	return half
}

// strataF32FromF16 is f16_bits.hpp's f32_from_f16: exact, subnormals kept.
func strataF32FromF16(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exponent, mantissa := uint32(bits>>10)&0x1F, uint32(bits)&0x3FF
	var out uint32
	switch {
	case exponent == 0 && mantissa == 0:
		out = sign
	case exponent == 0:
		shift := uint32(0)
		for mantissa&0x400 == 0 {
			mantissa <<= 1
			shift++
		}
		out = sign | (113-shift)<<23 | (mantissa&0x3FF)<<13
	case exponent == 31:
		out = sign | 0x7F800000 | mantissa<<13
	default:
		out = sign | (exponent-15+127)<<23 | mantissa<<13
	}
	return math.Float32frombits(out)
}

// strataDevice holds a parity's device buffers on the executor's worker, freed together.
type strataDevice struct {
	cuda     *Executor
	pointers []driver.DevicePtr
}

func newStrataDevice(t *testing.T, cuda *Executor) *strataDevice {
	t.Helper()
	return &strataDevice{cuda: cuda}
}

// zeros allocates bytes of zeroed device memory.
func (d *strataDevice) zeros(t *testing.T, bytes int) driver.DevicePtr {
	t.Helper()
	return d.upload(t, make([]byte, bytes))
}

func (d *strataDevice) upload(t *testing.T, data []byte) driver.DevicePtr {
	t.Helper()
	var pointer driver.DevicePtr
	d.do(t, func(state *device.State) error {
		var err error
		if pointer, err = state.Driver.MemAlloc(uint64(len(data))); err != nil {
			return err
		}
		d.pointers = append(d.pointers, pointer)
		return state.Driver.MemcpyHtoD(pointer, data)
	})
	return pointer
}

func (d *strataDevice) write(t *testing.T, pointer driver.DevicePtr, data []byte) {
	t.Helper()
	d.do(t, func(state *device.State) error { return state.Driver.MemcpyHtoD(pointer, data) })
}

func (d *strataDevice) read(t *testing.T, pointer driver.DevicePtr, bytes int) []byte {
	t.Helper()
	data := make([]byte, bytes)
	d.do(t, func(state *device.State) error { return state.Driver.MemcpyDtoH(data, pointer) })
	return data
}

// run launches on the executor's stream with its loaded kernels, then waits for the stream.
func (d *strataDevice) run(t *testing.T, launch func(*device.State, functionSet) error) {
	t.Helper()
	d.do(t, func(state *device.State) error {
		resources, err := d.cuda.ensureResources(state, false, 0, 0, 0)
		if err != nil {
			return err
		}
		if err := launch(state, resources.functions); err != nil {
			return err
		}
		return state.Driver.StreamSynchronize(state.Stream)
	})
}

func (d *strataDevice) do(t *testing.T, work func(*device.State) error) {
	t.Helper()
	if err := d.cuda.worker.Do(t.Context(), work); err != nil {
		t.Fatal(err)
	}
}

func (d *strataDevice) release(t *testing.T) {
	t.Helper()
	d.do(t, func(state *device.State) error {
		for _, pointer := range d.pointers {
			if err := state.Driver.MemFree(pointer); err != nil {
				return err
			}
		}
		return nil
	})
}

func int32Bytes(values []int32) []byte {
	data := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[4*index:], uint32(value))
	}
	return data
}

func float32Bytes(values []float32) []byte {
	data := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[4*index:], math.Float32bits(value))
	}
	return data
}

func uint16s(data []byte) []uint16 {
	values := make([]uint16, len(data)/2)
	for index := range values {
		values[index] = binary.LittleEndian.Uint16(data[2*index:])
	}
	return values
}
