//go:build windows

package executor

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// strataCellKernels are one cell attention instantiation's entries (kernels/cuda/strata_qsa.cu STRATA_CELL_ENTRIES).
type strataCellKernels struct {
	chunk, merge, prompt, sharedBytes kernelFunctionID
}

// strataCellInstantiations are the cell attention's entries by head width and query heads a K/V head.
var strataCellInstantiations = map[[2]int64]strataCellKernels{
	{128, 8}:  {kernelStrataCellAttnChunkQ8128x8, kernelStrataCellAttnMerge128x8, kernelStrataCellPromptAttnI8128x8, kernelStrataCellPromptAttnI8Smem128x8},
	{256, 2}:  {kernelStrataCellAttnChunkQ8256x2, kernelStrataCellAttnMerge256x2, kernelStrataCellPromptAttnI8256x2, kernelStrataCellPromptAttnI8Smem256x2},
	{512, 16}: {kernelStrataCellAttnChunkQ8512x16, kernelStrataCellAttnMerge512x16, kernelStrataCellPromptAttnI8512x16, kernelStrataCellPromptAttnI8Smem512x16},
}

// strataCellGeometry is an attention layer's cells: K/V heads of width, group query heads each, pages of pageCells.
type strataCellGeometry struct {
	width, group, kvHeads, pageCells int64
}

// launchStrataCellDecode is launchStrataDecodeAttention at the layer's geometry: the int8 chunks over each query's
// selection, or over the range of cells its step record names where ids is zero, then the merge.
func launchStrataCellDecode(state *device.State, functions functionSet, geometry strataCellGeometry, queries driver.DevicePtr,
	pools strataAttentionPools, ids, steps driver.DevicePtr, capacity int64, scratch, output driver.DevicePtr, queryCount int64,
) error {
	kernels := strataCellInstantiations[[2]int64{geometry.width, geometry.group}]
	heads := geometry.kvHeads * geometry.group
	chunks := int32((capacity + strataAttentionChunk - 1) / strataAttentionChunk)
	stride := int64(chunks) * heads * (geometry.width + 2)
	accumulators := scratch
	maxima := scratch + driver.DevicePtr(int64(chunks)*heads*geometry.width*f32Bytes)
	sums := maxima + driver.DevicePtr(int64(chunks)*heads*f32Bytes)
	scale := float32(1 / math.Sqrt(float64(geometry.width)))
	kvHeads, pageCells, capacity32 := int32(geometry.kvHeads), int32(geometry.pageCells), int32(capacity)
	if err := launchGridABI(state, functions[kernels.chunk],
		driver.Dim3{X: uint32(chunks), Y: uint32(kvHeads), Z: uint32(queryCount)}, driver.Dim3{X: strataAttentionThreads, Y: 1, Z: 1},
		&queries, &pools.kv.key, &pools.kv.value, &pools.kv.keyScale, &pools.kv.valueScale, &pools.table, &ids, &steps,
		&kvHeads, &pageCells, &scale, &accumulators, &maxima, &sums, &chunks, &capacity32, &stride); err != nil {
		return err
	}
	return launchGridABI(state, functions[kernels.merge],
		driver.Dim3{X: uint32(heads), Y: uint32(queryCount), Z: 1}, driver.Dim3{X: uint32(geometry.width), Y: 1, Z: 1},
		&accumulators, &maxima, &sums, &chunks, &output, &stride)
}

// launchStrataCellPrompt is launchStrataPromptAttention at the layer's geometry, its shared memory sharedBytes.
func launchStrataCellPrompt(state *device.State, functions functionSet, geometry strataCellGeometry, queries driver.DevicePtr,
	pools strataAttentionPools, ids, steps driver.DevicePtr, capacity int64, sharedBytes uint32, output driver.DevicePtr, queryCount int64,
) error {
	kernels := strataCellInstantiations[[2]int64{geometry.width, geometry.group}]
	if err := state.Driver.FuncSetMaxDynamicShared(functions[kernels.prompt].function, sharedBytes); err != nil {
		return err
	}
	scaleLog2 := float32(math.Log2E / math.Sqrt(float64(geometry.width)))
	kvHeads, pageCells, capacity32 := int32(geometry.kvHeads), int32(geometry.pageCells), int32(capacity)
	return launchGridSharedABI(state, functions[kernels.prompt],
		driver.Dim3{X: uint32(queryCount), Y: uint32(kvHeads), Z: 1}, driver.Dim3{X: uint32(geometry.width / 2), Y: 1, Z: 1},
		sharedBytes, &queries, &pools.kv.key, &pools.kv.value, &pools.kv.keyScale, &pools.kv.valueScale, &pools.table, &ids, &steps,
		&kvHeads, &pageCells, &scaleLog2, &output, &capacity32)
}

// TestStrataCellAttentionGeometries holds Strata's int8 cell attention, instantiated for an attention layer's own
// geometry (widen-kv-kernels), to Strata's parity program for its own (qsa_prompt_attn_parity.cpp, as
// strataAttentionParity transcribes it): random codes and scales behind a shuffled page table, the decode attention
// within 1e-3 of the output scale of an FP64 attention over the dequantized cells, the prompt attention within four
// times the decode attention's error and within 1e-4 of the output scale of it. Each geometry is read three ways: a
// selection, as Strata's QSA layer reads, a causal prefix and a sliding window, both a range of cells its step record
// names. The geometries are gemma-4-12B's sliding layers (8 K/V heads of 256, 2 query heads each, a 1,024-token
// window), its global layers (one K/V head of 512, 16 query heads) and a llama layer's (2 K/V heads of 128, 8 each);
// each instantiation's shared memory is the compiler's size of its layout.
func TestStrataCellAttentionGeometries(t *testing.T) {
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	const window = 1024
	for _, c := range []struct {
		name     string
		geometry strataCellGeometry
	}{
		{"gemma-4-12B sliding", strataCellGeometry{width: 256, group: 2, kvHeads: 8, pageCells: 16}},
		{"gemma-4-12B global", strataCellGeometry{width: 512, group: 16, kvHeads: 1, pageCells: 16}},
		{"llama", strataCellGeometry{width: 128, group: 8, kvHeads: 2, pageCells: 4}},
	} {
		for _, mode := range []string{"selection", "causal", "window"} {
			t.Run(c.name+"/"+mode, func(t *testing.T) { strataCellParity(t, cuda, c.geometry, mode, window) })
		}
	}
}

func strataCellParity(t *testing.T, cuda *Executor, geometry strataCellGeometry, mode string, window int64) {
	const cells, queries = 2500, 96
	width, heads := geometry.width, geometry.kvHeads*geometry.group
	pages := (cells + geometry.pageCells - 1) / geometry.pageCells
	rows := pages * geometry.kvHeads * geometry.pageCells
	groups := width / int64(strataKVQ8Group)
	draws := rand.New(rand.NewPCG(uint64(width), uint64(geometry.group)))
	keyCodes, valueCodes := make([]int8, rows*width), make([]int8, rows*width)
	for index := range keyCodes {
		keyCodes[index], valueCodes[index] = int8(draws.IntN(255)-127), int8(draws.IntN(255)-127)
	}
	keyScales, valueScales := make([]uint16, rows*groups), make([]uint16, rows*groups)
	for index := range keyScales {
		keyScales[index] = strataF16FromF32(0.005 + 0.025*draws.Float32())
		valueScales[index] = strataF16FromF32(0.005 + 0.025*draws.Float32())
	}
	table := make([]int32, pages)
	for index, page := range draws.Perm(int(pages)) {
		table[index] = int32(page)
	}
	// Each query at its position reads its cells: a selection of a recent stretch and older cells, its causal prefix,
	// or its sliding window; the capacity is the most any query reads.
	read := func(position int64) (first, count int64) {
		if mode == "window" {
			first = max(0, position+1-window)
		}
		return first, position + 1 - first
	}
	capacity := int64(0)
	for query := range int64(queries) {
		_, count := read(cells - queries + query)
		capacity = max(capacity, min(count, 700))
	}
	if mode != "selection" {
		capacity = 0
		for query := range int64(queries) {
			_, count := read(cells - queries + query)
			capacity = max(capacity, count)
		}
	}
	selections := make([][]int32, queries)
	steps := make([]int32, queries*strataStepCount)
	var ids []int32
	if mode == "selection" {
		ids = make([]int32, queries*capacity)
	}
	for query := range int64(queries) {
		position := cells - queries + query
		first, count := read(position)
		steps[query*strataStepCount] = int32(position)
		steps[query*strataStepCount+1] = int32(position + 1)
		if mode == "selection" {
			chosen := draws.Perm(int(count))[:min(count, capacity)]
			selection := make([]int32, len(chosen))
			for index, cell := range chosen {
				selection[index] = int32(cell)
			}
			selections[query] = selection
			copy(ids[query*capacity:], selection)
			steps[query*strataStepCount+3] = int32(len(selection))
			continue
		}
		selection := make([]int32, count)
		for index := range selection {
			selection[index] = int32(first + int64(index))
		}
		selections[query] = selection
		steps[query*strataStepCount+2] = int32(first)
		steps[query*strataStepCount+3] = int32(count)
	}
	q := make([]float32, queries*heads*width)
	for index := range q {
		q[index] = float32(draws.NormFloat64()) * 2
	}

	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	pools := strataAttentionPools{
		kv: strataKVPools{
			key: buffers.upload(t, int8Bytes(keyCodes)), value: buffers.upload(t, int8Bytes(valueCodes)),
			keyScale: buffers.upload(t, uint16Bytes(keyScales)), valueScale: buffers.upload(t, uint16Bytes(valueScales)),
		},
		table: buffers.upload(t, int32Bytes(table)),
	}
	var idBuffer driver.DevicePtr
	if ids != nil {
		idBuffer = buffers.upload(t, int32Bytes(ids))
	}
	stepBuffer, queryBuffer := buffers.upload(t, int32Bytes(steps)), buffers.upload(t, float32Bytes(q))
	outputBytes := int(queries * heads * width * 4)
	chunks := (capacity + strataAttentionChunk - 1) / strataAttentionChunk
	decodeOutput, promptOutput := buffers.zeros(t, outputBytes), buffers.zeros(t, outputBytes)
	scratch := buffers.zeros(t, int(queries*chunks*heads*(width+2)*4))
	kernels := strataCellInstantiations[[2]int64{width, geometry.group}]
	sizeBuffer := buffers.zeros(t, 4)
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launchGridABI(state, functions[kernels.sharedBytes], driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 1, Y: 1, Z: 1}, &sizeBuffer)
	})
	sharedBytes := binary.LittleEndian.Uint32(buffers.read(t, sizeBuffer, 4))
	// StrataSmemCells<W>: two stages of W warps' K and V 32-cell slices of 64 bytes and their scales, W warps' partials,
	// the probabilities, W maxima and three 16-row softmax states.
	warps := uint32(width / 64)
	if want := 2*warps*2*32*64 + 2*warps*2*32*4 + warps*16*33*4 + 16*33*4 + warps*4 + 3*16*4; sharedBytes != want {
		t.Fatalf("the prompt kernel's shared layout is %d bytes, %d warps take %d", sharedBytes, warps, want)
	}
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataCellDecode(state, functions, geometry, queryBuffer, pools, idBuffer, stepBuffer, capacity, scratch,
			decodeOutput, queries); err != nil {
			return err
		}
		return launchStrataCellPrompt(state, functions, geometry, queryBuffer, pools, idBuffer, stepBuffer, capacity, sharedBytes,
			promptOutput, queries)
	})
	decode := float32s(buffers.read(t, decodeOutput, outputBytes))
	prompt := float32s(buffers.read(t, promptOutput, outputBytes))
	value := func(codes []int8, scales []uint16, row, d int64) float64 {
		return float64(codes[row*width+d]) * float64(strataF32FromF16(scales[row*groups+d/int64(strataKVQ8Group)]))
	}
	errorDecode, errorPrompt, referenceScale := 0.0, 0.0, 0.0
	for query := int64(0); query < queries; query += max(1, queries/16) {
		selection := selections[query]
		for head := range heads {
			kvHead := head / geometry.group
			rowOf := func(cell int32) int64 {
				return (int64(table[int64(cell)/geometry.pageCells])*geometry.kvHeads+kvHead)*geometry.pageCells + int64(cell)%geometry.pageCells
			}
			scores := make([]float64, len(selection))
			maximum := math.Inf(-1)
			for index, cell := range selection {
				row, dot := rowOf(cell), 0.0
				for d := range width {
					dot += float64(q[(query*heads+head)*width+d]) * value(keyCodes, keyScales, row, d)
				}
				scores[index] = dot / math.Sqrt(float64(width))
				maximum = max(maximum, scores[index])
			}
			sum := 0.0
			for index := range scores {
				scores[index] = math.Exp(scores[index] - maximum)
				sum += scores[index]
			}
			for d := range width {
				accumulated := 0.0
				for index, cell := range selection {
					accumulated += scores[index] * value(valueCodes, valueScales, rowOf(cell), d)
				}
				reference := accumulated / sum
				at := (query*heads+head)*width + d
				referenceScale = max(referenceScale, math.Abs(reference))
				errorDecode = max(errorDecode, math.Abs(float64(decode[at])-reference))
				errorPrompt = max(errorPrompt, math.Abs(float64(prompt[at])-reference))
			}
		}
	}
	difference, scale := 0.0, 0.0
	for index := range decode {
		difference = max(difference, math.Abs(float64(decode[index]-prompt[index])))
		scale = max(scale, math.Abs(float64(decode[index])))
	}
	if !(errorDecode <= 1e-3*referenceScale) {
		t.Errorf("the decode attention stands %.3g from FP64, past 1e-3 of the output scale %.3g", errorDecode, referenceScale)
	}
	if !(errorPrompt <= max(4*errorDecode, 1e-6*referenceScale)) {
		t.Errorf("the prompt attention stands %.3g from FP64, past four times the decode attention's %.3g", errorPrompt, errorDecode)
	}
	if !(difference <= 1e-4*scale) {
		t.Errorf("the prompt and decode attention differ by %.3g, past 1e-4 of the output scale %.3g", difference, scale)
	}
	t.Logf("%s: decode %.3g, prompt %.3g from FP64 (scale %.3g); the two %.2g of the scale",
		fmt.Sprint(geometry), errorDecode, errorPrompt, referenceScale, difference/scale)
}
