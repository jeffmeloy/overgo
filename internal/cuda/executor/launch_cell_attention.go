package executor

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// The cell attention operation's launch (tensor.CellAttention): an attention layer over Strata's int8 K/V cells, its
// step's keys and values appended as cells (kv_append_q8_steps), then each token's attention over the cells it reads -
// a token or a verify window by the decode chunks and their merge, a prompt chunk by the int8 prompt attention - at the
// layer's own geometry (strata_cell_* in kernels/cuda/strata_qsa.cu, Strata's bodies at 296b59d2, MIT). Every per-token
// count is the step records' (the node's runtime words), so the launches stay constant across tokens.
const (
	cellInputQuery = iota
	cellInputKey
	cellInputValue
	cellInputKeyCodes
	cellInputValueCodes
	cellInputKeyScales
	cellInputValueScales
	cellInputPageTable
)

// strataCellGeometry is an attention layer's cells: K/V heads of width, group query heads each, pages of pageCells.
type strataCellGeometry struct {
	width, group, kvHeads, pageCells int64
}

// strataCellKernels are one cell attention instantiation's entries: the int8 decode chunks, their merge, the int8
// prompt attention and its shared memory's size.
type strataCellKernels struct {
	chunk, merge, prompt, sharedBytes kernelFunctionID
}

// strataCellEntries are the cell attention's entries, an instantiation each of tensor.CellAttentionGeometries in its
// order.
var strataCellEntries = []strataCellKernels{
	{kernelStrataCellAttnChunkQ8128x8, kernelStrataCellAttnMerge128x8, kernelStrataCellPromptAttnI8128x8, kernelStrataCellPromptAttnI8Smem128x8},
	{kernelStrataCellAttnChunkQ8256x2, kernelStrataCellAttnMerge256x2, kernelStrataCellPromptAttnI8256x2, kernelStrataCellPromptAttnI8Smem256x2},
	{kernelStrataCellAttnChunkQ8512x16, kernelStrataCellAttnMerge512x16, kernelStrataCellPromptAttnI8512x16, kernelStrataCellPromptAttnI8Smem512x16},
	{kernelStrataCellAttnChunkQ8256x4, kernelStrataCellAttnMerge256x4, kernelStrataCellPromptAttnI8256x4, kernelStrataCellPromptAttnI8Smem256x4},
	{kernelStrataCellAttnChunkQ8256x6, kernelStrataCellAttnMerge256x6, kernelStrataCellPromptAttnI8256x6, kernelStrataCellPromptAttnI8Smem256x6},
}

// strataCellKernelsOf is the entries instantiated for a geometry, if any.
func strataCellKernelsOf(geometry strataCellGeometry) (strataCellKernels, bool) {
	for index, held := range tensor.CellAttentionGeometries {
		if held == [tensor.PairedExtent]uint64{uint64(geometry.width), uint64(geometry.group)} {
			return strataCellEntries[index], true
		}
	}
	return strataCellKernels{}, false
}

// The prompt attention's geometry (qsa_prompt_attn.cu): warps of 64 head dimensions and 32 threads each, two cp.async
// stages of 32-cell chunks (STRATA_CH2), 16 MMA rows (the query heads a K/V head, padded) and three softmax states a
// row (its rescale, its sum and its maximum).
const (
	strataCellPromptWarpWidth   = 64
	strataCellPromptWarpThreads = 32
	strataPromptStages          = uint32(2)
	strataPromptChunkCells      = uint32(32)
	strataPromptRows            = uint32(16)
	strataPromptRowStates       = uint32(3)
)

// strataCellPromptSharedBytes is StrataSmemCells<width / 64>: two stages of each warp's K and V 32-cell slices of 64
// bytes and their scales, each warp's q.k partials for 16 rows, the probabilities, each warp's query maximum and the
// three 16-row softmax states (strataPromptSharedBytes at Strata's four warps).
func strataCellPromptSharedBytes(width int64) uint32 {
	warps, value, padded := uint32(width/strataCellPromptWarpWidth), uint32(f32Bytes), strataPromptChunkCells+1
	slices := strataPromptStages * warps * strataKVHalves * strataPromptChunkCells
	return slices*strataCellPromptWarpWidth + slices*value + warps*strataPromptRows*padded*value + strataPromptRows*padded*value +
		warps*value + strataPromptRowStates*strataPromptRows*value
}

// cellGeometry is a cell attention node's geometry.
func cellGeometry(node *tensor.Tensor) strataCellGeometry {
	width, heads, kvHeads := node.Inputs[cellInputQuery].Shape.Dims[0], node.Inputs[cellInputQuery].Shape.Dims[1], node.Inputs[cellInputKey].Shape.Dims[1]
	return strataCellGeometry{width: int64(width), group: int64(heads / kvHeads), kvHeads: int64(kvHeads), pageCells: tensor.CellPageCells}
}

// cellReadCapacity is the most cells a token of the node reads: its window where the state holds more, or the
// state's capacity.
func cellReadCapacity(node *tensor.Tensor) int64 {
	attributes := node.Attrs.(tensor.CellAttentionAttributes)
	if attributes.Window > 0 {
		return int64(min(attributes.Window, attributes.Capacity))
	}
	return int64(attributes.Capacity)
}

// cellDecodes reports whether the node's tokens attend by the decode kernels: a token or a verify window.
func cellDecodes(node *tensor.Tensor) bool {
	return node.Shape.Dims[tensor.CellTokenAxis] <= q8InputSpanColumns
}

// cellScratchBytes is a decoding node's partials in the score staging: each token's chunks, a partial accumulator a
// head dimension and a maximum and a sum a head; a prompt chunk's attention needs none.
func cellScratchBytes(node *tensor.Tensor) uint64 {
	if !cellDecodes(node) {
		return 0
	}
	geometry := cellGeometry(node)
	return uint64(node.Shape.Dims[tensor.CellTokenAxis]) * uint64(cellScratchFloats(geometry, cellReadCapacity(node))) * uint64(f32Bytes)
}

// cellScratchFloats is a query's partials for capacity cells.
func cellScratchFloats(geometry strataCellGeometry, capacity int64) int64 {
	chunks := (capacity + strataAttentionChunk - 1) / strataAttentionChunk
	return chunks * geometry.kvHeads * geometry.group * (geometry.width + strataAttentionRunning)
}

// launchCellAttention runs one cell attention step: its rows appended as cells, then each token's attention.
func launchCellAttention(
	state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, runtimeAttributes tensor.Attributes,
	pointers launchPointerFrame,
) error {
	attributes, ok := runtimeAttributes.(tensor.CellAttentionAttributes)
	if !ok || len(node.Inputs) != tensor.CellAttentionInputs || len(attributes.Cells) == 0 {
		return errors.New("cell attention node is invalid")
	}
	geometry := cellGeometry(node)
	if _, ok := strataCellKernelsOf(geometry); !ok {
		return errors.New("cell attention geometry has no kernels")
	}
	pools := strataAttentionPools{
		kv: strataKVPools{
			key: pointers.input(cellInputKeyCodes), value: pointers.input(cellInputValueCodes),
			keyScale: pointers.input(cellInputKeyScales), valueScale: pointers.input(cellInputValueScales),
		},
		table: pointers.input(cellInputPageTable),
	}
	steps, tokens := pointers.attribute, int64(len(attributes.Cells))
	kvGeometry := strataKVGeometry{kvHeads: int32(geometry.kvHeads), headDim: int32(geometry.width), pageSize: int32(geometry.pageCells)}
	if err := launchStrataKVAppendSteps(state, functions, pools.kv, strataKVPools{}, pools.table, steps,
		pointers.input(cellInputKey), pointers.input(cellInputValue), kvGeometry, int32(tokens), tensor.QSAStepWords,
		int32(geometry.kvHeads*geometry.width)); err != nil {
		return err
	}
	capacity, scale := cellReadCapacity(node), attributes.Scale
	if !cellDecodes(node) {
		return launchStrataCellPrompt(state, functions, geometry, pointers.input(cellInputQuery), pools, 0, steps, capacity,
			scale, pointers.output(), tokens)
	}
	if blas == nil || blas.scores == 0 || cellScratchBytes(node) > blas.scoreBytes {
		return errors.New("cell attention scratch is unavailable")
	}
	return launchStrataCellDecode(state, functions, geometry, pointers.input(cellInputQuery), pools, 0, steps, capacity, scale,
		blas.scores, pointers.output(), tokens)
}

// launchStrataCellDecode is launchStrataDecodeAttention at a layer's geometry: the int8 chunks over each query's
// selection, or over the range of cells its step record names where ids is zero, then the merge; the scores scaled by
// scale.
func launchStrataCellDecode(state *device.State, functions functionSet, geometry strataCellGeometry, queries driver.DevicePtr,
	pools strataAttentionPools, ids, steps driver.DevicePtr, capacity int64, scale float32, scratch, output driver.DevicePtr,
	queryCount int64,
) error {
	kernels, _ := strataCellKernelsOf(geometry)
	heads := geometry.kvHeads * geometry.group
	chunks := int32((capacity + strataAttentionChunk - 1) / strataAttentionChunk)
	stride := cellScratchFloats(geometry, capacity)
	accumulators := scratch
	maxima := scratch + driver.DevicePtr(int64(chunks)*heads*geometry.width*f32Bytes)
	sums := maxima + driver.DevicePtr(int64(chunks)*heads*f32Bytes)
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

// launchStrataCellPrompt is launchStrataPromptAttention at a layer's geometry: queryCount queries, each over its
// selection or the range of cells its step record names where ids is zero, one CTA a (query, K/V head), at most
// 65,535 queries a launch.
func launchStrataCellPrompt(state *device.State, functions functionSet, geometry strataCellGeometry, queries driver.DevicePtr,
	pools strataAttentionPools, ids, steps driver.DevicePtr, capacity int64, scale float32, output driver.DevicePtr, queryCount int64,
) error {
	kernels, _ := strataCellKernelsOf(geometry)
	heads := geometry.kvHeads * geometry.group
	scaleLog2 := float32(math.Log2E) * scale
	kvHeads, pageCells, capacity32 := int32(geometry.kvHeads), int32(geometry.pageCells), int32(capacity)
	for first := int64(0); first < queryCount; first += strataPromptQueryLaunch {
		count := min(queryCount-first, strataPromptQueryLaunch)
		q := queries + driver.DevicePtr(first*heads*geometry.width*f32Bytes)
		selection := ids
		if ids != 0 {
			selection += driver.DevicePtr(first * capacity * f32Bytes)
		}
		records := steps + driver.DevicePtr(first*int64(strataStepRecord)*f32Bytes)
		out := output + driver.DevicePtr(first*heads*geometry.width*f32Bytes)
		if err := launchGridSharedABI(state, functions[kernels.prompt],
			driver.Dim3{X: uint32(count), Y: uint32(kvHeads), Z: 1},
			driver.Dim3{X: uint32(geometry.width / strataCellPromptWarpWidth * strataCellPromptWarpThreads), Y: 1, Z: 1}, strataCellPromptSharedBytes(geometry.width),
			&q, &pools.kv.key, &pools.kv.value, &pools.kv.keyScale, &pools.kv.valueScale, &pools.table, &selection, &records,
			&kvHeads, &pageCells, &scaleLog2, &out, &capacity32); err != nil {
			return err
		}
	}
	return nil
}
