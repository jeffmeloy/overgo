package executor

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// Q8_0 K/V cells. A cache of Q8_0 blocks takes F32 rows quantized a block a
// warp (cache_append_q8_0_f32 and the rope-append variants) and is read back
// as F32 rows: a selected attention gathers only its step's slots
// (gather_q8_0_rows_f32), a prompt's attention dequantizes its page
// (dequantize_q8_0_rows_f32), and every attention kernel stays the F32 one.
// Strata reads its int8 cells in the attention kernels themselves (kv_q8.cu,
// qsa_decode_attn.cu); the gather is this landing's form, which the kv step's
// bench prices against reading the cells in place.

// q8CellTraits are the Q8_0 block's size and bytes.
var q8CellTraits, _ = dtype.Q8_0.Traits()

// q8StagingAlignment keeps each staged region at a boundary every kernel's
// vector loads accept.
const q8StagingAlignment = uint64(256)

// An attention's keys and values stand after its queries among its inputs, and a
// cache append's rows after its cache.
const (
	attentionKeyInput    = 1
	attentionValueInput  = attentionFlashValueInput
	cacheAppendRowsInput = 1
)

func alignStaging(bytes uint64) uint64 {
	return (bytes + q8StagingAlignment - 1) / q8StagingAlignment * q8StagingAlignment
}

// q8CellBytes is the byte size of a Q8_0 store of the shape, whose rows are whole blocks.
func q8CellBytes(shape tensor.Shape) (uint32, error) {
	elements, err := shape.Elements()
	if err != nil {
		return 0, err
	}
	bytes, err := dtype.Q8_0.StorageBytes(elements, shape.Dims[0])
	if err != nil {
		return 0, err
	}
	return uint32Checked(bytes, "Q8_0 cell store")
}

// copyQ8Cells copies a Q8_0 store when a run's output is not its input's storage.
func copyQ8Cells(state *device.State, functions functionSet, shape tensor.Shape, source, destination driver.DevicePtr) error {
	bytes, err := q8CellBytes(shape)
	if err != nil {
		return err
	}
	return launch1DABI(state, functions[kernelCopyU8], bytes, &source, &destination, &bytes)
}

// launchCacheAppendQ80 appends F32 rows to a cache of Q8_0 cells at the
// device-resident row offset, a block a warp.
func launchCacheAppendQ80(
	state *device.State,
	functions functionSet,
	node *tensor.Tensor,
	attributes tensor.CacheAppendAttributes,
	pointers launchPointerFrame,
) error {
	output, left, right := pointers.output(), pointers.input(0), pointers.input(1)
	if node.Inputs[cacheAppendRowsInput].Type != dtype.F32 {
		return errors.New("Q8_0 cache append takes F32 rows")
	}
	if output != left {
		if err := copyQ8Cells(state, functions, node.Inputs[0].Shape, left, output); err != nil {
			return err
		}
	}
	inner := uint64(1)
	for dimension := uint32(0); dimension < attributes.Axis; dimension++ {
		if inner > math.MaxUint64/node.Shape.Dims[dimension] {
			return errors.New("cache append offset overflows")
		}
		inner *= node.Shape.Dims[dimension]
	}
	rightCount, outputCount := shapeCount32(node.Inputs[cacheAppendRowsInput].Shape), shapeCount32(node.Shape)
	if inner%q8CellTraits.BlockSize != 0 || uint64(rightCount)%q8CellTraits.BlockSize != 0 {
		return errors.New("Q8_0 cache rows must be whole blocks")
	}
	if inner != 0 && uint64(attributes.Offset) > math.MaxUint64/inner {
		return errors.New("cache append offset overflows")
	}
	if offset := uint64(attributes.Offset) * inner; offset > uint64(outputCount) || uint64(rightCount) > uint64(outputCount)-offset {
		return errors.New("cache append range exceeds capacity")
	}
	offsetPointer := pointers.attribute
	if offsetPointer == 0 {
		return errors.New("cache append offset storage is unavailable")
	}
	innerCount, err := uint32Checked(inner, "cache append inner size")
	if err != nil {
		return err
	}
	return launch1DABI(
		state, functions[kernelCacheAppendQ80F32], rightCount,
		&right, &output, &offsetPointer, &innerCount, &rightCount,
	)
}

// q8SparseStaging lays out the score staging of a selected attention over
// Q8_0 cells: first what the F32 kernels take for the gathered rows (the
// tensor-core conversion or the split partials), then the gathered keys and
// values and the rewritten indices.
type q8SparseStaging struct {
	kernels, keys, values, indices, total uint64
	rows                                  uint32
}

func q8SparseStagingLayout(node *tensor.Tensor) (q8SparseStaging, error) {
	query, key := node.Inputs[sparseAttentionQuery].Shape, node.Inputs[sparseAttentionKey].Shape
	value, indices := node.Inputs[sparseAttentionValue].Shape, node.Inputs[sparseAttentionIndices].Shape
	keyWidth, valueWidth := uint32(query.Dims[0]), uint32(value.Dims[0])
	queryHeads, keyValueHeads := uint32(query.Dims[1]), uint32(key.Dims[1])
	queryTokens, selected := uint32(query.Dims[sparseAttentionTokenAxis]), uint32(indices.Dims[0])
	rows := uint64(queryTokens) * uint64(selected)
	if rows > math.MaxUint32 {
		return q8SparseStaging{}, errors.New("Q8_0 gathered rows exceed uint32")
	}
	layout := q8SparseStaging{rows: uint32(rows)}
	if mmaSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens) {
		// The gathered rows' FP16 keys and values, of one width, take one f32 a channel.
		layout.kernels = rows * uint64(keyValueHeads) * uint64(keyWidth) * f32ScalarBytes
	} else if geometry, ok := splitSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens, selected); ok && geometry.splits > 1 {
		layout.kernels = uint64(queryTokens) * uint64(queryHeads) * uint64(geometry.splits) * (uint64(valueWidth) + attentionDecodeMetaFloats) * f32ScalarBytes
	}
	layout.keys = alignStaging(layout.kernels)
	layout.values = alignStaging(layout.keys + rows*uint64(keyValueHeads)*uint64(keyWidth)*f32ScalarBytes)
	layout.indices = alignStaging(layout.values + rows*uint64(keyValueHeads)*uint64(valueWidth)*f32ScalarBytes)
	layout.total = alignStaging(layout.indices + rows*f32ScalarBytes)
	return layout, nil
}

// gatherQ80SparseRows stages a step's selected keys and values as F32 rows
// beside the kernels' own staging and returns the pointers and row count the
// F32 kernels then run on, the causal rule applied by the gather.
func gatherQ80SparseRows(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	key, value, indices, firstQuery driver.DevicePtr,
	keyWidth, valueWidth, keyValueHeads, queryTokens, keyValueTokens, selected, causal uint32,
) (keys, values, remapped driver.DevicePtr, rows uint32, err error) {
	layout, err := q8SparseStagingLayout(node)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	if blas == nil || blas.scores == 0 || layout.total > blas.scoreBytes {
		return 0, 0, 0, 0, errors.New("Q8_0 cell staging is unavailable")
	}
	keys = blas.scores + driver.DevicePtr(layout.keys)
	values = blas.scores + driver.DevicePtr(layout.values)
	remapped = blas.scores + driver.DevicePtr(layout.indices)
	rows = layout.rows
	keyCount, err := uint32Checked(uint64(rows)*uint64(keyValueHeads)*uint64(keyWidth), "Q8_0 gathered keys")
	if err != nil {
		return 0, 0, 0, 0, err
	}
	valueCount, err := uint32Checked(uint64(rows)*uint64(keyValueHeads)*uint64(valueWidth), "Q8_0 gathered values")
	if err != nil {
		return 0, 0, 0, 0, err
	}
	writeRemapped, keepRemapped := kernelBool(true), kernelBool(false)
	if err := launch1DABI(
		state, functions[kernelGatherQ80RowsF32], keyCount,
		&key, &indices, &keys, &remapped, &keyWidth, &keyValueHeads, &selected, &queryTokens,
		&keyValueTokens, &causal, &firstQuery, &writeRemapped, &keyCount,
	); err != nil {
		return 0, 0, 0, 0, err
	}
	if err := launch1DABI(
		state, functions[kernelGatherQ80RowsF32], valueCount,
		&value, &indices, &values, &remapped, &valueWidth, &keyValueHeads, &selected, &queryTokens,
		&keyValueTokens, &causal, &firstQuery, &keepRemapped, &valueCount,
	); err != nil {
		return 0, 0, 0, 0, err
	}
	return keys, values, remapped, rows, nil
}

// q8AttentionStaging lays out a prompt attention over Q8_0 cells: the F32
// kernels' own staging, then the page's keys and values dequantized to F32.
type q8AttentionStaging struct {
	kernels, keys, values, total uint64
}

func q8AttentionStagingLayout(node *tensor.Tensor) (q8AttentionStaging, error) {
	var kernels uint64
	if bytes, ok := blasAttentionScoreBytes(node); ok {
		kernels = max(kernels, bytes)
	}
	if bytes, ok := decodePartialBytes(node); ok {
		kernels = max(kernels, bytes)
	}
	keyElements, err := node.Inputs[attentionKeyInput].Shape.Elements()
	if err != nil {
		return q8AttentionStaging{}, err
	}
	valueElements, err := node.Inputs[attentionValueInput].Shape.Elements()
	if err != nil {
		return q8AttentionStaging{}, err
	}
	layout := q8AttentionStaging{kernels: kernels}
	layout.keys = alignStaging(kernels)
	layout.values = alignStaging(layout.keys + keyElements*f32ScalarBytes)
	layout.total = alignStaging(layout.values + valueElements*f32ScalarBytes)
	return layout, nil
}

// dequantizeQ80Page stages a page's keys and values as F32 rows for the prompt
// attention kernels.
func dequantizeQ80Page(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	key, value driver.DevicePtr,
) (keys, values driver.DevicePtr, err error) {
	layout, err := q8AttentionStagingLayout(node)
	if err != nil {
		return 0, 0, err
	}
	if blas == nil || blas.scores == 0 || layout.total > blas.scoreBytes {
		return 0, 0, errors.New("Q8_0 page staging is unavailable")
	}
	keys = blas.scores + driver.DevicePtr(layout.keys)
	values = blas.scores + driver.DevicePtr(layout.values)
	keyCount, valueCount := shapeCount32(node.Inputs[attentionKeyInput].Shape), shapeCount32(node.Inputs[attentionValueInput].Shape)
	if err := launch1DABI(state, functions[kernelDequantizeQ80RowsF32], keyCount, &key, &keys, &keyCount); err != nil {
		return 0, 0, err
	}
	if err := launch1DABI(state, functions[kernelDequantizeQ80RowsF32], valueCount, &value, &values, &valueCount); err != nil {
		return 0, 0, err
	}
	return keys, values, nil
}
