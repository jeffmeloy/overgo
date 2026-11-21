package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// The 8-bit attention's launch (tensor.AttentionAttributes.EightBit), after SageAttention2 (arXiv 2411.10958,
// Apache-2.0; kernels/cuda/ops_f32_sage_attention.cuh): K's channel means and V's channel scales from per-block
// statistics, Q and the smoothed K in INT8 per 64-token block, V in FP8 E4M3 transposed, then the attention on the 8-bit
// tensor cores, over F32 heads of 128 channels interleaved [sequence][token][head][channel]. Its codes, scales and
// statistics are its staging, kept in the score staging.
const (
	eightBitWidth   = uint32(tensor.EightBitAttentionWidth)
	eightBitBlock   = uint32(64)
	eightBitAligned = uint64(256)
)

// eightBitGeometry is one attention's shape: query and key/value heads, query and key/value tokens, and sequences.
type eightBitGeometry struct {
	queryHeads, keyValueHeads, queryTokens, keyValueTokens, sequences uint32
}

// eightBitGeometryOf is an attention node's shape, read from its queries and keys.
func eightBitGeometryOf(node *tensor.Tensor) eightBitGeometry {
	return headsGeometry(node.Inputs[0].Shape, node.Inputs[1].Shape)
}

// headsGeometry is an attention's shape read from its queries and keys, each [width][heads][tokens][sequences].
func headsGeometry(query, key tensor.Shape) eightBitGeometry {
	geometry := eightBitGeometry{
		queryHeads: uint32(query.Dims[1]), keyValueHeads: uint32(key.Dims[1]),
		queryTokens: uint32(query.Dims[2]), keyValueTokens: uint32(key.Dims[2]), sequences: 1,
	}
	if query.Rank == 4 {
		geometry.sequences = uint32(query.Dims[3])
	}
	return geometry
}

// blocks is the 64-token blocks of tokens.
func (geometry eightBitGeometry) blocks(tokens uint32) uint32 {
	return (tokens + eightBitBlock - 1) / eightBitBlock
}

// stagingParts are the bytes of the attention's staging, in order: Q's and K's INT8 codes, V's transposed FP8 codes
// padded to whole blocks, Q's and K's block scales, V's channel scales, K's channel means, and K's block sums and V's
// block maxima.
func (geometry eightBitGeometry) stagingParts() [9]uint64 {
	sequences, width := uint64(geometry.sequences), uint64(eightBitWidth)
	queryHeads, keyValueHeads := uint64(geometry.queryHeads), uint64(geometry.keyValueHeads)
	queryBlocks, keyBlocks := uint64(geometry.blocks(geometry.queryTokens)), uint64(geometry.blocks(geometry.keyValueTokens))
	channels, floatBytes := sequences*keyValueHeads*width, uint64(f32Bytes)
	return [9]uint64{
		sequences * uint64(geometry.queryTokens) * queryHeads * width,
		sequences * uint64(geometry.keyValueTokens) * keyValueHeads * width,
		channels * keyBlocks * uint64(eightBitBlock),
		sequences * queryHeads * queryBlocks * floatBytes,
		sequences * keyValueHeads * keyBlocks * floatBytes,
		channels * floatBytes,
		channels * floatBytes,
		channels * keyBlocks * floatBytes,
		channels * keyBlocks * floatBytes,
	}
}

// stagingBytes is the staging's whole size, each part aligned to 256 bytes.
func (geometry eightBitGeometry) stagingBytes() uint64 {
	parts := geometry.stagingParts()
	_, total := alignedParts(0, parts[:])
	return total
}

// alignedParts lays parts of a staging out from base, each aligned to eightBitAligned bytes: each part's place, and the
// bytes they take in all.
func alignedParts(base driver.DevicePtr, parts []uint64) ([]driver.DevicePtr, uint64) {
	offsets, total := make([]driver.DevicePtr, len(parts)), uint64(0)
	for index, bytes := range parts {
		offsets[index] = base + driver.DevicePtr(total)
		total += (bytes + eightBitAligned - 1) &^ (eightBitAligned - 1)
	}
	return offsets, total
}

// launchEightBitAttention launches a graph's 8-bit attention node over the score staging, its key bias the final input
// where it has one.
func launchEightBitAttention(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor,
	attributes tensor.AttentionAttributes, pointers launchPointerFrame) error {
	geometry := eightBitGeometryOf(node)
	if blas == nil || blas.scores == 0 || geometry.stagingBytes() > blas.scoreBytes {
		return errors.New("8-bit attention staging is unavailable")
	}
	var keyBias driver.DevicePtr
	if attributes.HasKeyBias {
		keyBias = pointers.input(len(node.Inputs) - 1)
	}
	return launchEightBitAttentionKernels(state, functions, pointers.input(0), pointers.input(1), pointers.input(2), keyBias,
		pointers.output(), blas.scores, geometry, attributes.Scale)
}

// launchEightBitAttentionKernels runs the attention's six launches over query, key and value, writing output in the
// query's layout, with staging of geometry.stagingBytes().
func launchEightBitAttentionKernels(state *device.State, functions functionSet, query, key, value, keyBias, output, staging driver.DevicePtr,
	geometry eightBitGeometry, scale float32) error {
	parts := geometry.stagingParts()
	offsets, _ := alignedParts(staging, parts[:])
	queryCodes, keyCodes, valueCodes, queryScales, keyScales := offsets[0], offsets[1], offsets[2], offsets[3], offsets[4]
	valueScales, keyMean, keySums, valueMaxima := offsets[5], offsets[6], offsets[7], offsets[8]
	queryHeads, keyValueHeads, sequences := geometry.queryHeads, geometry.keyValueHeads, geometry.sequences
	queryTokens, keyValueTokens := geometry.queryTokens, geometry.keyValueTokens
	queryBlocks, keyBlocks := geometry.blocks(queryTokens), geometry.blocks(keyValueTokens)
	padded := keyBlocks * eightBitBlock
	threads := driver.Dim3{X: eightBitWidth, Y: 1, Z: 1}
	keyGrid := driver.Dim3{X: keyBlocks, Y: keyValueHeads, Z: sequences}
	var none driver.DevicePtr
	for _, launch := range []struct {
		function kernelFunctionID
		grid     driver.Dim3
		args     []any
	}{
		{kernelSageBlockStatsF32, keyGrid, []any{&key, &value, &keySums, &valueMaxima, &keyValueHeads, &keyValueTokens, &sequences}},
		{kernelSageChannelStatsF32, driver.Dim3{X: keyValueHeads, Y: sequences, Z: 1},
			[]any{&keySums, &valueMaxima, &keyMean, &valueScales, &keyValueHeads, &keyValueTokens, &sequences}},
		{kernelSageQuantizeI8F32, driver.Dim3{X: queryBlocks, Y: queryHeads, Z: sequences},
			[]any{&query, &none, &queryCodes, &queryScales, &queryHeads, &queryTokens, &sequences}},
		{kernelSageQuantizeI8F32, keyGrid, []any{&key, &keyMean, &keyCodes, &keyScales, &keyValueHeads, &keyValueTokens, &sequences}},
		{kernelSageQuantizeVFp8F32, keyGrid,
			[]any{&value, &valueScales, &valueCodes, &keyValueHeads, &keyValueTokens, &padded, &sequences}},
		{kernelAttentionSage128F32, driver.Dim3{X: queryBlocks, Y: queryHeads, Z: sequences},
			[]any{&queryCodes, &queryScales, &keyCodes, &keyScales, &valueCodes, &valueScales, &keyBias, &output,
				&queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &sequences, &scale}},
	} {
		if err := launchGridABI(state, functions[launch.function], launch.grid, threads, launch.args...); err != nil {
			return err
		}
	}
	return nil
}
