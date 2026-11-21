package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
)

// The int8 attention's launch (tensor.AttentionAttributes.Int8): comfy_kitchen's int8_attention at v0.2.40
// (kernels/cuda/ops_f32_int8_attention.cuh) over F32 heads of 128 channels interleaved [sequence][token][head][channel],
// each value rounded to BF16 as its kernel reads it. Four launches, as comfy_kitchen's sage_sdpa makes them: each key
// head's anchor (int8_attention_anchor_f32), Q and K rotated and quantized to INT8 (int8_attention_quantize_qk_f32), V
// quantized to INT8 per channel (int8_attention_quantize_v_f32, or its short form for 256 keys or fewer), then the
// attention (int8_attention_bf16_f32). The codes, scales and anchors are its staging, kept in the score staging.
const (
	int8AttentionWidth = uint32(tensor.EightBitAttentionWidth)
	// int8AttentionQueryTile and int8AttentionKeyTile are the query rows a block of the attention takes and the keys a
	// tile holds; the quantizer's warps take int8AttentionQueryWarpRows query rows and int8AttentionKeyTile keys each.
	int8AttentionQueryTile     = uint32(128)
	int8AttentionKeyTile       = uint32(128)
	int8AttentionQueryWarpRows = uint32(32)
	// int8AttentionQueryScales and int8AttentionKeyScales are the scales a warp block of queries and a key tile hold.
	int8AttentionQueryScales = uint32(8)
	int8AttentionKeyScales   = uint32(4)
	// int8AttentionValueChannels are the channels a block of V's quantizer takes, int8AttentionShortKeys the most keys
	// its short form takes.
	int8AttentionValueChannels = uint32(8)
	int8AttentionShortKeys     = uint32(256)
	int8AttentionThreads       = uint32(128)
	int8AttentionValueThreads  = uint32(512)
	int8AttentionWarps         = uint32(8)
	// int8AttentionSharedBytes mirrors kitchen_attention::shared_bytes: two key tiles and a value tile of INT8 codes.
	int8AttentionSharedBytes = 3 * int8AttentionKeyTile * int8AttentionWidth
)

// int8AttentionTiles is the tiles of size that tokens fill, the last perhaps in part.
func int8AttentionTiles(tokens, size uint32) uint32 {
	return (tokens + size - 1) / size
}

// stagingParts are the bytes of the int8 attention's staging, in order: Q's and K's INT8 codes, V's transposed INT8
// codes padded to whole key tiles, Q's, K's and V's scales, and the key heads' anchors.
func (geometry eightBitGeometry) int8StagingParts() [7]uint64 {
	sequences, width := uint64(geometry.sequences), uint64(int8AttentionWidth)
	queryHeads, keyValueHeads := uint64(geometry.queryHeads), uint64(geometry.keyValueHeads)
	queryWarpBlocks := uint64(int8AttentionTiles(geometry.queryTokens, int8AttentionQueryTile) * (int8AttentionQueryTile / int8AttentionQueryWarpRows))
	keyTiles := uint64(int8AttentionTiles(geometry.keyValueTokens, int8AttentionKeyTile))
	return [7]uint64{
		sequences * queryHeads * uint64(geometry.queryTokens) * width,
		sequences * keyValueHeads * uint64(geometry.keyValueTokens) * width,
		sequences * keyValueHeads * width * keyTiles * uint64(int8AttentionKeyTile),
		sequences * queryHeads * queryWarpBlocks * uint64(int8AttentionQueryScales) * uint64(f32Bytes),
		sequences * keyValueHeads * keyTiles * uint64(int8AttentionKeyScales) * uint64(f32Bytes),
		sequences * keyValueHeads * width * uint64(f32Bytes),
		sequences * keyValueHeads * uint64(f32Bytes),
	}
}

// int8StagingBytes is the int8 attention's staging, each part aligned (alignedParts).
func (geometry eightBitGeometry) int8StagingBytes() uint64 {
	parts := geometry.int8StagingParts()
	_, total := alignedParts(0, parts[:])
	return total
}

// fusionGeometry is a fused attention's shape, read from its pre-round heads.
func fusionGeometry(fusion bf16AttentionFusion) eightBitGeometry {
	return headsGeometry(fusion.query.Shape, fusion.key.Shape)
}

// int8AttentionNode says an attention node takes the int8 attention: heads of 128 channels as wide as its values, query
// heads in whole groups of the key heads, and no key bias.
func int8AttentionNode(node *tensor.Tensor, attributes tensor.AttentionAttributes) bool {
	if !attributes.Int8 || attributes.HasKeyBias || len(node.Inputs) != 3 {
		return false
	}
	query, key, value := node.Inputs[0].Shape.Dims, node.Inputs[1].Shape.Dims, node.Inputs[2].Shape.Dims
	return query[0] == uint64(int8AttentionWidth) && key[0] == query[0] && value[0] == query[0] && key[1] != 0 && query[1]%key[1] == 0
}

// launchInt8Attention runs the int8 attention's four launches over the F32 heads query, key and value, writing output
// in the query's layout, its staging the score staging.
func launchInt8Attention(state *device.State, functions functionSet, blas *blasState, query, key, value, output driver.DevicePtr,
	geometry eightBitGeometry, scale float32) error {
	if blas == nil || blas.scores == 0 || geometry.int8StagingBytes() > blas.scoreBytes {
		return errors.New("int8 attention staging is unavailable")
	}
	parts := geometry.int8StagingParts()
	offsets, _ := alignedParts(blas.scores, parts[:])
	queryCodes, keyCodes, valueCodes, queryScales, keyScales, valueScales, anchors :=
		offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5], offsets[6]
	queryHeads, keyValueHeads, sequences := geometry.queryHeads, geometry.keyValueHeads, geometry.sequences
	queryTokens, keyValueTokens, width := geometry.queryTokens, geometry.keyValueTokens, int8AttentionWidth
	// The heads' element strides: a token's heads lie together, a sequence's tokens after one another.
	queryTokenStride, keyTokenStride := queryHeads*width, keyValueHeads*width
	querySequenceStride, keySequenceStride := queryTokens*queryTokenStride, keyValueTokens*keyTokenStride
	headStride := width
	keyTiles := int8AttentionTiles(keyValueTokens, int8AttentionKeyTile)
	padded := keyTiles * int8AttentionKeyTile
	queryWarpBlocks := int8AttentionTiles(queryTokens, int8AttentionQueryTile) * (int8AttentionQueryTile / int8AttentionQueryWarpRows)
	queryScalesPerHead, keyScalesPerHead := queryWarpBlocks*int8AttentionQueryScales, keyTiles*int8AttentionKeyScales
	signedQueryTokens, signedKeyTokens, signedWidth := int32(queryTokens), int32(keyValueTokens), int32(width)
	signedQueryHeads, signedKeyHeads := int32(queryHeads), int32(keyValueHeads)
	signedWarpBlocks, signedQueryScales, signedKeyScales := int32(queryWarpBlocks), int32(queryScalesPerHead), int32(keyScalesPerHead)
	signedPadded := int32(padded)
	threads := driver.Dim3{X: int8AttentionThreads, Y: 1, Z: 1}
	if err := launchGridABI(state, functions[kernelInt8AttentionAnchorF32], driver.Dim3{X: keyValueHeads, Y: sequences, Z: 1}, threads,
		&key, &anchors, &signedKeyTokens, &signedWidth, &signedKeyHeads, &keySequenceStride, &headStride, &keyTokenStride); err != nil {
		return err
	}
	if err := launchGridABI(state, functions[kernelInt8AttentionQuantizeQkF32],
		driver.Dim3{X: queryWarpBlocks + keyTiles, Y: max(queryHeads, keyValueHeads), Z: sequences}, threads,
		&query, &queryCodes, &queryScales, &key, &keyCodes, &keyScales, &anchors,
		&signedQueryTokens, &signedKeyTokens, &signedWidth, &signedWarpBlocks, &signedQueryHeads, &signedKeyHeads,
		&signedQueryScales, &signedKeyScales,
		&querySequenceStride, &headStride, &queryTokenStride, &keySequenceStride, &headStride, &keyTokenStride); err != nil {
		return err
	}
	valueFunction, valueThreads := kernelInt8AttentionQuantizeVF32, int8AttentionValueThreads
	if keyValueTokens <= int8AttentionShortKeys {
		valueFunction, valueThreads = kernelInt8AttentionQuantizeVShortF32, int8AttentionThreads
	}
	if err := launchGridABI(state, functions[valueFunction],
		driver.Dim3{X: sequences * keyValueHeads * (width / int8AttentionValueChannels), Y: 1, Z: 1}, driver.Dim3{X: valueThreads, Y: 1, Z: 1},
		&value, &valueCodes, &valueScales, &signedKeyTokens, &signedPadded, &signedKeyHeads, &signedWidth,
		&keySequenceStride, &headStride, &keyTokenStride); err != nil {
		return err
	}
	// The codes lie [sequence][head][token][channel], V's [sequence][head][channel][padded key].
	groups := queryHeads / keyValueHeads
	queryCodeHead, keyCodeHead := queryTokens*width, keyValueTokens*width
	queryCodeSequence, keyCodeSequence := queryHeads*queryCodeHead, keyValueHeads*keyCodeHead
	valueCodeHead := width * padded
	valueCodeSequence := keyValueHeads * valueCodeHead
	return launchGridSharedABI(state, functions[kernelInt8AttentionBf16F32],
		driver.Dim3{X: int8AttentionTiles(queryTokens, int8AttentionQueryTile), Y: queryHeads, Z: sequences},
		driver.Dim3{X: uint32(kernel.WarpThreads()), Y: int8AttentionWarps, Z: 1}, int8AttentionSharedBytes,
		&queryCodes, &keyCodes, &valueCodes, &output, &queryScales, &keyScales, &valueScales,
		&queryTokens, &keyValueTokens, &groups,
		&queryCodeSequence, &width, &queryCodeHead, &keyCodeSequence, &width, &keyCodeHead,
		&valueCodeSequence, &valueCodeHead, &padded,
		&querySequenceStride, &queryTokenStride, &headStride, &scale)
}
