package executor

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// BF16 tensor-core flash attention geometry (attention_tiled_bf16_f32):
// width fixed at 128, 64 query rows per 512-thread (16-warp) block,
// 64-token K/V panels (K double-buffered, V single-buffered through the
// cp.async pipeline; the O spill scratch aliases the dead Q+S region).
// Shared bytes mirror the kernel's layout exactly.
const (
	attentionBF16Width       = uint32(128)
	attentionBF16RowTile     = uint32(64)
	attentionBF16KVTile      = uint32(64)
	attentionBF16Threads     = uint32(512)
	attentionBF16SharedBytes = uint32(2*2*attentionBF16KVTile*(attentionBF16Width+8) + // K bf16 [2][64][136]
		2*attentionBF16KVTile*(attentionBF16Width+8) + // V bf16 [64][136]
		2*attentionBF16RowTile*(attentionBF16Width+8) + // Q bf16 [64][136] (O spill alias base)
		4*attentionBF16RowTile*(attentionBF16KVTile+4) + // S f32 [64][68] (O spill alias tail)
		2*attentionBF16RowTile*(attentionBF16KVTile+8) + // P bf16 [64][72]
		3*4*attentionBF16RowTile) // alpha/max/sum rows
)

// configureLargeSharedKernels: one-time dynamic-shared opt-in for kernels
// above the 48KB default.
func configureLargeSharedKernels(lib *driver.Library, functions functionSet) error {
	return lib.FuncSetMaxDynamicShared(
		functions[kernelAttentionTiledBf16F32].function, attentionBF16SharedBytes,
	)
}

// blasBF16AttentionStagingBytes: per-node BF16 K/V pack staging (the
// cp.async kernel consumes pre-rounded BF16 K/V from the shared score
// allocation).
func blasBF16AttentionStagingBytes(fusion bf16AttentionFusion) (uint64, bool) {
	elements, err := fusion.key.Shape.Elements()
	if err != nil || elements == 0 || elements > math.MaxUint64/4 {
		return 0, false
	}
	return elements * 2 * 2, true
}

// launchBF16Attention: the fused Attention(BF16Round(q/k/v)) tensor-core
// path; reads the pre-round F32 tensors, packs K/V to BF16 staging for the
// cp.async pipeline, and rounds Q on kernel stage-in.
func launchBF16Attention(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	fusion bf16AttentionFusion,
	pointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	attributes, ok := node.Attrs.(tensor.AttentionAttributes)
	if !ok {
		return errors.New("invalid fused attention attributes")
	}
	queryNode, keyNode := fusion.query, fusion.key
	keyWidth, err := uint32Checked(queryNode.Shape.Dims[0], "fused attention width")
	if err != nil {
		return err
	}
	if keyWidth != attentionBF16Width || fusion.value.Shape.Dims[0] != uint64(attentionBF16Width) {
		return errors.New("fused attention width must be 128")
	}
	queryHeads, err := uint32Checked(queryNode.Shape.Dims[1], "fused attention query heads")
	if err != nil {
		return err
	}
	keyValueHeads, err := uint32Checked(keyNode.Shape.Dims[1], "fused attention KV heads")
	if err != nil {
		return err
	}
	queryTokens, err := uint32Checked(queryNode.Shape.Dims[2], "fused attention query tokens")
	if err != nil {
		return err
	}
	keyValueTokens, err := uint32Checked(keyNode.Shape.Dims[2], "fused attention KV tokens")
	if err != nil {
		return err
	}
	sequences := uint32(1)
	if queryNode.Shape.Rank == 4 {
		sequences, err = uint32Checked(queryNode.Shape.Dims[3], "fused attention sequences")
		if err != nil {
			return err
		}
	}
	query := pointers[queryNode]
	key := pointers[keyNode]
	value := pointers[fusion.value]
	keyBias := pointers[fusion.keyBias]
	output := pointers[node]
	scale := attributes.Scale
	stagingBytes, ok := blasBF16AttentionStagingBytes(fusion)
	if !ok || blas == nil || blas.scores == 0 || stagingBytes > blas.scoreBytes {
		return errors.New("fused attention BF16 staging is unavailable")
	}
	keyValueCount := uint32(stagingBytes / 4)
	keyStage := blas.scores
	valueStage := blas.scores + driver.DevicePtr(uint64(keyValueCount)*2)
	for _, stage := range [...]struct {
		source, destination driver.DevicePtr
	}{{key, keyStage}, {value, valueStage}} {
		if err := launch1DABI(
			state, functions[kernelF32ToBf16], keyValueCount,
			&stage.source, &stage.destination, &keyValueCount,
		); err != nil {
			return err
		}
	}
	grid := driver.Dim3{
		X: (queryTokens + attentionBF16RowTile - 1) / attentionBF16RowTile,
		Y: queryHeads,
		Z: sequences,
	}
	return launchGridSharedABI(
		state, functions[kernelAttentionTiledBf16F32],
		grid, driver.Dim3{X: attentionBF16Threads, Y: 1, Z: 1}, attentionBF16SharedBytes,
		&query, &keyStage, &valueStage, &keyBias, &output,
		&queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens,
		&sequences, &scale,
	)
}

// attentionBlasQueryFloor: below this the SIMT tiled/online kernels win on
// launch overhead (same floor as the tiled kernel's 32-row tile).
const attentionBlasQueryFloor = uint32(32)

// blasAttentionGeometry: strided-batched SGEMM attention admission
// (non-causal featureless dense MHA, full logical KV) + the derived tile
// geometry. Query chunk: eight head-width row tiles saturate the SGEMM n
// dimension and cut K/V reloads 8x vs a single head-width tile. Key chunk:
// four head-width tiles keep the score tile (heads*chunk*keyChunk floats)
// L2-resident so the online softmax and PV accumulate never round-trip
// scores through DRAM. Returns (chunk, keyChunk, staging bytes, ok);
// staging = score tile + per-row stats (max+sum).
func blasAttentionGeometry(
	queryHeads, keyValueHeads, queryTokens, keyValueTokens, keyCapacityTokens, keyWidth, valueWidth uint32,
	attributes tensor.AttentionAttributes,
) (uint32, uint32, uint64, bool) {
	if attributes.Causal || attributes.HasBlockMask || attributes.HasSinks ||
		attributes.Softcap != 0 || attributes.MaxALiBiBias != 0 || attributes.Window != 0 ||
		attributes.SymmetricWindow || attributes.ChunkedWindow || attributes.QueryStart != 0 ||
		attributes.RelativeBuckets != 0 ||
		queryHeads == 0 || queryHeads != keyValueHeads || keyWidth == 0 || keyWidth != valueWidth ||
		keyValueTokens == 0 || keyValueTokens != keyCapacityTokens ||
		queryTokens < attentionBlasQueryFloor {
		return 0, 0, 0, false
	}
	chunk, keyChunk := blasAttentionTiles(keyWidth, queryTokens, keyValueTokens)
	bytes := uint64(queryHeads) * uint64(chunk) * (uint64(keyChunk) + 2) * 4
	return chunk, keyChunk, bytes, true
}

// blasAttentionTiles: derived tile geometry shared by the F32 and BF16
// strided-batched attention paths (see blasAttentionGeometry).
func blasAttentionTiles(keyWidth, queryTokens, keyValueTokens uint32) (uint32, uint32) {
	chunk := min(8*keyWidth, queryTokens)
	keyChunk := min(4*keyWidth, keyValueTokens)
	return chunk, keyChunk
}

// blasAttentionScoreBytes: compile-time score staging reservation for
// OpAttention nodes the strided-batched SGEMM path will accept.
func blasAttentionScoreBytes(node *tensor.Tensor) (uint64, bool) {
	attributes, ok := node.Attrs.(tensor.AttentionAttributes)
	if !ok || len(node.Inputs) < 3 || len(node.Inputs) > 4 ||
		(attributes.HasKeyBias != (len(node.Inputs) == 4)) {
		return 0, false
	}
	queryNode, keyNode, valueNode := node.Inputs[0], node.Inputs[1], node.Inputs[2]
	if queryNode.Shape.Rank < 3 || keyNode.Shape.Rank < 3 {
		return 0, false
	}
	dims := []uint64{
		queryNode.Shape.Dims[0], valueNode.Shape.Dims[0],
		queryNode.Shape.Dims[1], keyNode.Shape.Dims[1],
		queryNode.Shape.Dims[2], keyNode.Shape.Dims[2],
	}
	checked := make([]uint32, len(dims))
	for i, dim := range dims {
		value, err := uint32Checked(dim, "attention dimension")
		if err != nil {
			return 0, false
		}
		checked[i] = value
	}
	keyWidth, valueWidth := checked[0], checked[1]
	queryHeads, keyValueHeads := checked[2], checked[3]
	queryTokens, keyCapacityTokens := checked[4], checked[5]
	keyValueTokens := keyCapacityTokens
	if attributes.KeyValueTokens != 0 {
		keyValueTokens = attributes.KeyValueTokens
	}
	_, _, bytes, ok := blasAttentionGeometry(
		queryHeads, keyValueHeads, queryTokens, keyValueTokens, keyCapacityTokens,
		keyWidth, valueWidth, attributes,
	)
	return bytes, ok
}

// launchBlasFullAttention: exact F32 attention as strided-batched SGEMM
// QK^T -> attention_online_softmax_f32 -> strided-batched SGEMM PV
// (beta=1 accumulate), tiled over query rows and key tokens so the score
// tile stays L2-resident. Interleaved [token][head][channel] layout is
// consumed and produced in place (head stride = width, token stride =
// heads*width). Port of the proven external online tiled attention
// structure.
func launchBlasFullAttention(
	state *device.State,
	functions functionSet,
	blas *blasState,
	query, key, value, keyBias, output driver.DevicePtr,
	queryHeads, queryTokens, keyValueTokens, keyWidth, sequences, chunk, keyChunk uint32,
	scale float32,
) error {
	const softmaxThreads = uint32(256)
	const f64Bytes = uint32(8)
	d := uint64(queryHeads) * uint64(keyWidth)
	rowBytes := d * 4
	scores := blas.scores
	stats := scores + driver.DevicePtr(uint64(queryHeads)*uint64(chunk)*uint64(keyChunk)*4)
	statRows := queryHeads * chunk
	scoreStride := int64(uint64(chunk) * uint64(keyChunk))
	for sequence := uint32(0); sequence < sequences; sequence++ {
		qBase := query + driver.DevicePtr(uint64(sequence)*uint64(queryTokens)*rowBytes)
		kBase := key + driver.DevicePtr(uint64(sequence)*uint64(keyValueTokens)*rowBytes)
		vBase := value + driver.DevicePtr(uint64(sequence)*uint64(keyValueTokens)*rowBytes)
		oBase := output + driver.DevicePtr(uint64(sequence)*uint64(queryTokens)*rowBytes)
		for start := uint32(0); start < queryTokens; start += chunk {
			rows := min(chunk, queryTokens-start)
			qChunk := qBase + driver.DevicePtr(uint64(start)*rowBytes)
			oChunk := oBase + driver.DevicePtr(uint64(start)*rowBytes)
			outputCount := rows * uint32(d)
			initCount := max(outputCount, statRows)
			if err := launch1DABI(
				state, functions[kernelAttentionOnlineInitF32], initCount,
				&oChunk, &stats, &outputCount, &statRows, &initCount,
			); err != nil {
				return err
			}
			for keyStart := uint32(0); keyStart < keyValueTokens; keyStart += keyChunk {
				keys := min(keyChunk, keyValueTokens-keyStart)
				kTile := kBase + driver.DevicePtr(uint64(keyStart)*rowBytes)
				vTile := vBase + driver.DevicePtr(uint64(keyStart)*rowBytes)
				if !traceExternalCall(
					state, traceTagSGEMMStrided,
					uint64(kTile), uint64(qChunk), uint64(scores),
					uint64(keys), uint64(rows), uint64(keyWidth), uint64(queryHeads),
				) {
					if err := blas.library.SGEMMStridedBatched(
						blas.handle, cublas.OperationTranspose, cublas.OperationNone,
						int32(keys), int32(rows), int32(keyWidth), 1,
						kTile, int32(d), int64(keyWidth),
						qChunk, int32(d), int64(keyWidth),
						0,
						scores, int32(keyChunk), scoreStride,
						int32(queryHeads),
					); err != nil {
						return err
					}
				}
				if err := launchGridSharedABI(
					state, functions[kernelAttentionOnlineSoftmaxF32],
					driver.Dim3{X: queryHeads * rows, Y: 1, Z: 1},
					driver.Dim3{X: softmaxThreads, Y: 1, Z: 1}, softmaxThreads*f64Bytes,
					&scores, &oChunk, &stats,
					&keyBias, &keyStart, &keys, &keyChunk, &rows, &chunk, &keyWidth, &queryHeads, &scale,
				); err != nil {
					return err
				}
				if !traceExternalCall(
					state, traceTagSGEMMStrided,
					uint64(vTile), uint64(scores), uint64(oChunk),
					uint64(keyWidth), uint64(rows), uint64(keys), uint64(queryHeads),
				) {
					if err := blas.library.SGEMMStridedBatched(
						blas.handle, cublas.OperationNone, cublas.OperationNone,
						int32(keyWidth), int32(rows), int32(keys), 1,
						vTile, int32(d), int64(keyWidth),
						scores, int32(keyChunk), scoreStride,
						1,
						oChunk, int32(d), int64(keyWidth),
						int32(queryHeads),
					); err != nil {
						return err
					}
				}
			}
			if err := launch1DABI(
				state, functions[kernelAttentionOnlineFinalizeF32], outputCount,
				&oChunk, &stats, &chunk, &keyWidth, &queryHeads, &outputCount,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

func launchAttentionLayout(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	runtimeAttributes tensor.Attributes,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	output := pointers[node]
	switch node.Op {
	case tensor.OpReshape:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		return launch1DABI(state, functions[kernelCopyF32], count, &input, &output, &count)
	case tensor.OpAttention:
		attributes, ok := runtimeAttributes.(tensor.AttentionAttributes)
		if !ok {
			return errors.New("invalid attention attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		queryNode := node.Inputs[0]
		keyNode := node.Inputs[1]
		valueNode := node.Inputs[2]
		keyWidth, err := uint32Checked(queryNode.Shape.Dims[0], "attention key width")
		if err != nil {
			return err
		}
		valueWidth, err := uint32Checked(valueNode.Shape.Dims[0], "attention value width")
		if err != nil {
			return err
		}
		queryHeads, err := uint32Checked(queryNode.Shape.Dims[1], "attention query heads")
		if err != nil {
			return err
		}
		keyValueHeads, err := uint32Checked(keyNode.Shape.Dims[1], "attention KV heads")
		if err != nil {
			return err
		}
		queryTokens, err := uint32Checked(queryNode.Shape.Dims[2], "attention query tokens")
		if err != nil {
			return err
		}
		keyCapacityTokens, err := uint32Checked(keyNode.Shape.Dims[2], "attention KV capacity")
		if err != nil {
			return err
		}
		keyValueTokens := keyCapacityTokens
		if attributes.KeyValueTokens != 0 {
			keyValueTokens = attributes.KeyValueTokens
		}
		if keyValueTokens > keyCapacityTokens {
			return errors.New("attention logical KV tokens exceed capacity")
		}
		sequences := uint32(1)
		if queryNode.Shape.Rank == 4 {
			sequences, err = uint32Checked(queryNode.Shape.Dims[3], "attention sequences")
			if err != nil {
				return err
			}
		}
		if sequences > 1 && keyValueTokens != keyCapacityTokens {
			return errors.New("parameterized attention capacity requires one sequence")
		}
		query := pointers[queryNode]
		key := pointers[keyNode]
		value := pointers[valueNode]
		var relativeBias driver.DevicePtr
		var sinks driver.DevicePtr
		var blockIDs driver.DevicePtr
		var keyBias driver.DevicePtr
		// Optional inputs follow q/k/v from index 3 in a fixed order:
		// (bias|sinks), blockIDs, keyBias -- each present per its attribute flag.
		optIdx := 3
		if attributes.RelativeBuckets != 0 && optIdx < len(node.Inputs) {
			relativeBias = pointers[node.Inputs[optIdx]]
			optIdx++
		} else if attributes.HasSinks && optIdx < len(node.Inputs) {
			sinks = pointers[node.Inputs[optIdx]]
			optIdx++
		}
		if attributes.HasBlockMask && optIdx < len(node.Inputs) {
			blockIDs = pointers[node.Inputs[optIdx]]
			optIdx++
		}
		if attributes.HasKeyBias && optIdx < len(node.Inputs) {
			keyBias = pointers[node.Inputs[optIdx]]
			optIdx++
		}
		relativeBuckets := attributes.RelativeBuckets
		relativeBidirectional := kernelBool(attributes.RelativeBidirectional)
		scale := attributes.Scale
		softcap := attributes.Softcap
		maxALiBiBias := attributes.MaxALiBiBias
		causal := kernelBool(attributes.Causal)
		queryStart := attributes.QueryStart
		window := attributes.Window
		var symmetricWindow uint32
		if attributes.SymmetricWindow {
			const symmetricWindowABI = 1
			symmetricWindow = symmetricWindowABI
		} else if attributes.ChunkedWindow {
			const chunkedWindowABI = 2
			symmetricWindow = chunkedWindowABI
		}
		const (
			attentionDecodeThreads       = uint32(256)
			attentionDecodeSharedLimit   = uint64(48 * 1024)
			attentionDecodePartialFloats = uint64(attentionDecodeThreads)
			f32Bytes                     = uint64(4)
		)
		// shared memory sized to capacity; logical KV count is device-resident
		tokenCountPointer, hasTokenCount := attributePointers[node]
		sharedBytes := (uint64(keyCapacityTokens) + attentionDecodePartialFloats) * f32Bytes
		if queryTokens == 1 && causal != 0 && queryStart+1 == keyValueTokens &&
			relativeBias == 0 && sinks == 0 && blockIDs == 0 && keyBias == 0 && softcap == 0 &&
			maxALiBiBias == 0 && window == 0 && hasTokenCount &&
			sharedBytes <= attentionDecodeSharedLimit {
			blocks := uint64(queryHeads) * uint64(sequences)
			if blocks > math.MaxUint32 {
				return errors.New("decode attention launch size exceeds uint32")
			}
			return launchGridSharedABI(
				state, functions[kernelAttentionDecodeF32],
				driver.Dim3{X: uint32(blocks), Y: 1, Z: 1},
				driver.Dim3{X: attentionDecodeThreads, Y: 1, Z: 1}, uint32(sharedBytes),
				&query, &key, &value, &output, &keyWidth, &valueWidth,
				&queryHeads, &keyValueHeads, &tokenCountPointer, &keyCapacityTokens,
				&sequences, &scale,
			)
		}
		// Strided-batched SGEMM path: exact F32 gemm+softmax+gemm with the
		// score tile bounded by the derived query chunk; owns every
		// qualifying dense-MHA workload when the score staging is resident.
		if !attributes.NaiveF32 && blas != nil && blas.scores != 0 &&
			relativeBias == 0 && sinks == 0 && blockIDs == 0 {
			chunk, keyChunk, scoreBytes, ok := blasAttentionGeometry(
				queryHeads, keyValueHeads, queryTokens, keyValueTokens, keyCapacityTokens,
				keyWidth, valueWidth, attributes,
			)
			if ok && scoreBytes <= blas.scoreBytes {
				return launchBlasFullAttention(
					state, functions, blas,
					query, key, value, keyBias, output,
					queryHeads, queryTokens, keyValueTokens, keyWidth, sequences, chunk, keyChunk,
					scale,
				)
			}
		}
		// Tiled exact path: large featureless non-causal workloads stream
		// K/V through shared memory with an online softmax (scores never
		// reach global memory); one block per 32-query-row tile per head.
		const (
			attentionTiledTile     = uint32(32)
			attentionTiledMaxWidth = uint32(128)
			attentionTiledThreads  = uint32(256)
		)
		tiledShared := uint64(2*attentionTiledTile*keyWidth+attentionTiledTile*valueWidth) * f32Bytes
		if causal == 0 && relativeBias == 0 && sinks == 0 && blockIDs == 0 &&
			softcap == 0 && maxALiBiBias == 0 && window == 0 && symmetricWindow == 0 &&
			queryStart == 0 && keyValueTokens == keyCapacityTokens &&
			keyWidth <= attentionTiledMaxWidth && valueWidth <= attentionTiledMaxWidth &&
			queryTokens >= attentionTiledTile && tiledShared <= attentionDecodeSharedLimit {
			grid := driver.Dim3{
				X: (queryTokens + attentionTiledTile - 1) / attentionTiledTile,
				Y: queryHeads,
				Z: sequences,
			}
			return launchGridSharedABI(
				state, functions[kernelAttentionTiledF32],
				grid, driver.Dim3{X: attentionTiledThreads, Y: 1, Z: 1}, uint32(tiledShared),
				&query, &key, &value, &keyBias, &output, &keyWidth, &valueWidth,
				&queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens,
				&sequences, &scale,
			)
		}
		const attentionOnlineThreads = uint32(256)
		if valueWidth <= attentionOnlineThreads {
			blocks := uint64(queryHeads) * uint64(queryTokens) * uint64(sequences)
			if blocks > math.MaxUint32 {
				return errors.New("online attention launch size exceeds uint32")
			}
			return launchGridABI(
				state, functions[kernelAttentionOnlineF32],
				driver.Dim3{X: uint32(blocks), Y: 1, Z: 1},
				driver.Dim3{X: attentionOnlineThreads, Y: 1, Z: 1},
				&query, &key, &value, &relativeBias, &sinks, &blockIDs, &keyBias, &output,
				&keyWidth, &valueWidth, &queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &sequences,
				&scale, &softcap, &maxALiBiBias, &causal, &queryStart, &window, &symmetricWindow,
				&relativeBuckets, &relativeBidirectional,
			)
		}
		return launch1DABI(
			state, functions[kernelAttentionF32], count,
			&query, &key, &value, &relativeBias, &sinks, &blockIDs, &keyBias, &output,
			&keyWidth, &valueWidth, &queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &sequences,
			&scale, &softcap, &maxALiBiBias, &causal, &queryStart, &window, &symmetricWindow,
			&relativeBuckets, &relativeBidirectional, &count,
		)
	case tensor.OpConcat:
		attributes, ok := runtimeAttributes.(tensor.ConcatAttributes)
		if !ok || attributes.Axis >= uint32(node.Shape.Rank) {
			return errors.New("invalid concat attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		if output == left && attributes.Axis+1 == uint32(node.Shape.Rank) {
			leftBytes, sizeErr := node.Inputs[0].Shape.Bytes(dtype.F32)
			if sizeErr != nil || uint64(output) > math.MaxUint64-leftBytes {
				return errors.New("concat append offset overflows")
			}
			rightCount, countErr := elementCount32(node.Inputs[1].Shape)
			if countErr != nil {
				return countErr
			}
			destination := output + driver.DevicePtr(leftBytes)
			return launch1DABI(
				state, functions[kernelCopyF32], rightCount,
				&right, &destination, &rightCount,
			)
		}
		var inner uint64 = 1
		for dimension := uint32(0); dimension < attributes.Axis; dimension++ {
			inner *= node.Shape.Dims[dimension]
		}
		innerSize, err := uint32Checked(inner, "concat inner size")
		if err != nil {
			return err
		}
		leftAxis, err := uint32Checked(node.Inputs[0].Shape.Dims[attributes.Axis], "concat left axis")
		if err != nil {
			return err
		}
		rightAxis, err := uint32Checked(node.Inputs[1].Shape.Dims[attributes.Axis], "concat right axis")
		if err != nil {
			return err
		}
		axis := attributes.Axis
		return launch1DABI(
			state, functions[kernelConcatF32], count,
			&left, &right, &output, &innerSize, &leftAxis, &rightAxis, &axis, &count,
		)
	case tensor.OpCacheAppend:
		attributes, ok := runtimeAttributes.(tensor.CacheAppendAttributes)
		if !ok || attributes.Axis+1 != uint32(node.Shape.Rank) {
			return errors.New("invalid cache append attributes")
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		leftCount, err := elementCount32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		rightCount, err := elementCount32(node.Inputs[1].Shape)
		if err != nil {
			return err
		}
		outputCount, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		if output != left {
			if err := launch1DABI(
				state, functions[kernelCopyF32], leftCount, &left, &output, &leftCount,
			); err != nil {
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
		if inner != 0 && uint64(attributes.Offset) > math.MaxUint64/inner {
			return errors.New("cache append offset overflows")
		}
		offset := uint64(attributes.Offset) * inner
		if offset > uint64(outputCount) || uint64(rightCount) > uint64(outputCount)-offset {
			return errors.New("cache append range exceeds capacity")
		}
		// destination base stays fixed; the row offset is device-resident so
		// decode launches stay byte-identical across steps
		offsetPointer, ok := attributePointers[node]
		if !ok {
			return errors.New("cache append offset storage is unavailable")
		}
		innerCount, err := uint32Checked(inner, "cache append inner size")
		if err != nil {
			return err
		}
		return launch1DABI(
			state, functions[kernelCopyTokenOffsetF32], rightCount,
			&right, &output, &offsetPointer, &innerCount, &rightCount,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
