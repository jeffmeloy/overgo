package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's decode round's copies through mapped memory, doorbell, embedding gather, stream broadcast and experts' q8_1
// quantizer (kernels/cuda/strata_round_elementwise.cuh, ported from Strata at 296b59d2, MIT: elementwise.cu,
// verify_kernels.cu, iq_kernels.cu), their host sides translated.

// The launches' geometry, which each kernel's indexing fixes.
const (
	strataCopyThreads        = 256  // copy_from_mapped's block
	strataCopyMaxBlocks      = 64   // copy_from_mapped's most blocks, a grid-stride loop past them
	strataCopyFloatsAVector  = 4    // a float4
	strataCopyVectorAlign    = 16   // a float4's alignment
	strataCopyI32Threads     = 128  // copy_i32_from_mapped's one block
	strataCopyRowsThreads    = 128  // copy_rows_from_mapped's block a row
	strataDoorbellThreads    = 1024 // doorbell_publish_res's one block, and its most routed slots
	strataBroadcastThreads   = 256  // broadcast_streams' block
	strataEmbedRowValues     = 256  // embed_rows' block of a row: one QK_K superblock
	strataEmbedRowThreads    = 32   // embed_rows' threads a block
	strataQuantizeQ81Threads = 256  // quantize_q8_1_rows' block
)

// launchStrataCopyFromMapped is copy_from_mapped: n floats from src (mapped host memory, read volatile) to dst, both
// 16-byte aligned and n a multiple of 4.
func launchStrataCopyFromMapped(state *device.State, functions functionSet, dst, src driver.DevicePtr, n int64) error {
	if n <= 0 {
		return nil
	}
	if n%strataCopyFloatsAVector != 0 || dst%strataCopyVectorAlign != 0 || src%strataCopyVectorAlign != 0 {
		return fmt.Errorf("copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned")
	}
	vectors := n / strataCopyFloatsAVector
	blocks := min((vectors+strataCopyThreads-1)/strataCopyThreads, strataCopyMaxBlocks)
	return launchGridABI(state, functions[kernelStrataCopyFromMapped], driver.Dim3{X: uint32(blocks), Y: 1, Z: 1},
		driver.Dim3{X: strataCopyThreads, Y: 1, Z: 1}, &dst, &src, &vectors)
}

// launchStrataCopyI32FromMapped is copy_i32_from_mapped: n int32 words from src (mapped host memory) to dst.
func launchStrataCopyI32FromMapped(state *device.State, functions functionSet, dst, src driver.DevicePtr, n int64) error {
	if n <= 0 {
		return nil
	}
	words := int32(n)
	return launchGridABI(state, functions[kernelStrataCopyI32FromMapped], driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: strataCopyI32Threads, Y: 1, Z: 1}, &dst, &src, &words)
}

// launchStrataCopyRowsFromMapped is copy_rows_from_mapped (flag 0) and copy_rows_from_mapped_after: rows rows of width
// floats from src to dst, a row the plan's hit rows name (hitRows[0, *count)) written +0.0, after the flag reaches value
// where flag is set.
func launchStrataCopyRowsFromMapped(state *device.State, functions functionSet, dst, src driver.DevicePtr, rows, width int64,
	hitRows, count, flag driver.DevicePtr, value uint32) error {
	if rows <= 0 {
		return nil
	}
	if width%strataCopyFloatsAVector != 0 || dst%strataCopyVectorAlign != 0 || src%strataCopyVectorAlign != 0 {
		return fmt.Errorf("copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned")
	}
	vectors := width / strataCopyFloatsAVector
	return launchGridABI(state, functions[kernelStrataCopyRowsFromMapped], driver.Dim3{X: uint32(rows), Y: 1, Z: 1},
		driver.Dim3{X: strataCopyRowsThreads, Y: 1, Z: 1}, &dst, &src, &vectors, &hitRows, &count, &flag, &value)
}

// launchStrataDoorbellPublishRes is doorbell_publish_res: the window rows' routed experts (k of them) to idsOut, the n
// activations to xOut only where an expert is not resident by dRes (nExpert a layer), then the ring seq advanced.
func launchStrataDoorbellPublishRes(state *device.State, functions functionSet, x, ids, dRes driver.DevicePtr, nExpert int32,
	n, k int64, xOut, idsOut, seq driver.DevicePtr) error {
	if k > strataDoorbellThreads {
		return fmt.Errorf("doorbell_publish_res: k too large")
	}
	values, slots := int32(n), int32(k)
	return launchGridABI(state, functions[kernelStrataDoorbellPublishRes], driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: strataDoorbellThreads, Y: 1, Z: 1}, &x, &ids, &dRes, &nExpert, &values, &slots, &xOut, &idsOut, &seq)
}

// launchStrataBroadcastStreams is broadcast_streams: each of tokens rows of embedding values copied into its hc streams.
func launchStrataBroadcastStreams(state *device.State, functions functionSet, x, r driver.DevicePtr, embedding int64, hc, tokens int32) error {
	return launchGridABI(state, functions[kernelStrataBroadcastStreams],
		driver.Dim3{X: uint32((embedding*int64(hc) + strataBroadcastThreads - 1) / strataBroadcastThreads), Y: uint32(tokens), Z: 1},
		driver.Dim3{X: strataBroadcastThreads, Y: 1, Z: 1}, &x, &r, &embedding, &hc)
}

// launchStrataEmbedRowsQ80 is iq_embed_rows over a Q8_0 table (NativeEmbed::gather_dev): each token's row, rowBytes a
// row, dequantized into out.
func launchStrataEmbedRowsQ80(state *device.State, functions functionSet, table driver.DevicePtr, rowBytes uint64,
	tokens driver.DevicePtr, count int32, embedding int64, out driver.DevicePtr) error {
	if count <= 0 {
		return nil
	}
	if embedding%strataEmbedRowValues != 0 {
		return fmt.Errorf("iq_embed_rows: bad arguments")
	}
	return launchGridABI(state, functions[kernelStrataEmbedRowsQ80],
		driver.Dim3{X: uint32(embedding / strataEmbedRowValues), Y: uint32(count), Z: 1},
		driver.Dim3{X: strataEmbedRowThreads, Y: 1, Z: 1}, &table, &rowBytes, &tokens, &embedding, &out)
}

// verify_kernels.cu's head picks: the multi-block argmax (kArgMaxBlocks, kArgThreads, a block per 4,096 values) and the
// #879 guard (kNonfiniteBlocks slices a row).
const (
	strataArgMaxBlocks       = 128
	strataArgThreads         = 256
	strataArgValuesABlock    = 4096
	strataNonfiniteBlocks    = 16
	strataNonfiniteThreads   = 256
	strataArgRowHeaderWords  = 4 // ArgRow's counter and its padding
	strataArgRowWordBytes    = 4
	strataArgRowBlockEntries = 2 // a block's value and index
)

// strataArgmaxScratchSize is argmax_rows_scratch_bytes: an ArgRow a row, its counter zero before the first launch and
// left zero by each.
func strataArgmaxScratchSize(rows int32) uint64 {
	return uint64(rows) * (strataArgRowHeaderWords + strataArgRowBlockEntries*strataArgMaxBlocks) * strataArgRowWordBytes
}

// launchStrataArgmaxRows is argmax_rows: each of rows rows of n logits to its largest value's index (the lower on a tie,
// 0 for a row of no value above -inf), over at most kArgMaxBlocks blocks a row, the last block to finish picking.
func launchStrataArgmaxRows(state *device.State, functions functionSet, logits driver.DevicePtr, rows, n int32, scratch, out driver.DevicePtr) error {
	if rows <= 0 {
		return nil
	}
	blocks := min(int32(strataArgMaxBlocks), max(1, (n+strataArgValuesABlock-1)/strataArgValuesABlock))
	perBlock := (n + blocks - 1) / blocks
	return launchGridABI(state, functions[kernelStrataArgmaxRows], driver.Dim3{X: uint32(blocks), Y: uint32(rows), Z: 1},
		driver.Dim3{X: strataArgThreads, Y: 1, Z: 1}, &logits, &n, &perBlock, &scratch, &out)
}

// launchStrataLogitsNonfinite is logits_nonfinite_rows: flags[t * kNonfiniteBlocks + b] = 1 where slice b of row t
// holds a NaN or an infinity.
func launchStrataLogitsNonfinite(state *device.State, functions functionSet, logits driver.DevicePtr, rows, n int32, flags driver.DevicePtr) error {
	if rows <= 0 || n <= 0 {
		return nil
	}
	perBlock := (n + strataNonfiniteBlocks - 1) / strataNonfiniteBlocks
	return launchGridABI(state, functions[kernelStrataLogitsNonfinite],
		driver.Dim3{X: strataNonfiniteBlocks, Y: uint32(rows), Z: 1}, driver.Dim3{X: strataNonfiniteThreads, Y: 1, Z: 1},
		&logits, &n, &perBlock, &flags)
}

// strataSigmoidScaleThreads is shared_expert.cu's THREADS.
const strataSigmoidScaleThreads = 256

// launchStrataSigmoidScaleRows is launch_sigmoid_scale_rows: each of tokens rows of embedding outputs times the sigmoid
// of its gate g[t], four floats a thread where the row is a whole number of float4s and out is 16-byte aligned.
func launchStrataSigmoidScaleRows(state *device.State, functions functionSet, out, g driver.DevicePtr, embedding, tokens int32) error {
	if embedding%strataCopyFloatsAVector == 0 && out%strataCopyVectorAlign == 0 {
		vectors := embedding / strataCopyFloatsAVector
		return launchGridABI(state, functions[kernelStrataSigmoidScaleRowsVec4],
			driver.Dim3{X: (uint32(vectors) + strataSigmoidScaleThreads - 1) / strataSigmoidScaleThreads, Y: uint32(tokens), Z: 1},
			driver.Dim3{X: strataSigmoidScaleThreads, Y: 1, Z: 1}, &out, &g, &vectors)
	}
	return launchGridABI(state, functions[kernelStrataSigmoidScaleRows],
		driver.Dim3{X: (uint32(embedding) + strataSigmoidScaleThreads - 1) / strataSigmoidScaleThreads, Y: uint32(tokens), Z: 1},
		driver.Dim3{X: strataSigmoidScaleThreads, Y: 1, Z: 1}, &out, &g, &embedding)
}

// launchStrataQuantizeQ81Rows is quantize_q8_1_rows: rows rows of cols values as q8_1 blocks, IEEE arithmetic.
func launchStrataQuantizeQ81Rows(state *device.State, functions functionSet, x driver.DevicePtr, rows, cols int64, y driver.DevicePtr) error {
	n := rows * cols
	if n <= 0 {
		return nil
	}
	return launchGridABI(state, functions[kernelStrataQuantizeQ81],
		driver.Dim3{X: uint32((n + strataQuantizeQ81Threads - 1) / strataQuantizeQ81Threads), Y: 1, Z: 1},
		driver.Dim3{X: strataQuantizeQ81Threads, Y: 1, Z: 1}, &x, &y, &n)
}

// The expert tier's launches' geometry, which each kernel's indexing fixes.
const (
	strataFetchBlobsBlocks  = 48 * 8 // fetch_blobs' grid
	strataFetchBlobsThreads = 256
	strataFetchBlobsWord    = 16  // a uint4
	strataRebaseThreads     = 128 // rebase_ptrs' one block: the most blobs it rebases
	strataHitAddThreads     = 256 // moe_hit_add's block
	strataHitAddMaxBlocks   = 8   // moe_hit_add's most blocks a row
)

// launchStrataWaitFlagGE is wait_flag_ge: the stream waits, one thread spinning, until the mapped flag reaches value.
func launchStrataWaitFlagGE(state *device.State, functions functionSet, flag driver.DevicePtr, value uint32) error {
	return launchGridABI(state, functions[kernelStrataWaitFlagGe], driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 1, Y: 1, Z: 1},
		&flag, &value)
}

// launchStrataFetchBlobs is fetch_blobs: the first *n of the blobs src points at (blobBytes each, a multiple of 16),
// copied to dst back to back; capacity is the most there may be.
func launchStrataFetchBlobs(state *device.State, functions functionSet, src, n, dst driver.DevicePtr, blobBytes int64, capacity int) error {
	if capacity <= 0 {
		return nil
	}
	if blobBytes%strataFetchBlobsWord != 0 {
		return fmt.Errorf("fetch_blobs: blob size must be a multiple of 16")
	}
	per := blobBytes / strataFetchBlobsWord
	return launchGridABI(state, functions[kernelStrataFetchBlobs], driver.Dim3{X: strataFetchBlobsBlocks, Y: 1, Z: 1},
		driver.Dim3{X: strataFetchBlobsThreads, Y: 1, Z: 1}, &src, &n, &dst, &per)
}

// launchStrataRebasePtrs is rebase_ptrs: the first *n pointers of ptr set to the blobs fetch_blobs packed at base.
func launchStrataRebasePtrs(state *device.State, functions functionSet, ptr, n, base driver.DevicePtr, blobBytes int64) error {
	return launchGridABI(state, functions[kernelStrataRebasePtrs], driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: strataRebaseThreads, Y: 1, Z: 1}, &ptr, &n, &base, &blobBytes)
}

// launchStrataMoEHitAdd is moe_hit_add: the rows dst[0, *count) of parts (embedding floats each) gain hitOut's;
// capacity is the most rows there may be.
func launchStrataMoEHitAdd(state *device.State, functions functionSet, parts, hitOut, dst, count driver.DevicePtr,
	capacity, embedding int64) error {
	if capacity <= 0 {
		return nil
	}
	blocks := min((embedding+strataHitAddThreads-1)/strataHitAddThreads, strataHitAddMaxBlocks)
	width := int32(embedding)
	return launchGridABI(state, functions[kernelStrataAddHits], driver.Dim3{X: uint32(blocks), Y: uint32(capacity), Z: 1},
		driver.Dim3{X: strataHitAddThreads, Y: 1, Z: 1}, &parts, &hitOut, &dst, &count, &width)
}
