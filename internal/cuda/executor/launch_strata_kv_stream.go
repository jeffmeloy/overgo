package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's K/V streaming for a QSA layer whose cells outgrow its VRAM slots (kernels/cuda/strata_qsa.cu's
// strata_kv_stream_*, ported from Strata at 296b59d2, MIT: src/kernels/cuda/kv_stream.cu), and its prompt read's
// append (strata_prefill_kv_append_q8: src/prefill/kernels.cu kv_append_kernel). The cells' authority is a host copy
// in pinned, device-mapped memory at the identity layout; the VRAM pools hold n_slots pages behind the page table,
// now a residency map (block -> slot, or -1).

// strataKVStreamMap is kv_stream.hpp's KvStreamMap: the residency map's arrays at fixed addresses.
type strataKVStreamMap struct {
	table, slotBlock, slotStamp, slotRef, ctl, missBlock, missSlot driver.DevicePtr
	blocks, slots                                                  int64
}

// The stream's launch constants (kv_stream.cu): the resolve's one block of 1,024 threads (RT), whose clock sweep
// would take a slot twice with fewer slots than threads; the copy's 96 blocks of 128 threads. Provisional: the load-costs step prices these launch shapes, Strata's as tuned on its author's cards.
const (
	strataStreamResolveThreads = uint32(1024)
	strataStreamCopyBlocks     = uint32(96)
	strataStreamCopyThreads    = uint32(128)
	// strataStreamControlInts is kKvCtlInts: epoch, hand, the last call's misses, overflow, and three u64 counters.
	strataStreamControlInts = 16
	// strataStreamSlotArrays is the map's per-slot arrays: slot_block, slot_stamp, slot_ref, miss_block, miss_slot.
	strataStreamSlotArrays = 5
	// strataStreamCopyRuns is an int8 block's runs: K codes, V codes, K scales, V scales.
	strataStreamCopyRuns = 4
	// strataStreamRunAlignment is the copy's 16-byte vector a thread; every run is a whole number of them.
	strataStreamRunAlignment = int64(16)
)

// strataStreamRunBytes are an int8 block's runs: its codes (all K/V heads' page_size cells) and its scales.
func strataStreamRunBytes(geometry strataKVGeometry) (codes, scales int32) {
	rows := geometry.kvHeads * geometry.pageSize
	return rows * geometry.headDim, rows * (geometry.headDim / int32(strataKVQ8Group)) * int32(f16ScalarBytes)
}

// launchStrataKVStreamResolve is kv_stream_resolve: the blocks the queryCount selections name (ids [query][capacity],
// widths from the step records) made resident, then the missed blocks copied from the host copy into their slots.
func launchStrataKVStreamResolve(
	state *device.State, functions functionSet, stream strataKVStreamMap, slots, host strataKVPools,
	ids, steps driver.DevicePtr, queryCount, capacity int64, geometry strataKVGeometry,
) error {
	if queryCount <= 0 {
		return nil
	}
	codes, scales := strataStreamRunBytes(geometry)
	if int64(scales)%strataStreamRunAlignment != 0 || int64(codes)%strataStreamRunAlignment != 0 {
		return errors.New("a streamed block's runs must be whole 16-byte vectors")
	}
	if stream.slots < int64(strataStreamResolveThreads) {
		return errors.New("a streamed layer takes at least as many slots as the resolve's threads")
	}
	count, width, page := int32(queryCount), int32(capacity), geometry.pageSize
	if err := launchGridABI(state, functions[kernelStrataKvStreamResolve],
		driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: strataStreamResolveThreads, Y: 1, Z: 1},
		&stream.table, &stream.slotBlock, &stream.slotStamp, &stream.slotRef, &stream.ctl, &stream.missBlock,
		&stream.missSlot, &stream.slots, &ids, &steps, &count, &width, &page); err != nil {
		return err
	}
	return launchGridABI(state, functions[kernelStrataKvStreamCopy],
		driver.Dim3{X: strataStreamCopyBlocks, Y: 1, Z: 1}, driver.Dim3{X: strataStreamCopyThreads, Y: 1, Z: 1},
		&stream.ctl, &stream.missBlock, &stream.missSlot,
		&host.key, &host.value, &host.keyScale, &host.valueScale,
		&slots.key, &slots.value, &slots.keyScale, &slots.valueScale, &codes, &codes, &scales, &scales)
}

// launchStrataPrefillKVAppend is prefill's kv_append: a chunk's tokens cells from first into the resident slots through
// the page table, the host copy and the staging pool, one launch.
func launchStrataPrefillKVAppend(
	state *device.State, functions functionSet, keys, values driver.DevicePtr, first, tokens int64,
	table driver.DevicePtr, slots, host, stage strataKVPools, geometry strataKVGeometry,
) error {
	if tokens <= 0 {
		return nil
	}
	page := int64(geometry.pageSize)
	return launchGridABI(state, functions[kernelStrataPrefillKvAppendQ8],
		driver.Dim3{X: uint32(tokens), Y: uint32(geometry.kvHeads), Z: strataKVHalves * (uint32(geometry.headDim) / strataKVQ8Group)},
		driver.Dim3{X: strataKVQ8Group, Y: 1, Z: 1},
		&keys, &values, &first, &table, &page,
		&slots.key, &slots.value, &slots.keyScale, &slots.valueScale,
		&host.key, &host.value, &host.keyScale, &host.valueScale,
		&stage.key, &stage.value, &stage.keyScale, &stage.valueScale)
}

// launchStrataKVStage is kv_stage_from_host: the host copy's first blocks into the staging pool, the prompt read's
// one layer at a time, here by the copy kernel reading the mapped host memory (the driver's copies take only the
// card's own allocations).
func launchStrataKVStage(
	state *device.State, functions functionSet, host, stage strataKVPools, blocks int64, geometry strataKVGeometry,
) error {
	if blocks <= 0 {
		return nil
	}
	codes, scales := strataStreamRunBytes(geometry)
	for _, run := range []struct {
		source, target driver.DevicePtr
		bytes          int64
	}{
		{host.key, stage.key, blocks * int64(codes)}, {host.value, stage.value, blocks * int64(codes)},
		{host.keyScale, stage.keyScale, blocks * int64(scales)}, {host.valueScale, stage.valueScale, blocks * int64(scales)},
	} {
		count := uint32(run.bytes / f32Bytes)
		if err := launch1DABI(state, functions[kernelCopyF32], count, &run.source, &run.target, &count); err != nil {
			return err
		}
	}
	return nil
}
