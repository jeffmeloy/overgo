//go:build windows

package executor

import (
	"encoding/binary"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// TestKVStreamMatchesResident is Strata's src/kernels/kv_stream_parity.cpp for its int8 pools, transcribed: a
// synthetic sequence decoded twice, into a fully resident pool behind the identity page table and into a streamed
// layer (the host copy and a VRAM slot pool too small for the sequence, so it evicts all along), and every few cells
// a batch of one to eight queries, each with a selection shaped like the indexer's (whole 4-cell blocks, one partial
// block, the tail block, half of them recent), attended in both with the resolve before the streamed attention.
// The attention outputs are bitwise equal, the residency map inverts itself after every call, and no call overflows
// while the counters add up with misses among the lookups.
func TestKVStreamMatchesResident(t *testing.T) {
	const (
		cells         = 40000
		blocks        = (cells + strataPageCells - 1) / strataPageCells
		slots         = 8*516 + 700 // fewer than the blocks: the layer must evict
		queriesMost   = 8
		batchEvery    = 131
		spikeEvery    = 17
		spike         = 30
		recentBlocks  = 2048
		decisionCoins = 2
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	capacity := strataSelectionWidth(strataTopKMaxCells)
	geometry := strataKVGeometry{kvHeads: strataKVHeads, headDim: strataHeadWidth, pageSize: strataPageCells}
	codes, scales := strataStreamRunBytes(geometry)
	pools := func(pages int) strataKVPools {
		return strataKVPools{
			key: buffers.zeros(t, pages*int(codes)), value: buffers.zeros(t, pages*int(codes)),
			keyScale: buffers.zeros(t, pages*int(scales)), valueScale: buffers.zeros(t, pages*int(scales)),
		}
	}
	resident, slotPools := pools(blocks), pools(slots)
	// The host copy: pinned, device-mapped memory at the identity layout.
	var hostBuffers [][]byte
	host := strataKVPools{}
	for _, target := range []*driver.DevicePtr{&host.key, &host.value, &host.keyScale, &host.valueScale} {
		bytes := blocks * int(codes)
		if target == &host.keyScale || target == &host.valueScale {
			bytes = blocks * int(scales)
		}
		buffers.do(t, func(state *device.State) error {
			data, pointer, err := state.Driver.MemHostAlloc(uint64(bytes))
			clear(data)
			hostBuffers = append(hostBuffers, data)
			*target = pointer
			return err
		})
	}
	defer buffers.do(t, func(state *device.State) error {
		for _, data := range hostBuffers {
			if err := state.Driver.MemFreeHost(data); err != nil {
				return err
			}
		}
		return nil
	})
	identity := make([]int32, blocks)
	for block := range identity {
		identity[block] = int32(block)
	}
	identityTable := buffers.upload(t, int32Bytes(identity))
	stream := strataKVStreamMap{
		table: buffers.zeros(t, 4*blocks), slotBlock: buffers.zeros(t, 4*slots), slotStamp: buffers.zeros(t, 4*slots),
		slotRef: buffers.zeros(t, 4*slots), missBlock: buffers.zeros(t, 4*slots), missSlot: buffers.zeros(t, 4*slots),
		ctl: buffers.zeros(t, 4*strataStreamControlInts), blocks: blocks, slots: slots,
	}
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launchStrataKVStreamReset(state, functions, stream)
	})

	kvValues := strataKVHeads * strataHeadWidth
	queryValues := strataQueryHeads * strataHeadWidth
	// One upload a cell: its keys, its values and its step record.
	step := buffers.zeros(t, 4*(2*kvValues+tensor.QSAStepWords))
	keys, values := step, step+driver.DevicePtr(4*kvValues)
	record := values + driver.DevicePtr(4*kvValues)
	ids := buffers.zeros(t, 4*queriesMost*int(capacity))
	steps := buffers.zeros(t, 4*queriesMost*tensor.QSAStepWords)
	queries := buffers.zeros(t, 4*queriesMost*queryValues)
	scratch := buffers.zeros(t, 4*queriesMost*int(strataAttentionScratchFloats(capacity, strataQueryHeads)))
	residentOut, streamedOut := buffers.zeros(t, 4*queriesMost*queryValues), buffers.zeros(t, 4*queriesMost*queryValues)

	draws := rand.New(rand.NewPCG(7, 11))
	normal := func(scale float64) float32 { return float32(draws.NormFloat64() * scale) }
	batches := 0
	for position := range cells {
		upload := make([]float32, 2*kvValues)
		for index := range kvValues {
			scale := 1.0
			if position%spikeEvery == 0 {
				scale = spike
			}
			upload[index], upload[kvValues+index] = normal(scale), normal(1)
		}
		held := int32(position + 1)
		data := append(float32Bytes(upload), int32Bytes([]int32{int32(position), held, held / strataPageCells, 0})...)
		buffers.write(t, step, data)
		buffers.run(t, func(state *device.State, functions functionSet) error {
			if err := launchStrataKVAppend(state, functions, resident, strataKVPools{}, identityTable, record, keys, values, geometry, true); err != nil {
				return err
			}
			return launchStrataKVAppend(state, functions, slotPools, host, stream.table, record, keys, values, geometry, true)
		})
		if position%batchEvery != batchEvery-1 && position != cells-1 {
			continue
		}
		// A batch: queryCount queries at the last positions, as a verify window, each with its own selection.
		queryCount := 1 + draws.IntN(queriesMost)
		selected := make([]int32, queriesMost*int(capacity))
		records := make([]int32, queriesMost*tensor.QSAStepWords)
		for query := range queryCount {
			at := max(0, position-(queryCount-1-query))
			held := int64(at + 1)
			width := min(held, capacity)
			selection := streamSelection(held, width, draws, recentBlocks, decisionCoins)
			if int64(len(selection)) != width {
				t.Fatalf("selection of %d cells, want %d", len(selection), width)
			}
			copy(selected[query*int(capacity):], selection)
			copy(records[query*tensor.QSAStepWords:], []int32{int32(at), int32(held), int32(held / strataPageCells), int32(width)})
		}
		queryData := make([]float32, queriesMost*queryValues)
		for index := range queryData {
			queryData[index] = normal(1)
		}
		buffers.write(t, ids, int32Bytes(selected))
		buffers.write(t, steps, int32Bytes(records))
		buffers.write(t, queries, float32Bytes(queryData))
		buffers.run(t, func(state *device.State, functions functionSet) error {
			if err := launchStrataDecodeAttention(state, functions, queries, strataAttentionPools{kv: resident, table: identityTable},
				ids, steps, capacity, strataKVHeads, strataPageCells, scratch, residentOut, int64(queryCount), true); err != nil {
				return err
			}
			if err := launchStrataKVStreamResolve(state, functions, stream, slotPools, host, ids, steps, int64(queryCount), capacity, geometry); err != nil {
				return err
			}
			return launchStrataDecodeAttention(state, functions, queries, strataAttentionPools{kv: slotPools, table: stream.table},
				ids, steps, capacity, strataKVHeads, strataPageCells, scratch, streamedOut, int64(queryCount), true)
		})
		bytes := 4 * queryCount * queryValues
		if !slices.Equal(buffers.read(t, residentOut, bytes), buffers.read(t, streamedOut, bytes)) {
			t.Fatalf("position %d, %d queries: the streamed attention differs from the resident one", position, queryCount)
		}
		// The map inverts itself.
		table := int32s(buffers.read(t, stream.table, 4*blocks))
		slotBlocks := int32s(buffers.read(t, stream.slotBlock, 4*slots))
		for block, slot := range table {
			if slot < -1 || slot >= slots || slot >= 0 && slotBlocks[slot] != int32(block) {
				t.Fatalf("position %d: the map is broken at block %d (table %d)", position, block, slot)
			}
		}
		for slot, block := range slotBlocks {
			if block >= 0 && table[block] != int32(slot) {
				t.Fatalf("position %d: slot %d holds block %d, which the table does not name", position, slot, block)
			}
		}
		batches++
	}
	control := buffers.read(t, stream.ctl, 4*strataStreamControlInts)
	overflow := binary.LittleEndian.Uint32(control[3*4:])
	misses, lookups, calls := binary.LittleEndian.Uint64(control[4*4:]), binary.LittleEndian.Uint64(control[6*4:]), binary.LittleEndian.Uint64(control[8*4:])
	t.Logf("%d batches, %d block lookups, %d misses (%.1f%% hit)", batches, lookups, misses, 100*float64(lookups-misses)/float64(lookups))
	if overflow != 0 || calls != uint64(batches) || misses == 0 || misses >= lookups {
		t.Fatalf("the counters: overflow %d, %d calls of %d batches, %d misses of %d lookups", overflow, calls, batches, misses, lookups)
	}
}

// streamSelection is the parity program's selection(): ascending cells of held, width of them - whole blocks, half
// drawn from the recent blocks, one block taken in part, and the tail block's cells.
func streamSelection(held, width int64, draws *rand.Rand, recentBlocks int64, coins int) []int32 {
	ids := make([]int32, 0, width)
	if held <= width {
		for cell := range held {
			ids = append(ids, int32(cell))
		}
		return ids
	}
	completed := held / strataPageCells
	tail := held - completed*strataPageCells
	whole, part := (width-tail)/strataPageCells, (width-tail)%strataPageCells
	want := whole
	if part != 0 {
		want++
	}
	chosen := map[int64]bool{}
	for int64(len(chosen)) < want {
		if draws.IntN(coins) == 0 {
			chosen[max(0, completed-recentBlocks)+draws.Int64N(min(recentBlocks, completed))] = true
		} else {
			chosen[draws.Int64N(completed)] = true
		}
	}
	ordered := make([]int64, 0, len(chosen))
	for block := range chosen {
		ordered = append(ordered, block)
	}
	slices.Sort(ordered)
	partial := int64(-1)
	if part != 0 {
		partial = ordered[draws.IntN(len(ordered))]
	}
	for _, block := range ordered {
		take := int64(strataPageCells)
		if block == partial {
			take = part
		}
		for cell := range take {
			ids = append(ids, int32(block*strataPageCells+cell))
		}
	}
	for cell := completed * strataPageCells; cell < held; cell++ {
		ids = append(ids, int32(cell))
	}
	slices.Sort(ids)
	return ids
}

// launchStrataKVStreamReset is kv_stream_reset: every block evicted, every slot free, the control ints zero, on its
// 128 blocks of 256 threads. The runtime writes a reset map from the host when it allocates a streamed state.
func launchStrataKVStreamReset(state *device.State, functions functionSet, stream strataKVStreamMap) error {
	blocks, threads := uint32(128), uint32(256)
	return launchGridABI(state, functions[kernelStrataKvStreamReset],
		driver.Dim3{X: blocks, Y: 1, Z: 1}, driver.Dim3{X: threads, Y: 1, Z: 1},
		&stream.table, &stream.slotBlock, &stream.slotStamp, &stream.slotRef, &stream.ctl, &stream.blocks, &stream.slots)
}

func int32s(data []byte) []int32 {
	values := make([]int32, len(data)/4)
	for index := range values {
		values[index] = int32(binary.LittleEndian.Uint32(data[4*index:]))
	}
	return values
}
