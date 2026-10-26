package inference

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
)

// A sequence's QSA cells sit on the card while its capacity fits the ring, as Strata's kv_plan holds a state fully
// resident (mode 0); past the ring they stream (mode 1, Strata's kv_stream.hpp): each layer keeps the ring's pages as
// VRAM slots behind its own page table, now a residency map, the cells' authority is a host copy in pinned,
// device-mapped memory, and a step's selected blocks are made resident before its attention reads them. The pooled
// rows, the block position and the image position table stay on the card.

// qsaRingCells is the cells a layer keeps on the card once its sequence streams: Strata's benched --kv-resident,
// above its qsa_kv_resident_min() of 20,480. Provisional: the load-costs step derives it from the card's room against
// a resolve's misses.
const qsaRingCells = 32768

// qsaCells is one sequence's QSA cells at a capacity: its card buffer and, when it streams, its host copy.
type qsaCells struct {
	buffer   *executor.DeviceBuffer
	host     *executor.PinnedHostBuffer
	capacity uint32
	// slots is a streamed sequence's VRAM pages a layer, zero for a sequence on the card.
	slots     uint32
	layers    []qsaLayerCells
	pageTable executor.DeviceValue
	positions executor.DeviceValue
	// mirror is the image position table as the device holds it: a cell's temporal, height and width positions.
	mirror []int32
}

// qsaLayerCells is one QSA layer's cells; zero for a layer that is not one. A streamed layer adds its own page table,
// its stream map, and its host copy's device-mapped regions with their offsets in the host buffer.
type qsaLayerCells struct {
	keyCodes, valueCodes, keyScales, valueScales, pooled, blockPosition executor.DeviceValue
	pageTable, streamMap                                                executor.DeviceValue
	host                                                                [qsaHostRegions]executor.DeviceValue
	hostOffsets                                                         [qsaHostRegions]uint64
}

// The host copy's regions a layer, in the order the copy kernels take them: K codes, V codes, K scales, V scales.
const qsaHostRegions = 4

// qsaRegionAlignment keeps each region of the cells at a boundary the kernels' vector loads accept.
const qsaRegionAlignment = uint64(256)

// qsaPoolHalves is a layer's pools of each kind: its keys' and its values'.
const qsaPoolHalves = 2

// qsaCodePageBytes and qsaScalePageBytes are a 4-cell page's codes and scales over the K/V heads.
const (
	qsaCodePageBytes  = tensor.QSAKeyValueHeads * tensor.QSABlockCells * tensor.QSAHeadWidth
	qsaScalePageBytes = tensor.QSAKeyValueHeads * tensor.QSABlockCells * tensor.QSAHeadWidth / tensor.QSAScaleGroup * f16Bytes
	qsaPooledRowBytes = tensor.QSAIndexerWidth * f32Bytes
	qsaPositionBytes  = tensor.QSAPositionAxes * i32Bytes
	f16Bytes          = 2
	f32Bytes          = 4
	i32Bytes          = 4
)

func alignQSARegion(bytes uint64) uint64 {
	return (bytes + qsaRegionAlignment - 1) / qsaRegionAlignment * qsaRegionAlignment
}

// qsaCellLayout is the cells' regions at a capacity: on the card, a layer's codes and scales (its pages, or its slots
// when it streams), pooled rows, block position, and a streamed layer's page table and map, then the shared page
// table and image position table; in the host copy, a streamed layer's codes and scales of every page.
type qsaCellLayout struct {
	codes, scales, pooled, blockPosition, table, streamMap, layer uint64
	pageTable, positions, total                                   uint64
	hostCodes, hostScales, hostLayer, hostTotal                   uint64
}

func newQSACellLayout(capacity, slots uint32, layers int) qsaCellLayout {
	pages := uint64(capacity) / tensor.QSABlockCells
	pooledPages := pages
	layout := qsaCellLayout{}
	if slots > 0 {
		pooledPages = uint64(slots)
		layout.table = alignQSARegion(pages * i32Bytes)
		layout.streamMap = alignQSARegion((tensor.QSAStreamSlotArrays*uint64(slots) + tensor.QSAStreamControlInts) * i32Bytes)
		layout.hostCodes, layout.hostScales = alignQSARegion(pages*qsaCodePageBytes), alignQSARegion(pages*qsaScalePageBytes)
		layout.hostLayer = qsaPoolHalves*layout.hostCodes + qsaPoolHalves*layout.hostScales
		layout.hostTotal = uint64(layers) * layout.hostLayer
	}
	layout.codes, layout.scales = alignQSARegion(pooledPages*qsaCodePageBytes), alignQSARegion(pooledPages*qsaScalePageBytes)
	layout.pooled = alignQSARegion((pages + tensor.QSAPooledSpare) * qsaPooledRowBytes)
	layout.blockPosition = qsaRegionAlignment
	layout.layer = qsaPoolHalves*layout.codes + qsaPoolHalves*layout.scales + layout.pooled + layout.blockPosition +
		layout.table + layout.streamMap
	layout.pageTable = alignQSARegion(pages * i32Bytes)
	layout.positions = alignQSARegion(uint64(capacity) * qsaPositionBytes)
	layout.total = uint64(layers)*layout.layer + layout.pageTable + layout.positions
	return layout
}

// qsaSlots is the VRAM pages a layer keeps for a capacity: none while the capacity fits the ring, the ring's pages
// past it.
func (r *Runner) qsaSlots(capacity uint32) uint32 {
	ring := cmp.Or(r.qsaRing, qsaRingCells)
	if capacity <= ring {
		return 0
	}
	return (ring + tensor.QSABlockCells - 1) / tensor.QSABlockCells
}

// qsaSequenceCapacity is the cells a sequence of tokens holds, fixed as Strata's session holds max_cells from its start
// (kv_plan over --max-context): the ring while the tokens fit it and the context passes it, the context otherwise, so a
// sequence keeps one capacity, and its steps their graphs, from its first token until it streams past the ring.
func (r *Runner) qsaSequenceCapacity(tokens uint32) uint32 {
	context, ring := r.qsaRopeRows(), qsaCapacity(cmp.Or(r.qsaRing, qsaRingCells))
	if tokens <= ring && ring < context {
		return ring
	}
	return context
}

// qsaLayerCount is the runner's QSA layers.
func (r *Runner) qsaLayerCount() int {
	count := 0
	for _, holds := range r.qsaLayers {
		if holds {
			count++
		}
	}
	return count
}

// qsaCellBytes is the card bytes a sequence's QSA cells take at a capacity.
func (r *Runner) qsaCellBytes(capacity uint32) uint64 {
	return newQSACellLayout(capacity, r.qsaSlots(capacity), r.qsaLayerCount()).total
}

// newQSACells allocates a sequence's cells at a capacity and writes its page tables and image position table: the
// identity for a sequence on the card, as qsa_state_init writes it, and for a streamed one every block evicted and every
// slot free, as kv_stream_reset leaves a map; the image positions the identity, as generate.cpp's vision setup does.
func (r *Runner) newQSACells(ctx context.Context, capacity uint32) (*qsaCells, error) {
	if capacity == 0 || capacity%tensor.QSABlockCells != 0 || uint64(capacity) > uint64(math.MaxInt32) {
		return nil, fmt.Errorf("inference: QSA capacity %d is not whole pages", capacity)
	}
	slots := r.qsaSlots(capacity)
	layout := newQSACellLayout(capacity, slots, r.qsaLayerCount())
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, layout.total)
	if err != nil {
		return nil, err
	}
	cells := &qsaCells{buffer: buffer, capacity: capacity, slots: slots, layers: make([]qsaLayerCells, len(r.qsaLayers))}
	fail := func(cause error) (*qsaCells, error) {
		return nil, errors.Join(cause, cells.release(context.WithoutCancel(ctx)))
	}
	whole, err := buffer.Value(tensor.MustShape(layout.total / f32Bytes))
	if err != nil {
		return fail(err)
	}
	var hostBase driver.DevicePtr
	if slots > 0 {
		if cells.host, err = r.cuda.AllocatePinnedHost(ctx, layout.hostTotal); err != nil {
			return fail(err)
		}
		hostBase = cells.host.Device
	}
	capacityCells := uint64(capacity)
	pages := capacityCells / tensor.QSABlockCells
	pooledCells := capacityCells
	if slots > 0 {
		pooledCells = uint64(slots) * tensor.QSABlockCells
	}
	region := func(base driver.DevicePtr, offset, bytes uint64, shape tensor.Shape) executor.DeviceValue {
		return executor.DeviceValue{Pointer: base + driver.DevicePtr(offset), Shape: shape, CapacityBytes: bytes}
	}
	codesOf := func(cellCount uint64) tensor.Shape {
		return tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads*cellCount)
	}
	scalesOf := func(cellCount uint64) tensor.Shape {
		return tensor.MustShape(tensor.QSAHeadWidth/tensor.QSAScaleGroup, tensor.QSAKeyValueHeads*cellCount)
	}
	var writes []executor.DeviceBytes
	offset, hostOffset := uint64(0), uint64(0)
	for layer, holds := range r.qsaLayers {
		if !holds {
			continue
		}
		at := offset
		next := func(bytes uint64, shape tensor.Shape) executor.DeviceValue {
			value := region(whole.Pointer, at, bytes, shape)
			at += bytes
			return value
		}
		held := qsaLayerCells{
			keyCodes: next(layout.codes, codesOf(pooledCells)), valueCodes: next(layout.codes, codesOf(pooledCells)),
			keyScales: next(layout.scales, scalesOf(pooledCells)), valueScales: next(layout.scales, scalesOf(pooledCells)),
			pooled:        next(layout.pooled, tensor.MustShape(tensor.QSAIndexerWidth, pages+tensor.QSAPooledSpare)),
			blockPosition: next(layout.blockPosition, tensor.MustShape(tensor.SingletonExtent)),
		}
		if slots > 0 {
			held.pageTable = next(layout.table, tensor.MustShape(pages))
			mapInts := tensor.QSAStreamSlotArrays*uint64(slots) + tensor.QSAStreamControlInts
			held.streamMap = next(layout.streamMap, tensor.MustShape(mapInts))
			writes = append(writes,
				executor.DeviceBytes{Pointer: held.pageTable.Pointer, Data: filledInt32s(pages, -1)},
				executor.DeviceBytes{Pointer: held.streamMap.Pointer, Data: qsaResetMap(slots)})
			regions := [qsaHostRegions]uint64{layout.hostCodes, layout.hostCodes, layout.hostScales, layout.hostScales}
			shapes := [qsaHostRegions]tensor.Shape{codesOf(capacityCells), codesOf(capacityCells), scalesOf(capacityCells), scalesOf(capacityCells)}
			for index, bytes := range regions {
				held.host[index] = region(hostBase, hostOffset, bytes, shapes[index])
				held.hostOffsets[index] = hostOffset
				hostOffset += bytes
			}
		}
		cells.layers[layer] = held
		offset += layout.layer
	}
	cells.pageTable = region(whole.Pointer, offset, layout.pageTable, tensor.MustShape(pages))
	cells.positions = region(whole.Pointer, offset+layout.pageTable, layout.positions, tensor.MustShape(tensor.QSAPositionAxes, capacityCells))
	identity := make([]int32, pages)
	for page := range identity {
		identity[page] = int32(page)
	}
	cells.mirror = make([]int32, tensor.QSAPositionAxes*capacityCells)
	for cell := range capacityCells {
		for axis := range uint64(tensor.QSAPositionAxes) {
			cells.mirror[cell*tensor.QSAPositionAxes+axis] = int32(cell)
		}
	}
	writes = append(writes,
		executor.DeviceBytes{Pointer: cells.pageTable.Pointer, Data: int32Bytes(identity)},
		executor.DeviceBytes{Pointer: cells.positions.Pointer, Data: int32Bytes(cells.mirror)})
	if err := r.cuda.WriteDeviceBytes(ctx, writes); err != nil {
		return fail(err)
	}
	return cells, nil
}

// qsaUnsetSlotArrays is the map's leading arrays a reset sets to -1: slot_block and slot_stamp.
const qsaUnsetSlotArrays = 2

// qsaResetMap is a stream map as kv_stream_reset leaves it: every slot holding no block (slot_block and slot_stamp -1,
// slot_ref 0), the miss lists unset and the control ints zero.
func qsaResetMap(slots uint32) []byte {
	words := make([]int32, tensor.QSAStreamSlotArrays*uint64(slots)+tensor.QSAStreamControlInts)
	for index := range qsaUnsetSlotArrays * uint64(slots) {
		words[index] = -1
	}
	return int32Bytes(words)
}

func filledInt32s(count uint64, value int32) []byte {
	words := make([]int32, count)
	for index := range words {
		words[index] = value
	}
	return int32Bytes(words)
}

// forkQSACells allocates cells at a capacity holding the source's first count cells, as a fork or a growth keeps a
// prefix: each layer's pooled rows and block position and the image positions on the card, and the cells' pages -
// card to card for a state on the card, into the new host copy (from the source's host copy, or from its pages on the
// card) for a streamed one, whose slots then start empty.
func (r *Runner) forkQSACells(ctx context.Context, source *qsaCells, count, capacity uint32) (*qsaCells, error) {
	if source == nil || count > source.capacity || count > capacity {
		return nil, errors.New("inference: QSA cells fork exceeds its source")
	}
	cells, err := r.newQSACells(ctx, capacity)
	if err != nil || count == 0 {
		return cells, err
	}
	if cells.slots == 0 && source.slots > 0 {
		return nil, errors.Join(errors.New("inference: QSA cells do not move from streamed to resident"), cells.release(context.WithoutCancel(ctx)))
	}
	pages := uint64(count+tensor.QSABlockCells-1) / tensor.QSABlockCells
	pooled := min(uint64(count)/tensor.QSABlockCells+tensor.QSAPooledSpare, uint64(source.capacity)/tensor.QSABlockCells+tensor.QSAPooledSpare)
	var copies []executor.DeviceRangeCopy
	var reads []executor.DeviceBytes
	add := func(target, origin executor.DeviceValue, bytes uint64) {
		copies = append(copies, executor.DeviceRangeCopy{Target: target.Pointer, Source: origin.Pointer, Bytes: bytes})
	}
	runs := [qsaHostRegions]uint64{pages * qsaCodePageBytes, pages * qsaCodePageBytes, pages * qsaScalePageBytes, pages * qsaScalePageBytes}
	for layer, holds := range r.qsaLayers {
		if !holds {
			continue
		}
		to, from := cells.layers[layer], source.layers[layer]
		add(to.pooled, from.pooled, pooled*qsaPooledRowBytes)
		add(to.blockPosition, from.blockPosition, i32Bytes)
		cardPools := [qsaHostRegions]executor.DeviceValue{from.keyCodes, from.valueCodes, from.keyScales, from.valueScales}
		for index, bytes := range runs {
			switch {
			case cells.slots == 0:
				add([qsaHostRegions]executor.DeviceValue{to.keyCodes, to.valueCodes, to.keyScales, to.valueScales}[index], cardPools[index], bytes)
			case source.slots == 0:
				// A resident state's pages are at the identity layout the host copy keeps.
				reads = append(reads, executor.DeviceBytes{Pointer: cardPools[index].Pointer, Data: cells.host.Host[to.hostOffsets[index]:][:bytes]})
			default:
				copy(cells.host.Host[to.hostOffsets[index]:][:bytes], source.host.Host[from.hostOffsets[index]:][:bytes])
			}
		}
	}
	rows := min(source.capacity, capacity)
	add(cells.positions, source.positions, uint64(rows)*qsaPositionBytes)
	copy(cells.mirror, source.mirror[:uint64(rows)*tensor.QSAPositionAxes])
	err = r.cuda.CopyDeviceRanges(ctx, copies)
	if err == nil && len(reads) > 0 {
		err = r.cuda.ReadDeviceBytes(ctx, reads)
	}
	if err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	return cells, nil
}

func (c *qsaCells) release(ctx context.Context) error {
	if c == nil {
		return nil
	}
	var errs []error
	if c.buffer != nil {
		if err := c.buffer.Release(ctx); err != nil {
			errs = append(errs, err)
		} else {
			c.buffer = nil
		}
	}
	if err := c.host.Release(ctx); err != nil {
		errs = append(errs, err)
	} else {
		c.host = nil
	}
	return errors.Join(errs...)
}

// qsaStage is the runner's staging pool for a streamed prompt read (Strata's prefill stage): one layer's cells of a
// capacity at the identity layout, with its identity page table, which a chunk stages the cells before it into.
type qsaStage struct {
	buffer   *executor.DeviceBuffer
	capacity uint32
	pools    [qsaHostRegions]executor.DeviceValue
	table    executor.DeviceValue
}

// qsaStageBytes is a staging pool's card bytes at a capacity.
func qsaStageBytes(capacity uint32) uint64 {
	pages := uint64(capacity) / tensor.QSABlockCells
	return qsaPoolHalves*alignQSARegion(pages*qsaCodePageBytes) + qsaPoolHalves*alignQSARegion(pages*qsaScalePageBytes) +
		alignQSARegion(pages*i32Bytes)
}

// qsaStagePool answers the runner's staging pool for a capacity, grown when a sequence outgrows it.
func (r *Runner) qsaStagePool(ctx context.Context, capacity uint32) (*qsaStage, error) {
	if r.qsaStage != nil && r.qsaStage.capacity >= capacity {
		return r.qsaStage, nil
	}
	if err := r.releaseQSAStage(ctx); err != nil {
		return nil, err
	}
	bytes := qsaStageBytes(capacity)
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, bytes)
	if err != nil {
		return nil, err
	}
	whole, err := buffer.Value(tensor.MustShape(bytes / f32Bytes))
	if err != nil {
		return nil, errors.Join(err, buffer.Release(context.WithoutCancel(ctx)))
	}
	pages := uint64(capacity) / tensor.QSABlockCells
	stage := &qsaStage{buffer: buffer, capacity: capacity}
	sizes := [qsaHostRegions]uint64{
		alignQSARegion(pages * qsaCodePageBytes), alignQSARegion(pages * qsaCodePageBytes),
		alignQSARegion(pages * qsaScalePageBytes), alignQSARegion(pages * qsaScalePageBytes),
	}
	offset := uint64(0)
	for index, size := range sizes {
		stage.pools[index] = executor.DeviceValue{Pointer: whole.Pointer + driver.DevicePtr(offset), CapacityBytes: size}
		offset += size
	}
	stage.table = executor.DeviceValue{Pointer: whole.Pointer + driver.DevicePtr(offset), CapacityBytes: alignQSARegion(pages * i32Bytes)}
	identity := make([]int32, pages)
	for page := range identity {
		identity[page] = int32(page)
	}
	if err := r.cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{{Pointer: stage.table.Pointer, Data: int32Bytes(identity)}}); err != nil {
		return nil, errors.Join(err, buffer.Release(context.WithoutCancel(ctx)))
	}
	r.qsaStage = stage
	return stage, nil
}

// releaseQSAStage releases the runner's staging pool.
func (r *Runner) releaseQSAStage(ctx context.Context) error {
	if r.qsaStage == nil {
		return nil
	}
	err := r.qsaStage.buffer.Release(ctx)
	if err == nil {
		r.qsaStage = nil
	}
	return err
}
