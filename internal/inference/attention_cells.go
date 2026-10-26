package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// The attention cells: Strata's int8 K/V cells (kv_q8's codes with an fp16 scale a 64-value group, behind a page
// table) for a runner whose every attention layer holds a geometry the cell attention's kernels are instantiated for
// (model.CompiledLayerProgram.HoldsAttentionCells). A sequence's cells - each such layer's codes and scales in pages of
// tensor.CellPageCells cells behind a page table - are its cache storage's token-aligned part, which each step appends
// to in place (tensor.CellAttention) and which its caches hold as they hold the storage. A full-attention layer's pools
// hold the capacity's pages behind the identity; a sliding layer's hold a ring of the pages its window and a prompt
// chunk read, behind a table that hands each page of the capacity the ring's page it takes in turn, so the cells its
// window has passed are written over. A layer over cells keeps no cache value of its own; a capacity its cells
// outgrow takes new cells, the past's live cells copied into them.

// cellLayerSet answers, a layer each, whether the runner builds the layer over attention cells: on the device cache
// path, where every layer that holds tokens holds them as cells and every other layer keeps only fixed state, so the
// cells are a sequence's only token-aligned storage; nil where none is, and for a QSA runner, whose cells are its own.
func (r *Runner) cellLayerSet() []bool {
	if !r.persistentDeviceCache() || r.qsaLayers != nil {
		return nil
	}
	holds := make([]bool, len(r.weights.Layers))
	held := false
	for layer := range holds {
		program := r.layerProgram(layer)
		if program.HoldsAttentionCells() {
			holds[layer], held = true, true
			continue
		}
		if program.Layer().SharedKV {
			return nil
		}
		schema, err := r.program.Model.CacheSchema(uint32(layer), tensor.SingletonExtent)
		if err != nil || schema.Primary.Key.Mode.TokenAligned() || schema.Primary.Value.Mode.TokenAligned() {
			return nil
		}
		token := false
		schema.RangeStates(func(_ model.CacheStateName, state model.CacheState[model.CacheValueSchema]) {
			token = token || state.Mode.TokenAligned()
		})
		if token {
			return nil
		}
	}
	if !held {
		return nil
	}
	return holds
}

// holdsCells reports whether the runner builds the layer over attention cells.
func (r *Runner) holdsCells(layer int) bool {
	return layer < len(r.cellLayers) && r.cellLayers[layer]
}

// cellsRunner reports whether the runner's caches keep their tokens as cells, QSA cells or attention cells: such a
// cache neither shifts its context nor moves to host memory.
func (r *Runner) cellsRunner() bool {
	return r.qsaLayers != nil || r.cellLayers != nil
}

// cellCapacity is the cells a sequence's attention cells hold for a cache capacity of tokens: the capacity in whole
// pages.
func cellCapacity(tokens uint32) uint32 {
	return (tokens + tensor.CellPageCells - 1) / tensor.CellPageCells * tensor.CellPageCells
}

// cellRingWindow is the window of the runner's sliding cell layers, whose pools ring past it, and the pages a ring
// holds: every page a step reads or writes, its window's cells before its first token and its own tokens, a prompt
// read's chunk at most, where they straddle one page more than they fill. Zero without a sliding cell layer or a
// prompt chunk; a ring past a capacity's pages is no ring.
func (r *Runner) cellRingWindow() (window uint32, pages uint64) {
	for layer, holds := range r.cellLayers {
		if _, _, layerWindow := r.layerProgram(layer).AttentionCells(); holds && layerWindow > 0 {
			window = max(window, layerWindow)
		}
	}
	if window == 0 || r.promptChunkTokens <= 0 {
		return 0, 0
	}
	cells := uint64(r.promptChunkTokens) + uint64(window) - tensor.SingletonExtent
	return window, (cells+tensor.CellPageCells-tensor.SingletonExtent)/tensor.CellPageCells + tensor.SingletonExtent
}

// attentionCells is one sequence's attention cells at a capacity, in one card buffer.
type attentionCells struct {
	buffer   *executor.DeviceBuffer
	capacity uint32
	layers   []attentionLayerCells
	// pageTable is the identity a full layer's pools sit behind; ringTable the ring's, which hands logical page p the
	// pool's page p mod ringPages, zero without a ring.
	pageTable, ringTable executor.DeviceValue
	// window and ringPages are the sliding layers' window and their pools' pages where they ring, zero where none do.
	window    uint32
	ringPages uint64
	// written is the cells the sequence's steps have appended through these cells, after which a ring has written over
	// the pages its window passed.
	written uint32
}

// attentionLayerCells is one layer's cells, zero for a layer that keeps none; ring marks a sliding layer's pools that
// ring.
type attentionLayerCells struct {
	keyCodes, valueCodes, keyScales, valueScales executor.DeviceValue
	ring                                         bool
}

// cellPoolHalves is a layer's pools of each kind, its keys' and its values'; cellPageTables are the cells' page tables,
// the identity and the ring's.
const (
	cellPoolHalves = 2
	cellPageTables = 2
)

// cellLayerBytes is a layer's codes and scales a cell: a byte a value and two a 64-value group, over its K/V heads.
func (r *Runner) cellLayerBytes(layer int) (codes, scales uint64) {
	kvHeads, width, _ := r.layerProgram(layer).AttentionCells()
	return kvHeads * width, kvHeads * width / tensor.QSAScaleGroup * f16Bytes
}

// cellPoolPages is the pages a layer's pools hold at a capacity: a sliding layer's ring where it is shorter than the
// capacity's pages, else every page.
func (r *Runner) cellPoolPages(layer int, capacity uint32) (pages uint64, ring bool) {
	pages = uint64(capacity) / tensor.CellPageCells
	_, _, window := r.layerProgram(layer).AttentionCells()
	ringWindow, ringPages := r.cellRingWindow()
	if window == 0 || window != ringWindow || ringPages >= pages {
		return pages, false
	}
	return ringPages, true
}

// attentionCellBytes is the card bytes a sequence's attention cells take at a capacity.
func (r *Runner) attentionCellBytes(capacity uint32) uint64 {
	table := alignQSARegion(uint64(capacity) / tensor.CellPageCells * i32Bytes)
	total := cellPageTables * table
	for layer, holds := range r.cellLayers {
		if holds {
			codes, scales := r.cellLayerBytes(layer)
			pages, _ := r.cellPoolPages(layer, capacity)
			cells := pages * tensor.CellPageCells
			total += cellPoolHalves*alignQSARegion(cells*codes) + cellPoolHalves*alignQSARegion(cells*scales)
		}
	}
	return total
}

// newAttentionCells allocates a sequence's attention cells at a capacity and writes its page tables: the identity, and
// a ring's where a sliding layer's pools ring.
func (r *Runner) newAttentionCells(ctx context.Context, capacity uint32) (*attentionCells, error) {
	if capacity == 0 || capacity%tensor.CellPageCells != 0 || uint64(capacity) > uint64(math.MaxInt32) {
		return nil, fmt.Errorf("inference: attention cell capacity %d is not whole pages", capacity)
	}
	total := r.attentionCellBytes(capacity)
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, total)
	if err != nil {
		return nil, err
	}
	cells := &attentionCells{buffer: buffer, capacity: capacity, layers: make([]attentionLayerCells, len(r.cellLayers))}
	whole, err := buffer.Value(tensor.MustShape(total / f32Bytes))
	if err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	offset := uint64(0)
	next := func(bytes uint64, shape tensor.Shape) executor.DeviceValue {
		value := executor.DeviceValue{Pointer: whole.Pointer + driver.DevicePtr(offset), Shape: shape, CapacityBytes: bytes}
		offset += alignQSARegion(bytes)
		return value
	}
	for layer, holds := range r.cellLayers {
		if !holds {
			continue
		}
		kvHeads, width, _ := r.layerProgram(layer).AttentionCells()
		codes, scales := r.cellLayerBytes(layer)
		pages, ring := r.cellPoolPages(layer, capacity)
		pooled := pages * tensor.CellPageCells
		codeShape := tensor.MustShape(width, kvHeads*pooled)
		scaleShape := tensor.MustShape(width/tensor.QSAScaleGroup, kvHeads*pooled)
		cells.layers[layer] = attentionLayerCells{
			keyCodes: next(pooled*codes, codeShape), valueCodes: next(pooled*codes, codeShape),
			keyScales: next(pooled*scales, scaleShape), valueScales: next(pooled*scales, scaleShape), ring: ring,
		}
		if ring {
			cells.window, cells.ringPages = r.cellRingWindow()
		}
	}
	pages := uint64(capacity) / tensor.CellPageCells
	cells.pageTable = next(pages*i32Bytes, tensor.MustShape(pages))
	cells.ringTable = next(pages*i32Bytes, tensor.MustShape(pages))
	identity, ring := make([]int32, pages), make([]int32, pages)
	for page := range identity {
		identity[page], ring[page] = int32(page), int32(page)
		if cells.ringPages > 0 {
			ring[page] = int32(uint64(page) % cells.ringPages)
		}
	}
	if err := r.cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{
		{Pointer: cells.pageTable.Pointer, Data: int32Bytes(identity)}, {Pointer: cells.ringTable.Pointer, Data: int32Bytes(ring)},
	}); err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	return cells, nil
}

// poolPage is the pool page a layer's cells keep a logical page at.
func (c *attentionCells) poolPage(layer int, page uint64) uint64 {
	if c.layers[layer].ring {
		return page % c.ringPages
	}
	return page
}

// stale reports whether the cells no longer hold what a cache of tokens reads: a ring that later steps carried past
// the cache has written over the pages of the cells its window reads back from it.
func (c *attentionCells) stale(tokens uint32) bool {
	if c == nil || c.ringPages == 0 || c.written <= tokens {
		return false
	}
	first := (tokens - min(tokens, c.window)) / tensor.CellPageCells
	last := (c.written - tensor.SingletonExtent) / tensor.CellPageCells
	return uint64(last-first)+tensor.SingletonExtent > c.ringPages
}

// staleCells reports whether the cache's attention cells no longer hold what it reads (attentionCells.stale).
func (c *deviceKVCache) staleCells() bool {
	return c != nil && c.storage != nil && c.storage.cells.stale(c.Tokens)
}

// errCellsStale refuses a cache whose sliding layers' rings later steps have written over.
var errCellsStale = errors.New("inference: the cache's sliding cells are written over")

// forkAttentionCells is new cells at a capacity holding source's first count cells: each layer's pages a step after
// them reads - a full layer's every page, a ring's last ones - copied from where the source keeps them to where the new
// cells do, a run of pages at a time.
func (r *Runner) forkAttentionCells(ctx context.Context, source *attentionCells, count, capacity uint32) (*attentionCells, error) {
	if source == nil || count > source.capacity || count > capacity || source.stale(count) {
		return nil, errors.New("inference: attention cells fork exceeds its source")
	}
	cells, err := r.newAttentionCells(ctx, capacity)
	if err != nil || count == 0 {
		return cells, err
	}
	cells.written = count
	pages := uint64(count+tensor.CellPageCells-tensor.SingletonExtent) / tensor.CellPageCells
	var copies []executor.DeviceRangeCopy
	for layer, holds := range r.cellLayers {
		if !holds {
			continue
		}
		codes, scales := r.cellLayerBytes(layer)
		to, from := cells.layers[layer], source.layers[layer]
		first := uint64(0)
		if to.ring || from.ring {
			first = pages - min(pages, min(cells.poolPages(layer, capacity), source.poolPages(layer, source.capacity)))
		}
		for page := first; page < pages; page++ {
			target, origin := cells.poolPage(layer, page), source.poolPage(layer, page)
			for _, pair := range []struct {
				target, origin executor.DeviceValue
				bytes          uint64
			}{
				{to.keyCodes, from.keyCodes, codes}, {to.valueCodes, from.valueCodes, codes},
				{to.keyScales, from.keyScales, scales}, {to.valueScales, from.valueScales, scales},
			} {
				span := tensor.CellPageCells * pair.bytes
				copies = appendCellRun(copies, pair.target.Pointer+driver.DevicePtr(target*span),
					pair.origin.Pointer+driver.DevicePtr(origin*span), span)
			}
		}
	}
	if err := r.cuda.CopyDeviceRanges(ctx, copies); err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	return cells, nil
}

// poolPages is the pages a layer's pools hold.
func (c *attentionCells) poolPages(layer int, capacity uint32) uint64 {
	if c.layers[layer].ring {
		return c.ringPages
	}
	return uint64(capacity) / tensor.CellPageCells
}

// appendCellRun adds a page's copy to the copies, joined to the copy before it where both follow on from it.
func appendCellRun(copies []executor.DeviceRangeCopy, target, source driver.DevicePtr, bytes uint64) []executor.DeviceRangeCopy {
	if last := len(copies) - tensor.SingletonExtent; last >= 0 {
		run := &copies[last]
		if run.Target+driver.DevicePtr(run.Bytes) == target && run.Source+driver.DevicePtr(run.Bytes) == source {
			run.Bytes += bytes
			return copies
		}
	}
	return append(copies, executor.DeviceRangeCopy{Target: target, Source: source, Bytes: bytes})
}

// release frees the cells' buffer.
func (c *attentionCells) release(ctx context.Context) error {
	if c == nil || c.buffer == nil {
		return nil
	}
	err := c.buffer.Release(ctx)
	if err == nil {
		c.buffer = nil
	}
	return err
}

// prepareAttentionCells gives a branch's storage its attention cells at the step's capacity - its past's where the
// storage is the past's, a fork of the past's live cells where the storage is new, fresh cells without a past - and
// binds the step's cell inputs to them. A step past a ring's chunk, which its ring cannot hold beside its window, is
// refused.
func (r *Runner) prepareAttentionCells(
	ctx context.Context, compiled *executor.CompiledGraph, inputs *executor.DeviceInputs, storage *deviceCacheStorage,
	past *deviceKVCache, graph deviceBatchGraph, capacity uint32,
) error {
	if storage.cells == nil {
		var err error
		switch {
		case past != nil && past.Tokens > 0 && (past.storage == nil || past.storage.cells == nil):
			return errors.New("inference: an attention cell step's past holds no cells")
		case past != nil && past.Tokens > 0:
			storage.cells, err = r.forkAttentionCells(ctx, past.storage.cells, past.Tokens, capacity)
		default:
			storage.cells, err = r.newAttentionCells(ctx, capacity)
		}
		if err != nil {
			return err
		}
	}
	if storage.cells.capacity != capacity {
		return errors.New("inference: an attention cell step's cells differ from its capacity")
	}
	if storage.cells.ringPages > 0 && int(graph.tokenCount) > r.promptChunkTokens {
		return fmt.Errorf("inference: a step of %d tokens outruns the sliding cells' ring of a %d-token chunk", graph.tokenCount, r.promptChunkTokens)
	}
	storage.cells.written = max(storage.cells.written, graph.pastTokens+graph.tokenCount)
	return bindCellInputs(compiled, inputs, graph.cellInputs, storage.cells)
}

// cellStateInputs builds a layer's cells as the step's inputs, in tensor.CellState's order, its pools of pooled cells,
// which the step binds to the sequence's cells once its storage is chosen (bindCellInputs).
func cellStateInputs(builder *tensor.Builder, name func(string) string, kvHeads, width uint64, capacity uint32, pooled uint64) (tensor.CellState, []*tensor.Tensor) {
	input := func(suffix string, kind dtype.Type, dimensions ...uint64) *tensor.Tensor {
		return builder.Input(name(suffix), kind, tensor.MustShape(dimensions...))
	}
	state := tensor.CellState{
		KeyCodes:    input("cell_key_codes", dtype.I8, width, kvHeads*pooled),
		ValueCodes:  input("cell_value_codes", dtype.I8, width, kvHeads*pooled),
		KeyScales:   input("cell_key_scales", dtype.F16, width/tensor.QSAScaleGroup, kvHeads*pooled),
		ValueScales: input("cell_value_scales", dtype.F16, width/tensor.QSAScaleGroup, kvHeads*pooled),
		PageTable:   input("cell_page_table", dtype.I32, uint64(capacity)/tensor.CellPageCells),
	}
	return state, []*tensor.Tensor{state.KeyCodes, state.ValueCodes, state.KeyScales, state.ValueScales, state.PageTable}
}

// bindCellInputs points a step's cell layers' inputs at the sequence's cells, a ring's pools behind the ring's table.
func bindCellInputs(compiled *executor.CompiledGraph, inputs *executor.DeviceInputs, nodes [][]*tensor.Tensor, cells *attentionCells) error {
	for layer, layerNodes := range nodes {
		if len(layerNodes) == 0 {
			continue
		}
		if cells == nil || layer >= len(cells.layers) {
			return errors.New("inference: attention cell step has no cells to bind")
		}
		held := cells.layers[layer]
		table := cells.pageTable.Pointer
		if held.ring {
			table = cells.ringTable.Pointer
		}
		pointers := []driver.DevicePtr{held.keyCodes.Pointer, held.valueCodes.Pointer, held.keyScales.Pointer, held.valueScales.Pointer, table}
		if len(layerNodes) != len(pointers) {
			return errors.New("inference: attention cell step inputs differ from the cells")
		}
		for index, node := range layerNodes {
			slot, ok := compiled.InputSlot(node)
			if !ok {
				return fmt.Errorf("inference: attention cell input %q is not compiled", node.Name)
			}
			if pointers[index] == 0 {
				return fmt.Errorf("inference: attention cell input %q has no cells", node.Name)
			}
			inputs.Pointers[slot] = pointers[index]
		}
	}
	return nil
}
