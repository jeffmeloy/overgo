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
// tensor.CellPageCells cells, and the page table the layers share, the identity - are its cache storage's
// token-aligned part, which each step appends to in place (tensor.CellAttention) and which its caches hold as they hold
// the storage. A layer over cells keeps no cache value of its own; a capacity its cells outgrow takes new cells, the
// past's first cells copied into them.

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

// attentionCells is one sequence's attention cells at a capacity, in one card buffer.
type attentionCells struct {
	buffer    *executor.DeviceBuffer
	capacity  uint32
	layers    []attentionLayerCells
	pageTable executor.DeviceValue
}

// attentionLayerCells is one layer's cells, zero for a layer that keeps none.
type attentionLayerCells struct {
	keyCodes, valueCodes, keyScales, valueScales executor.DeviceValue
}

// cellPoolHalves is a layer's pools of each kind, its keys' and its values'.
const cellPoolHalves = 2

// cellLayerBytes is a layer's codes and scales a cell: a byte a value and two a 64-value group, over its K/V heads.
func (r *Runner) cellLayerBytes(layer int) (codes, scales uint64) {
	kvHeads, width, _ := r.layerProgram(layer).AttentionCells()
	return kvHeads * width, kvHeads * width / tensor.QSAScaleGroup * f16Bytes
}

// attentionCellBytes is the card bytes a sequence's attention cells take at a capacity.
func (r *Runner) attentionCellBytes(capacity uint32) uint64 {
	total := alignQSARegion(uint64(capacity) / tensor.CellPageCells * i32Bytes)
	for layer, holds := range r.cellLayers {
		if holds {
			codes, scales := r.cellLayerBytes(layer)
			total += cellPoolHalves*alignQSARegion(uint64(capacity)*codes) + cellPoolHalves*alignQSARegion(uint64(capacity)*scales)
		}
	}
	return total
}

// newAttentionCells allocates a sequence's attention cells at a capacity and writes its page table, the identity.
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
	cellCount := uint64(capacity)
	for layer, holds := range r.cellLayers {
		if !holds {
			continue
		}
		kvHeads, width, _ := r.layerProgram(layer).AttentionCells()
		codes, scales := r.cellLayerBytes(layer)
		codeShape := tensor.MustShape(width, kvHeads*cellCount)
		scaleShape := tensor.MustShape(width/tensor.QSAScaleGroup, kvHeads*cellCount)
		cells.layers[layer] = attentionLayerCells{
			keyCodes: next(cellCount*codes, codeShape), valueCodes: next(cellCount*codes, codeShape),
			keyScales: next(cellCount*scales, scaleShape), valueScales: next(cellCount*scales, scaleShape),
		}
	}
	pages := cellCount / tensor.CellPageCells
	cells.pageTable = next(pages*i32Bytes, tensor.MustShape(pages))
	identity := make([]int32, pages)
	for page := range identity {
		identity[page] = int32(page)
	}
	if err := r.cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{{Pointer: cells.pageTable.Pointer, Data: int32Bytes(identity)}}); err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	return cells, nil
}

// forkAttentionCells is new cells at a capacity holding source's first count cells: each layer's pools' first pages,
// where the identity page table lays them out alike in both.
func (r *Runner) forkAttentionCells(ctx context.Context, source *attentionCells, count, capacity uint32) (*attentionCells, error) {
	if source == nil || count > source.capacity || count > capacity {
		return nil, errors.New("inference: attention cells fork exceeds its source")
	}
	cells, err := r.newAttentionCells(ctx, capacity)
	if err != nil || count == 0 {
		return cells, err
	}
	pages := uint64(count+tensor.CellPageCells-1) / tensor.CellPageCells
	var copies []executor.DeviceRangeCopy
	for layer, holds := range r.cellLayers {
		if !holds {
			continue
		}
		codes, scales := r.cellLayerBytes(layer)
		to, from := cells.layers[layer], source.layers[layer]
		for _, pair := range []struct {
			target, origin executor.DeviceValue
			bytes          uint64
		}{
			{to.keyCodes, from.keyCodes, codes}, {to.valueCodes, from.valueCodes, codes},
			{to.keyScales, from.keyScales, scales}, {to.valueScales, from.valueScales, scales},
		} {
			copies = append(copies, executor.DeviceRangeCopy{
				Target: pair.target.Pointer, Source: pair.origin.Pointer, Bytes: pages * tensor.CellPageCells * pair.bytes,
			})
		}
	}
	if err := r.cuda.CopyDeviceRanges(ctx, copies); err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	return cells, nil
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
// storage is the past's, a fork of the past's first cells where the storage is new, fresh cells without a past - and
// binds the step's cell inputs to them.
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
	return bindCellInputs(compiled, inputs, graph.cellInputs, storage.cells)
}

// cellStateInputs builds a layer's cells as the step's inputs, in tensor.CellState's order, which the step binds to the
// sequence's cells once its storage is chosen (bindCellInputs).
func cellStateInputs(builder *tensor.Builder, name func(string) string, kvHeads, width uint64, capacity uint32) (tensor.CellState, []*tensor.Tensor) {
	cells := uint64(capacity)
	input := func(suffix string, kind dtype.Type, dimensions ...uint64) *tensor.Tensor {
		return builder.Input(name(suffix), kind, tensor.MustShape(dimensions...))
	}
	state := tensor.CellState{
		KeyCodes:    input("cell_key_codes", dtype.I8, width, kvHeads*cells),
		ValueCodes:  input("cell_value_codes", dtype.I8, width, kvHeads*cells),
		KeyScales:   input("cell_key_scales", dtype.F16, width/tensor.QSAScaleGroup, kvHeads*cells),
		ValueScales: input("cell_value_scales", dtype.F16, width/tensor.QSAScaleGroup, kvHeads*cells),
		PageTable:   input("cell_page_table", dtype.I32, cells/tensor.CellPageCells),
	}
	return state, []*tensor.Tensor{state.KeyCodes, state.ValueCodes, state.KeyScales, state.ValueScales, state.PageTable}
}

// bindCellInputs points a step's cell layers' inputs at the sequence's cells.
func bindCellInputs(compiled *executor.CompiledGraph, inputs *executor.DeviceInputs, nodes [][]*tensor.Tensor, cells *attentionCells) error {
	for layer, layerNodes := range nodes {
		if len(layerNodes) == 0 {
			continue
		}
		if cells == nil || layer >= len(cells.layers) {
			return errors.New("inference: attention cell step has no cells to bind")
		}
		held := cells.layers[layer]
		pointers := []driver.DevicePtr{held.keyCodes.Pointer, held.valueCodes.Pointer, held.keyScales.Pointer, held.valueScales.Pointer, cells.pageTable.Pointer}
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
