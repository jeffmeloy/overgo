package inference

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// The QSA cells: Strata's QSA layer state (src/core/layer.cpp qsa_state_bytes and qsa_state_init, kv_plan's mode 0,
// fully resident; ported from Strata at 296b59d2, MIT) for a runner whose block-sparse attention layers hold the
// geometry Strata's kernels are compiled for (model.Spec.HoldsQSA). A sequence's cells - each QSA layer's int8 K/V
// codes and fp16 scales in 4-cell pages behind the page table, its indexer's pooled rows and block position, and the
// image position table the layers share - are its cache storage's token-aligned part, which each step appends to in
// place (tensor.QSAAttention) and which its caches hold as they hold the storage; a layer's fixed state, the indexer's
// tail and spare, is its cache pair. The rotation table is the runner's, at the context's positions.

// qsaLayerSet answers, a layer each, whether the runner builds the layer as Strata's QSA layer: on the device cache
// path, a layer at Strata's compiled geometry, where every other layer keeps only fixed state, so the QSA cells are a
// sequence's only token-aligned storage; nil where none is.
func (r *Runner) qsaLayerSet() []bool {
	if !r.persistentDeviceCache() {
		return nil
	}
	holds := make([]bool, len(r.weights.Layers))
	held := false
	for layer := range holds {
		if r.spec.HoldsQSA(r.layerProgram(layer).Layer()) {
			holds[layer], held = true, true
			continue
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

// holdsQSA reports whether the runner builds the layer as Strata's QSA layer.
func (r *Runner) holdsQSA(layer int) bool {
	return layer < len(r.qsaLayers) && r.qsaLayers[layer]
}

// deviceCacheSchema is the layer's cache on the device cache path: a QSA layer's fixed state, otherwise the model's.
func (r *Runner) deviceCacheSchema(layer int, tokens uint32) (model.LayerPlan, model.LayerCacheSchema, error) {
	if r.holdsQSA(layer) {
		return r.layerProgram(layer).Layer(), model.QSACacheSchema(), nil
	}
	return r.cacheSchema(layer, tokens)
}

// A QSA runner's caches neither shift nor move to host memory: Strata's QSA state holds every cell from the
// sequence's first (no context shift), and its conversation file, which a prompt state's save and restore would
// port, is not ported yet.
var (
	errQSAContextShift = errors.New("inference: a QSA runner's cache cannot shift its context")
	errQSAHostCache    = errors.New("inference: a QSA runner's cache cannot move to host memory")
)

// qsaCapacity is the cells a sequence's QSA cells hold for a cache capacity of tokens: the capacity in whole pages.
func qsaCapacity(tokens uint32) uint32 {
	return (tokens + tensor.QSABlockCells - 1) / tensor.QSABlockCells * tensor.QSABlockCells
}

// qsaCells is one sequence's QSA cells at a capacity, in one device buffer.
type qsaCells struct {
	buffer    *executor.DeviceBuffer
	capacity  uint32
	layers    []qsaLayerCells
	pageTable executor.DeviceValue
	positions executor.DeviceValue
	// mirror is the image position table as the device holds it: a cell's temporal, height and width positions.
	mirror []int32
}

// qsaLayerCells is one QSA layer's cells; zero for a layer that is not one.
type qsaLayerCells struct {
	keyCodes, valueCodes, keyScales, valueScales, pooled, blockPosition executor.DeviceValue
}

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

// qsaCellLayout is the cells' regions at a capacity: a QSA layer's codes, scales, pooled rows and block position, then
// the page table and the image position table, each aligned; and their total.
type qsaCellLayout struct {
	codes, scales, pooled, blockPosition, layer uint64
	pageTable, positions, total                 uint64
}

func newQSACellLayout(capacity uint32, layers int) qsaCellLayout {
	align := func(bytes uint64) uint64 {
		return (bytes + qsaRegionAlignment - 1) / qsaRegionAlignment * qsaRegionAlignment
	}
	pages := uint64(capacity) / tensor.QSABlockCells
	layout := qsaCellLayout{
		codes: align(pages * qsaCodePageBytes), scales: align(pages * qsaScalePageBytes),
		pooled: align((pages + tensor.QSAPooledSpare) * qsaPooledRowBytes), blockPosition: qsaRegionAlignment,
	}
	layout.layer = qsaPoolHalves*layout.codes + qsaPoolHalves*layout.scales + layout.pooled + layout.blockPosition
	layout.pageTable = align(pages * i32Bytes)
	layout.positions = align(uint64(capacity) * qsaPositionBytes)
	layout.total = uint64(layers)*layout.layer + layout.pageTable + layout.positions
	return layout
}

// qsaCellBytes is the device bytes a sequence's QSA cells take at a capacity.
func (r *Runner) qsaCellBytes(capacity uint32) uint64 {
	count := 0
	for _, holds := range r.qsaLayers {
		if holds {
			count++
		}
	}
	return newQSACellLayout(capacity, count).total
}

// newQSACells allocates a sequence's cells at a capacity and writes its page table and image position table, both the
// identity, as qsa_state_init and generate.cpp's vision setup do.
func (r *Runner) newQSACells(ctx context.Context, capacity uint32) (*qsaCells, error) {
	if capacity == 0 || capacity%tensor.QSABlockCells != 0 || uint64(capacity) > uint64(math.MaxInt32) {
		return nil, fmt.Errorf("inference: QSA capacity %d is not whole pages", capacity)
	}
	count := 0
	for _, holds := range r.qsaLayers {
		if holds {
			count++
		}
	}
	layout := newQSACellLayout(capacity, count)
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, layout.total)
	if err != nil {
		return nil, err
	}
	whole, err := buffer.Value(tensor.MustShape(layout.total / f32Bytes))
	if err != nil {
		return nil, errors.Join(err, buffer.Release(context.WithoutCancel(ctx)))
	}
	capacityCells := uint64(capacity)
	pages := capacityCells / tensor.QSABlockCells
	region := func(offset, bytes uint64, shape tensor.Shape) executor.DeviceValue {
		return executor.DeviceValue{Pointer: whole.Pointer + driver.DevicePtr(offset), Shape: shape, CapacityBytes: bytes}
	}
	cells := &qsaCells{buffer: buffer, capacity: capacity, layers: make([]qsaLayerCells, len(r.qsaLayers))}
	offset := uint64(0)
	for layer, holds := range r.qsaLayers {
		if !holds {
			continue
		}
		codes := tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads*capacityCells)
		scales := tensor.MustShape(tensor.QSAHeadWidth/tensor.QSAScaleGroup, tensor.QSAKeyValueHeads*capacityCells)
		cells.layers[layer] = qsaLayerCells{
			keyCodes:      region(offset, layout.codes, codes),
			valueCodes:    region(offset+layout.codes, layout.codes, codes),
			keyScales:     region(offset+qsaPoolHalves*layout.codes, layout.scales, scales),
			valueScales:   region(offset+qsaPoolHalves*layout.codes+layout.scales, layout.scales, scales),
			pooled:        region(offset+qsaPoolHalves*layout.codes+qsaPoolHalves*layout.scales, layout.pooled, tensor.MustShape(tensor.QSAIndexerWidth, pages+tensor.QSAPooledSpare)),
			blockPosition: region(offset+qsaPoolHalves*layout.codes+qsaPoolHalves*layout.scales+layout.pooled, layout.blockPosition, tensor.MustShape(tensor.SingletonExtent)),
		}
		offset += layout.layer
	}
	cells.pageTable = region(offset, layout.pageTable, tensor.MustShape(pages))
	cells.positions = region(offset+layout.pageTable, layout.positions, tensor.MustShape(tensor.QSAPositionAxes, capacityCells))
	table := make([]byte, pages*i32Bytes)
	for page := range pages {
		binary.LittleEndian.PutUint32(table[page*i32Bytes:], uint32(page))
	}
	cells.mirror = make([]int32, tensor.QSAPositionAxes*capacityCells)
	for cell := range capacityCells {
		for axis := range uint64(tensor.QSAPositionAxes) {
			cells.mirror[cell*tensor.QSAPositionAxes+axis] = int32(cell)
		}
	}
	err = r.cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{
		{Pointer: cells.pageTable.Pointer, Data: table},
		{Pointer: cells.positions.Pointer, Data: int32Bytes(cells.mirror)},
	})
	if err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	return cells, nil
}

// forkQSACells allocates cells at a capacity holding the source's first count cells: each layer's pages and pooled
// rows of them, its block position and the image position table's rows, as a fork or a growth keeps a prefix.
func (r *Runner) forkQSACells(ctx context.Context, source *qsaCells, count, capacity uint32) (*qsaCells, error) {
	if source == nil || count > source.capacity || count > capacity {
		return nil, errors.New("inference: QSA cells fork exceeds its source")
	}
	cells, err := r.newQSACells(ctx, capacity)
	if err != nil || count == 0 {
		return cells, err
	}
	pages := uint64(count+tensor.QSABlockCells-1) / tensor.QSABlockCells
	pooled := min(uint64(count)/tensor.QSABlockCells+tensor.QSAPooledSpare, uint64(source.capacity)/tensor.QSABlockCells+tensor.QSAPooledSpare)
	var copies []executor.DeviceRangeCopy
	add := func(target, origin executor.DeviceValue, bytes uint64) {
		copies = append(copies, executor.DeviceRangeCopy{Target: target.Pointer, Source: origin.Pointer, Bytes: bytes})
	}
	for layer, holds := range r.qsaLayers {
		if !holds {
			continue
		}
		to, from := cells.layers[layer], source.layers[layer]
		add(to.keyCodes, from.keyCodes, pages*qsaCodePageBytes)
		add(to.valueCodes, from.valueCodes, pages*qsaCodePageBytes)
		add(to.keyScales, from.keyScales, pages*qsaScalePageBytes)
		add(to.valueScales, from.valueScales, pages*qsaScalePageBytes)
		add(to.pooled, from.pooled, pooled*qsaPooledRowBytes)
		add(to.blockPosition, from.blockPosition, i32Bytes)
	}
	rows := min(source.capacity, capacity)
	add(cells.positions, source.positions, uint64(rows)*qsaPositionBytes)
	copy(cells.mirror, source.mirror[:uint64(rows)*tensor.QSAPositionAxes])
	if err := r.cuda.CopyDeviceRanges(ctx, copies); err != nil {
		return nil, errors.Join(err, cells.release(context.WithoutCancel(ctx)))
	}
	return cells, nil
}

func (c *qsaCells) release(ctx context.Context) error {
	if c == nil || c.buffer == nil {
		return nil
	}
	err := c.buffer.Release(ctx)
	if err == nil {
		c.buffer = nil
	}
	return err
}

// syncPositions writes the image position table's rows for the cells a step appends from first, a row its tokens'
// temporal, height and width positions, where they differ from the table's, and the rows after them as the text that
// follows continues from after, as generate.cpp lays out a picture request's cells: nothing for text whose positions
// are its cells.
func (r *Runner) syncQSAPositions(ctx context.Context, cells *qsaCells, first uint32, rows [][tensor.QSAPositionAxes]uint32, after uint32) error {
	if cells == nil || uint64(first)+uint64(len(rows)) > uint64(cells.capacity) {
		return errors.New("inference: QSA positions exceed the cells")
	}
	same := true
	for index, row := range rows {
		for axis, position := range row {
			same = same && cells.mirror[(int(first)+index)*tensor.QSAPositionAxes+axis] == int32(position)
		}
	}
	if same {
		return nil
	}
	for index, row := range rows {
		for axis, position := range row {
			cells.mirror[(int(first)+index)*tensor.QSAPositionAxes+axis] = int32(position)
		}
	}
	for cell := int(first) + len(rows); cell < int(cells.capacity); cell++ {
		position := int32(after) + int32(cell-int(first)-len(rows))
		for axis := range tensor.QSAPositionAxes {
			cells.mirror[cell*tensor.QSAPositionAxes+axis] = position
		}
	}
	return r.cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{{
		Pointer: cells.positions.Pointer + driver.DevicePtr(uint64(first)*qsaPositionBytes),
		Data:    int32Bytes(cells.mirror[int(first)*tensor.QSAPositionAxes:]),
	}})
}

func int32Bytes(values []int32) []byte {
	bytes := make([]byte, len(values)*i32Bytes)
	for index, value := range values {
		binary.LittleEndian.PutUint32(bytes[index*i32Bytes:], uint32(value))
	}
	return bytes
}

// qsaRope is the runner's rotation table for the pooling, cos then sin at `[position][32]` for the context's positions.
type qsaRope struct {
	buffer   *executor.DeviceBuffer
	cos, sin driver.DevicePtr
	rows     uint32
}

// qsaRopeRows is the rotation table's positions: the context's, in whole pages.
func (r *Runner) qsaRopeRows() uint32 {
	return qsaCapacity(max(r.spec.ContextLength, tensor.QSABlockCells))
}

// qsaRopeBytes is the rotation table's bytes at rows positions: cos and sin of 32 pairs a position.
func qsaRopeBytes(rows uint32) uint64 {
	return tensor.QSARotated * uint64(rows) * f32Bytes
}

// qsaRopeTable answers the runner's rotation table, built at the first QSA step from Strata's table builder.
func (r *Runner) qsaRopeTable(ctx context.Context) (*qsaRope, error) {
	if r.qsaRope != nil {
		return r.qsaRope, nil
	}
	rows := r.qsaRopeRows()
	cosines, sines := tensor.QSARopeTable(int(rows))
	half := uint64(tensor.QSARotated / tensor.PairedExtent)
	shape := tensor.MustShape(half, uint64(rows))
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, qsaRopeBytes(rows))
	if err != nil {
		return nil, err
	}
	whole, err := buffer.Value(tensor.MustShape(tensor.PairedExtent * half * uint64(rows)))
	if err != nil {
		return nil, errors.Join(err, buffer.Release(context.WithoutCancel(ctx)))
	}
	rope := &qsaRope{buffer: buffer, cos: whole.Pointer, sin: whole.Pointer + driver.DevicePtr(half*uint64(rows)*f32Bytes), rows: rows}
	err = r.cuda.WriteDeviceValues(ctx, []executor.DeviceValue{
		{Pointer: rope.cos, Shape: shape}, {Pointer: rope.sin, Shape: shape},
	}, [][]float32{cosines, sines})
	if err != nil {
		return nil, errors.Join(err, buffer.Release(context.WithoutCancel(ctx)))
	}
	r.qsaRope = rope
	return rope, nil
}

// releaseQSARope releases the runner's rotation table.
func (r *Runner) releaseQSARope(ctx context.Context) error {
	if r.qsaRope == nil {
		return nil
	}
	err := r.qsaRope.buffer.Release(ctx)
	if err == nil {
		r.qsaRope = nil
	}
	return err
}

// prepareQSACells gives a branch's storage its QSA cells at the step's capacity - its past's where the storage is the
// past's, a fork of the past's first cells into new ones where the storage is new, fresh cells without a past - binds
// the step's QSA inputs to them and writes the image positions of the step's cells.
func (r *Runner) prepareQSACells(
	ctx context.Context, compiled *executor.CompiledGraph, inputs *executor.DeviceInputs, storage *deviceCacheStorage,
	past *deviceKVCache, graph deviceBatchGraph, capacity uint32,
) error {
	if storage.qsa == nil {
		var err error
		switch {
		case past != nil && past.Tokens > 0 && (past.storage == nil || past.storage.qsa == nil):
			return errors.New("inference: a QSA step's past holds no QSA cells")
		case past != nil && past.Tokens > 0:
			storage.qsa, err = r.forkQSACells(ctx, past.storage.qsa, past.Tokens, capacity)
		default:
			storage.qsa, err = r.newQSACells(ctx, capacity)
		}
		if err != nil {
			return err
		}
	}
	if storage.qsa.capacity != capacity {
		return errors.New("inference: a QSA step's cells differ from its capacity")
	}
	rope, err := r.qsaRopeTable(ctx)
	if err != nil {
		return err
	}
	if err := bindQSAInputs(compiled, inputs, graph.qsaInputs, storage.qsa, rope); err != nil {
		return err
	}
	rows := graph.qsaPositions
	if rows == nil {
		rows = make([][tensor.QSAPositionAxes]uint32, graph.tokenCount)
		for index := range rows {
			position := graph.nextPosition + uint32(index)
			rows[index] = [tensor.QSAPositionAxes]uint32{position, position, position}
		}
	}
	after := graph.nextPosition + graph.tokenCount
	if graph.positionAfter != 0 {
		after = graph.positionAfter
	}
	return r.syncQSAPositions(ctx, storage.qsa, graph.pastTokens, rows, after)
}

// qsaStateInputs builds a QSA layer's cells as the step's inputs, in tensor.QSAState's order, which the step binds to
// the sequence's cells once its storage is chosen (bindQSAInputs).
func qsaStateInputs(builder *tensor.Builder, name func(string) string, capacity uint32) (tensor.QSAState, []*tensor.Tensor) {
	cells := uint64(capacity)
	pages := cells / tensor.QSABlockCells
	input := func(suffix string, kind dtype.Type, dimensions ...uint64) *tensor.Tensor {
		return builder.Input(name(suffix), kind, tensor.MustShape(dimensions...))
	}
	state := tensor.QSAState{
		KeyCodes:      input("qsa_key_codes", dtype.I8, tensor.QSAHeadWidth, tensor.QSAKeyValueHeads*cells),
		ValueCodes:    input("qsa_value_codes", dtype.I8, tensor.QSAHeadWidth, tensor.QSAKeyValueHeads*cells),
		KeyScales:     input("qsa_key_scales", dtype.F16, tensor.QSAHeadWidth/tensor.QSAScaleGroup, tensor.QSAKeyValueHeads*cells),
		ValueScales:   input("qsa_value_scales", dtype.F16, tensor.QSAHeadWidth/tensor.QSAScaleGroup, tensor.QSAKeyValueHeads*cells),
		PageTable:     input("qsa_page_table", dtype.I32, pages),
		IndexerPooled: input("qsa_indexer_pooled", dtype.F32, tensor.QSAIndexerWidth, pages+tensor.QSAPooledSpare),
		BlockPosition: input("qsa_block_position", dtype.I32, tensor.SingletonExtent),
		Positions:     input("qsa_positions", dtype.I32, tensor.QSAPositionAxes, cells),
		RopeCos:       input("qsa_rope_cos", dtype.F32, tensor.QSARotated/tensor.PairedExtent, cells),
		RopeSin:       input("qsa_rope_sin", dtype.F32, tensor.QSARotated/tensor.PairedExtent, cells),
	}
	return state, []*tensor.Tensor{
		state.KeyCodes, state.ValueCodes, state.KeyScales, state.ValueScales, state.PageTable,
		state.IndexerPooled, state.BlockPosition, state.Positions, state.RopeCos, state.RopeSin,
	}
}

// bindQSAInputs points a step's QSA layers' inputs at the sequence's cells and the runner's rotation table.
func bindQSAInputs(compiled *executor.CompiledGraph, inputs *executor.DeviceInputs, nodes [][]*tensor.Tensor, cells *qsaCells, rope *qsaRope) error {
	for layer, layerNodes := range nodes {
		if len(layerNodes) == 0 {
			continue
		}
		if cells == nil || layer >= len(cells.layers) || rope == nil || cells.capacity > rope.rows {
			return errors.New("inference: QSA step has no cells to bind")
		}
		held := cells.layers[layer]
		pointers := []driver.DevicePtr{
			held.keyCodes.Pointer, held.valueCodes.Pointer, held.keyScales.Pointer, held.valueScales.Pointer,
			cells.pageTable.Pointer, held.pooled.Pointer, held.blockPosition.Pointer, cells.positions.Pointer, rope.cos, rope.sin,
		}
		if len(layerNodes) != len(pointers) {
			return errors.New("inference: QSA step inputs differ from the cells")
		}
		for index, node := range layerNodes {
			slot, ok := compiled.InputSlot(node)
			if !ok {
				// An input no output reads is not compiled; a QSA step reads every one.
				return fmt.Errorf("inference: QSA input %q is not compiled", node.Name)
			}
			if pointers[index] == 0 {
				return fmt.Errorf("inference: QSA input %q has no cells", node.Name)
			}
			inputs.Pointers[slot] = pointers[index]
		}
	}
	return nil
}
