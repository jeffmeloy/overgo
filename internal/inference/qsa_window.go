package inference

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/modeldevice"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// Strata's verifier over a QSA runner's sequence (src/core/verify.cpp, Strata at 296b59d2, MIT): the sequence's fixed
// state sits at fixed addresses beside its cells (SessionState); each decode step is a verify window keyed by its size
// that reads that state where it sits (record_window), its rows kept at the runner's own addresses; a commit advances
// the state over the tokens the window keeps (capture_commit, commit), except in a one-token window, which advances it
// itself (one_token_self_commit).

// qsaLive is a sequence's fixed state where its windows read it: each layer's fixed cache values - a gated-delta
// layer's history and state, a QSA layer's indexer tail and spare, the lexical injection's convolution and token
// histories - and the tokens it holds.
type qsaLive struct {
	buffer       *executor.DeviceBuffer
	keys, values []executor.DeviceValue
	states       []deviceLayerStates
	tokens       uint32
}

// errQSAStaleLive refuses a cache whose fixed state is its sequence's live state at other tokens: a later window moved
// it on.
var errQSAStaleLive = errors.New("inference: the sequence's live state holds other tokens than the cache")

// names reports whether cache's fixed values are the live state's.
func (l *qsaLive) names(cache *deviceKVCache) bool {
	if l == nil || len(cache.Keys) != len(l.keys) {
		return false
	}
	for layer := range l.keys {
		if l.keys[layer].Pointer != 0 && cache.Keys[layer].Pointer != l.keys[layer].Pointer {
			return false
		}
	}
	return true
}

// staleLive reports whether the cache names its sequence's live state at other tokens than its own: a later window
// moved that state on, so the cache's fixed state is gone.
func (c *deviceKVCache) staleLive() bool {
	if c == nil || c.storage == nil || c.storage.qsa == nil {
		return false
	}
	live := c.storage.qsa.live
	return live.names(c) && live.tokens != c.Tokens
}

// release frees the live state's buffer.
func (l *qsaLive) release(ctx context.Context) error {
	if l == nil || l.buffer == nil {
		return nil
	}
	err := l.buffer.Release(ctx)
	if err == nil {
		l.buffer = nil
	}
	return err
}

// liveDeviceCache answers past as a cache whose fixed state is its sequence's live state: past itself where it names
// the live state at its tokens, else a cache beside it that holds the storage and names the live state, into which
// past's fixed values are copied first (the prompt read hands its state to the verifier).
func (r *Runner) liveDeviceCache(ctx context.Context, past *deviceKVCache) (*deviceKVCache, error) {
	if past == nil || past.storage == nil || past.storage.qsa == nil {
		return nil, errors.New("inference: a verify window's past holds no QSA cells")
	}
	cells := past.storage.qsa
	if cells.live.names(past) {
		if cells.live.tokens != past.Tokens {
			return nil, errQSAStaleLive
		}
		return past, nil
	}
	if cells.live == nil {
		live, err := r.newQSALive(ctx, past)
		if err != nil {
			return nil, err
		}
		cells.live = live
	}
	live := cells.live
	var copies []executor.DeviceRangeCopy
	add := func(target, source executor.DeviceValue) error {
		if target.Pointer == 0 {
			return nil
		}
		bytes, err := source.Shape.Bytes(dtype.F32)
		if err != nil || source.Pointer == 0 || bytes > target.CapacityBytes {
			return errors.New("inference: a cache's fixed state does not fit its sequence's live state")
		}
		copies = append(copies, executor.DeviceRangeCopy{Target: target.Pointer, Source: source.Pointer, Bytes: bytes})
		return nil
	}
	for layer := range live.keys {
		if err := errors.Join(add(live.keys[layer], past.Keys[layer]), add(live.values[layer], past.Values[layer])); err != nil {
			return nil, err
		}
		for name, state := range live.states[layer] {
			if err := add(state.Value, past.States[layer][name].Value); err != nil {
				return nil, err
			}
		}
	}
	if err := r.cuda.CopyDeviceRanges(ctx, copies); err != nil {
		return nil, err
	}
	live.tokens = past.Tokens
	if !past.storage.retain() {
		return nil, errors.New("inference: a verify window's storage is released")
	}
	next := &deviceKVCache{
		storage: past.storage, Keys: append([]executor.DeviceValue(nil), live.keys...), Values: append([]executor.DeviceValue(nil), live.values...),
		States: make([]deviceLayerStates, len(live.states)), Tokens: past.Tokens, Position: past.Position,
		Logits: past.Logits, Candidates: past.Candidates, Selected: past.Selected,
		ngramHistory: append([]float32(nil), past.ngramHistory...),
	}
	for layer, states := range live.states {
		if len(states) != 0 {
			next.States[layer] = states.Clone()
		}
	}
	if err := rebuildDeviceCachePages(next, r.program.Decode.Session); err != nil {
		return nil, errors.Join(err, next.Release(context.Background()))
	}
	return next, nil
}

// newQSALive allocates a sequence's live state for the fixed values past holds, each at its value's shape.
func (r *Runner) newQSALive(ctx context.Context, past *deviceKVCache) (*qsaLive, error) {
	live := &qsaLive{
		keys: make([]executor.DeviceValue, len(past.Keys)), values: make([]executor.DeviceValue, len(past.Values)),
		states: make([]deviceLayerStates, len(past.States)),
	}
	type region struct {
		into  *executor.DeviceValue
		shape tensor.Shape
	}
	var regions []region
	var total uint64
	add := func(into *executor.DeviceValue, value executor.DeviceValue) error {
		if value.Pointer == 0 {
			return nil
		}
		bytes, err := value.Shape.Bytes(dtype.F32)
		if err != nil {
			return err
		}
		regions = append(regions, region{into: into, shape: value.Shape})
		total += alignQSARegion(bytes)
		return nil
	}
	for layer := range past.Keys {
		_, schema, err := r.deviceCacheSchema(layer, past.Tokens)
		if err != nil {
			return nil, err
		}
		if schema.Primary.Key.Mode.TokenAligned() || schema.Primary.Value.Mode.TokenAligned() {
			return nil, fmt.Errorf("inference: layer %d holds token-aligned cache values a verify window cannot keep in place", layer)
		}
		if err := errors.Join(add(&live.keys[layer], past.Keys[layer]), add(&live.values[layer], past.Values[layer])); err != nil {
			return nil, err
		}
	}
	stateValues := make([]map[model.CacheStateName]*executor.DeviceValue, len(past.States))
	for layer, states := range past.States {
		stateValues[layer] = make(map[model.CacheStateName]*executor.DeviceValue, len(states))
		for name, state := range states {
			if state.Mode.TokenAligned() {
				return nil, fmt.Errorf("inference: layer %d state %q is token-aligned, which a verify window cannot keep in place", layer, name)
			}
			value := new(executor.DeviceValue)
			stateValues[layer][name] = value
			if err := add(value, state.Value); err != nil {
				return nil, err
			}
		}
	}
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, max(total, f32Bytes))
	if err != nil {
		return nil, err
	}
	live.buffer = buffer
	whole, err := buffer.Value(tensor.MustShape(max(total, f32Bytes) / f32Bytes))
	if err != nil {
		return nil, errors.Join(err, live.release(context.WithoutCancel(ctx)))
	}
	var offset uint64
	for _, region := range regions {
		bytes, _ := region.shape.Bytes(dtype.F32)
		*region.into = executor.DeviceValue{Pointer: whole.Pointer + driver.DevicePtr(offset), Shape: region.shape, CapacityBytes: bytes}
		offset += alignQSARegion(bytes)
	}
	for layer, states := range past.States {
		if len(states) == 0 {
			continue
		}
		live.states[layer] = make(deviceLayerStates, len(states))
		for name, state := range states {
			live.states[layer][name] = deviceLayerState{Mode: state.Mode, Value: *stateValues[layer][name]}
		}
	}
	return live, nil
}

// qsaWindowPart is a part of the rows a window keeps a layer (model.LayerWindow).
type qsaWindowPart uint8

const (
	qsaWindowQKV qsaWindowPart = iota
	qsaWindowRows
	qsaWindowGates
	qsaWindowRaw
	qsaWindowFull
	qsaWindowParts
)

// qsaWindowScratch is the runner's rows a window keeps, at fixed addresses so every window's graph and every commit's
// reads the same (verify.cpp's qkv_L_, h_L_, gate_L_, beta_L_, idx_raw_L_ and hist_snap_), each part a layer sized for
// the most tokens a window holds; a QSA layer's tail snapshot (tail_snap_); the one-token window's kept count (one_);
// the commit's words, its kept count then each row's cell, in pinned host memory the commit reads mapped, where the host
// writes them before each commit (commit_ staged through m_commit_).
type qsaWindowScratch struct {
	control, rows *executor.DeviceBuffer
	words         *executor.PinnedHostBuffer
	one           executor.DeviceValue
	snapshots     []executor.DeviceValue
	parts         [][qsaWindowParts]executor.DeviceValue
}

// qsaWindowControl allocates the runner's window control at its first window: the kept count of one, the commit's
// words and each QSA layer's tail snapshot.
func (r *Runner) qsaWindowControl(ctx context.Context) (*qsaWindowScratch, error) {
	if r.qsaWindow != nil {
		return r.qsaWindow, nil
	}
	tailBytes := uint64(tensor.QSATailRows * tensor.QSAIndexerWidth * f32Bytes)
	words := uint64(tensor.SingletonExtent + tensor.GatedDeltaWindowTokens)
	total := alignQSARegion(i32Bytes) + uint64(len(r.qsaLayers))*alignQSARegion(tailBytes)
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, total)
	if err != nil {
		return nil, err
	}
	scratch := &qsaWindowScratch{control: buffer, snapshots: make([]executor.DeviceValue, len(r.qsaLayers))}
	if scratch.words, err = r.cuda.AllocatePinnedHost(ctx, words*i32Bytes); err != nil {
		return nil, errors.Join(err, buffer.Release(context.WithoutCancel(ctx)))
	}
	whole, err := buffer.Value(tensor.MustShape(total / f32Bytes))
	if err != nil {
		return nil, errors.Join(err, scratch.release(context.WithoutCancel(ctx)))
	}
	offset := uint64(0)
	place := func(shape tensor.Shape, bytes uint64) executor.DeviceValue {
		value := executor.DeviceValue{Pointer: whole.Pointer + driver.DevicePtr(offset), Shape: shape, CapacityBytes: bytes}
		offset += alignQSARegion(bytes)
		return value
	}
	scratch.one = place(tensor.MustShape(tensor.SingletonExtent), i32Bytes)
	for layer, holds := range r.qsaLayers {
		if holds {
			scratch.snapshots[layer] = place(tensor.MustShape(tensor.QSAIndexerWidth, tensor.QSATailRows), tailBytes)
		}
	}
	one := []byte{1, 0, 0, 0}
	if err := r.cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{{Pointer: scratch.one.Pointer, Data: one}}); err != nil {
		return nil, errors.Join(err, scratch.release(context.WithoutCancel(ctx)))
	}
	r.qsaWindow = scratch
	return scratch, nil
}

// qsaWindowRowBytes is a window part's bytes for the most tokens a window holds, from its node at tokens tokens: a
// token's rows times the most tokens, the lexical rows after their history's.
func qsaWindowRowBytes(part qsaWindowPart, node *tensor.Tensor, tokens uint64) (uint64, error) {
	elements, err := node.Shape.Elements()
	if err != nil || tokens == 0 {
		return 0, errors.New("inference: a window's kept rows are invalid")
	}
	most := uint64(tensor.GatedDeltaWindowTokens)
	if part == qsaWindowFull {
		width := node.Shape.Dims[0]
		return (elements + width*(most-tokens)) * f32Bytes, nil
	}
	return elements / tokens * most * f32Bytes, nil
}

// windowParts are a layer's window rows by part.
func windowParts(window model.LayerWindow) [qsaWindowParts]*tensor.Tensor {
	return [qsaWindowParts]*tensor.Tensor{window.QKV, window.Rows, window.Gates, window.Raw, window.Full}
}

// bindWindowRows binds each window row a graph keeps to the runner's rows, allocating them at the first window, so
// every window and commit reads and writes them at the same addresses.
func (r *Runner) bindWindowRows(ctx context.Context, targets *executor.RetainedTargets, graphs []deviceBatchGraph) error {
	for _, graph := range graphs {
		if graph.window == nil {
			continue
		}
		if err := r.ensureWindowRows(ctx, graph); err != nil {
			return err
		}
		for layer, window := range graph.window {
			for part, node := range windowParts(window) {
				if node == nil {
					continue
				}
				held := r.qsaWindow.parts[layer][part]
				bytes, err := node.Shape.Bytes(dtype.F32)
				if err != nil || held.Pointer == 0 || bytes > held.CapacityBytes {
					return errors.New("inference: a window's rows outgrow the runner's")
				}
				if err := targets.Set(node, executor.DeviceValue{Pointer: held.Pointer, Shape: node.Shape, CapacityBytes: held.CapacityBytes}); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// ensureWindowRows allocates the runner's window rows from a window graph's parts.
func (r *Runner) ensureWindowRows(ctx context.Context, graph deviceBatchGraph) error {
	scratch := r.qsaWindow
	if scratch == nil {
		return errors.New("inference: a window's control is not allocated")
	}
	if scratch.rows != nil {
		return nil
	}
	sizes := make([][qsaWindowParts]uint64, len(graph.window))
	var total uint64
	for layer, window := range graph.window {
		for part, node := range windowParts(window) {
			if node == nil {
				continue
			}
			bytes, err := qsaWindowRowBytes(qsaWindowPart(part), node, uint64(graph.tokenCount))
			if err != nil {
				return err
			}
			sizes[layer][part] = bytes
			total += alignQSARegion(bytes)
		}
	}
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, max(total, f32Bytes))
	if err != nil {
		return err
	}
	whole, err := buffer.Value(tensor.MustShape(max(total, f32Bytes) / f32Bytes))
	if err != nil {
		return errors.Join(err, buffer.Release(context.WithoutCancel(ctx)))
	}
	scratch.rows, scratch.parts = buffer, make([][qsaWindowParts]executor.DeviceValue, len(graph.window))
	offset := uint64(0)
	for layer := range sizes {
		for part, bytes := range sizes[layer] {
			if bytes == 0 {
				continue
			}
			scratch.parts[layer][part] = executor.DeviceValue{Pointer: whole.Pointer + driver.DevicePtr(offset), CapacityBytes: bytes}
			offset += alignQSARegion(bytes)
		}
	}
	return nil
}

// release frees the runner's window rows and control.
func (s *qsaWindowScratch) release(ctx context.Context) error {
	if s == nil {
		return nil
	}
	errs := []error{s.words.Release(ctx)}
	for _, buffer := range []*executor.DeviceBuffer{s.rows, s.control} {
		if buffer != nil {
			errs = append(errs, buffer.Release(ctx))
		}
	}
	return errors.Join(errs...)
}

// qsaCommitKey is a commit graph's identity: its window's tokens and the cells it binds, their capacity and slots.
type qsaCommitKey struct {
	tokens, capacity, slots uint32
}

// qsaCommitGraph is a compiled commit (verify.cpp capture_commit), one a window size and cell layout: the live state's
// inputs by layer, rebound to each sequence's, and its QSA state inputs.
type qsaCommitGraph struct {
	execution *executor.IndexedGraph
	output    *tensor.Tensor
	keys      []*tensor.Tensor
	values    []*tensor.Tensor
	states    []model.CacheStates[*tensor.Tensor]
	qsa       []tensor.QSAState
}

// commitWindowLocked commits a window over a live past: the window's first kept tokens advance the sequence's live
// state, which holds past's tokens and then past's tokens and the kept ones; a one-token window committed itself.
func (r *Runner) commitWindowLocked(ctx context.Context, cells *qsaCells, pastTokens uint32, tokens, kept int) error {
	layout := r.qsaWindowLayouts[tokens]
	if cells == nil || cells.live == nil || kept < 1 || kept > tokens {
		return errors.New("inference: a window commit is invalid")
	}
	live := cells.live
	if tokens == tensor.SingletonExtent {
		live.tokens = pastTokens + 1
		return nil
	}
	if layout == nil {
		return errors.New("inference: a window commit's window was never built")
	}
	key := qsaCommitKey{tokens: uint32(tokens), capacity: cells.capacity, slots: cells.slots}
	commit := r.qsaCommits[key]
	if commit == nil {
		built, err := r.buildWindowCommit(layout, uint64(tokens), live, cells)
		if err != nil {
			return err
		}
		if r.qsaCommits == nil {
			r.qsaCommits = make(map[qsaCommitKey]*qsaCommitGraph)
		}
		r.qsaCommits[key], commit = built, built
	}
	inputs := commit.execution.Inputs
	// The live state's values the commit reads, rebound to this sequence's; a value no layer's commit reads is not an
	// input of the compiled commit.
	set := func(node *tensor.Tensor, pointer driver.DevicePtr) error {
		if _, read := commit.execution.Graph.InputSlot(node); !read {
			return nil
		}
		return inputs.Set(node, pointer)
	}
	for layer := range commit.keys {
		if err := errors.Join(set(commit.keys[layer], live.keys[layer].Pointer), set(commit.values[layer], live.values[layer].Pointer)); err != nil {
			return err
		}
		for name, node := range commit.states[layer] {
			if err := set(node.Value, live.states[layer][name].Value.Pointer); err != nil {
				return err
			}
		}
	}
	// A QSA layer's commit reads its pooled rows, its block position, the position and rotation tables.
	for layer, state := range commit.qsa {
		if state.IndexerPooled == nil {
			continue
		}
		if layer >= len(cells.layers) || r.qsaRope == nil {
			return errors.New("inference: a QSA commit has no cells to bind")
		}
		held := cells.layers[layer]
		if err := errors.Join(
			set(state.IndexerPooled, held.pooled.Pointer), set(state.BlockPosition, held.blockPosition.Pointer),
			set(state.Positions, cells.positions.Pointer), set(state.RopeCos, r.qsaRope.cos), set(state.RopeSin, r.qsaRope.sin),
		); err != nil {
			return err
		}
	}
	// The words go straight to the mapped host memory the commit reads: the window before it has run, its results read.
	words := r.qsaWindow.words.Host
	putWord := func(index int, value int32) { binary.LittleEndian.PutUint32(words[index*i32Bytes:], uint32(value)) }
	putWord(0, int32(kept))
	for row := range tokens {
		cell := int32(-1)
		if row < kept {
			cell = int32(pastTokens) + int32(row)
		}
		putWord(tensor.SingletonExtent+row, cell)
	}
	retained, err := r.cuda.ExecuteRetainedCompiled(ctx, commit.execution.Graph, nil, inputs, nil, nil)
	if err != nil {
		return err
	}
	if err := retained.Release(ctx); err != nil {
		return err
	}
	live.tokens = pastTokens + uint32(kept)
	return nil
}

// buildWindowCommit compiles the commit of a window graph's size over a sequence's live state and cells: each layer's
// commit (model.CompiledLayerProgram.Commit) in order, the window's rows read where the window wrote them, the kept count
// and the cells from the commit's words.
func (r *Runner) buildWindowCommit(layout []model.LayerWindow, tokens uint64, live *qsaLive, cells *qsaCells) (*qsaCommitGraph, error) {
	builder := r.newGraphBuilder()
	commit := &qsaCommitGraph{
		keys: make([]*tensor.Tensor, len(live.keys)), values: make([]*tensor.Tensor, len(live.values)),
		states: make([]model.CacheStates[*tensor.Tensor], len(live.states)), qsa: make([]tensor.QSAState, len(r.weights.Layers)),
	}
	var feeds tensor.InputBindings[driver.DevicePtr]
	kept := builder.Input("window.kept", dtype.I32, tensor.MustShape(tensor.SingletonExtent))
	rowCells := builder.Input("window.cells", dtype.I32, tensor.MustShape(tokens))
	feeds = append(feeds, tensor.InputBinding[driver.DevicePtr]{Node: kept, Value: r.qsaWindow.words.Device},
		tensor.InputBinding[driver.DevicePtr]{Node: rowCells, Value: r.qsaWindow.words.Device + driver.DevicePtr(i32Bytes)})
	var after *tensor.Tensor
	// Every gated-delta layer's commit is one node, launched once a kernel for all of them (Strata's gdn_commit_layers).
	var deferred []tensor.GatedDeltaCommitLayer
	for layer, window := range layout {
		parts := windowParts(window)
		holdsQSA := r.holdsQSA(layer)
		if !holdsQSA && window.Rows == nil && window.Full == nil {
			continue
		}
		name := func(suffix string) string { return fmt.Sprintf("window.blk.%d.%s", layer, suffix) }
		inputs := model.LayerCommitInputs{Kept: kept, Deferred: &deferred}
		if live.keys[layer].Pointer != 0 {
			commit.keys[layer] = builder.Input(name("key"), dtype.F32, live.keys[layer].Shape)
			commit.values[layer] = builder.Input(name("value"), dtype.F32, live.values[layer].Shape)
			inputs.Key, inputs.Value = commit.keys[layer], commit.values[layer]
		}
		if len(live.states[layer]) != 0 {
			commit.states[layer] = make(model.CacheStates[*tensor.Tensor], len(live.states[layer]))
			for stateName, state := range live.states[layer] {
				commit.states[layer][stateName] = model.CacheState[*tensor.Tensor]{
					Mode: state.Mode, Value: builder.Input(name(string(stateName)), dtype.F32, state.Value.Shape),
				}
			}
			inputs.States = commit.states[layer]
		}
		rows := make([]*tensor.Tensor, qsaWindowParts)
		for part, node := range parts {
			if node == nil {
				continue
			}
			rows[part] = builder.Input(name(fmt.Sprintf("rows.%d", part)), dtype.F32, node.Shape)
			feeds = append(feeds, tensor.InputBinding[driver.DevicePtr]{Node: rows[part], Value: r.qsaWindow.parts[layer][part].Pointer})
		}
		inputs.Window = model.LayerWindow{
			QKV: rows[qsaWindowQKV], Rows: rows[qsaWindowRows], Gates: rows[qsaWindowGates], Raw: rows[qsaWindowRaw], Full: rows[qsaWindowFull],
		}
		info := r.weights.Layers[layer]
		weights, layerFeeds, err := modeldevice.BindDeviceLayerGraphInputs(builder, info, modeldevice.DeviceTensorBinder(r.deviceInput))
		if err != nil {
			return nil, err
		}
		feeds = append(feeds, layerFeeds...)
		if holdsQSA {
			state, _ := qsaStateInputs(builder, name, cells.capacity, cells.slots)
			commit.qsa[layer] = state
			inputs.QSA = &model.QSABinding{State: state, Capacity: cells.capacity}
			inputs.Cells = rowCells
			inputs.Snapshot = builder.Input(name("snapshot"), dtype.F32, r.qsaWindow.snapshots[layer].Shape)
			feeds = append(feeds, tensor.InputBinding[driver.DevicePtr]{Node: inputs.Snapshot, Value: r.qsaWindow.snapshots[layer].Pointer})
		}
		if after, err = r.layerProgram(layer).Commit(builder, after, inputs, weights); err != nil {
			return nil, fmt.Errorf("inference: window commit layer %d: %w", layer, err)
		}
	}
	if len(deferred) != 0 {
		after = builder.GatedDeltaCommitLayers(after, deferred, kept, r.spec.SSMGroupCount, r.spec.SSMTimeStepRank)
	}
	if after == nil {
		return nil, errors.New("inference: a window commit holds no layer")
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	execution, err := executor.CompileIndexed(after)
	if err != nil {
		return nil, err
	}
	if err := execution.Inputs.Bind(feeds); err != nil {
		return nil, err
	}
	commit.execution, commit.output = execution, after
	return commit, nil
}
