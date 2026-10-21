package inference

import (
	"context"
	"errors"

	"overgo/internal/checked"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// speculatingGraphExecs: the captured graphs a speculating runner keeps.
// Each span length on each cache capacity is a graph with two frames, one
// a slot parity, and the default four held too few: a speculative
// Flash-Next short chat made 19 instantiations and 25 updates a request,
// each capture of a 4,000-node span costing 13 to 60 ms. A request whose
// frames outnumber the kept graphs still recaptures nearly all of them,
// since it cycles through them in order: Flash-Next's one-draft short chat
// (six span graphs, two frames each, and the draft step) made six
// instantiations and three updates a request with eight kept and none with
// sixteen (1.45 against 1.32 s); the 27B's three-draft loop (23 span
// graphs, 46 frames) outnumbers both. Requests of several lengths cross more
// capacities: the probe's third chat after two others recaptured six graphs
// with sixteen kept and none with 48.
const speculatingGraphExecs = 48

// spanSlotAlignment: each output's offset in a span slot, the device allocation alignment.
const spanSlotAlignment = 256

// spanSlotCount: the span slots a speculating runner keeps, one its newest past holds and one the next span writes.
const spanSlotCount = 2

// spanSlot is one of the two buffers a speculating runner's verify spans write every output into that is not a token cache: the recurrent states, the hidden rows and the selections.
//   - a span reads its past's states from one slot and writes the other, so a span session replays with two frames whichever span made its past
//   - in the buffer pool the outputs took a lease of their own, whose address followed the pool's reuse order and the layout of the span that made it, so a session's frames rarely repeated
type spanSlot struct {
	buffer *executor.DeviceBuffer
	bytes  uint64
	refs   int
}

// startSpeculatingLocked gives the runner a speculating runner's frames when a request first speculates, so a runner never asked keeps the default executor and pool.
func (r *Runner) startSpeculatingLocked() error {
	if r.speculating {
		return nil
	}
	if err := r.cuda.SetGraphExecCapacity(speculatingGraphExecs); err != nil {
		return err
	}
	r.speculating = true
	return nil
}

// spanSlotOutput is one output a span writes into its slot.
type spanSlotOutput struct {
	slot   executor.OutputSlot
	shape  tensor.Shape
	offset uint64
}

// bindSpanSlot binds a speculating runner's single-branch verify span's outputs that are not token caches to a span slot its past does not hold, and returns the slot; nil keeps them in the pool.
//   - the outputs keep the graph's output order, the recurrent states before the hidden rows and selections, so a state sits at one offset whatever the span's length
func (r *Runner) bindSpanSlot(
	ctx context.Context,
	compiled *executor.CompiledGraph,
	graphs []deviceBatchGraph,
	appends []deviceBatchAppend,
	plans []deviceCacheTargetPlan,
	targets *executor.RetainedTargets,
	output deviceOutputPlan,
) (*spanSlot, error) {
	if !r.speculating || !output.is(deviceOutputGreedySpan) || len(graphs) != 1 || len(appends) != 1 || len(plans) != 1 {
		return nil, nil
	}
	tokenTargets := make(map[*tensor.Tensor]bool)
	for layer, target := range plans[0].layers {
		tokenTargets[graphs[0].keys[layer]] = tokenTargets[graphs[0].keys[layer]] || target.key.enabled
		tokenTargets[graphs[0].values[layer]] = tokenTargets[graphs[0].values[layer]] || target.value.enabled
		for name, state := range target.states {
			if state.enabled {
				tokenTargets[graphs[0].states[layer][name].Value] = true
			}
		}
	}
	var outputs []spanSlotOutput
	var bytes uint64
	for _, node := range decodeGraphOutputs(graphs[0], output) {
		if tokenTargets[node] {
			continue
		}
		slot, ok := compiled.OutputSlot(node)
		if !ok {
			return nil, errors.New("inference: span slot output is not a compiled output")
		}
		size, err := node.Shape.Bytes(dtype.F32)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, spanSlotOutput{slot: slot, shape: node.Shape, offset: bytes})
		if bytes, ok = checked.Align(bytes+size, spanSlotAlignment); !ok {
			return nil, errors.New("inference: span slot size overflows")
		}
	}
	past := appends[0].Past
	var chosen *spanSlot
	for index := range r.spanSlots {
		if r.spanSlots[index] == nil {
			r.spanSlots[index] = &spanSlot{}
		}
		if candidate := r.spanSlots[index]; candidate.refs == 0 && (past == nil || past.spanSlot != candidate) {
			chosen = candidate
			break
		}
	}
	if chosen == nil || len(outputs) == 0 {
		return nil, nil
	}
	if chosen.bytes < bytes {
		if chosen.buffer != nil {
			if err := chosen.buffer.Release(ctx); err != nil {
				return nil, err
			}
		}
		buffer, err := r.cuda.AllocateDeviceBuffer(ctx, bytes)
		if err != nil {
			chosen.buffer, chosen.bytes = nil, 0
			return nil, err
		}
		chosen.buffer, chosen.bytes = buffer, bytes
	}
	whole, err := chosen.buffer.Value(tensor.MustShape(chosen.bytes / cacheScalarBytes))
	if err != nil {
		return nil, err
	}
	for _, item := range outputs {
		value, err := deviceCacheStorageView(whole, item.offset/cacheScalarBytes, item.shape)
		if err != nil {
			return nil, err
		}
		if err := targets.SetSlot(item.slot, value); err != nil {
			return nil, err
		}
	}
	return chosen, nil
}

// holdSpanSlot gives the cache a span made the slot its outputs were written to.
func holdSpanSlot(caches []*deviceKVCache, slot *spanSlot) {
	if slot == nil || len(caches) != 1 {
		return
	}
	slot.refs++
	caches[0].spanSlot = slot
}

// releaseSpanSlots returns the slot buffers to the pool once no cache holds them.
func (r *Runner) releaseSpanSlots(ctx context.Context) error {
	var errs []error
	for index, slot := range r.spanSlots {
		if slot != nil && slot.refs == 0 && slot.buffer != nil {
			errs = append(errs, slot.buffer.Release(ctx))
			r.spanSlots[index] = nil
		}
	}
	return errors.Join(errs...)
}
