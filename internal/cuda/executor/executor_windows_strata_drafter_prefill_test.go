//go:build windows

package executor

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// The prefill's fixture: the drafter's widest window, the prompt cells it reads, and the bytes the executor's stream
// clears ahead of it, so the prefill begins while that stream still runs, as a prompt read's does.
const (
	strataPrefillMaxT      = 6
	strataPrefillCells     = 4096
	strataPrefillBusyBytes = 4 << 30
)

// TestPrefillStrataDrafterWritesEveryCell holds a prompt's draft prefill to writing each of its rows' cells into the
// draft layer's K/V: the records go up on the drafter's stream ahead of the copies that read them, so each group of up
// to max_t rows appends at its own cells. A synchronous upload on the legacy stream, unordered against the drafter's
// non-blocking stream, had let the copies read the records before they landed: every group wrote at cell 0 and the
// draft layer drafted from an empty prompt. The prefill graphs here are the K/V append alone, of fixed K and V rows.
func TestPrefillStrataDrafterWritesEveryCell(t *testing.T) {
	e := newFixtureExecutor(t)
	heads, width, page := uint64(tensor.QSAKeyValueHeads), uint64(tensor.QSAHeadWidth), uint64(tensor.QSABlockCells)
	pages := int64(strataPrefillCells / page)
	d := &StrataDrafter{spec: StrataDrafterSpec{MaxTokens: strataPrefillMaxT}, resources: &e.resources, pages: pages, slots: pages}
	rowBytes := uint64(strataGRStreams*strataGREmbedding) * strataRoundWordBytes
	kvBytes := strataPrefillMaxT * heads * width * strataRoundWordBytes
	var kv, rows, busy driver.DevicePtr
	if err := e.worker.Do(t.Context(), func(state *device.State) error {
		resources, err := e.ensureResources(state, false, 0, 0, 0)
		if err != nil {
			return err
		}
		carve := func(b *strataRoundBump) {
			d.dTok, d.dStep = b.take(strataPrefillMaxT*strataRoundWordBytes), b.take(strataDrafterStepSets*strataPrefillMaxT*strataDrafterStepWords*strataRoundWordBytes)
			d.dPos = b.take(strataDrafterStepSets * strataPrefillMaxT * strataDrafterHeads * strataRoundWordBytes)
			d.rin = b.take(strataPrefillMaxT * rowBytes)
		}
		var count strataRoundBump
		carve(&count)
		if d.arena, err = e.resources.allocate(state, count.used); err != nil {
			return err
		}
		carve(&strataRoundBump{base: d.arena})
		if err := d.loadKV(state, page, heads, width); err != nil {
			return err
		}
		// one K row and one V row of ones for every token of a group: each cell's codes come out 127
		ones := make([]float32, 2*kvBytes/strataRoundWordBytes)
		for index := range ones {
			ones[index] = 1
		}
		if kv, err = e.resources.allocate(state, 2*kvBytes); err != nil {
			return err
		}
		if err := state.Driver.MemcpyHtoD(kv, float32Bytes(ones)); err != nil {
			return err
		}
		if rows, err = e.resources.allocate(state, strataPrefillCells*rowBytes); err != nil {
			return err
		}
		if busy, err = e.resources.allocate(state, strataPrefillBusyBytes); err != nil {
			return err
		}
		if d.cs, err = state.Driver.StreamCreate(strataRoundStreamNonBlocking); err != nil {
			return err
		}
		geometry := strataKVGeometry{kvHeads: tensor.QSAKeyValueHeads, headDim: tensor.QSAHeadWidth, pageSize: tensor.QSABlockCells}
		for rowsInGroup := int64(1); rowsInGroup <= strataPrefillMaxT; rowsInGroup++ {
			if err := d.capture(state, resources.functions, &d.prefillExec[rowsInGroup], func(w *strataDrafterRecord) error {
				return launchStrataKVAppendSteps(w.state, w.functions, w.pools, w.host, w.table, w.stepAt(0), kv, kv+driver.DevicePtr(kvBytes),
					geometry, int32(rowsInGroup), strataDrafterStepWords, int32(heads*width))
			}); err != nil {
				return err
			}
		}
		// the executor's stream still clearing a gigabyte, as a prompt read still runs when its rows are recorded
		return state.Driver.MemsetD32Async(busy, 0, strataPrefillBusyBytes/strataRoundWordBytes, state.Stream)
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := e.worker.Do(context.Background(), func(state *device.State) error {
			for _, pointer := range []driver.DevicePtr{kv, rows, busy} {
				if pointer != 0 {
					if err := state.Driver.MemFree(pointer); err != nil {
						return err
					}
				}
			}
			return nil
		}); err != nil {
			t.Error(err)
		}
		if err := e.CloseStrataDrafter(context.Background(), d); err != nil {
			t.Error(err)
		}
	})
	next := make([]int32, strataPrefillCells-1)
	if err := e.PrefillStrataDrafter(t.Context(), d, rows, next, 0); err != nil {
		t.Fatal(err)
	}
	codes := make([]byte, width)
	if err := e.worker.Do(t.Context(), func(state *device.State) error {
		for cell := range uint64(len(next)) {
			for head := range heads {
				row := ((cell/page)*heads+head)*page + cell%page
				if err := state.Driver.MemcpyDtoH(codes, d.pools.key+driver.DevicePtr(row*width)); err != nil {
					return err
				}
				if int8(codes[0]) != math.MaxInt8 {
					t.Errorf("cell %d head %d holds key code %d, want %d: the prefill did not write it", cell, head, int8(codes[0]), math.MaxInt8)
				}
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
