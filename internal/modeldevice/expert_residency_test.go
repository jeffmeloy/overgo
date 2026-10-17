package modeldevice

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func TestFlashNextExpertResidency(t *testing.T) {
	f := writeFixture(t, rand.New(rand.NewPCG(19, 31)), []fixtureTensor{
		{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "down", shape: []uint64{2, 2, 3}, kind: dtype.F32},
	}, 0, 0)
	file := f.parse(t)
	const tableBytes, pairBytes = 2 * 3 * 8, 4*2 + 4*4
	open := func(t *testing.T, limit uint64) (*expertResidency, *fakeDevice) {
		t.Helper()
		mem := newFakeDevice()
		cache, err := newExpertResidency(t.Context(), file, file.Tensors, limit, mem)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := cache.close(); err != nil {
				t.Error(err)
			}
			if mem.liveCount() != 0 {
				t.Error("expert allocations survived close")
			}
		})
		return cache, mem
	}
	t.Run("encoded-hit-miss-eviction", func(t *testing.T) {
		cache, mem := open(t, tableBytes+2*pairBytes)
		names := []string{"gate", "down"}
		check := func(indices []uint32) {
			t.Helper()
			tables, err := cache.resolve(t.Context(), names, indices)
			if err != nil {
				t.Fatal(err)
			}
			for bankIndex, name := range names {
				bank := cache.banks[name]
				table := mem.contents(t, DeviceTensor{Info: bank.info, Pointer: tables[bankIndex]})
				for _, index := range indices {
					pointer := driver.DevicePtr(binary.LittleEndian.Uint64(table[index*8:]))
					actual := mem.contents(t, DeviceTensor{Info: bank.info, Pointer: pointer})
					start := uint64(index) * bank.bytes
					if !bytes.Equal(actual, f.payloads[name][start:start+bank.bytes]) {
						t.Fatalf("%s expert %d changed encoded bytes", name, index)
					}
				}
			}
			if cache.bytes > cache.limit {
				t.Fatal("cache exceeded byte limit")
			}
		}
		check([]uint32{0, 1})
		allocs := mem.allocs
		check([]uint32{1, 1})
		if mem.allocs != allocs {
			t.Fatal("duplicate hit allocated another expert")
		}
		check([]uint32{1, 2})
		for _, name := range names {
			if cache.entries[expertKey{name, 0}] != nil {
				t.Fatal("old unused expert was not evicted")
			}
		}
		if mem.allocs != allocs+len(names) {
			t.Fatal("miss did not allocate exactly one expert per bank")
		}
	})
	for _, name := range []string{"cancelled-before", "cancelled-copy", "failed-copy", "out-of-range", "over-limit"} {
		t.Run(name, func(t *testing.T) {
			cache, mem := open(t, tableBytes+pairBytes)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			indices := []uint32{0}
			switch name {
			case "cancelled-before":
				cancel()
			case "cancelled-copy":
				mem.onWrite = cancel
			case "failed-copy":
				mem.failWrite = mem.writes + 1
			case "out-of-range":
				indices = []uint32{3}
			case "over-limit":
				indices = []uint32{0, 1}
			}
			tables, err := cache.resolve(ctx, []string{"gate", "down"}, indices)
			if err == nil || tables != nil {
				t.Fatal("failed resolution published a table")
			}
			if (name == "cancelled-before" || name == "cancelled-copy") && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation = %v", err)
			}
			if len(cache.entries) != 0 || mem.liveCount() != len(file.Tensors) || cache.bytes != tableBytes {
				t.Fatal("failed first upload retained an expert allocation")
			}
			mem.onWrite = nil
			mem.failWrite = 0
			if _, err := cache.resolve(t.Context(), []string{"gate", "down"}, []uint32{0}); err != nil {
				t.Fatalf("retry: %v", err)
			}
		})
	}
	t.Run("descriptor-and-table-allocation", func(t *testing.T) {
		for _, fail := range []int{1, 2} {
			mem := newFakeDevice()
			mem.failAlloc = fail
			if _, err := newExpertResidency(t.Context(), file, file.Tensors, tableBytes+pairBytes, mem); err == nil {
				t.Fatal("failed table allocation accepted")
			}
			if mem.liveCount() != 0 {
				t.Fatal("partial table allocation leaked")
			}
		}
		changed := append([]gguf.TensorInfo(nil), file.Tensors...)
		changed[0].Offset++
		mem := newFakeDevice()
		if _, err := newExpertResidency(t.Context(), file, changed, tableBytes+pairBytes, mem); err == nil || mem.liveCount() != 0 {
			t.Fatal("foreign descriptor accepted or allocated")
		}
	})
}
