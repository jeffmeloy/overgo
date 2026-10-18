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
		weights := &DeviceWeights{experts: cache}
		reclaimable, capacity, err := weights.ExpertMemory(nil)
		if err != nil || reclaimable != 2*pairBytes || capacity != 2*pairBytes {
			t.Fatalf("expert admission includes tables or omits resident experts: %d/%d: %v", reclaimable, capacity, err)
		}
		allocs := mem.allocs
		check([]uint32{1, 1})
		if mem.allocs != allocs {
			t.Fatal("duplicate hit allocated another expert")
		}
		victims := make(map[string]driver.DevicePtr, len(names))
		for _, name := range names {
			victims[name] = cache.entries[expertKey{name, 0}].Value.(residentExpert).pointer
		}
		check([]uint32{1, 2})
		for _, name := range names {
			if cache.entries[expertKey{name, 0}] != nil {
				t.Fatal("old unused expert was not evicted")
			}
			if cache.entries[expertKey{name, 2}].Value.(residentExpert).pointer != victims[name] {
				t.Fatal("replacement did not reuse the victim allocation")
			}
		}
		if mem.allocs != allocs {
			t.Fatal("equal-sized replacements allocated instead of reusing completed victims")
		}
		if _, err := cache.resolve(t.Context(), []string{"down", "gate"}, []uint32{0}); err != nil {
			t.Fatalf("replacement across different bank sizes: %v", err)
		}
		if mem.allocs != allocs {
			t.Fatal("bank order change discarded available equal-sized allocations")
		}
		check([]uint32{0})
	})
	t.Run("physical-allocation-pressure", func(t *testing.T) {
		cache, mem := open(t, tableBytes+3*pairBytes)
		names := []string{"gate", "down"}
		mem.maxLive = len(names) + 2*len(names)
		if _, err := cache.resolve(t.Context(), names, []uint32{0, 1}); err != nil {
			t.Fatal(err)
		}
		retained := cache.entries[expertKey{"gate", 1}].Value.(residentExpert).pointer
		if _, err := cache.resolve(t.Context(), names, []uint32{1, 2}); err != nil {
			t.Fatalf("physical capacity did not reclaim unused experts: %v", err)
		}
		if cache.entries[expertKey{"gate", 1}].Value.(residentExpert).pointer != retained {
			t.Fatal("the current request's expert was displaced")
		}
		for _, name := range names {
			if cache.entries[expertKey{name, 0}] != nil || cache.entries[expertKey{name, 2}] == nil {
				t.Fatal("physical pressure did not replace only the unused experts")
			}
		}
		if tables, err := cache.resolve(t.Context(), names, []uint32{0, 1, 2}); !driver.IsOutOfMemory(err) || tables != nil {
			t.Fatalf("an entirely required working set must refuse without a table: %v, %v", tables, err)
		}
		mem.maxLive = 0
		if _, err := cache.resolve(t.Context(), names, []uint32{0, 1, 2}); err != nil {
			t.Fatalf("capacity recovery: %v", err)
		}
	})
	for _, name := range []string{"cancelled-before", "cancelled-copy", "failed-copy", "cancelled-replacement", "failed-replacement", "out-of-range", "over-limit"} {
		t.Run(name, func(t *testing.T) {
			cache, mem := open(t, tableBytes+pairBytes)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			indices := []uint32{0}
			retainedEntries, retainedBytes := 0, uint64(0)
			if name == "cancelled-replacement" || name == "failed-replacement" {
				if _, err := cache.resolve(t.Context(), []string{"gate", "down"}, indices); err != nil {
					t.Fatal(err)
				}
				indices = []uint32{1}
				retainedEntries, retainedBytes = 1, cache.banks["down"].bytes
			}
			switch name {
			case "cancelled-before":
				cancel()
			case "cancelled-copy", "cancelled-replacement":
				mem.onWrite = cancel
			case "failed-copy", "failed-replacement":
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
			if (name == "cancelled-before" || name == "cancelled-copy" || name == "cancelled-replacement") && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation = %v", err)
			}
			if len(cache.entries) != retainedEntries || mem.liveCount() != len(file.Tensors)+retainedEntries || cache.bytes != tableBytes+retainedBytes {
				t.Fatal("failed upload retained its allocation or discarded an unrelated expert")
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
