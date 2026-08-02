package inference

import (
	"testing"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/tensor"
)

func TestRebuildDeviceCachePagesCreatesPointerViews(t *testing.T) {
	cache := &deviceKVCache{
		Keys: []executor.DeviceValue{{
			Pointer: driver.DevicePtr(1000),
			Shape:   tensor.MustShape(2, 1, 5),
		}},
		Values: []executor.DeviceValue{{
			Pointer: driver.DevicePtr(2000),
			Shape:   tensor.MustShape(3, 1, 5),
		}},
		Tokens: 5,
	}
	if err := rebuildDeviceCachePages(cache, 2); err != nil {
		t.Fatal(err)
	}
	if cache.PageTokens != 2 || len(cache.Pages) != 3 {
		t.Fatalf("page table = %+v", cache.Pages)
	}
	wantStarts := []uint32{0, 2, 4}
	wantCounts := []uint32{2, 2, 1}
	for index, page := range cache.Pages {
		if page.Start != wantStarts[index] || page.Tokens != wantCounts[index] ||
			page.Keys[0].Shape.Dims[2] != uint64(wantCounts[index]) ||
			page.Values[0].Shape.Dims[2] != uint64(wantCounts[index]) ||
			page.Keys[0].Pointer != driver.DevicePtr(1000+8*wantStarts[index]) ||
			page.Values[0].Pointer != driver.DevicePtr(2000+12*wantStarts[index]) {
			t.Fatalf("page %d = %+v", index, page)
		}
	}
}

func TestRebuildDeviceCachePagesPreservesFixedState(t *testing.T) {
	fixed := executor.DeviceValue{
		Pointer: driver.DevicePtr(3000),
		Shape:   tensor.MustShape(4, 4),
	}
	cache := &deviceKVCache{
		Keys: []executor.DeviceValue{fixed}, Values: []executor.DeviceValue{fixed},
		Tokens: 3,
	}
	if err := rebuildDeviceCachePages(cache, 2); err != nil {
		t.Fatal(err)
	}
	for index, page := range cache.Pages {
		if page.Keys[0] != fixed || page.Values[0] != fixed {
			t.Fatalf("page %d fixed state changed", index)
		}
	}
}
