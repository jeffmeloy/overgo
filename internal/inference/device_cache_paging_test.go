package inference

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
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

func BenchmarkRebuildDeviceCachePages(b *testing.B) {
	const (
		layers      = 64
		cacheTokens = 8192
		keyWidth    = 128
		keyHeads    = 8
		keyBase     = 1000
		valueBase   = 2000
		layerStride = 8192
	)
	cache := &deviceKVCache{
		Keys: make([]executor.DeviceValue, layers), Values: make([]executor.DeviceValue, layers),
		Tokens: cacheTokens,
	}
	for layer := range layers {
		cache.Keys[layer] = executor.DeviceValue{
			Pointer: driver.DevicePtr(keyBase + layer*layerStride),
			Shape:   tensor.MustShape(keyWidth, keyHeads, uint64(cache.Tokens)),
		}
		cache.Values[layer] = executor.DeviceValue{
			Pointer: driver.DevicePtr(valueBase + layer*layerStride),
			Shape:   tensor.MustShape(keyWidth, keyHeads, uint64(cache.Tokens)),
		}
	}
	if err := rebuildDeviceCachePages(cache, DefaultCachePageTokens); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := rebuildDeviceCachePages(cache, DefaultCachePageTokens); err != nil {
			b.Fatal(err)
		}
	}
}

func TestShiftDeviceHybridCacheEditsOnlyTokenAlignedState(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "falcon-h1", ContextLength: 4,
	}},
		weights: model.Weights{Layers: []model.LayerWeights{{}}}},
	}
	cache := &deviceKVCache{
		Keys: []executor.DeviceValue{{
			Pointer: driver.DevicePtr(1000), Shape: tensor.MustShape(2, 1, 4),
		}},
		Values: []executor.DeviceValue{{
			Pointer: driver.DevicePtr(2000), Shape: tensor.MustShape(3, 1, 4),
		}},
		States: []deviceLayerStates{{
			"fixed": {
				Mode: CacheStateFixed,
				Value: executor.DeviceValue{
					Pointer: driver.DevicePtr(3000), Shape: tensor.MustShape(4, 4),
				},
			},
			"token": {
				Mode: CacheStateToken,
				Value: executor.DeviceValue{
					Pointer: driver.DevicePtr(4000), Shape: tensor.MustShape(1, 1, 4),
				},
			},
		}},
		Tokens: 4,
	}
	if err := runner.shiftDeviceCacheForAppendPolicy(cache, 1, 1); err != nil {
		t.Fatal(err)
	}
	if cache.Tokens != 3 || cache.Keys[0].Pointer != driver.DevicePtr(1008) ||
		cache.Values[0].Pointer != driver.DevicePtr(2012) {
		t.Fatalf("shifted primary cache = %+v", cache)
	}
	if fixed := cache.States[0]["fixed"].Value; fixed.Pointer != driver.DevicePtr(3000) ||
		!fixed.Shape.Equal(tensor.MustShape(4, 4)) {
		t.Fatalf("fixed state = %+v", fixed)
	}
	if token := cache.States[0]["token"].Value; token.Pointer != driver.DevicePtr(4004) ||
		token.Shape.Dims[2] != 3 {
		t.Fatalf("token state = %+v", token)
	}
}
