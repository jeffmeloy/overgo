package inference

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestCompileDeviceCacheTargetPlanPinsSlotsAndCapacity(t *testing.T) {
	const (
		fixturePastTokens = uint32(2)
		fixtureNewTokens  = uint32(1)
		fixturePageTokens = uint32(4)
		fixtureContext    = uint32(16)
		fixtureKeyWidth   = uint32(2)
		fixtureValueWidth = uint32(3)
		fixtureHeads      = uint32(1)
	)
	runner := &Runner{preparedModel: preparedModel{
		spec: model.Spec{
			CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1, ContextLength: fixtureContext},
			AttentionSpec: model.AttentionSpec{
				KeyLength: fixtureKeyWidth, ValueLength: fixtureValueWidth,
				HeadCountKV: fixtureHeads,
			},
		},
		weights: model.Weights{Layers: []model.LayerWeights{{}}},
	}}
	runner = attachFixtureProgram(runner)
	builder := tensor.NewBuilder()
	keyPast := builder.Input("key_past", dtype.F32, tensor.MustShape(
		uint64(fixtureKeyWidth), uint64(fixtureHeads), uint64(fixturePastTokens),
	))
	keyNew := builder.Input("key_new", dtype.F32, tensor.MustShape(
		uint64(fixtureKeyWidth), uint64(fixtureHeads), uint64(fixtureNewTokens),
	))
	valuePast := builder.Input("value_past", dtype.F32, tensor.MustShape(
		uint64(fixtureValueWidth), uint64(fixtureHeads), uint64(fixturePastTokens),
	))
	valueNew := builder.Input("value_new", dtype.F32, tensor.MustShape(
		uint64(fixtureValueWidth), uint64(fixtureHeads), uint64(fixtureNewTokens),
	))
	const tokenAxis = uint32(2)
	key := builder.Concat(keyPast, keyNew, tokenAxis)
	value := builder.Concat(valuePast, valueNew, tokenAxis)
	compiled, err := executor.Compile(key, value)
	if err != nil {
		t.Fatal(err)
	}
	plans, err := runner.compileDeviceCacheTargetPlans(
		compiled,
		[]deviceBatchGraph{{
			keys: []*tensor.Tensor{key}, values: []*tensor.Tensor{value},
			pastTokens: fixturePastTokens, tokenCount: fixtureNewTokens,
		}},
		[]deviceBatchAppend{{PageTokens: fixturePageTokens}},
	)
	if err != nil {
		t.Fatal(err)
	}
	got := plans[0].layers[0]
	if !got.enabled || got.keySlot == got.valueSlot ||
		got.keyCapacity.Dims[tokenAxis] != uint64(fixturePageTokens) ||
		got.valueCapacity.Dims[tokenAxis] != uint64(fixturePageTokens) {
		t.Fatalf("cache target plan = %+v", got)
	}
}

func TestRebuildDeviceCachePagesCreatesPointerViews(t *testing.T) {
	const (
		fixtureKeyWidth     = uint32(2)
		fixtureValueWidth   = uint32(3)
		fixtureHeads        = uint32(1)
		fixtureTokens       = uint32(5)
		fixturePageTokens   = uint32(2)
		fixtureKeyPointer   = driver.DevicePtr(1000)
		fixtureValuePointer = driver.DevicePtr(2000)
	)
	fixture := deviceAttentionCacheFixture{
		KeyWidth: fixtureKeyWidth, ValueWidth: fixtureValueWidth,
		Heads: fixtureHeads, Tokens: fixtureTokens,
		KeyPointer: fixtureKeyPointer, ValuePointer: fixtureValuePointer,
	}
	cache := fixture.cache()
	if err := rebuildDeviceCachePages(cache, fixturePageTokens); err != nil {
		t.Fatal(err)
	}
	wantPages := int((fixtureTokens + fixturePageTokens - 1) / fixturePageTokens)
	if cache.PageTokens != fixturePageTokens || len(cache.Pages) != wantPages {
		t.Fatalf("page table = %+v", cache.Pages)
	}
	for index, page := range cache.Pages {
		wantStart := uint32(index) * fixturePageTokens
		wantCount := min(fixturePageTokens, fixtureTokens-wantStart)
		if page.Start != wantStart || page.Tokens != wantCount ||
			page.Keys[0].Shape != fixture.shape(fixtureKeyWidth, wantCount) ||
			page.Values[0].Shape != fixture.shape(fixtureValueWidth, wantCount) ||
			page.Keys[0].Pointer != fixture.advanced(fixtureKeyPointer, fixtureKeyWidth, wantStart) ||
			page.Values[0].Pointer != fixture.advanced(fixtureValuePointer, fixtureValueWidth, wantStart) {
			t.Fatalf("page %d = %+v", index, page)
		}
	}
}

func TestCachePageCapacityRoundsAndClamps(t *testing.T) {
	const (
		page  = uint32(4)
		limit = uint32(10)
	)
	for _, fixture := range []struct {
		tokens uint32
		want   uint32
	}{
		{1, 4},
		{4, 4},
		{5, 8},
		{9, limit},
		{limit, limit},
	} {
		if got := cachePageCapacity(fixture.tokens, page, limit); got != fixture.want {
			t.Fatalf("capacity(%d) = %d, want %d", fixture.tokens, got, fixture.want)
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
	const (
		fixtureLayerCount     = 1
		fixtureContextTokens  = uint32(4)
		fixtureEmbeddingWidth = 4
		fixtureKeyWidth       = 2
		fixtureValueWidth     = 3
		fixtureKVHeads        = 1
		fixtureConvKernel     = 2
		fixtureSSMInnerWidth  = 2
		fixtureSSMGroups      = 1
		fixtureSSMStateWidth  = 1
		fixtureRemovedTokens  = 1
		fixtureKeyPointer     = driver.DevicePtr(1000)
		fixtureValuePointer   = driver.DevicePtr(2000)
		fixtureFixedPointer   = driver.DevicePtr(3000)
		fixtureTokenPointer   = driver.DevicePtr(4000)
		fixtureFixedState     = "fixed"
		fixtureTokenState     = "token"
	)
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "falcon-h1", BlockCount: fixtureLayerCount,
		ContextLength: fixtureContextTokens, EmbeddingLength: fixtureEmbeddingWidth,
	}, AttentionSpec: model.AttentionSpec{
		KeyLength: fixtureKeyWidth, ValueLength: fixtureValueWidth, HeadCountKV: fixtureKVHeads,
	}, RecurrentSpec: model.RecurrentSpec{
		SSMConvKernel: fixtureConvKernel, SSMInnerSize: fixtureSSMInnerWidth,
		SSMGroupCount: fixtureSSMGroups, SSMStateSize: fixtureSSMStateWidth,
	}},
		weights: model.Weights{Layers: []model.LayerWeights{{}}}},
	}
	runner = attachFixtureProgram(runner)
	fixture := deviceAttentionCacheFixture{
		KeyWidth: fixtureKeyWidth, ValueWidth: fixtureValueWidth,
		Heads: fixtureKVHeads, Tokens: fixtureContextTokens,
		KeyPointer: fixtureKeyPointer, ValuePointer: fixtureValuePointer,
	}
	cache := fixture.cache()
	cache.States = []deviceLayerStates{{
		fixtureFixedState: {
			Mode: CacheStateFixed,
			Value: executor.DeviceValue{
				Pointer: fixtureFixedPointer,
				Shape:   tensor.MustShape(fixtureEmbeddingWidth, uint64(fixtureContextTokens)),
			},
		},
		fixtureTokenState: {
			Mode: CacheStateToken,
			Value: executor.DeviceValue{
				Pointer: fixtureTokenPointer,
				Shape: tensor.MustShape(
					fixtureSSMStateWidth, fixtureKVHeads, uint64(fixtureContextTokens),
				),
			},
		},
	}}
	if err := runner.shiftDeviceCacheForAppendPolicy(cache, fixtureRemovedTokens, fixtureRemovedTokens); err != nil {
		t.Fatal(err)
	}
	wantTokens := fixtureContextTokens - uint32(fixtureRemovedTokens)
	wantKeyPointer := fixture.advanced(fixtureKeyPointer, fixtureKeyWidth, uint32(fixtureRemovedTokens))
	wantValuePointer := fixture.advanced(fixtureValuePointer, fixtureValueWidth, uint32(fixtureRemovedTokens))
	if cache.Tokens != wantTokens || cache.Keys[0].Pointer != wantKeyPointer ||
		cache.Values[0].Pointer != wantValuePointer {
		t.Fatalf("shifted primary cache = %+v", cache)
	}
	if fixed := cache.States[0][fixtureFixedState].Value; fixed.Pointer != fixtureFixedPointer ||
		!fixed.Shape.Equal(tensor.MustShape(fixtureEmbeddingWidth, uint64(fixtureContextTokens))) {
		t.Fatalf("fixed state = %+v", fixed)
	}
	wantTokenPointer := fixture.advanced(fixtureTokenPointer, fixtureSSMStateWidth, uint32(fixtureRemovedTokens))
	if token := cache.States[0][fixtureTokenState].Value; token.Pointer != wantTokenPointer ||
		token.Shape.Dims[2] != uint64(wantTokens) {
		t.Fatalf("token state = %+v", token)
	}
}
