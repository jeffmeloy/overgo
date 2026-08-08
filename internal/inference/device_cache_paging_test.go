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
		fixtureScalarBytes    = 4
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
	cache := &deviceKVCache{
		Keys: []executor.DeviceValue{{
			Pointer: fixtureKeyPointer,
			Shape:   tensor.MustShape(fixtureKeyWidth, fixtureKVHeads, uint64(fixtureContextTokens)),
		}},
		Values: []executor.DeviceValue{{
			Pointer: fixtureValuePointer,
			Shape:   tensor.MustShape(fixtureValueWidth, fixtureKVHeads, uint64(fixtureContextTokens)),
		}},
		States: []deviceLayerStates{{
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
		}},
		Tokens: fixtureContextTokens,
	}
	if err := runner.shiftDeviceCacheForAppendPolicy(cache, fixtureRemovedTokens, fixtureRemovedTokens); err != nil {
		t.Fatal(err)
	}
	wantTokens := fixtureContextTokens - uint32(fixtureRemovedTokens)
	wantKeyPointer := fixtureKeyPointer + driver.DevicePtr(fixtureKeyWidth*fixtureScalarBytes)
	wantValuePointer := fixtureValuePointer + driver.DevicePtr(fixtureValueWidth*fixtureScalarBytes)
	if cache.Tokens != wantTokens || cache.Keys[0].Pointer != wantKeyPointer ||
		cache.Values[0].Pointer != wantValuePointer {
		t.Fatalf("shifted primary cache = %+v", cache)
	}
	if fixed := cache.States[0][fixtureFixedState].Value; fixed.Pointer != fixtureFixedPointer ||
		!fixed.Shape.Equal(tensor.MustShape(fixtureEmbeddingWidth, uint64(fixtureContextTokens))) {
		t.Fatalf("fixed state = %+v", fixed)
	}
	wantTokenPointer := fixtureTokenPointer + driver.DevicePtr(fixtureScalarBytes)
	if token := cache.States[0][fixtureTokenState].Value; token.Pointer != wantTokenPointer ||
		token.Shape.Dims[2] != uint64(wantTokens) {
		t.Fatalf("token state = %+v", token)
	}
}
