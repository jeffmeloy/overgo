package inference

import (
	"encoding/binary"
	"reflect"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const fixtureCacheStateName model.CacheStateName = "fixture_state"

func TestKVCacheStateRoundTrip(t *testing.T) {
	runner := cacheTestRunner()
	cache := cacheTestValue(t)
	data, err := runner.SaveCache(cache)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := runner.LoadCache(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded, cache) {
		t.Fatalf("loaded cache = %+v, want %+v", loaded, cache)
	}
}

func TestKVCacheNamedStatesRoundTripAndEdit(t *testing.T) {
	runner := cacheTestRunner()
	cache := cacheTestValue(t)
	tokenState, err := reference.NewValue(
		tensor.MustShape(1, 1, 2), []float32{11, 12},
	)
	if err != nil {
		t.Fatal(err)
	}
	fixedState, err := reference.NewValue(
		tensor.MustShape(2), []float32{21, 22},
	)
	if err != nil {
		t.Fatal(err)
	}
	cache.Layers[0].States = LayerStates{
		model.CacheStateIndexerKey:  {Mode: CacheStateToken, Value: tokenState},
		model.CacheStateConvolution: {Mode: CacheStateFixed, Value: fixedState},
		fixtureCacheStateName:       {Mode: CacheStateFixed, Value: fixedState},
	}
	payload, err := runner.SaveCache(cache)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadCache(payload)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, cache) {
		t.Fatalf("restored cache = %+v, want %+v", restored, cache)
	}

	reordered := cacheTestValue(t)
	reordered.Layers[0].States = LayerStates{
		model.CacheStateConvolution: {Mode: CacheStateFixed, Value: fixedState},
		model.CacheStateIndexerKey:  {Mode: CacheStateToken, Value: tokenState},
		fixtureCacheStateName:       {Mode: CacheStateFixed, Value: fixedState},
	}
	reorderedPayload, err := runner.SaveCache(reordered)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reorderedPayload, payload) {
		t.Fatal("named cache serialization depends on map insertion order")
	}

	trimmed, err := runner.RemoveCacheRange(restored, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := trimmed.Layers[0].States[model.CacheStateIndexerKey].Value.Data; !reflect.DeepEqual(got, []float32{12}) {
		t.Fatalf("trimmed token state = %v, want [12]", got)
	}
	if got := trimmed.Layers[0].States[model.CacheStateConvolution].Value.Data; !reflect.DeepEqual(got, []float32{21, 22}) {
		t.Fatalf("fixed state = %v, want [21 22]", got)
	}
	trimmed.Layers[0].States[model.CacheStateConvolution].Value.Data[0] = 99
	if restored.Layers[0].States[model.CacheStateConvolution].Value.Data[0] != 21 {
		t.Fatal("edited fixed state aliases source cache")
	}
}

func TestKVCacheNamedStateValidation(t *testing.T) {
	fixed, err := reference.NewValue(tensor.MustShape(1), []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	token, err := reference.NewValue(tensor.MustShape(1, 1, 1), []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		state LayerStates
	}{
		{"reserved name", LayerStates{"key": {Mode: CacheStateFixed, Value: fixed}}},
		{"invalid name", LayerStates{"Bad Name": {Mode: CacheStateFixed, Value: fixed}}},
		{"invalid mode", LayerStates{"state": {Mode: 99, Value: fixed}}},
		{"wrong token extent", LayerStates{"state": {Mode: CacheStateToken, Value: token}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache := cacheTestValue(t)
			cache.Layers[0].States = test.state
			if _, err := cacheTestRunner().SaveCache(cache); err == nil {
				t.Fatal("invalid named state was accepted")
			}
		})
	}
}

func TestDeepSeek2AbsorbedCacheValidation(t *testing.T) {
	testDeepSeek2FamilyAbsorbedCacheValidation(t, "deepseek2")
}

func TestMistral4AbsorbedCacheValidation(t *testing.T) {
	testDeepSeek2FamilyAbsorbedCacheValidation(t, "mistral4")
}

func TestDeepSeek32CacheStateRoundTrip(t *testing.T) {
	runner, cache := deepSeek32CacheFixture()
	payload, err := runner.SaveCache(cache)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadCache(payload)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, cache) {
		t.Fatalf("restored cache = %+v, want %+v", restored, cache)
	}
	delete(cache.Layers[0].States, model.CacheStateIndexerKey)
	if err := runner.validateCache(cache); err == nil {
		t.Fatal("DeepSeek 3.2 cache without indexer state was accepted")
	}
}

func TestDeepSeek32NamedStateGraphBinding(t *testing.T) {
	runner, cache := deepSeek32CacheFixture()
	builder := tensor.NewBuilder()
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	host, err := runner.hostLayerCacheInputs(builder, 0, &cache.Layers[0], hostFeeds)
	if err != nil {
		t.Fatal(err)
	}
	hostIndexer := host.states[model.CacheStateIndexerKey]
	if hostIndexer.Mode != model.CacheStateToken ||
		!reflect.DeepEqual(hostFeeds[hostIndexer.Value], cache.Layers[0].States[model.CacheStateIndexerKey].Value) {
		t.Fatalf("host indexer binding = %+v", hostIndexer)
	}

	keyPointer := driver.DevicePtr(101)
	valuePointer := driver.DevicePtr(202)
	indexerPointer := driver.DevicePtr(303)
	deviceCache := &deviceKVCache{
		Keys:   []executor.DeviceValue{{Pointer: keyPointer, Shape: cache.Layers[0].Key.Shape}},
		Values: []executor.DeviceValue{{Pointer: valuePointer, Shape: cache.Layers[0].Value.Shape}},
		States: []deviceLayerStates{{
			model.CacheStateIndexerKey: {
				Mode: model.CacheStateToken,
				Value: executor.DeviceValue{
					Pointer: indexerPointer,
					Shape:   cache.Layers[0].States[model.CacheStateIndexerKey].Value.Shape,
				},
			},
		}},
	}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	device, err := runner.deviceBatchLayerCacheInputs(
		builder, "fixture.", 0, deviceCache, hostFeeds, deviceFeeds,
	)
	if err != nil {
		t.Fatal(err)
	}
	deviceIndexer := device.states[model.CacheStateIndexerKey]
	if deviceIndexer.Mode != model.CacheStateToken || deviceFeeds[deviceIndexer.Value] != indexerPointer {
		t.Fatalf("device indexer binding = %+v", deviceIndexer)
	}
}

func deepSeek32CacheFixture() (*Runner, *KVCache) {
	attentionKB := gguf.TensorInfo{Name: "blk.0.attn_k_b.weight"}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek32", BlockCount: 1, ContextLength: 16}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KVLoRARank: 3, RopeDimensionCount: 2,
		IndexerKeyLength: 4, IndexerFullLayers: []bool{true}}}
	weights := model.Weights{Layers: []model.LayerWeights{{AttentionKB: &attentionKB}}}
	runner := fixtureRunner(spec, weights)
	key, _ := reference.NewValue(tensor.MustShape(5, 1, 2), make([]float32, 10))
	value, _ := reference.NewValue(tensor.MustShape(3, 1, 2), make([]float32, 6))
	indexerKey, _ := reference.NewValue(tensor.MustShape(4, 1, 2), []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
	})
	cache := &KVCache{
		Layers: []LayerCache{{
			Key: key, Value: value,
			States: LayerStates{
				model.CacheStateIndexerKey: {Mode: CacheStateToken, Value: indexerKey},
			},
		}},
		Tokens: 2, Position: 2,
	}
	return runner, cache
}

func TestDeepSeek4CacheStateRoundTrip(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek4", BlockCount: 1, ContextLength: 16}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		IndexerKeyLength: 8, CompressRatios: []uint32{4}},
	}}}
	runner = attachFixtureProgram(runner)
	key, _ := reference.NewValue(tensor.MustShape(4, 1, 2), make([]float32, 8))
	state := func(width uint64) LayerState {
		value, _ := reference.NewValue(tensor.MustShape(width, 1, 2), make([]float32, int(2*width)))
		return LayerState{Mode: CacheStateToken, Value: value}
	}
	cache := &KVCache{Layers: []LayerCache{{Key: key, Value: key, States: LayerStates{
		model.CacheStatePositions:              {Mode: CacheStateToken, Value: reference.Value{Shape: tensor.MustShape(1, 1, 2), Data: []float32{0, 1}}},
		model.CacheStateCompressorKV:           state(8),
		model.CacheStateCompressorScore:        state(8),
		model.CacheStateIndexerCompressorKV:    state(16),
		model.CacheStateIndexerCompressorScore: state(16),
	}}}, Tokens: 2, Position: 2}
	payload, err := runner.SaveCache(cache)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadCache(payload)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, cache) {
		t.Fatalf("restored cache = %+v, want %+v", restored, cache)
	}
	prefixShifted, err := runner.RemoveCacheRange(cache, 0, 1)
	if err != nil || prefixShifted.Tokens != 1 || prefixShifted.Position != 2 {
		t.Fatalf("DeepSeek 4 prefix shift = %+v, %v", prefixShifted, err)
	}
	cache.Tokens = 4
	cache.Position = 4
	key, _ = reference.NewValue(tensor.MustShape(4, 1, 4), make([]float32, 16))
	cache.Layers[0].Key, cache.Layers[0].Value = key, key
	for name, state := range cache.Layers[0].States {
		width := state.Value.Shape.Dims[0]
		data := make([]float32, int(4*width))
		if name == model.CacheStatePositions {
			data = []float32{0, 1, 2, 3}
		}
		state.Value, _ = reference.NewValue(tensor.MustShape(width, 1, 4), data)
		cache.Layers[0].States[name] = state
	}
	edited, err := runner.RemoveCacheRange(cache, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := edited.Layers[0].States[model.CacheStatePositions].Value.Data; !reflect.DeepEqual(got, []float32{0, 2, 3}) {
		t.Fatalf("DeepSeek 4 edited positions = %v", got)
	}
	delete(cache.Layers[0].States, model.CacheStateIndexerCompressorScore)
	if err := runner.validateCache(cache); err == nil {
		t.Fatal("DeepSeek 4 cache without indexer compressor score was accepted")
	}
}

func TestMambaCacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "mamba", BlockCount: 1}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2}}}}
	runner = attachFixtureProgram(runner)
	conv, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
	trimmed, err := runner.RemoveCacheRange(cache, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if trimmed.Tokens != 1 || !trimmed.Layers[0].Key.Shape.Equal(conv.Shape) ||
		!trimmed.Layers[0].Value.Shape.Equal(ssm.Shape) {
		t.Fatalf("trimmed Mamba cache = %+v", trimmed)
	}
	trimmed.Layers[0].Key.Data[0] = 1
	if cache.Layers[0].Key.Data[0] != 0 {
		t.Fatal("trimmed Mamba cache aliases source state")
	}
}

func TestMamba2CacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "mamba2", BlockCount: 1}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMGroupCount: 2}}}}
	runner = attachFixtureProgram(runner)
	conv, _ := reference.NewValue(tensor.MustShape(2, 16), make([]float32, 32))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestFalconH1CacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "falcon-h1", BlockCount: 1}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 2, HeadCountKV: 1}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMGroupCount: 2}}}}
	runner = attachFixtureProgram(runner)
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	conv, _ := reference.NewValue(tensor.MustShape(2, 16), make([]float32, 32))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	cache := &KVCache{Layers: []LayerCache{{Key: key, Value: value, States: LayerStates{
		model.CacheStateConvolution: {Mode: CacheStateFixed, Value: conv},
		model.CacheStateSSM:         {Mode: CacheStateFixed, Value: ssm},
	}}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
	cache.Layers[0].States[model.CacheStateConvolution] = LayerState{Mode: CacheStateToken, Value: conv}
	if err := runner.validateCache(cache); err == nil {
		t.Fatal("Falcon-H1 token-mode recurrent state was accepted")
	}
}

func TestT5CacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "t5", ContextLength: 16}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 3, HeadCountKV: 1}, EncoderSpec: model.EncoderSpec{DecoderBlockCount: 1}}}}
	runner = attachFixtureProgram(runner)
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(3, 1, 2), make([]float32, 6))
	crossKey, _ := reference.NewValue(tensor.MustShape(2, 1, 4), make([]float32, 8))
	crossValue, _ := reference.NewValue(tensor.MustShape(3, 1, 4), make([]float32, 12))
	cache := &KVCache{Layers: []LayerCache{{
		Key: key, Value: value,
		States: LayerStates{
			model.CacheStateCrossKey:   {Mode: CacheStateFixed, Value: crossKey},
			model.CacheStateCrossValue: {Mode: CacheStateFixed, Value: crossValue},
		},
	}}, Tokens: 2, Position: 2}
	if err := runner.validateEncoderDecoderCache(cache, 4); err != nil {
		t.Fatal(err)
	}
	if err := runner.validateEncoderDecoderCache(cache, 3); err == nil {
		t.Fatal("mismatched T5 encoder extent was accepted")
	}
	shifted, err := runner.RemoveCacheRange(cache, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if shifted.Tokens != 1 || shifted.Position != 1 ||
		!shifted.Layers[0].Key.Shape.Equal(tensor.MustShape(2, 1, 1)) ||
		!shifted.Layers[0].Value.Shape.Equal(tensor.MustShape(3, 1, 1)) {
		t.Fatalf("shifted T5 cache = %+v", shifted)
	}
	if !shifted.Layers[0].States[model.CacheStateCrossKey].Value.Shape.Equal(crossKey.Shape) ||
		!shifted.Layers[0].States[model.CacheStateCrossValue].Value.Shape.Equal(crossValue.Shape) {
		t.Fatalf("shifted T5 cross cache changed: %+v", shifted.Layers[0].States)
	}
	state := cache.Layers[0].States[model.CacheStateCrossKey]
	state.Mode = CacheStateToken
	cache.Layers[0].States[model.CacheStateCrossKey] = state
	if err := runner.validateCache(cache); err == nil {
		t.Fatal("token-aligned T5 cross cache was accepted")
	}
}

func TestJambaHybridCacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "jamba", BlockCount: 2}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 2,
		HeadCountKV: 1}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize: 8, SSMStateSize: 2,
		RecurrentLayers: []bool{true, false}}},
		weights: model.Weights{Layers: []model.LayerWeights{{Recurrent: true}, {}}}},
	}
	runner = attachFixtureProgram(runner)
	conv, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}, {Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestGraniteHybridCacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "granitehybrid", BlockCount: 2}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 2,
		HeadCountKV: 1, LayerKVHeadCounts: []uint32{0, 1}}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize: 8, SSMStateSize: 2, SSMGroupCount: 2,
		RecurrentLayers: []bool{true, false}}},
		weights: model.Weights{Layers: []model.LayerWeights{{Recurrent: true}, {}}}},
	}
	runner = attachFixtureProgram(runner)
	conv, _ := reference.NewValue(tensor.MustShape(2, 16), make([]float32, 32))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}, {Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestPLaMo2HybridCacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "plamo2", BlockCount: 2}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 2,
		HeadCountKV: 1, LayerKVHeadCounts: []uint32{0, 1}}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize: 8, SSMStateSize: 2,
		RecurrentLayers: []bool{true, false}}},
		weights: model.Weights{Layers: []model.LayerWeights{{Recurrent: true}, {}}}},
	}
	runner = attachFixtureProgram(runner)
	conv, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}, {Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestNemotronHThreeWayCacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "nemotron_h_moe", BlockCount: 3}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 2,
		HeadCountKV: 1, LayerKVHeadCounts: []uint32{1, 0, 1}}, MoESpec: model.MoESpec{LayerFeedForward: []uint32{0, 0, 6}}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize: 8, SSMStateSize: 2, SSMGroupCount: 2,

		RecurrentLayers: []bool{false, true, false}}},
		weights: model.Weights{Layers: []model.LayerWeights{{}, {Recurrent: true}, {}}}},
	}
	runner = attachFixtureProgram(runner)
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	conv, _ := reference.NewValue(tensor.MustShape(2, 16), make([]float32, 32))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	sentinel, _ := reference.NewValue(tensor.MustShape(1, 1, 2), make([]float32, 2))
	cache := &KVCache{Layers: []LayerCache{
		{Key: key, Value: value}, {Key: conv, Value: ssm}, {Key: sentinel, Value: sentinel},
	}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestKimiLinearHybridCacheValidation(t *testing.T) {
	attentionKB := gguf.TensorInfo{Name: "blk.1.attn_k_b.weight"}
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "kimi-linear", BlockCount: 2}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 2, KVLoRARank: 3, RopeDimensionCount: 2}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize: 4, KDAHeadDim: 2},
	},
		weights: model.Weights{Layers: []model.LayerWeights{{Recurrent: true}, {AttentionKB: &attentionKB}}}},
	}
	runner = attachFixtureProgram(runner)
	conv, _ := reference.NewValue(tensor.MustShape(2, 12), make([]float32, 24))
	state, _ := reference.NewValue(tensor.MustShape(2, 2, 2, 1), make([]float32, 8))
	key, _ := reference.NewValue(tensor.MustShape(5, 1, 2), make([]float32, 10))
	value, _ := reference.NewValue(tensor.MustShape(3, 1, 2), make([]float32, 6))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: state}, {Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestRWKV6Qwen2CacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "rwkv6qwen2", BlockCount: 1, EmbeddingLength: 8}, AttentionSpec: model.AttentionSpec{HeadCount: 2}, RecurrentSpec: model.RecurrentSpec{WKVHeadSize: 4}}}}
	runner = attachFixtureProgram(runner)
	shift, _ := reference.NewValue(tensor.MustShape(8), make([]float32, 8))
	state, _ := reference.NewValue(tensor.MustShape(4, 4, 2, 1), make([]float32, 32))
	cache := &KVCache{Layers: []LayerCache{{Key: shift, Value: state}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestRWKV6CacheValidation(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "rwkv6", BlockCount: 1, EmbeddingLength: 8}, AttentionSpec: model.AttentionSpec{HeadCount: 2}, RecurrentSpec: model.RecurrentSpec{WKVHeadSize: 4}}}}
	runner = attachFixtureProgram(runner)
	shift, _ := reference.NewValue(tensor.MustShape(8, 2), make([]float32, 16))
	state, _ := reference.NewValue(tensor.MustShape(4, 4, 2, 1), make([]float32, 32))
	cache := &KVCache{Layers: []LayerCache{{Key: shift, Value: state}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestRWKV7CacheValidation(t *testing.T) {
	for _, test := range []struct {
		architecture string
		shiftCount   uint32
	}{{"rwkv7", 2}, {"arwkv7", 1}} {
		t.Run(test.architecture, func(t *testing.T) {
			runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: test.architecture, BlockCount: 1, EmbeddingLength: 8}, AttentionSpec: model.AttentionSpec{HeadCount: 2}, RecurrentSpec: model.RecurrentSpec{WKVHeadSize: 4, TokenShiftCount: test.shiftCount}}}}
			runner = attachFixtureProgram(runner)
			shift, _ := reference.NewValue(tensor.MustShape(8, uint64(test.shiftCount)), make([]float32, 8*test.shiftCount))
			state, _ := reference.NewValue(tensor.MustShape(4, 4, 2, 1), make([]float32, 32))
			auxiliary, _ := reference.NewValue(tensor.MustShape(8, 2), make([]float32, 16))
			cache := &KVCache{Layers: []LayerCache{{Key: shift, Value: state, Auxiliary: &auxiliary}}, Tokens: 2, Position: 2}
			if err := runner.validateCache(cache); err != nil {
				t.Fatal(err)
			}
			payload, err := runner.SaveCache(cache)
			if err != nil {
				t.Fatal(err)
			}
			restored, err := runner.LoadCache(payload)
			if err != nil {
				t.Fatal(err)
			}
			if restored.Layers[0].Auxiliary != nil {
				t.Fatal("WKV7 transient value residual was serialized")
			}
		})
	}
}

func testDeepSeek2FamilyAbsorbedCacheValidation(t *testing.T, architecture string) {
	attentionKB := gguf.TensorInfo{Name: "blk.0.attn_k_b.weight"}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture, BlockCount: 1}, AttentionSpec: model.AttentionSpec{KeyLength: 6, ValueLength: 4,
		HeadCount: 2, HeadCountKV: 2, KVLoRARank: 3, RopeDimensionCount: 2}}
	weights := model.Weights{Layers: []model.LayerWeights{{AttentionKB: &attentionKB}}}
	runner := fixtureRunner(spec, weights)
	key, _ := reference.NewValue(tensor.MustShape(5, 1, 2), make([]float32, 10))
	value, _ := reference.NewValue(tensor.MustShape(3, 1, 2), make([]float32, 6))
	cache := &KVCache{Layers: []LayerCache{{Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestDeepSeek2LegacyCacheValidation(t *testing.T) {
	attentionKVB := gguf.TensorInfo{Name: "blk.0.attn_kv_b.weight"}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek2", BlockCount: 1}, AttentionSpec: model.AttentionSpec{KeyLength: 6, ValueLength: 4,
		HeadCount: 2, HeadCountKV: 1, KVLoRARank: 3, RopeDimensionCount: 2}}
	weights := model.Weights{Layers: []model.LayerWeights{{AttentionKVB: &attentionKVB}}}
	runner := fixtureRunner(spec, weights)
	key, _ := reference.NewValue(tensor.MustShape(6, 2, 2), make([]float32, 24))
	value, _ := reference.NewValue(tensor.MustShape(4, 2, 2), make([]float32, 16))
	cache := &KVCache{Layers: []LayerCache{{Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestDeciSentinelCacheValidationAndRangeRemoval(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "deci", BlockCount: 1}, AttentionSpec: model.AttentionSpec{KeyLength: 4, ValueLength: 4,
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2},
		LayerKVHeadCounts: []uint32{0}},
	}}}
	runner = attachFixtureProgram(runner)
	sentinel, err := reference.NewValue(
		tensor.MustShape(1, 1, 3), []float32{1, 2, 3},
	)
	if err != nil {
		t.Fatal(err)
	}
	cache := &KVCache{
		Layers: []LayerCache{{Key: sentinel, Value: sentinel}},
		Tokens: 3, Position: 3,
	}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
	trimmed, err := runner.RemoveCacheRange(cache, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	wantShape := tensor.MustShape(1, 1, 2)
	if !trimmed.Layers[0].Key.Shape.Equal(wantShape) ||
		!trimmed.Layers[0].Value.Shape.Equal(wantShape) ||
		!reflect.DeepEqual(trimmed.Layers[0].Key.Data, []float32{1, 3}) {
		t.Fatalf("trimmed Deci sentinel cache = %+v", trimmed)
	}
}

func TestKVCacheStateRejectsTruncationAndTrailingData(t *testing.T) {
	runner := cacheTestRunner()
	data, err := runner.SaveCache(cacheTestValue(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, length := range []int{0, 15, 16, 19, 20, 63, len(data) - 1} {
		if _, loadErr := runner.LoadCache(data[:length]); loadErr == nil {
			t.Fatalf("truncated length %d was accepted", length)
		}
	}
	withTrailing := append(append([]byte(nil), data...), 0)
	if _, err := runner.LoadCache(withTrailing); err == nil {
		t.Fatal("trailing cache data was accepted")
	}
}

func TestKVCacheStateRejectsLayerCountBeyondPayload(t *testing.T) {
	data := make([]byte, cacheStateHeaderSize)
	copy(data, cacheStateMagic)
	binary.LittleEndian.PutUint32(data[8:], 1)
	binary.LittleEndian.PutUint32(data[12:], 1)
	binary.LittleEndian.PutUint32(data[16:], cacheTestRunner().program.Model.CacheLayerCount())
	if _, err := cacheTestRunner().LoadCache(data); err == nil {
		t.Fatal("layer count beyond payload was accepted")
	}
}

func TestShiftCacheDropsAttentionPrefixAndPreservesPosition(t *testing.T) {
	runner := cacheTestRunner()
	cache := cacheTestValue(t)
	cache.Position = 7
	shifted, err := runner.RemoveCacheRange(cache, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if shifted.Tokens != 1 || shifted.Position != 7 {
		t.Fatalf(
			"shifted count/position = %d/%d, want 1/7",
			shifted.Tokens,
			shifted.Position,
		)
	}
	if got := shifted.Layers[0].Key.Data; !reflect.DeepEqual(got, []float32{3, 4}) {
		t.Fatalf("shifted key = %v, want [3 4]", got)
	}
	if got := shifted.Layers[0].Value.Data; !reflect.DeepEqual(got, []float32{8, 9, 10}) {
		t.Fatalf("shifted value = %v, want [8 9 10]", got)
	}
	shifted.Layers[0].Key.Data[0] = 99
	if cache.Layers[0].Key.Data[2] == 99 {
		t.Fatal("shifted key aliases the input cache")
	}
}

func TestRemoveCacheRangePreservesPrefixAndSuffix(t *testing.T) {
	runner := cacheTestRunner()
	key, err := reference.NewValue(
		tensor.MustShape(2, 1, 4),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
	)
	if err != nil {
		t.Fatal(err)
	}
	value, err := reference.NewValue(
		tensor.MustShape(3, 1, 4),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
	)
	if err != nil {
		t.Fatal(err)
	}
	cache := &KVCache{
		Layers:   []LayerCache{{Key: key, Value: value}},
		Tokens:   4,
		Position: 9,
	}
	edited, err := runner.RemoveCacheRange(cache, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if edited.Tokens != 2 || edited.Position != 9 {
		t.Fatalf("edited count/position = %d/%d, want 2/9",
			edited.Tokens, edited.Position)
	}
	if got := edited.Layers[0].Key.Data; !reflect.DeepEqual(
		got,
		[]float32{1, 2, 7, 8},
	) {
		t.Fatalf("edited key = %v, want [1 2 7 8]", got)
	}
	if got := edited.Layers[0].Value.Data; !reflect.DeepEqual(
		got,
		[]float32{1, 2, 3, 10, 11, 12},
	) {
		t.Fatalf("edited value = %v, want [1 2 3 10 11 12]", got)
	}
	edited.Layers[0].Key.Data[0] = 99
	if cache.Layers[0].Key.Data[0] == 99 {
		t.Fatal("edited key aliases the input cache")
	}
}

func TestShiftHybridCacheRetainsIndependentRecurrentState(t *testing.T) {
	runner := hybridCacheTestRunner()
	cache := hybridCacheTestValue(t)
	cache.Position = 9
	shifted, err := runner.RemoveCacheRange(cache, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(shifted.Layers[0], cache.Layers[0]) {
		t.Fatal("hybrid recurrent state changed during attention shift")
	}
	if got := shifted.Layers[1].Key.Data; !reflect.DeepEqual(got, []float32{5, 6, 7, 8}) {
		t.Fatalf("shifted hybrid attention key = %v, want [5 6 7 8]", got)
	}
	shifted.Layers[0].Value.Data[0] = 99
	if cache.Layers[0].Value.Data[0] == 99 {
		t.Fatal("shifted recurrent state aliases the input cache")
	}
}

func TestCacheForAppendShiftsOnlyWhenEnabled(t *testing.T) {
	runner := cacheTestRunner()
	runner.spec.ContextLength = 2
	cache := cacheTestValue(t)
	cache.Position = 5
	unchanged, err := runner.cacheForAppendKeeping(cache, 1, false, 0, -1)
	if err != nil {
		t.Fatal(err)
	}
	if unchanged != cache {
		t.Fatal("disabled context shift replaced the cache")
	}
	shifted, err := runner.cacheForAppendKeeping(cache, 1, true, 0, -1)
	if err != nil {
		t.Fatal(err)
	}
	if shifted.Tokens != 1 || shifted.Position != 5 {
		t.Fatalf(
			"auto-shifted count/position = %d/%d, want 1/5",
			shifted.Tokens,
			shifted.Position,
		)
	}
}

func TestCacheForAppendPreservesRequestedPrefix(t *testing.T) {
	runner := cacheTestRunner()
	runner.spec.ContextLength = 4
	key, err := reference.NewValue(
		tensor.MustShape(2, 1, 4),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
	)
	if err != nil {
		t.Fatal(err)
	}
	value, err := reference.NewValue(
		tensor.MustShape(3, 1, 4),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
	)
	if err != nil {
		t.Fatal(err)
	}
	cache := &KVCache{
		Layers:   []LayerCache{{Key: key, Value: value}},
		Tokens:   4,
		Position: 7,
	}
	edited, err := runner.cacheForAppendKeeping(cache, 1, true, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := edited.Layers[0].Key.Data; !reflect.DeepEqual(
		got,
		[]float32{1, 2, 3, 4, 7, 8},
	) {
		t.Fatalf("edited key = %v, want prefix plus suffix", got)
	}
	if edited.Tokens != 3 || edited.Position != 7 {
		t.Fatalf("edited count/position = %d/%d, want 3/7",
			edited.Tokens, edited.Position)
	}
}

func TestContextDiscardCount(t *testing.T) {
	for _, test := range []struct {
		name      string
		tokens    uint32
		incoming  int
		context   uint32
		keep      uint32
		requested int
		want      uint32
	}{
		{"minimum", 8, 1, 8, 0, -1, 1},
		{"default-half", 8, 1, 8, 0, 0, 4},
		{"explicit", 8, 1, 8, 0, 3, 3},
		{"explicit-below-overflow", 8, 3, 8, 0, 1, 3},
		{"keep-prefix", 8, 1, 8, 3, 0, 2},
		{"clamp-leave-suffix", 8, 1, 8, 3, 99, 4},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := contextDiscardCount(
				test.tokens,
				test.incoming,
				test.context,
				test.keep,
				test.requested,
			)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("discard = %d, want %d", got, test.want)
			}
		})
	}
}

func TestEffectiveKeepTokens(t *testing.T) {
	for _, test := range []struct {
		requested int
		prompt    int
		context   uint32
		want      uint32
	}{
		{0, 8, 16, 0},
		{3, 8, 16, 3},
		{20, 8, 16, 8},
		{-1, 8, 16, 8},
		{-1, 20, 16, 12},
	} {
		if got := effectiveKeepTokens(
			test.requested,
			test.prompt,
			test.context,
		); got != test.want {
			t.Fatalf(
				"effectiveKeepTokens(%d,%d,%d) = %d, want %d",
				test.requested,
				test.prompt,
				test.context,
				got,
				test.want,
			)
		}
	}
}

func TestKVCacheStateRejectsWrongModelShape(t *testing.T) {
	source := cacheTestRunner()
	data, err := source.SaveCache(cacheTestValue(t))
	if err != nil {
		t.Fatal(err)
	}
	other := cacheTestRunner()
	other.spec.KeyLength = 4
	other = attachFixtureProgram(other)
	if _, err := other.LoadCache(data); err == nil {
		t.Fatal("cache for a different key width was accepted")
	}
}

func TestHybridCacheStateRoundTrip(t *testing.T) {
	runner := hybridCacheTestRunner()
	cache := hybridCacheTestValue(t)
	data, err := runner.SaveCache(cache)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := runner.LoadCache(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded, cache) {
		t.Fatalf("loaded hybrid cache = %+v, want %+v", loaded, cache)
	}
}

func hybridCacheTestRunner() *Runner {
	fixture := modeltest.Qwen35DenseRecurrentPair()
	spec := fixture.Spec
	return &Runner{preparedModel: preparedModel{spec: spec,
		program: fixtureProgram(spec, fixture.Weights),
		weights: fixture.Weights},
	}
}

func hybridCacheTestValue(t *testing.T) *KVCache {
	t.Helper()
	conv, err := reference.NewValue(
		tensor.MustShape(2, 8),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	)
	if err != nil {
		t.Fatal(err)
	}
	ssm, err := reference.NewValue(
		tensor.MustShape(2, 2, 2, 1),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
	)
	if err != nil {
		t.Fatal(err)
	}
	key, err := reference.NewValue(
		tensor.MustShape(4, 1, 2),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
	)
	if err != nil {
		t.Fatal(err)
	}
	value, err := reference.NewValue(
		tensor.MustShape(4, 1, 2),
		[]float32{8, 7, 6, 5, 4, 3, 2, 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	return &KVCache{
		Layers: []LayerCache{
			{Key: conv, Value: ssm},
			{Key: key, Value: value},
		},
		Tokens:   2,
		Position: 2,
	}
}

func cacheTestRunner() *Runner {
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1}, AttentionSpec: model.AttentionSpec{KeyLength: 2,
		ValueLength: 3,
		HeadCountKV: 1},
	}
	weights := model.Weights{Layers: []model.LayerWeights{{}}}
	return &Runner{preparedModel: preparedModel{spec: spec, weights: weights, program: fixtureProgram(spec, weights)}}
}

func cacheTestValue(t testing.TB) *KVCache {
	t.Helper()
	key, err := reference.NewValue(
		tensor.MustShape(2, 1, 2),
		[]float32{1, 2, 3, 4},
	)
	if err != nil {
		t.Fatal(err)
	}
	value, err := reference.NewValue(
		tensor.MustShape(3, 1, 2),
		[]float32{5, 6, 7, 8, 9, 10},
	)
	if err != nil {
		t.Fatal(err)
	}
	return &KVCache{
		Layers:   []LayerCache{{Key: key, Value: value}},
		Tokens:   2,
		Position: 2,
	}
}
