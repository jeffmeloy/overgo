package inference

import (
	"encoding/binary"
	"reflect"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

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

func TestDeepSeek2AbsorbedCacheValidation(t *testing.T) {
	testDeepSeek2FamilyAbsorbedCacheValidation(t, "deepseek2")
}

func TestMistral4AbsorbedCacheValidation(t *testing.T) {
	testDeepSeek2FamilyAbsorbedCacheValidation(t, "mistral4")
}

func TestMambaCacheValidation(t *testing.T) {
	runner := &Runner{spec: model.Spec{Architecture: "mamba", BlockCount: 1,
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2}}
	conv, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestMamba2CacheValidation(t *testing.T) {
	runner := &Runner{spec: model.Spec{Architecture: "mamba2", BlockCount: 1,
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMGroupCount: 2}}
	conv, _ := reference.NewValue(tensor.MustShape(2, 16), make([]float32, 32))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestJambaHybridCacheValidation(t *testing.T) {
	runner := &Runner{
		spec: model.Spec{Architecture: "jamba", BlockCount: 2, SSMConvKernel: 3,
			SSMInnerSize: 8, SSMStateSize: 2, KeyLength: 2, ValueLength: 2,
			HeadCountKV: 1, RecurrentLayers: []bool{true, false}},
		weights: model.Weights{Layers: []model.LayerWeights{{Recurrent: true}, {}}},
	}
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
	runner := &Runner{
		spec: model.Spec{Architecture: "granitehybrid", BlockCount: 2, SSMConvKernel: 3,
			SSMInnerSize: 8, SSMStateSize: 2, SSMGroupCount: 2, KeyLength: 2, ValueLength: 2,
			HeadCountKV: 1, LayerKVHeadCounts: []uint32{0, 1}, RecurrentLayers: []bool{true, false}},
		weights: model.Weights{Layers: []model.LayerWeights{{Recurrent: true}, {}}},
	}
	conv, _ := reference.NewValue(tensor.MustShape(2, 16), make([]float32, 32))
	ssm, _ := reference.NewValue(tensor.MustShape(2, 8), make([]float32, 16))
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	cache := &KVCache{Layers: []LayerCache{{Key: conv, Value: ssm}, {Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func testDeepSeek2FamilyAbsorbedCacheValidation(t *testing.T, architecture string) {
	attentionKB := gguf.TensorInfo{Name: "blk.0.attn_k_b.weight"}
	runner := &Runner{
		spec: model.Spec{Architecture: architecture, BlockCount: 1, KeyLength: 6, ValueLength: 4,
			HeadCount: 2, HeadCountKV: 2, KVLoRARank: 3, RopeDimensionCount: 2},
		weights: model.Weights{Layers: []model.LayerWeights{{AttentionKB: &attentionKB}}},
	}
	key, _ := reference.NewValue(tensor.MustShape(5, 1, 2), make([]float32, 10))
	value, _ := reference.NewValue(tensor.MustShape(3, 1, 2), make([]float32, 6))
	cache := &KVCache{Layers: []LayerCache{{Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestDeepSeek2LegacyCacheValidation(t *testing.T) {
	attentionKVB := gguf.TensorInfo{Name: "blk.0.attn_kv_b.weight"}
	runner := &Runner{
		spec: model.Spec{Architecture: "deepseek2", BlockCount: 1, KeyLength: 6, ValueLength: 4,
			HeadCount: 2, HeadCountKV: 1, KVLoRARank: 3, RopeDimensionCount: 2},
		weights: model.Weights{Layers: []model.LayerWeights{{AttentionKVB: &attentionKVB}}},
	}
	key, _ := reference.NewValue(tensor.MustShape(6, 2, 2), make([]float32, 24))
	value, _ := reference.NewValue(tensor.MustShape(4, 2, 2), make([]float32, 16))
	cache := &KVCache{Layers: []LayerCache{{Key: key, Value: value}}, Tokens: 2, Position: 2}
	if err := runner.validateCache(cache); err != nil {
		t.Fatal(err)
	}
}

func TestDeciSentinelCacheValidationAndRangeRemoval(t *testing.T) {
	runner := &Runner{spec: model.Spec{
		Architecture: "deci", BlockCount: 1, KeyLength: 4, ValueLength: 4,
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2},
		LayerKVHeadCounts: []uint32{0},
	}}
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

func TestKVCacheStateLoadsLegacyVersion(t *testing.T) {
	runner := cacheTestRunner()
	current, err := runner.SaveCache(cacheTestValue(t))
	if err != nil {
		t.Fatal(err)
	}
	legacy := make([]byte, legacyCacheHeaderSize+len(current)-cacheStateHeaderSize)
	copy(legacy, legacyCacheStateMagic)
	copy(legacy[8:12], current[8:12])
	copy(legacy[12:16], current[16:20])
	copy(legacy[16:], current[20:])
	loaded, err := runner.LoadCache(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Position != loaded.Tokens {
		t.Fatalf(
			"legacy next position = %d, want token count %d",
			loaded.Position,
			loaded.Tokens,
		)
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

func TestKVCacheStateRejectsExcessiveLayers(t *testing.T) {
	data := make([]byte, cacheStateHeaderSize)
	copy(data, cacheStateMagic)
	binary.LittleEndian.PutUint32(data[8:], 1)
	binary.LittleEndian.PutUint32(data[12:], 1)
	binary.LittleEndian.PutUint32(data[16:], maxCacheStateLayers+1)
	if _, err := cacheTestRunner().LoadCache(data); err == nil {
		t.Fatal("excessive layer count was accepted")
	}
}

func TestShiftCacheDropsAttentionPrefixAndPreservesPosition(t *testing.T) {
	runner := cacheTestRunner()
	cache := cacheTestValue(t)
	cache.Position = 7
	shifted, err := runner.ShiftCache(cache, 1)
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
	shifted, err := runner.ShiftCache(cache, 1)
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
	unchanged, err := runner.cacheForAppend(cache, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if unchanged != cache {
		t.Fatal("disabled context shift replaced the cache")
	}
	shifted, err := runner.cacheForAppend(cache, 1, true)
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
	spec := model.Spec{
		Architecture:    "qwen35",
		BlockCount:      2,
		KeyLength:       4,
		ValueLength:     4,
		HeadCountKV:     1,
		SSMConvKernel:   3,
		SSMInnerSize:    4,
		SSMStateSize:    2,
		SSMTimeStepRank: 2,
		SSMGroupCount:   1,
	}
	return &Runner{
		spec: spec,
		weights: model.Weights{Layers: []model.LayerWeights{
			{Recurrent: true},
			{Recurrent: false},
		}},
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
	return &Runner{spec: model.Spec{
		BlockCount:  1,
		KeyLength:   2,
		ValueLength: 3,
		HeadCountKV: 1,
	}}
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
