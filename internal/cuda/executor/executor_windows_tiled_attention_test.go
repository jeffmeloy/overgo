//go:build windows

package executor

import (
	"cmp"
	"context"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestExecutorBF16AttentionMatchesRoundedReference: the fused
// Attention(BF16Round(q/k/v)) tensor-core kernel vs the reference backend
// executing the same rounded graph. Residual disagreement is the in-kernel
// BF16 probability rounding plus F32 accumulation order; the gate bounds it.
func TestExecutorBF16AttentionMatchesRoundedReference(t *testing.T) {
	cases := []struct {
		name                    string
		queryHeads, keyValHeads uint64
		queryTokens, kvTokens   uint64
		scale                   float32
		tolerance               float64
		masked                  bool
	}{
		{"mha_dense", 4, 4, 128, 96, 0.0883883, 5e-3, false},
		{"mha_tail", 4, 4, 100, 77, 0.0883883, 5e-3, false},
		{"gqa_tail", 4, 2, 70, 33, 0.0883883, 5e-3, false},
		{"cross_long", 12, 12, 64, 512, 0.0883883, 5e-3, false},
		{"gqa_key_mask", 4, 2, 70, 96, 0.0883883, 5e-3, true},
	}
	cuda := newFixtureExecutor(t)
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			const width = uint64(128)
			builder := tensor.NewBuilder()
			query := builder.Input("q", dtype.F32, tensor.MustShape(width, testCase.queryHeads, testCase.queryTokens))
			key := builder.Input("k", dtype.F32, tensor.MustShape(width, testCase.keyValHeads, testCase.kvTokens))
			value := builder.Input("v", dtype.F32, tensor.MustShape(width, testCase.keyValHeads, testCase.kvTokens))
			var keyBias *tensor.Tensor
			if testCase.masked {
				keyBias = builder.Input("key_bias", dtype.F32, tensor.MustShape(testCase.kvTokens))
			}
			roundedQuery, roundedKey, roundedValue := builder.BF16Round(query), builder.BF16Round(key), builder.BF16Round(value)
			var attention *tensor.Tensor
			if keyBias != nil {
				attention = builder.AttentionWithOptions(roundedQuery, roundedKey, roundedValue, tensor.AttentionOptions{KeyBias: keyBias, Scale: testCase.scale, Causal: false})
			} else {
				attention = builder.AttentionWithOptions(roundedQuery, roundedKey, roundedValue, tensor.AttentionOptions{Scale: testCase.scale, Causal: false})
			}
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			compiled, err := Compile(attention)
			if err != nil {
				t.Fatal(err)
			}
			if fusion := compiled.nodes[compiled.orderIndexes[attention]].fusion; fusion == nil || fusion.kind != compiledFusionBF16Attention {
				t.Fatalf("bf16 attention fusion did not apply: %+v", fusion)
			}
			keyElements, err := key.Shape.Elements()
			if err != nil {
				t.Fatal(err)
			}
			if want := keyElements * 4; compiled.attentionScoreBytes != want {
				t.Fatalf("bf16 attention staging=%d want=%d", compiled.attentionScoreBytes, want)
			}
			feeds := map[*tensor.Tensor]reference.Value{
				query: patternedValue(query.Shape, 1, 0.11, 0),
				key:   patternedValue(key.Shape, 2, 0.09, 0),
				value: patternedValue(value.Shape, 3, 0.07, 0),
			}
			if keyBias != nil {
				bias := make([]float32, testCase.kvTokens)
				for index := testCase.kvTokens / 2; index < testCase.kvTokens; index++ {
					bias[index] = -3.402823466e38
				}
				feeds[keyBias] = reference.Value{Shape: keyBias.Shape, Data: bias}
			}
			outputs := []*tensor.Tensor{attention}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), compiled, feeds, nil)
			if err != nil {
				t.Fatal(err)
			}
			replay, err := cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), compiled, feeds, nil)
			if err != nil {
				t.Fatal(err)
			}
			replayWorst, replayAt := maxAbsDifference(t, replay[attention].Data, got[attention].Data)
			if replayWorst != 0 {
				t.Fatalf("replay max abs diff %.6g at=%d", replayWorst, replayAt)
			}
			worst, at := maxAbsDifference(t, got[attention].Data, want[attention].Data)
			t.Logf("%s max_abs_diff=%.6g at=%d", testCase.name, worst, at)
			if worst > testCase.tolerance {
				t.Fatalf("max abs diff %.6g > %.6g", worst, testCase.tolerance)
			}
		})
	}
}

// TestExecutorTiledAttentionMatchesReference: the tiled exact attention path
// (non-causal, featureless, >=32 query rows) vs the reference backend across
// widths, GQA groupings, and non-multiple-of-32 KV extents.
func TestExecutorTiledAttentionMatchesReference(t *testing.T) {
	cases := []struct {
		name                    string
		width                   uint64
		queryHeads, keyValHeads uint64
		queryTokens, kvTokens   uint64
		scale                   float32
		tolerance               float64
		masked                  bool
	}{
		{"w128_mha_dense", 128, 4, 4, 96, 96, 0.0883883, 5e-5, false},
		{"w128_gqa_tail", 128, 4, 2, 100, 77, 0.0883883, 5e-5, false},
		{"w128_gqa_mask", 128, 4, 2, 100, 77, 0.0883883, 5e-5, true},
		{"w64_mha_tail", 64, 6, 6, 33, 41, 0.125, 5e-5, false},
		{"w128_cross_long", 128, 12, 12, 64, 512, 0.0883883, 5e-5, false},
	}
	cuda := newFixtureExecutor(t)
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			query := builder.Input("q", dtype.F32, tensor.MustShape(testCase.width, testCase.queryHeads, testCase.queryTokens))
			key := builder.Input("k", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, testCase.kvTokens))
			value := builder.Input("v", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, testCase.kvTokens))
			var attention *tensor.Tensor
			var keyBias *tensor.Tensor
			if testCase.masked {
				keyBias = builder.Input("key_bias", dtype.F32, tensor.MustShape(testCase.kvTokens))
				attention = builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{KeyBias: keyBias, Scale: testCase.scale, Causal: false})
			} else {
				attention = builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{Scale: testCase.scale, Causal: false})
			}
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			feeds := map[*tensor.Tensor]reference.Value{
				query: patternedValue(query.Shape, 1, 0.11, 0),
				key:   patternedValue(key.Shape, 2, 0.09, 0),
				value: patternedValue(value.Shape, 3, 0.07, 0),
			}
			if keyBias != nil {
				bias := make([]float32, testCase.kvTokens)
				for index := testCase.kvTokens / 2; index < testCase.kvTokens; index++ {
					bias[index] = -3.402823466e38
				}
				feeds[keyBias] = reference.Value{Shape: keyBias.Shape, Data: bias}
			}
			outputs := []*tensor.Tensor{attention}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cuda.Execute(context.WithoutCancel(t.Context()), outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			worst, at := maxAbsDifference(t, got[attention].Data, want[attention].Data)
			t.Logf("%s max_abs_diff=%.6g at=%d", testCase.name, worst, at)
			if worst > testCase.tolerance {
				t.Fatalf("max abs diff %.6g > %.6g", worst, testCase.tolerance)
			}
		})
	}
}

func TestExecutorBLASAttentionKeyBiasMatchesReference(t *testing.T) {
	const (
		width       = uint64(128)
		heads       = uint64(12)
		queryTokens = uint64(64)
		keyTokens   = uint64(512)
		scale       = float32(0.0883883)
	)
	builder := tensor.NewBuilder()
	query := builder.Input("q", dtype.F32, tensor.MustShape(width, heads, queryTokens))
	key := builder.Input("k", dtype.F32, tensor.MustShape(width, heads, keyTokens))
	value := builder.Input("v", dtype.F32, tensor.MustShape(width, heads, keyTokens))
	keyBias := builder.Input("key_bias", dtype.F32, tensor.MustShape(keyTokens))
	attention := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{KeyBias: keyBias, Scale: scale, Causal: false})
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	compiled, err := Compile(attention)
	if err != nil {
		t.Fatal(err)
	}
	if !compiled.needBlas {
		t.Fatal("masked attention did not request BLAS")
	}
	bias := make([]float32, keyTokens)
	for index := uint64(29); index < keyTokens; index++ {
		bias[index] = -3.402823466e38
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query:   patternedValue(query.Shape, 1, 0.11, 0),
		key:     patternedValue(key.Shape, 2, 0.09, 0),
		value:   patternedValue(value.Shape, 3, 0.07, 0),
		keyBias: {Shape: keyBias.Shape, Data: bias},
	}
	want, err := reference.Execute([]*tensor.Tensor{attention}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), compiled, feeds, nil)
	if err != nil {
		t.Fatal(err)
	}
	worst, at := maxAbsDifference(t, got[attention].Data, want[attention].Data)
	t.Logf("max_abs_diff=%.6g at=%d", worst, at)
	if worst > 5e-5 {
		t.Fatalf("max abs diff %.6g > 5e-5", worst)
	}
}

// TestExecutorPrefillAttentionMatchesReference: causal prompt prefill on
// the strided-batched SGEMM path vs the reference backend: plain causal,
// grouped-query heads, a past context (query start), a causal sliding
// window, a softcap, and a logical KV count below the cache capacity
// page, every case at or above the SGEMM query floor.
func TestExecutorPrefillAttentionMatchesReference(t *testing.T) {
	cases := []struct {
		name                    string
		width                   uint64
		queryHeads, keyValHeads uint64
		queryTokens, pastTokens uint64
		capacity                uint64
		window                  uint32
		softcap                 float32
	}{
		{"causal_mha", 128, 4, 4, 96, 0, 0, 0, 0},
		{"causal_gqa", 128, 4, 2, 100, 0, 0, 0, 0},
		{"causal_past", 64, 6, 3, 64, 40, 0, 0, 0},
		{"causal_window", 128, 4, 2, 100, 0, 0, 48, 0},
		{"causal_window_past", 64, 4, 1, 40, 70, 0, 32, 0},
		{"causal_softcap", 128, 4, 4, 96, 0, 0, 0, 50},
		{"causal_capacity", 128, 4, 2, 64, 40, 160, 0, 0},
		{"causal_window_capacity", 64, 8, 2, 48, 100, 256, 64, 30},
	}
	cuda := newFixtureExecutor(t)
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			logical := testCase.pastTokens + testCase.queryTokens
			capacity := cmp.Or(testCase.capacity, logical)
			query := builder.Input("q", dtype.F32, tensor.MustShape(testCase.width, testCase.queryHeads, testCase.queryTokens))
			feeds := map[*tensor.Tensor]reference.Value{query: patternedValue(query.Shape, 1, 0.11, 0)}
			var key, value *tensor.Tensor
			if testCase.capacity == 0 {
				key = builder.Input("k", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, logical))
				value = builder.Input("v", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, logical))
				feeds[key] = patternedValue(key.Shape, 2, 0.09, 0)
				feeds[value] = patternedValue(value.Shape, 3, 0.07, 0)
			} else {
				builder.SetCacheAppendPlan(tensor.CacheAppendPlan{
					CapacityTokens: uint32(capacity), ActiveTokens: uint32(testCase.pastTokens),
				})
				pastKey := builder.Input("past_k", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, capacity))
				newKey := builder.Input("new_k", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, testCase.queryTokens))
				pastValue := builder.Input("past_v", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, capacity))
				newValue := builder.Input("new_v", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, testCase.queryTokens))
				key = builder.AppendCache(pastKey, newKey, 2)
				value = builder.AppendCache(pastValue, newValue, 2)
				feeds[pastKey] = patternedValue(pastKey.Shape, 2, 0.09, 0)
				feeds[newKey] = patternedValue(newKey.Shape, 4, 0.09, 0)
				feeds[pastValue] = patternedValue(pastValue.Shape, 3, 0.07, 0)
				feeds[newValue] = patternedValue(newValue.Shape, 5, 0.07, 0)
			}
			attention := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
				Scale: 0.0883883, Causal: true, QueryStart: uint32(testCase.pastTokens),
				Window: testCase.window, Softcap: testCase.softcap,
			})
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			compiled, err := Compile(attention)
			if err != nil {
				t.Fatal(err)
			}
			if !compiled.needBlas {
				t.Fatal("causal prefill attention did not request BLAS")
			}
			want, err := reference.Execute([]*tensor.Tensor{attention}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), compiled, feeds, nil)
			if err != nil {
				t.Fatal(err)
			}
			worst, at := maxAbsDifference(t, got[attention].Data, want[attention].Data)
			t.Logf("%s max_abs_diff=%.6g at=%d", testCase.name, worst, at)
			if worst > 5e-5 {
				t.Fatalf("max abs diff %.6g > 5e-5", worst)
			}
		})
	}
}
