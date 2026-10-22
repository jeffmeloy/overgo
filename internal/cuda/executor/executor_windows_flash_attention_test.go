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

// flashAttentionTolerance bounds the tensor-core kernel's distance from the f32 reference: its queries, keys, values
// and probabilities round to FP16 (11 significant bits) before their f32-accumulated products, which moves these
// outputs (values up to 0.63) by up to 1.74e-4.
const flashAttentionTolerance = 5e-4

// TestExecutorFlashAttentionMatchesReference holds a prompt's dense attention on tensor cores to the reference across
// the head widths the kernel takes: plain and grouped heads (groups of 7 and 8, the 0.5B and 1B models' shapes), a past
// context, a causal window, a softcap, a logical key count below the cache's capacity, a key bias and two sequences.
// The graph reserves no score staging, and a replay reproduces every bit. A non-causal attention, an encoder's, keeps
// the f32 paths and the staging they read.
func TestExecutorFlashAttentionMatchesReference(t *testing.T) {
	cases := []struct {
		name                    string
		width                   uint64
		queryHeads, keyValHeads uint64
		queryTokens, pastTokens uint64
		capacity, sequences     uint64
		window                  uint32
		softcap                 float32
		causal, keyBias         bool
	}{
		{name: "causal_mha_64", width: 64, queryHeads: 4, keyValHeads: 4, queryTokens: 96, causal: true},
		{name: "causal_group7_64", width: 64, queryHeads: 14, keyValHeads: 2, queryTokens: 100, causal: true},
		{name: "causal_group8_128", width: 128, queryHeads: 16, keyValHeads: 2, queryTokens: 70, causal: true},
		{name: "causal_gqa_256", width: 256, queryHeads: 8, keyValHeads: 2, queryTokens: 100, causal: true},
		{name: "causal_past_128", width: 128, queryHeads: 6, keyValHeads: 3, queryTokens: 64, pastTokens: 40, causal: true},
		{name: "causal_window_256", width: 256, queryHeads: 4, keyValHeads: 2, queryTokens: 100, window: 48, causal: true},
		{name: "causal_window_past_64", width: 64, queryHeads: 4, keyValHeads: 1, queryTokens: 40, pastTokens: 70, window: 32, causal: true},
		{name: "causal_softcap_128", width: 128, queryHeads: 4, keyValHeads: 4, queryTokens: 96, softcap: 50, causal: true},
		{name: "causal_capacity_256", width: 256, queryHeads: 4, keyValHeads: 2, queryTokens: 64, pastTokens: 40, capacity: 160, causal: true},
		{name: "causal_window_capacity_64", width: 64, queryHeads: 8, keyValHeads: 2, queryTokens: 48, pastTokens: 100, capacity: 256, window: 64, softcap: 30, causal: true},
		{name: "key_bias_128", width: 128, queryHeads: 12, keyValHeads: 12, queryTokens: 64, pastTokens: 448, keyBias: true, causal: true},
		{name: "sequences_128", width: 128, queryHeads: 4, keyValHeads: 2, queryTokens: 40, pastTokens: 30, sequences: 2, causal: true},
	}
	cuda := newFixtureExecutor(t)
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			logical := testCase.pastTokens + testCase.queryTokens
			capacity := cmp.Or(testCase.capacity, logical)
			shape := func(heads, tokens uint64) tensor.Shape {
				if testCase.sequences != 0 {
					return tensor.MustShape(testCase.width, heads, tokens, testCase.sequences)
				}
				return tensor.MustShape(testCase.width, heads, tokens)
			}
			query := builder.Input("q", dtype.F32, shape(testCase.queryHeads, testCase.queryTokens))
			feeds := map[*tensor.Tensor]reference.Value{query: patternedValue(query.Shape, 1, 0.11, 0)}
			var key, value *tensor.Tensor
			if testCase.capacity == 0 {
				key = builder.Input("k", dtype.F32, shape(testCase.keyValHeads, logical))
				value = builder.Input("v", dtype.F32, shape(testCase.keyValHeads, logical))
				feeds[key] = patternedValue(key.Shape, 2, 0.09, 0)
				feeds[value] = patternedValue(value.Shape, 3, 0.07, 0)
			} else {
				builder.SetCacheAppendPlan(tensor.CacheAppendPlan{
					CapacityTokens: uint32(capacity), ActiveTokens: uint32(testCase.pastTokens),
				})
				pastKey := builder.Input("past_k", dtype.F32, shape(testCase.keyValHeads, capacity))
				newKey := builder.Input("new_k", dtype.F32, shape(testCase.keyValHeads, testCase.queryTokens))
				pastValue := builder.Input("past_v", dtype.F32, shape(testCase.keyValHeads, capacity))
				newValue := builder.Input("new_v", dtype.F32, shape(testCase.keyValHeads, testCase.queryTokens))
				key = builder.AppendCache(pastKey, newKey, 2)
				value = builder.AppendCache(pastValue, newValue, 2)
				feeds[pastKey] = patternedValue(pastKey.Shape, 2, 0.09, 0)
				feeds[newKey] = patternedValue(newKey.Shape, 4, 0.09, 0)
				feeds[pastValue] = patternedValue(pastValue.Shape, 3, 0.07, 0)
				feeds[newValue] = patternedValue(newValue.Shape, 5, 0.07, 0)
			}
			options := tensor.AttentionOptions{Scale: 0.0883883, Causal: testCase.causal, Window: testCase.window, Softcap: testCase.softcap}
			if testCase.causal {
				options.QueryStart = uint32(testCase.pastTokens)
			}
			if testCase.keyBias {
				options.KeyBias = builder.Input("key_bias", dtype.F32, tensor.MustShape(logical))
				bias := make([]float32, logical)
				for index := logical / 3; index < logical; index++ {
					bias[index] = -3.402823466e38
				}
				feeds[options.KeyBias] = reference.Value{Shape: options.KeyBias.Shape, Data: bias}
			}
			attention := builder.AttentionWithOptions(query, key, value, options)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			compiled, err := Compile(attention)
			if err != nil {
				t.Fatal(err)
			}
			if compiled.attentionScoreBytes != 0 {
				t.Fatalf("prompt attention on tensor cores reserved %d bytes of score staging", compiled.attentionScoreBytes)
			}
			want, err := reference.Execute([]*tensor.Tensor{attention}, feeds)
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
			if worst, at := maxAbsDifference(t, replay[attention].Data, got[attention].Data); worst != 0 {
				t.Fatalf("replay max abs diff %.6g at=%d", worst, at)
			}
			worst, at := maxAbsDifference(t, got[attention].Data, want[attention].Data)
			t.Logf("%s max_abs_diff=%.6g at=%d", testCase.name, worst, at)
			if worst > flashAttentionTolerance {
				t.Fatalf("max abs diff %.6g > %.6g", worst, flashAttentionTolerance)
			}
		})
	}
	builder := tensor.NewBuilder()
	query := builder.Input("q", dtype.F32, tensor.MustShape(128, 4, 96))
	key := builder.Input("k", dtype.F32, tensor.MustShape(128, 4, 96))
	value := builder.Input("v", dtype.F32, tensor.MustShape(128, 4, 96))
	encoder, err := Compile(builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{Scale: 0.0883883}))
	if err != nil || encoder.attentionScoreBytes == 0 {
		t.Fatalf("an encoder's attention reserved no score staging (%v), so the tensor-core kernel read it", err)
	}
}
