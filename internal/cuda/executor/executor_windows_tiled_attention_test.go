//go:build windows

package executor

import (
	"context"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestExecutorBF16AttentionMatchesRoundedReference: the fused
// Attention(BF16Round(q/k/v)) tensor-core kernel vs the reference backend
// executing the same rounded graph. Residual disagreement is the in-kernel
// BF16 probability rounding plus F32 accumulation order; the gate bounds it.
func TestExecutorBF16AttentionMatchesRoundedReference(t *testing.T) {
	cases := []struct {
		name                    string
		width                   uint64
		queryHeads, keyValHeads uint64
		queryTokens, kvTokens   uint64
		scale                   float32
		tolerance               float64
		masked                  bool
	}{
		{"mha_dense", 128, 4, 4, 128, 96, 0.0883883, 5e-3, false},
		{"mha_tail", 128, 4, 4, 100, 77, 0.0883883, 5e-3, false},
		{"gqa_tail", 128, 4, 2, 70, 33, 0.0883883, 5e-3, false},
		{"cross_long", 128, 12, 12, 64, 512, 0.0883883, 5e-3, false},
		{"gqa_key_mask", 128, 4, 2, 70, 96, 0.0883883, 5e-3, true},
		{"narrow_one_copy", 8, 4, 2, 2, 3, 0.3535534, 5e-3, false},
		{"narrow_panel_tail", 64, 4, 2, 65, 129, 0.125, 5e-3, false},
		{"narrow_key_mask", 120, 4, 2, 67, 73, 0.0912871, 5e-3, true},
	}
	cuda := newFixtureExecutor(t)
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			width := testCase.width
			builder := tensor.NewBuilder()
			query := builder.Input("q", dtype.F32, tensor.MustShape(width, testCase.queryHeads, testCase.queryTokens))
			key := builder.Input("k", dtype.F32, tensor.MustShape(width, testCase.keyValHeads, testCase.kvTokens))
			value := builder.Input("v", dtype.F32, tensor.MustShape(width, testCase.keyValHeads, testCase.kvTokens))
			var keyBias *tensor.Tensor
			if testCase.masked {
				keyBias = builder.Input("key_bias", dtype.F32, tensor.MustShape(testCase.kvTokens))
			}
			round := builder.BF16Round
			if width < uint64(attentionBF16Width) {
				round = builder.BF16RoundNative
			}
			roundedQuery, roundedKey, roundedValue := round(query), round(key), round(value)
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
// widths the tensor-core kernel does not take, GQA groupings, and
// non-multiple-of-32 KV extents. A key bias over a long cross attention takes the BLAS path.
func TestExecutorTiledAttentionMatchesReference(t *testing.T) {
	cases := []struct {
		name                    string
		width                   uint64
		queryHeads, keyValHeads uint64
		queryTokens, kvTokens   uint64
		scale                   float32
		tolerance               float64
		// maskFrom is the first key a key bias masks, none where zero; needBlas holds the graph to the BLAS path.
		maskFrom uint64
		needBlas bool
	}{
		{"w96_mha_dense", 96, 4, 4, 96, 96, 0.0883883, 5e-5, 0, false},
		{"w96_gqa_tail", 96, 4, 2, 100, 77, 0.0883883, 5e-5, 0, false},
		{"w96_gqa_mask", 96, 4, 2, 100, 77, 0.0883883, 5e-5, 38, false},
		{"w80_mha_tail", 80, 6, 6, 33, 41, 0.125, 5e-5, 0, false},
		{"w96_cross_long", 96, 12, 12, 64, 512, 0.0883883, 5e-5, 0, false},
		{"w96_blas_key_bias", 96, 12, 12, 64, 512, 0.0883883, 5e-5, 29, true},
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
			if testCase.maskFrom != 0 {
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
				for index := testCase.maskFrom; index < testCase.kvTokens; index++ {
					bias[index] = -3.402823466e38
				}
				feeds[keyBias] = reference.Value{Shape: keyBias.Shape, Data: bias}
			}
			outputs := []*tensor.Tensor{attention}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			compiled := testcheck.Must(Compile(outputs...))(t)
			if testCase.needBlas && !compiled.needBlas {
				t.Fatal("masked attention did not request BLAS")
			}
			got := testcheck.Must(cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), compiled, feeds, nil))(t)
			worst, at := maxAbsDifference(t, got[attention].Data, want[attention].Data)
			t.Logf("%s max_abs_diff=%.6g at=%d", testCase.name, worst, at)
			if worst > testCase.tolerance {
				t.Fatalf("max abs diff %.6g > %.6g", worst, testCase.tolerance)
			}
		})
	}
}
