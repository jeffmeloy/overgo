//go:build windows

package executor

import (
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
	}{
		{"mha_dense", 4, 4, 128, 96, 0.0883883, 5e-3},
		{"mha_tail", 4, 4, 100, 77, 0.0883883, 5e-3},
		{"gqa_tail", 4, 2, 70, 33, 0.0883883, 5e-3},
		{"cross_long", 12, 12, 64, 512, 0.0883883, 5e-3},
	}
	cuda := newFixtureExecutor(t)
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			const width = uint64(128)
			builder := tensor.NewBuilder()
			query := builder.Input("q", dtype.F32, tensor.MustShape(width, testCase.queryHeads, testCase.queryTokens))
			key := builder.Input("k", dtype.F32, tensor.MustShape(width, testCase.keyValHeads, testCase.kvTokens))
			value := builder.Input("v", dtype.F32, tensor.MustShape(width, testCase.keyValHeads, testCase.kvTokens))
			attention := builder.Attention(
				builder.BF16Round(query), builder.BF16Round(key), builder.BF16Round(value),
				testCase.scale, false,
			)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			compiled, err := Compile(attention)
			if err != nil {
				t.Fatal(err)
			}
			if len(compiled.bf16Attention) != 1 {
				t.Fatalf("bf16 attention fusion did not apply: %d entries", len(compiled.bf16Attention))
			}
			feeds := map[*tensor.Tensor]reference.Value{
				query: patternedValue(query.Shape, 1, 0.11, 0),
				key:   patternedValue(key.Shape, 2, 0.09, 0),
				value: patternedValue(value.Shape, 3, 0.07, 0),
			}
			outputs := []*tensor.Tensor{attention}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cuda.ExecuteCompiled(context.Background(), compiled, feeds)
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
	}{
		{"w128_mha_dense", 128, 4, 4, 96, 96, 0.0883883, 5e-5},
		{"w128_gqa_tail", 128, 4, 2, 100, 77, 0.0883883, 5e-5},
		{"w64_mha_tail", 64, 6, 6, 33, 41, 0.125, 5e-5},
		{"w128_cross_long", 128, 12, 12, 64, 512, 0.0883883, 5e-5},
	}
	cuda := newFixtureExecutor(t)
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			query := builder.Input("q", dtype.F32, tensor.MustShape(testCase.width, testCase.queryHeads, testCase.queryTokens))
			key := builder.Input("k", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, testCase.kvTokens))
			value := builder.Input("v", dtype.F32, tensor.MustShape(testCase.width, testCase.keyValHeads, testCase.kvTokens))
			attention := builder.Attention(query, key, value, testCase.scale, false)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			feeds := map[*tensor.Tensor]reference.Value{
				query: patternedValue(query.Shape, 1, 0.11, 0),
				key:   patternedValue(key.Shape, 2, 0.09, 0),
				value: patternedValue(value.Shape, 3, 0.07, 0),
			}
			outputs := []*tensor.Tensor{attention}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cuda.Execute(context.Background(), outputs, feeds)
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
