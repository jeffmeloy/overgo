//go:build windows

package executor

import (
	"fmt"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// BenchmarkDecodeAttention times one decode attention step in the
// Qwen2.5-0.5B head geometry (14 query heads over 2 key-value heads of
// width 64) at growing cache capacities, so a kernel change is measured
// in seconds against the same graph rather than inferred from a model
// run: the split-key rewrite of 2026-09-04 was tuned on it.
func BenchmarkDecodeAttention(b *testing.B) {
	cudatest.Require(b)
	const (
		width      = 64
		queryHeads = 14
		keyHeads   = 2
	)
	for _, capacity := range []uint32{1024, 2048, 4096, 8192, 16384} {
		active := capacity - 256
		b.Run(fmt.Sprintf("capacity=%d", capacity), func(b *testing.B) {
			builder := tensor.NewBuilder()
			builder.SetCacheAppendPlan(tensor.CacheAppendPlan{CapacityTokens: capacity, ActiveTokens: active})
			query := builder.Input("query", dtype.F32, tensor.MustShape(width, queryHeads, 1))
			pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(width, keyHeads, uint64(capacity)))
			newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(width, keyHeads, 1))
			pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(width, keyHeads, uint64(capacity)))
			newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(width, keyHeads, 1))
			key := builder.AppendCache(pastKey, newKey, 2)
			value := builder.AppendCache(pastValue, newValue, 2)
			output := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
				Scale: 0.125, Causal: true, QueryStart: active,
			})
			if err := builder.Err(); err != nil {
				b.Fatal(err)
			}
			program, err := tensor.CompileProgram(output)
			if err != nil {
				b.Fatal(err)
			}
			compiled, err := CompileProgram(program)
			if err != nil {
				b.Fatal(err)
			}
			feeds := map[*tensor.Tensor]reference.Value{
				query:     patternedValue(query.Shape, 1, 0.2, 0),
				pastKey:   patternedValue(pastKey.Shape, 2, 0.15, 0),
				newKey:    patternedValue(newKey.Shape, 3, 0.15, 0),
				pastValue: patternedValue(pastValue.Shape, 4, 0.2, 0),
				newValue:  patternedValue(newValue.Shape, 5, 0.2, 0),
			}
			cuda := newFixtureExecutor(b)
			ctx := b.Context()
			if _, err := cuda.ExecuteCompiled(ctx, compiled, feeds, nil); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for b.Loop() {
				if _, err := cuda.ExecuteCompiled(ctx, compiled, feeds, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
