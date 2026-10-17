//go:build windows

package executor

import (
	"fmt"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// BenchmarkBlockIndexerScore measures complete resident scoring and selection.
// Setup is excluded and output synchronization is included.
func BenchmarkBlockIndexerScore(b *testing.B) {
	cudatest.Require(b)
	device, err := driver.Open()
	if err != nil {
		b.Fatal(err)
	}
	defer device.Close()
	if _, err := device.ReserveDevice(0); err != nil {
		b.Fatal(err)
	}
	// The target model declares width 128, four heads, four-token blocks and
	// a 2048-token budget. The cases cover a prefill chunk and 256K decode.
	const width, heads, block, budget = 128, 4, 4, 2048
	for _, geometry := range []struct{ total, queries uint64 }{{8192, 128}, {262144, 1}} {
		b.Run(fmt.Sprintf("tokens=%d/queries=%d", geometry.total, geometry.queries), func(b *testing.B) {
			builder := tensor.NewBuilder()
			blocks, start := geometry.total/block, uint32(geometry.total-geometry.queries)
			query := builder.Input("query", dtype.F32, tensor.MustShape(width, heads, geometry.queries))
			key := builder.Input("key", dtype.F32, tensor.MustShape(width, 1, blocks))
			var scores *tensor.Tensor
			expanded := builder.ReLU(builder.MulMat(builder.Reshape(key, width, blocks), builder.Reshape(query, width, heads*geometry.queries)))
			expanded = builder.Reshape(expanded, blocks*heads, geometry.queries)
			for head := range uint64(heads) {
				part := builder.Reshape(builder.GroupSlice(expanded, head*blocks, blocks, 1, blocks), blocks, geometry.queries)
				if scores == nil {
					scores = part
				} else {
					scores = builder.Add(scores, part)
				}
			}
			scores = builder.Scale(scores, hostmath.InvSqrt32(width))
			output := builder.BlockTopK(scores, budget, block, start, uint32(geometry.total))
			if err := builder.Err(); err != nil {
				b.Fatal(err)
			}
			compiled, err := Compile(output)
			if err != nil {
				b.Fatal(err)
			}
			cuda := newFixtureExecutor(b)
			feeds := make(map[*tensor.Tensor]reference.Value)
			for ordinal, node := range []*tensor.Tensor{query, key} {
				count, _ := node.Shape.Elements()
				feeds[node] = reference.Value{Shape: node.Shape, Data: moePinnedValues("block-indexer", uint64(ordinal), count, 1)}
			}
			inputs := residentDecodeInputs(b, cuda, compiled, feeds)
			if _, err := cuda.ExecuteCompiled(b.Context(), compiled, nil, inputs); err != nil {
				b.Fatal(err)
			}
			metrics, err := cuda.Metrics(b.Context())
			if err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for b.Loop() {
				if _, err := cuda.ExecuteCompiled(b.Context(), compiled, nil, inputs); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(metrics.ArenaRequiredBytes), "arena-B")
		})
	}
}
