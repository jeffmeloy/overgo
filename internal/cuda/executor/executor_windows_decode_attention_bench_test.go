//go:build windows

package executor

import (
	"fmt"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
)

// BenchmarkDecodeAttention times one decode attention step in the
// Qwen2.5-0.5B head geometry (14 query heads over 2 key-value heads of
// width 64) at growing cache capacities, so a kernel change is measured
// in seconds against the same graph rather than inferred from a model
// run. Inputs remain on device; setup uploads and timed scalar traffic are
// reported separately. Output download/synchronization remains in ns/op.
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
			output, feeds := decodeGuardGraph(b, width, queryHeads, keyHeads, capacity, active)
			compiled, err := Compile(output)
			if err != nil {
				b.Fatal(err)
			}
			cuda := newFixtureExecutor(b)
			ctx := b.Context()
			inputs := residentDecodeInputs(b, cuda, compiled, feeds)
			if _, err := cuda.ExecuteCompiled(ctx, compiled, nil, inputs); err != nil {
				b.Fatal(err)
			}
			before, err := cuda.worker.ExecutionStats(ctx)
			if err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for b.Loop() {
				if _, err := cuda.ExecuteCompiled(ctx, compiled, nil, inputs); err != nil {
					b.Fatal(err)
				}
			}
			after, err := cuda.worker.ExecutionStats(ctx)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportMetric(float64(before.HostToDeviceBytes), "setup-upload-B")
			b.ReportMetric(float64(after.HostToDeviceBytes-before.HostToDeviceBytes)/float64(b.N), "upload-B/op")
			b.ReportMetric(float64(after.DeviceToHostBytes-before.DeviceToHostBytes)/float64(b.N), "download-B/op")
		})
	}
}
