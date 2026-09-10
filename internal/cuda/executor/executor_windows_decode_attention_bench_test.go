//go:build windows

package executor

import (
	"fmt"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

// BenchmarkDecodeAttention times resident decode across narrow and wide heads.
// Setup uploads and timed scalar traffic are reported separately; output
// download/synchronization remains in ns/op. Measurement owns the device.
func BenchmarkDecodeAttention(b *testing.B) {
	cudatest.Require(b)
	device, err := driver.Open()
	if err != nil {
		b.Fatal(err)
	}
	defer device.Close()
	if _, err := device.ReserveDevice(0); err != nil {
		b.Fatal(err)
	}
	for _, geometry := range []struct{ width, queryHeads, keyHeads uint64 }{{64, 14, 2}, {128, 16, 2}, {256, 4, 1}} {
		b.Run(fmt.Sprintf("width=%d/heads=%d/kv=%d", geometry.width, geometry.queryHeads, geometry.keyHeads), func(b *testing.B) {
			for _, capacity := range []uint32{256, 512, 1024, 2048, 4096, 8192, 16384} {
				active := capacity - 1
				b.Run(fmt.Sprintf("capacity=%d", capacity), func(b *testing.B) {
					output, feeds := decodeGuardGraph(b, geometry.width, geometry.queryHeads, geometry.keyHeads, capacity, active)
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
		})
	}
}
