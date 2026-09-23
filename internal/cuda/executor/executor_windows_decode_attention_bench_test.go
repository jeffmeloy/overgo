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
			// A short context in a long cache is the chat decode's common
			// case: the cache is sized for the model's context while the
			// conversation has read a few dozen tokens.
			for _, shape := range []struct{ capacity, active uint32 }{
				{256, 255}, {512, 511}, {1024, 1023}, {2048, 2047}, {4096, 4095}, {8192, 8191}, {16384, 16383},
				{4096, 63}, {16384, 63}, {32768, 63}, {32768, 1023},
			} {
				capacity, active := shape.capacity, shape.active
				b.Run(fmt.Sprintf("capacity=%d/active=%d", capacity, active), func(b *testing.B) {
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
					// Device-side traffic and launches per step: a cost that
					// grows with capacity rather than the live context shows here.
					b.ReportMetric(float64(after.KernelLaunches-before.KernelLaunches)/float64(b.N), "launches/op")
					b.ReportMetric(float64(after.DeviceToDeviceBytes-before.DeviceToDeviceBytes)/float64(b.N), "copy-B/op")
					b.ReportMetric(float64(after.DeviceMemsetBytes-before.DeviceMemsetBytes)/float64(b.N), "memset-B/op")
				})
			}
		})
	}
}
