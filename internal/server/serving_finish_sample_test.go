package server

import (
	"context"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/runrecord"
)

// cancelSensitiveMemory answers a device memory read as the device does: not at all under a cancelled context.
type cancelSensitiveMemory struct{}

func (cancelSensitiveMemory) DeviceMemoryStats(ctx context.Context) (driver.MemoryStats, error) {
	if err := ctx.Err(); err != nil {
		return driver.MemoryStats{}, err
	}
	return driver.MemoryStats{CurrentBytes: 5, PeakBytes: 9, Allocations: 2}, nil
}

// TestCancelledGenerationStillReadsItsFinishSample holds the sample that closes a generation to a cancel:
// the device refuses a read under a cancelled context, so the finish of a run its person cancelled, the
// one whose release they wait for, was dropped and its start sample had no pair. The finish is read
// under a context the cancel does not end, and a stage is still sampled only when the device answers.
func TestCancelledGenerationStillReadsItsFinishSample(t *testing.T) {
	t.Parallel()
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	collector := &servingHardwareCollector{api: cancelSensitiveMemory{}}
	collector.sample(cancelled, runrecord.ServingHardwarePrefill)
	if len(collector.samples) != 0 {
		t.Fatal("a sample was taken under a cancelled context")
	}
	collector.finish(cancelled)
	if len(collector.samples) != 1 || collector.samples[0].Stage != runrecord.ServingHardwareFinish || collector.samples[0].DevicePeakBytes != 9 {
		t.Fatalf("a cancelled generation's finish sample = %+v, want one finish sample of the device's peak", collector.samples)
	}
	var none *servingHardwareCollector
	none.finish(cancelled)
}
