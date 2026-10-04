package trainingworkflow

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
)

// TestObserverSamplesTheProcessLedger holds a training session's device
// samples to the memory the process itself holds, read from the driver's
// ledger, and no longer to the device's use beside other processes, which
// made a session's peak a number its scorecard row could not own. The
// samples exist for a session that allocates nothing too, a host run, whose
// zero is a measurement; the peak is the ledger's high-water mark and never
// below what the process held at an earlier step.
func TestObserverSamplesTheProcessLedger(t *testing.T) {
	t.Parallel()
	observer := &Observer{clock: processmeasure.NewStopwatch()}
	peakBefore := driver.ProcessPeakBytes()
	observer.SampleStep()
	observer.sampleHardware(runrecord.ServingHardwareFinish)
	peakAfter := driver.ProcessPeakBytes()
	if len(observer.hardware) != 1 {
		t.Fatalf("a session samples its device once at its finish, got %+v", observer.hardware)
	}
	sample := observer.hardware[0]
	if sample.Stage != runrecord.ServingHardwareFinish || sample.DevicePeakBytes < peakBefore || sample.DevicePeakBytes > peakAfter {
		t.Fatalf("finish sample = %+v, want the process ledger's peak between %d and %d", sample, peakBefore, peakAfter)
	}
	if sample.DeviceCurrentBytes > sample.DevicePeakBytes {
		t.Fatalf("finish sample holds %d bytes now against a peak of %d", sample.DeviceCurrentBytes, sample.DevicePeakBytes)
	}
}
