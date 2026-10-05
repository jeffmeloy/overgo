package trainingworkflow

import (
	"testing"

	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
)

// TestObserverSamplesTheProcessLedger holds a device session's samples to the
// process ledger it reads: the memory the process holds now and the most it has
// held, at the session's start and finish, and the step at which it held the
// most, never below an earlier step.
func TestObserverSamplesTheProcessLedger(t *testing.T) {
	t.Parallel()
	current, peak := uint64(40), uint64(100)
	observer := &Observer{clock: processmeasure.NewStopwatch(), process: func() (uint64, uint64) { return current, peak }}
	observer.SampleStep()
	current = 25
	observer.SampleStep()
	observer.sampleHardware(runrecord.ServingHardwareFinish)
	if observer.stepUsed != 40 {
		t.Errorf("the step that held the most held %d, want 40, not the later 25", observer.stepUsed)
	}
	if len(observer.hardware) != 1 {
		t.Fatalf("a session samples its device once at its finish, got %+v", observer.hardware)
	}
	if sample := observer.hardware[0]; sample.Stage != runrecord.ServingHardwareFinish || sample.DeviceCurrentBytes != 25 || sample.DevicePeakBytes != 100 {
		t.Fatalf("finish sample = %+v, want the ledger's 25 now and 100 at most", sample)
	}
}

// TestAHostSessionHoldsNoDeviceMemory holds a session that runs on the host to
// zero device memory when its process holds some, as a workbench that trains on
// the host beside a resident inference runner does: its samples and its step
// read zero, not the runner's allocation, and a device session beside it reads
// the ledger. NewObserver declares which a session is.
func TestAHostSessionHoldsNoDeviceMemory(t *testing.T) {
	t.Parallel()
	host, err := NewObserver(nil, true)
	if err != nil {
		t.Fatal(err)
	}
	device, err := NewObserver(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !host.host || device.host {
		t.Fatalf("NewObserver declared host=%t for a host session and host=%t for a device one", host.host, device.host)
	}
	const resident = 5 << 30
	for _, observer := range []*Observer{host, device} {
		observer.process = func() (uint64, uint64) { return resident, 2 * resident }
		observer.SampleStep()
		observer.sampleHardware(runrecord.ServingHardwareFinish)
	}
	if sample := host.hardware[0]; sample.DeviceCurrentBytes != 0 || sample.DevicePeakBytes != 0 || host.stepUsed != 0 {
		t.Errorf("a host session holds %+v and %d at its step beside a resident runner, want zero", sample, host.stepUsed)
	}
	if sample := device.hardware[0]; sample.DeviceCurrentBytes != resident || sample.DevicePeakBytes != 2*resident || device.stepUsed != resident {
		t.Errorf("a device session holds %+v and %d at its step, want the ledger's", sample, device.stepUsed)
	}
}
