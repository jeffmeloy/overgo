//go:build windows && modeltest

package diffusionimage

import (
	"context"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
)

func TestRealResidentForwardLeadership(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, H, W int
		X, Out  []float32
	}](t, "real_forward")
	if golden.B != 1 {
		t.Fatalf("resident graph currently requires batch 1, got %d", golden.B)
	}
	hostDurations := make([]time.Duration, 3)
	var hostOutput []float32
	for index := range hostDurations {
		started := time.Now()
		var err error
		hostOutput, err = model.Forward(golden.X, 1, golden.H, golden.W)
		if err != nil {
			t.Fatal(err)
		}
		hostDurations[index] = time.Since(started)
	}
	compileStarted := time.Now()
	forward, err := CompileResidentForward(t.Context(), model, 0, golden.H, golden.W)
	if err != nil {
		t.Fatal(err)
	}
	defer forward.Close(context.Background())
	compileWall := time.Since(compileStarted)
	before, err := forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	coldStarted := time.Now()
	deviceOutput, err := forward.Execute(t.Context(), golden.X)
	if err != nil {
		t.Fatal(err)
	}
	coldWall := time.Since(coldStarted)
	warmDurations := make([]time.Duration, 3)
	for index := range warmDurations {
		started := time.Now()
		deviceOutput, err = forward.Execute(t.Context(), golden.X)
		if err != nil {
			t.Fatal(err)
		}
		warmDurations[index] = time.Since(started)
	}
	after, err := forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(hostDurations)
	slices.Sort(warmDurations)
	hostMedian, warmMedian := hostDurations[1], warmDurations[1]
	hostDiff := maxAbsDiff(hostOutput, golden.Out)
	deviceDiff := maxAbsDiff(deviceOutput, golden.Out)
	hostToDeviceCopies := after.Execution.HostToDeviceCopies - before.Execution.HostToDeviceCopies
	hostToDeviceBytes := after.Execution.HostToDeviceBytes - before.Execution.HostToDeviceBytes
	t.Logf("real SimpleDiffusion resident forward: compile=%s cold=%s warm_median=%s host_median=%s", compileWall, coldWall, warmMedian, hostMedian)
	t.Logf("quality: host_max=%.3e cuda_max=%.3e static=%.3f MiB peak=%.3f MiB h2d=%d copies/%.3f MiB", hostDiff, deviceDiff, float64(after.StaticBytes)/(1<<20), float64(after.Device.PeakBytes)/(1<<20), hostToDeviceCopies, float64(hostToDeviceBytes)/(1<<20))
	if hostDiff > tolReal || deviceDiff > 2e-5 {
		t.Fatalf("resident quality exceeds gates: host=%.3e cuda=%.3e", hostDiff, deviceDiff)
	}
	if coldWall >= hostMedian {
		t.Fatalf("resident cold execution %s does not beat host median %s", coldWall, hostMedian)
	}
	if warmMedian*4 >= hostMedian {
		t.Fatalf("resident warm median %s does not beat host %s by 4x", warmMedian, hostMedian)
	}
	if hostToDeviceBytes >= 1<<20 {
		t.Fatalf("resident execution uploaded %.3f MiB after static binding", float64(hostToDeviceBytes)/(1<<20))
	}
	if after.StaticBytes < 350<<20 || after.Device.PeakBytes > after.StaticBytes+140<<20 {
		t.Fatalf("resident memory static=%d peak=%d exceeds evidence range", after.StaticBytes, after.Device.PeakBytes)
	}
}
