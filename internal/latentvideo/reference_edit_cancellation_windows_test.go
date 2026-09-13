//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

func TestLiveEditCancellationRecovery(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	runtime, source, initial := newLiveEditRuntimeFixture(t)
	workers := []*device.Worker{runtime.encoder.codec.worker, runtime.denoiser.worker, runtime.decoder.worker}
	libraries := make([]*driver.Library, len(workers))
	for index, worker := range workers {
		if err := worker.Do(t.Context(), func(state *device.State) error { libraries[index] = state.Driver; return nil }); err != nil {
			t.Fatal(err)
		}
	}
	generate := func(t testing.TB) {
		t.Helper()
		var frames [][]float32
		result, err := runtime.Run(t.Context(), source, initial, func(_ int, frame []float32, _, _ int) error {
			frames = append(frames, append([]float32(nil), frame...))
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(frames) != runtime.sourcePlan.Source.Frames || hashReferenceEditValues(result.Latent) != liveEditDeviceLatentOracle || hashReferenceEditFrames(frames) != liveEditDeviceFrameOracle {
			t.Fatal("recovery changed the native latent or decoded frames")
		}
	}
	generate(t)
	idle := make([]uint64, len(libraries))
	for index, library := range libraries {
		idle[index] = library.MemoryStats().CurrentBytes
	}
	for index, phase := range []string{"source", "denoise", "decode"} {
		t.Run(phase, func(t *testing.T) {
			// Re-encode the same source to exercise cancellation after native
			// encoder work. A canceled encode must not populate the cache.
			if phase == "source" {
				runtime.sourceLatent = nil
			}
			ctx, cancel := cudatest.CancelAfterDeviceWork(t.Context(), libraries[index])
			defer cancel(context.Canceled)
			frames := 0
			_, err := runtime.Run(ctx, source, initial, func(_ int, _ []float32, _, _ int) error { frames++; return nil })
			if !errors.Is(err, context.Canceled) || frames != 0 {
				t.Fatalf("%s cancellation err=%v published_frames=%d", phase, err, frames)
			}
			if phase == "source" && runtime.sourceLatent != nil {
				t.Fatal("canceled encoding populated the source cache")
			}
			generate(t)
			for owner, library := range libraries {
				if got := library.MemoryStats().CurrentBytes; got != idle[owner] {
					t.Fatalf("%s recovery changed owner %d retained memory: %d -> %d", phase, owner, idle[owner], got)
				}
			}
			t.Logf("%s canceled after CUDA work; retry preserved native latent and frames; owned bytes=%v", phase, idle)
		})
	}
	t.Run("frame_publication", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(context.Canceled)
		frames := 0
		_, err := runtime.Run(ctx, source, initial, func(_ int, _ []float32, _, _ int) error { frames++; cancel(context.Canceled); return nil })
		if !errors.Is(err, context.Canceled) || frames != 1 {
			t.Fatalf("frame publication cancellation err=%v frames=%d", err, frames)
		}
		generate(t)
	})
	if err := runtime.Close(); err != nil {
		t.Fatal(err)
	}
	for owner, library := range libraries {
		stats := library.MemoryStats()
		if stats.CurrentBytes != 0 {
			t.Fatalf("closed owner %d retains %d bytes", owner, stats.CurrentBytes)
		}
		t.Logf("closed owner=%d peak_bytes=%d current_bytes=%d", owner, stats.PeakBytes, stats.CurrentBytes)
	}
}
