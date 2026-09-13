//go:build windows

package sensenovarecipe

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
)

func TestSenseNovaCancellationRecovery(t *testing.T) {
	cudatest.Require(t)
	// Assertions use owned allocations and exact outputs; shared admission suffices.
	_, _, generator := newProductionGenerationFixture(t)
	var oracle generationLeadershipOracle
	if err := jsonfile.Decode(senseNovaGenerationGold, &oracle); err != nil {
		t.Fatal(err)
	}
	request := GenerationRequest{Prompt: oracle.Request.Prompt, Width: oracle.Request.Width, Height: oracle.Request.Height, Steps: oracle.Request.Steps, Seed: oracle.Request.Seed, CFGScale: oracle.Request.CFGScale, TimestepShift: oracle.Request.TimestepShift}
	var library *driver.Library
	if err := generator.worker.Do(t.Context(), func(state *device.State) error { library = state.Driver; return nil }); err != nil {
		t.Fatal(err)
	}
	checkImage := func(t testing.TB, image latentimage.EncodedImage) {
		t.Helper()
		if hash := fmt.Sprintf("%x", sha256.Sum256(image.Data)); hash != senseNovaProductionPNG || image.Width != request.Width || image.Height != request.Height {
			t.Fatalf("recovered native PNG differs: %s %dx%d", hash, image.Width, image.Height)
		}
	}
	generate := func(t testing.TB) generationFeatures {
		t.Helper()
		plan, err := generator.prepare(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		features, err := generator.integrate(t.Context(), plan)
		if err != nil {
			t.Fatal(err)
		}
		image, err := generator.decode(t.Context(), features)
		if err != nil {
			t.Fatal(err)
		}
		checkImage(t, image)
		return features
	}
	features := generate(t)
	idle := library.MemoryStats()
	t.Logf("initial idle bytes=%d", idle.CurrentBytes)
	for _, phase := range []string{"prepare", "integrate"} {
		t.Run(phase, func(t *testing.T) {
			var plan *generationPlan
			var err error
			if phase == "integrate" {
				plan, err = generator.prepare(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
			}
			probe, cancel := cudatest.CancelAfterDeviceWork(t.Context(), library)
			defer cancel(context.Canceled)
			if phase == "prepare" {
				_, err = generator.prepare(probe, request)
			} else {
				_, err = generator.integrate(probe, plan)
			}
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("%s did not cancel after executed device work: %v", phase, err)
			}
			if err := generator.Reset(t.Context(), request); err != nil {
				t.Fatal(err)
			}
			after := library.MemoryStats()
			if after.CurrentBytes != idle.CurrentBytes {
				t.Errorf("%s cancellation retained device allocation: before=%d after=%d", phase, idle.CurrentBytes, after.CurrentBytes)
			}
			features = generate(t)
			recovered := library.MemoryStats()
			if recovered.CurrentBytes != idle.CurrentBytes {
				t.Fatal("recovered request increased retained device bytes")
			}
			t.Logf("%s canceled after CUDA work; idle bytes=%d; recovered native PNG=%s", phase, after.CurrentBytes, senseNovaProductionPNG)
		})
	}
	canceled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := generator.decode(canceled, features); !errors.Is(err, context.Canceled) {
		t.Fatal("decode ignored canceled request", err)
	}
	image, err := generator.decode(t.Context(), features)
	if err != nil {
		t.Fatal(err)
	}
	checkImage(t, image)
	if err := generator.Close(canceled); err != nil {
		t.Fatal("close failed with canceled request context", err)
	}
	closed := library.MemoryStats()
	if closed.CurrentBytes != 0 {
		t.Fatalf("closed generator retains %d bytes", closed.CurrentBytes)
	}
	t.Logf("canceled decode recovered; peak_device=%d closed_device=%d", closed.PeakBytes, closed.CurrentBytes)
}
