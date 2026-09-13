//go:build windows && integration

package latentvideo

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func newWanRuntimeCase(t *testing.T, fixture string) (*Generator, GenerateRequest, []float32) {
	t.Helper()
	manifest, fixtureDirectory := loadDenoiseManifest(t, fixture)
	generator, err := NewGenerator(GeneratorConfig{
		ModelDirectory: wanModelDir(t), Policy: referenceDenoiserPolicy,
		LatentStats: vaeG0LatentStats(t),
		Frames:      manifest.Request.Frames, Width: manifest.Request.Width, Height: manifest.Request.Height,
		Precision: DenoiserPrecision{MatmulWeights: dtype.F32}, DeviceOrdinal: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := generator.Close(); err != nil {
			t.Errorf("close generator: %v", err)
		}
	})
	contextElements := generator.ContextElements()
	request := GenerateRequest{
		Steps: manifest.Request.Steps, Shift: manifest.Request.Shift, GuideScale: manifest.Request.GuideScale,
		CondContext:   loadRawContext(t, fixtureDirectory, manifest.Request.CondContext, contextElements),
		UncondContext: loadRawContext(t, fixtureDirectory, manifest.Request.UncondContext, contextElements),
		InitialSample: manifest.trace(t, fixtureDirectory, "step_00_sample_in"),
	}
	wantLatent := loadG1Tensor(t, fixtureDirectory, manifest.FinalLatent)
	return generator, request, wantLatent
}

func TestWanProductionRuntime(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("integration excluded by -short: loads the real Wan denoiser and VAE")
	}
	generator, request, wantLatent := newWanRuntimeCase(t, "g3_denoise.json")
	var firstMemory uint64
	var warmCaptures uint64
	for run := range 3 {
		var frames [][]float32
		request.Sink = func(index int, frame []float32, height, width int) error {
			frames = append(frames, append([]float32(nil), frame...))
			return nil
		}
		result, err := generator.Generate(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		requireParity(t, "resident final latent", result.Denoise.Latent, wantLatent, goldenFinalTolerance)
		if len(frames) != result.Decode.OutputFrames || len(frames) != 1 || result.WeightBytes == 0 {
			t.Fatalf("run %d result=%+v frames=%d", run, result, len(frames))
		}
		captures := result.Execution.GraphInstantiations + result.Execution.GraphUpdates
		if run == 0 {
			firstMemory = result.DecodeMemory.CurrentBytes
		} else {
			if result.DecodeMemory.CurrentBytes != firstMemory {
				t.Fatalf("decoder residency grew: first=%d second=%d", firstMemory, result.DecodeMemory.CurrentBytes)
			}
			if run == 1 {
				warmCaptures = captures
			} else if captures != warmCaptures {
				t.Fatalf("compiled graph recaptured after warmup: warm=%d steady=%d", warmCaptures, captures)
			}
		}
	}
}

func TestWanPartialInitialization(t *testing.T) {
	cudatest.Require(t)
	session, err := NewDenoiserCUDASession(&DenoiserProgram{}, 0)
	if session != nil {
		t.Cleanup(func() {
			if err := session.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err == nil || session != nil {
		t.Fatalf("incomplete program returned session=%v error=%v", session, err)
	}
}

func TestWanChangedNegativeConditioning(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("integration excluded by -short: compares two real Wan sessions")
	}
	reused, request, _ := newWanRuntimeCase(t, "g3_denoise.json")
	request.Sink = func(int, []float32, int, int) error { return nil }
	if _, err := reused.Generate(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	request.UncondContext = make([]float32, reused.ContextElements())
	got, err := reused.Generate(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	fresh, _, _ := newWanRuntimeCase(t, "g3_denoise.json")
	want, err := fresh.Generate(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	requireParity(t, "changed-negative retained vs fresh latent", got.Denoise.Latent, want.Denoise.Latent, goldenFinalTolerance)
	// t.Context is canceled before cleanup; both sessions must still close.
}

func TestWanDecodeCancellation(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("integration excluded by -short: cancels and reuses a real Wan decoder")
	}
	generator, request, want := newWanRuntimeCase(t, "g4_denoise.json")
	generateFrames := func() ([][]float32, GenerationResult) {
		t.Helper()
		var result GenerationResult
		frames, _ := decodeVAEFrames(t, func(sink VideoFrameSink) (VAEDecodeStats, error) {
			request.Sink = sink
			var err error
			result, err = generator.Generate(t.Context(), request)
			return result.Decode, err
		})
		return frames, result
	}
	before, _ := generateFrames()
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(context.Canceled)
	frames := 0
	request.Sink = func(int, []float32, int, int) error {
		frames++
		cancel(context.Canceled)
		return nil
	}
	_, err := generator.Generate(ctx, request)
	if !errors.Is(err, context.Canceled) || frames != tensor.SingletonExtent {
		t.Fatalf("decode cancellation returned %v after %d frames", err, frames)
	}
	after, got := generateFrames()
	manifest, _ := loadDenoiseManifest(t, "g4_denoise.json")
	if len(after) != manifest.Request.Frames || len(after) != got.Decode.OutputFrames || len(after) != len(before) {
		t.Fatalf("recovered decode returned %d frames, result=%d", len(after), got.Decode.OutputFrames)
	}
	for index, frame := range before {
		if !slices.Equal(after[index], frame) {
			t.Fatalf("recovered frame %d differs from the same request before cancellation", index)
		}
	}
	requireParity(t, "decode cancellation recovery latent", got.Denoise.Latent, want, goldenFinalTolerance)
	t.Logf("canceled after %d frame; recovered all %d frames exactly", frames, len(after))
}

func TestWanLifecycleAcceptance(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("integration excluded by -short: checks real Wan lifecycle and references")
	}
	if os.Getenv("OVERGO_WAN_MODEL") == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("OVERGO_WAN_MODEL", filepath.Join(roots.Models, "Wan2.1-T2V-1.3B"))
	}
	t.Run("partial-initialization", TestWanPartialInitialization)
	t.Run("production-reference", TestWanProductionRuntime)
	t.Run("changed-negative-and-close", TestWanChangedNegativeConditioning)
	t.Run("decode-cancellation-and-recovery", TestWanDecodeCancellation)
}
