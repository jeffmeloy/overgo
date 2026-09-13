//go:build windows && integration

package latentvideo

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
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
	cond, uncond, err := reused.ProjectBranchContexts(request.CondContext, request.UncondContext)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{cond, uncond} {
		branch, ok := value.(*sessionBranchContext)
		if !ok {
			t.Fatal("unexpected projected branch type")
		}
		for _, node := range contextGraphOutputs(reused.denoiser.Program) {
			if _, live := branch.retained.Value(node); !live {
				t.Fatal("projected guidance pair contains a released context")
			}
		}
	}
	if len(reused.branches) != tensor.PairedExtent || len(reused.denoiser.branches) != tensor.PairedExtent {
		t.Fatal("changed guidance pair retained obsolete projections")
	}
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
	cond, uncond, err = reused.ProjectBranchContexts(request.CondContext, request.CondContext)
	if err != nil || cond != uncond || len(reused.branches) != tensor.SingletonExtent || len(reused.denoiser.branches) != tensor.SingletonExtent {
		t.Fatalf("identical guidance contexts did not share one live projection: %v", err)
	}
	if _, _, err := reused.ProjectBranchContexts(request.CondContext, nil); err == nil {
		t.Fatal("accepted missing negative conditioning")
	}
	if len(reused.branches) != 0 || len(reused.denoiser.branches) != 0 {
		t.Fatal("failed guidance pair retained a partial projection")
	}
	recovered, err := reused.Generate(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	requireParity(t, "failed-pair recovery vs fresh latent", recovered.Denoise.Latent, want.Denoise.Latent, goldenFinalTolerance)
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
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
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

func TestWanContextHandleOwnership(t *testing.T) {
	cudatest.Require(t)
	values := []float32{1, 2, 3, 4}
	build := func() *DenoiserCUDASession {
		t.Helper()
		b := tensor.NewBuilder()
		shape := tensor.MustShape(uint64(len(values)))
		input := b.Input("context", dtype.F32, shape)
		key, value := b.Add(input, input), b.Multiply(input, input)
		p := &DenoiserProgram{contextInput: input, contextKeys: []*tensor.Tensor{key}, contextValues: []*tensor.Tensor{value}, weights: &DenoiserWeights{}}
		p.stepPatch = b.Input("patch", dtype.F32, shape)
		p.stepBlockE = b.Input("block", dtype.F32, shape)
		p.stepHeadE = b.Input("head", dtype.F32, shape)
		crossKey, crossValue := b.Input("key", dtype.F32, shape), b.Input("value", dtype.F32, shape)
		p.stepCrossKeys = []*tensor.Tensor{crossKey}
		p.stepCrossValues = []*tensor.Tensor{crossValue}
		p.Head = b.Add(b.Add(b.Add(p.stepPatch, p.stepBlockE), p.stepHeadE), b.Add(crossKey, crossValue))
		s, err := NewDenoiserCUDASession(p, device.DefaultOrdinal())
		if err != nil {
			t.Fatal(err)
		}
		var library *driver.Library
		if err := s.worker.Do(t.Context(), func(state *device.State) error { library = state.Driver; return nil }); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.Close(); err != nil {
				t.Error(err)
			}
			if library.MemoryStats().CurrentBytes != 0 {
				t.Error("closed session retained device ownership")
			}
		})
		return s
	}
	s, other := build(), build()
	cond, _, err := s.ProjectBranchContexts(values, values)
	if err != nil {
		t.Fatal(err)
	}
	foreign, _, err := other.ProjectBranchContexts(values, values)
	if err != nil {
		t.Fatal(err)
	}
	output, err := s.ForwardHead(values, values, values, cond)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(output, []float32{6, 14, 24, 36}) {
		t.Fatalf("unexpected valid output: %v", output)
	}
	refuse := func(handle any) {
		t.Helper()
		before, err := s.ExecutionStats()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.ForwardHead(values, values, values, handle); err == nil {
			t.Fatal("accepted foreign or released conditioning handle")
		}
		after, err := s.ExecutionStats()
		if err != nil {
			t.Fatal(err)
		}
		if before != after {
			t.Fatal("invalid handle submitted device work")
		}
	}
	refuse(foreign)
	canceled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	s.ctx = canceled
	if _, err := s.ForwardHead(values, values, values, cond); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled step: %v", err)
	}
	s.ctx = t.Context()
	if err := s.ReleaseRequestResources(); err != nil {
		t.Fatal(err)
	}
	refuse(cond)
	cond, _, err = s.ProjectBranchContexts(values, values)
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := s.ForwardHead(values, values, values, cond)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(recovered, output) {
		t.Fatal("context recovery changed output")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ForwardHead(values, values, values, cond); err == nil {
		t.Fatal("closed session accepted context handle")
	}
}
