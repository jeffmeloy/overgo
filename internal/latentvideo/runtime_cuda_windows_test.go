//go:build windows && integration

package latentvideo

import (
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/dtype"
)

func TestWanProductionRuntime(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("integration excluded by -short: loads the real Wan denoiser and VAE")
	}
	manifest, fixtureDirectory := loadDenoiseManifest(t, "g3_denoise.json")
	generator, err := NewGenerator(GeneratorConfig{
		ModelDirectory: wanModelDir(t), Policy: referenceDenoiserPolicy,
		LatentStats: vaeG0LatentStats(t),
		Frames:      manifest.Request.Frames, Width: manifest.Request.Width, Height: manifest.Request.Height,
		Precision: DenoiserPrecision{MatmulWeights: dtype.F32}, DeviceOrdinal: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := generator.Close(); err != nil {
			t.Errorf("close generator: %v", err)
		}
	}()
	contextElements := generator.denoiser.Program.Config.TextLen * generator.denoiser.Program.Config.Dim
	request := GenerateRequest{
		Steps: manifest.Request.Steps, Shift: manifest.Request.Shift, GuideScale: manifest.Request.GuideScale,
		CondContext:   loadRawContext(t, fixtureDirectory, manifest.Request.CondContext, contextElements),
		UncondContext: loadRawContext(t, fixtureDirectory, manifest.Request.UncondContext, contextElements),
		InitialSample: manifest.trace(t, fixtureDirectory, "step_00_sample_in"),
	}
	wantLatent := loadG1Tensor(t, fixtureDirectory, manifest.FinalLatent)
	var firstMemory uint64
	var warmCaptures uint64
	for run := 0; run < 3; run++ {
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
