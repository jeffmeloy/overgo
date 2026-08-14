//go:build windows

package latentvideo

import (
	"context"
	"fmt"
	"math"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/dtype"
)

// runSessionGoldenDenoise: the full guided trajectory on the CUDA session vs
// one captured manifest, plus retained-graph replay evidence.
func runSessionGoldenDenoise(t *testing.T, name string) {
	t.Helper()
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("integration excluded by -short: runs the full 30-block denoiser on CUDA")
	}
	manifest, dir := loadDenoiseManifest(t, name)
	program := newGoldenDenoiserProgram(t, manifest)
	session, err := NewDenoiserCUDASession(program, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Close(); err != nil {
			t.Errorf("close session: %v", err)
		}
	}()
	config := program.Config
	contextElements := config.TextLen * config.Dim
	condContext := loadRawContext(t, dir, manifest.Request.CondContext, contextElements)
	uncondContext := loadRawContext(t, dir, manifest.Request.UncondContext, contextElements)
	sampleIn := manifest.trace(t, dir, "step_00_sample_in")

	result, err := session.Denoise(context.Background(), DenoiseRequest{
		Steps: manifest.Request.Steps, Shift: manifest.Request.Shift,
		GuideScale:    manifest.Request.GuideScale,
		CondContext:   condContext,
		UncondContext: uncondContext,
		InitialSample: sampleIn,
		TraceSteps:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, step := range result.Steps {
		prefix := fmt.Sprintf("step_%02d_", i)
		requireParity(t, prefix+"sample_in", step.SampleIn, manifest.trace(t, dir, prefix+"sample_in"), goldenSampleTolerance)
		requireParity(t, prefix+"branch_output_cond", step.CondOutput, manifest.trace(t, dir, prefix+"branch_output_cond"), goldenGuidedTolerance)
		requireParity(t, prefix+"branch_output_uncond", step.UncondOutput, manifest.trace(t, dir, prefix+"branch_output_uncond"), goldenGuidedTolerance)
		requireParity(t, prefix+"model_output", step.ModelOutput, manifest.trace(t, dir, prefix+"model_output"), goldenGuidedTolerance)
		requireParity(t, prefix+"sample_out", step.SampleOut, manifest.trace(t, dir, prefix+"sample_out"), goldenSampleTolerance)
	}
	requireParity(t, "final_latent", result.Latent, loadG1Tensor(t, dir, manifest.FinalLatent), goldenFinalTolerance)

	execution, err := session.ExecutionStats()
	if err != nil {
		t.Fatal(err)
	}
	memory, err := session.MemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("graph_launches=%d graph_instantiations=%d graph_updates=%d kernel_launches=%d htod_bytes=%d dtoh_bytes=%d",
		execution.GraphLaunches, execution.GraphInstantiations, execution.GraphUpdates,
		execution.KernelLaunches, execution.HostToDeviceBytes, execution.DeviceToHostBytes)
	t.Logf("device peak_bytes=%d current_bytes=%d allocations=%d weight_bytes=%d",
		memory.PeakBytes, memory.CurrentBytes, memory.Allocations, session.WeightBytes)
	// Every retained execution goes through the graph path: 2 context runs +
	// 2 branch runs per step; replays dominate once the per-branch traces
	// stabilize (parity of the pooled feed buffers admits at most two traces
	// per branch plus the two context captures).
	executions := uint64(2 + 2*manifest.Request.Steps)
	if execution.GraphLaunches != executions {
		t.Fatalf("graph launches %d, want %d (one per retained execution)", execution.GraphLaunches, executions)
	}
	captureCeiling := uint64(6)
	if captures := execution.GraphInstantiations + execution.GraphUpdates; captures > captureCeiling {
		t.Fatalf("graph captures %d exceed %d; retained replay is not engaging", captures, captureCeiling)
	}
}

// TestDenoiserCUDASessionGoldenG3: seed-31 single-frame trajectory on the
// full-CUDA session path.
func TestDenoiserCUDASessionGoldenG3(t *testing.T) {
	runSessionGoldenDenoise(t, "g3_denoise.json")
}

// TestDenoiserCUDASessionBF16G3: BF16 projection-weight storage rounding vs
// the exact F32 golden trajectory. BF16 cannot meet the F32 golden
// tolerances element-wise; the storage-rounding evidence is the final-latent
// agreement (cosine/normalized RMS) logged verbatim and gated loosely.
func TestDenoiserCUDASessionBF16G3(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("integration excluded by -short: runs the full 30-block denoiser on CUDA")
	}
	manifest, dir := loadDenoiseManifest(t, "g3_denoise.json")
	config, weights := denoiserFixture(t)
	geometry, err := config.CompileLatentGeometry(manifest.Request.Frames, manifest.Request.Width, manifest.Request.Height)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileDenoiserProgramPrecision(config, weights, geometry, DenoiserPrecision{
		MatmulWeights: dtype.BF16, RoundAttentionStorage: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	session, err := NewDenoiserCUDASession(program, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Close(); err != nil {
			t.Errorf("close session: %v", err)
		}
	}()
	contextElements := config.TextLen * config.Dim
	result, err := session.Denoise(context.Background(), DenoiseRequest{
		Steps: manifest.Request.Steps, Shift: manifest.Request.Shift,
		GuideScale:    manifest.Request.GuideScale,
		CondContext:   loadRawContext(t, dir, manifest.Request.CondContext, contextElements),
		UncondContext: loadRawContext(t, dir, manifest.Request.UncondContext, contextElements),
		InitialSample: manifest.trace(t, dir, "step_00_sample_in"),
	})
	if err != nil {
		t.Fatal(err)
	}
	want := loadG1Tensor(t, dir, manifest.FinalLatent)
	var dot, gotNorm, wantNorm, deltaSquares, maxDelta float64
	for i := range want {
		g, w := float64(result.Latent[i]), float64(want[i])
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
		delta := g - w
		deltaSquares += delta * delta
		if math.Abs(delta) > maxDelta {
			maxDelta = math.Abs(delta)
		}
	}
	cosine := dot / math.Sqrt(gotNorm*wantNorm)
	normalizedRMS := math.Sqrt(deltaSquares / wantNorm)
	t.Logf("bf16 vs f32 golden final latent: cosine=%.9f normalized_rms=%.6g max_abs_delta=%.6g", cosine, normalizedRMS, maxDelta)
	if cosine < 0.999 {
		t.Fatalf("bf16 final latent cosine %.9f < 0.999", cosine)
	}
	if session.WeightBytes >= 5573689600 {
		t.Fatalf("bf16 weight upload %d bytes did not shrink below the F32 5573689600", session.WeightBytes)
	}
	t.Logf("bf16 weight upload bytes=%d", session.WeightBytes)
}

// TestDenoiserCUDASessionGoldenG4: seed-42 temporal five-frame trajectory on
// the full-CUDA session path.
func TestDenoiserCUDASessionGoldenG4(t *testing.T) {
	runSessionGoldenDenoise(t, "g4_denoise.json")
}
