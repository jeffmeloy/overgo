//go:build integration

package latentvideo

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/pytorchzip"
	"overgo/internal/testutil"
)

// VAE decode parity vs the committed CUDA-captured goldens. Tolerance
// derivation: the goldens are f32 CUDA captures (the production decode's
// default fp32-FMA convolution accumulation, f64 RMS norm) while this port
// accumulates convolutions in f64, so residual disagreement is purely
// engine accumulation order/precision across the 21-op graph on O(1)
// clamped outputs — a handful of final-bit ulps, not compounding error.
// Measured 2026-08-09 (logged verbatim by the tests): g3 frame_00 max
// 1.01e-6; g4 worst frame max 1.31e-6 (frame_01); means <= 2.9e-7.
// Gate: measured worst x ~7 (host arithmetic is deterministic — the
// parallel channel split preserves serial accumulation order — so the
// margin only covers a future golden recapture on different CUDA hardware).
const goldenVAEDecodeTolerance = 1e-5

func wanVAECheckpointPath(t testing.TB) string {
	t.Helper()
	path := filepath.Join(wanModelDir(t), "Wan2.1_VAE.pth")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("UNAVAILABLE: VAE checkpoint absent: %v", err)
	}
	return path
}

// vaeG0LatentStats: published per-channel latent statistics mirrored by the
// g0 capture (the checkpoint carries no config; g0 is the committed source).
func vaeG0LatentStats(t testing.TB) VAELatentStats {
	t.Helper()
	raw, err := os.ReadFile(testutil.FixturePath(t, "wan", "g0_config.json"))
	if err != nil {
		t.Fatalf("missing required g0 config: %v", err)
	}
	var g0 struct {
		VAELatentStats struct {
			Mean []float32 `json:"mean"`
			Std  []float32 `json:"std"`
			ZDim int       `json:"z_dim"`
		} `json:"vae_latent_stats"`
	}
	if err := json.Unmarshal(raw, &g0); err != nil {
		t.Fatal(err)
	}
	stats := VAELatentStats{Mean: g0.VAELatentStats.Mean, Std: g0.VAELatentStats.Std}
	if len(stats.Mean) != g0.VAELatentStats.ZDim || len(stats.Std) != g0.VAELatentStats.ZDim || g0.VAELatentStats.ZDim <= 0 {
		t.Fatalf("g0 latent stats arity mean=%d std=%d z_dim=%d", len(stats.Mean), len(stats.Std), g0.VAELatentStats.ZDim)
	}
	return stats
}

func compileRealVAEDecoderPlan(t testing.TB) (VAEDecoderPlan, string) {
	t.Helper()
	checkpoint := wanVAECheckpointPath(t)
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	metas := catalog.Tensors
	plan, err := CompileVAEDecoderPlan(metas)
	if err != nil {
		t.Fatal(err)
	}
	return plan, checkpoint
}

// TestVAEDecoderPlanStructureReal: derived graph vs the reference plan
// invariants (adaptive TestCompileCausalVideoCodecProgramsRealGated: 21 ops,
// 108 tensors, 293182412 weight bytes, largest op 31856640, conv2 ->
// decoder.head endpoints).
func TestVAEDecoderPlanStructureReal(t *testing.T) {
	plan, _ := compileRealVAEDecoderPlan(t)
	if plan.Ops() != 21 || plan.UsedTensorCount != 108 || plan.UsedWeightBytes != 293182412 || plan.LargestOpWeightBytes != 31856640 {
		t.Fatalf("bad decoder plan: ops=%d tensors=%d bytes=%d largest=%d", plan.Ops(), plan.UsedTensorCount, plan.UsedWeightBytes, plan.LargestOpWeightBytes)
	}
	prefixes := plan.OpPrefixes()
	if prefixes[0] != "conv2" || prefixes[len(prefixes)-1] != "decoder.head" {
		t.Fatalf("bad op endpoints: %v", prefixes)
	}
	if plan.ZDim != 16 || plan.OutputChannels != 3 || plan.Stride != [3]int{4, 8, 8} {
		t.Fatalf("bad derived facts: z_dim=%d out=%d stride=%v", plan.ZDim, plan.OutputChannels, plan.Stride)
	}
	stats := vaeG0LatentStats(t)
	if len(stats.Mean) != plan.ZDim {
		t.Fatalf("g0 latent stats arity=%d vs derived z_dim=%d", len(stats.Mean), plan.ZDim)
	}
	t.Logf("vae decoder ops=%d tensors=%d bytes=%d largest_op=%d stride=%v", plan.Ops(), plan.UsedTensorCount, plan.UsedWeightBytes, plan.LargestOpWeightBytes, plan.Stride)
}

type vaeGoldenFrames struct {
	shape  goldenLatentShape
	latent []float32
	frames [][]float32
}

func loadVAEGoldenFrames(t *testing.T, manifestName string) vaeGoldenFrames {
	t.Helper()
	dir := testutil.FixturePath(t, "wan")
	raw, err := os.ReadFile(filepath.Join(dir, manifestName))
	if err != nil {
		t.Fatalf("missing required golden manifest: %v", err)
	}
	var manifest struct {
		FinalLatent g1Tensor            `json:"final_latent"`
		Frames      map[string]g1Tensor `json:"frames"`
		LatentShape goldenLatentShape   `json:"latent_shape"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	golden := vaeGoldenFrames{
		shape:  manifest.LatentShape,
		latent: loadG1Tensor(t, dir, manifest.FinalLatent),
		frames: make([][]float32, len(manifest.Frames)),
	}
	for index := range golden.frames {
		spec, ok := manifest.Frames[fmt.Sprintf("frame_%02d", index)]
		if !ok {
			t.Fatalf("golden manifest %s has no frame_%02d", manifestName, index)
		}
		golden.frames[index] = loadG1Tensor(t, dir, spec)
	}
	if len(golden.frames) == 0 {
		t.Fatalf("golden manifest %s has no decoded frames", manifestName)
	}
	return golden
}

func runVAEDecodeGolden(t *testing.T, manifestName string) {
	if testing.Short() {
		t.Skip("integration excluded by -short: loads the 293MB VAE decoder weight set")
	}
	golden := loadVAEGoldenFrames(t, manifestName)
	plan, checkpoint := compileRealVAEDecoderPlan(t)
	if golden.shape.Channels != plan.ZDim {
		t.Fatalf("golden latent channels=%d plan z_dim=%d", golden.shape.Channels, plan.ZDim)
	}
	decoded := make([][]float32, 0, len(golden.frames))
	stats, err := DecodeLatentVideo(checkpoint, plan, vaeG0LatentStats(t), golden.latent,
		golden.shape.LatentFrames, golden.shape.LatentHeight, golden.shape.LatentWidth,
		func(index int, frame []float32, height, width int) error {
			if index != len(decoded) {
				return fmt.Errorf("frame index %d, want %d", index, len(decoded))
			}
			decoded = append(decoded, append([]float32(nil), frame...))
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(golden.frames) {
		t.Fatalf("decoded %d frames, want %d", len(decoded), len(golden.frames))
	}
	for index, want := range golden.frames {
		requireParity(t, fmt.Sprintf("%s frame_%02d", manifestName, index), decoded[index], want, goldenVAEDecodeTolerance)
	}
	t.Logf("%s decode ops=%d frames=%d output=%dx%dx%d weight_bytes=%d max_op_bytes=%d wall=%.3fs peak_heap_delta=%.1fMB",
		manifestName, stats.Ops, stats.OutputFrames, stats.OutputChannels, stats.OutputHeight, stats.OutputWidth,
		stats.WeightBytesRead, stats.MaxOpWeightBytes, stats.DecodeWallSec, float64(stats.PeakHeapAllocBytes)/(1<<20))
}

// TestVAEDecodeGoldenG3: single-latent-frame decode vs the committed g3
// frame (no temporal time-conv path).
func TestVAEDecodeGoldenG3(t *testing.T) {
	runVAEDecodeGolden(t, "g3_denoise.json")
}

// TestVAEDecodeGoldenG4: two-latent-frame decode vs all five committed g4
// frames (exercises the temporal upsample 'Rep' and cached time convs).
func TestVAEDecodeGoldenG4(t *testing.T) {
	runVAEDecodeGolden(t, "g4_denoise.json")
}
