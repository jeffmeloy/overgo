//go:build windows

package latentvideo

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

// CUDA-vs-host decode tolerance derivation: the host decode is the in-repo
// oracle (itself 1.3e-6 vs the CUDA-captured goldens). The CUDA engine here
// is the SAME accumulation class as those captures — fp32-FMA convolutions,
// f64 channel RMS norm, fp32 attention — so the disagreement vs the f64-
// accumulating host is engine accumulation order/precision on O(1) clamped
// outputs, not compounding error: the same class the golden gate bounds at
// 1e-5 (measured 1.3e-6 worst there). Gate at the same 1e-5; measured maxima
// are logged verbatim by the tests. Measured 2026-08-09: g3 frame_00 1.73e-6;
// g4 worst 3.81e-6 (frame_04); production frame_00 vs banked host 8.04e-6.
const cudaVAEDecodeTolerance = goldenVAEDecodeTolerance

// decodeVAEFrames: run one engine over a latent volume, collecting frames.
func decodeVAEFrames(t *testing.T, decode func(sink VideoFrameSink) (VAEDecodeStats, error)) ([][]float32, VAEDecodeStats) {
	t.Helper()
	var frames [][]float32
	stats, err := decode(func(index int, frame []float32, height, width int) error {
		if index != len(frames) {
			return fmt.Errorf("frame index %d, want %d", index, len(frames))
		}
		frames = append(frames, append([]float32(nil), frame...))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return frames, stats
}

func runVAEDecodeCUDAAgainstHost(t *testing.T, manifestName string) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("loads the 293MB VAE decoder weight set; skipped in -short")
	}
	golden := loadVAEGoldenFrames(t, manifestName)
	plan, checkpoint := compileRealVAEDecoderPlan(t)
	stats := vaeG0LatentStats(t)
	shape := golden.shape
	hostFrames, _ := decodeVAEFrames(t, func(sink VideoFrameSink) (VAEDecodeStats, error) {
		return DecodeLatentVideo(checkpoint, plan, stats, golden.latent, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth, sink)
	})
	session, err := NewVAEDecoderCUDASession(checkpoint, plan, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Close(); err != nil {
			t.Errorf("close session: %v", err)
		}
	}()
	cudaFrames, cudaStats := decodeVAEFrames(t, func(sink VideoFrameSink) (VAEDecodeStats, error) {
		return session.Decode(stats, golden.latent, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth, sink)
	})
	if len(cudaFrames) != len(hostFrames) {
		t.Fatalf("cuda decoded %d frames, host %d", len(cudaFrames), len(hostFrames))
	}
	for index, want := range hostFrames {
		requireParity(t, fmt.Sprintf("%s cuda frame_%02d", manifestName, index), cudaFrames[index], want, cudaVAEDecodeTolerance)
	}
	t.Logf("%s cuda decode engine=%s frames=%d wall=%.3fs peak_device=%.1fMB",
		manifestName, cudaStats.Engine, cudaStats.OutputFrames, cudaStats.DecodeWallSec,
		float64(cudaStats.PeakDeviceBytes)/(1<<20))
}

// TestVAEDecodeCUDAGoldenG3: single-latent-frame CUDA decode vs the host
// oracle (no temporal time-conv path).
func TestVAEDecodeCUDAGoldenG3(t *testing.T) {
	runVAEDecodeCUDAAgainstHost(t, "g3_denoise.json")
}

// TestVAEDecodeCUDAGoldenG4: two-latent-frame CUDA decode vs the host oracle
// on all five frames (temporal upsample 'Rep' and cached time convs).
func TestVAEDecodeCUDAGoldenG4(t *testing.T) {
	runVAEDecodeCUDAAgainstHost(t, "g4_denoise.json")
}

func loadF32LE(t testing.TB, path string, elements int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4*elements {
		t.Fatalf("%s: %d bytes, want %d", path, len(raw), 4*elements)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// TestVAEDecodeCUDAProduction: the production 81-frame decode (latent
// 16x21x60x104 -> 81 frames at 480x832) on CUDA. Latents come from the
// banked full_bf16 run artifact when present (with a host-decoded frame_00
// cross-check), else a seeded latent of the production shape (wall/memory
// only); the log states which. Gated like every real-model CUDA test.
func TestVAEDecodeCUDAProduction(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("production-scale decode; skipped in -short")
	}
	plan, checkpoint := compileRealVAEDecoderPlan(t)
	stats := vaeG0LatentStats(t)
	const latentFrames, latentH, latentW = 21, 60, 104
	elements := plan.ZDim * latentFrames * latentH * latentW
	artifactDir := filepath.Join(testutil.RepoRoot(t), "build", "latentvideo", "full_bf16")
	latentPath := filepath.Join(artifactDir, "final_latent.f32le")
	framePath := filepath.Join(artifactDir, "frame_00.f32le")
	var z []float32
	source := "seeded latent (production shape)"
	if _, err := os.Stat(latentPath); err == nil {
		z = loadF32LE(t, latentPath, elements)
		source = latentPath
	} else {
		z = make([]float32, elements)
		for i := range z {
			z[i] = float32(math.Sin(float64(i)*0.61 + 0.17))
		}
	}
	session, err := NewVAEDecoderCUDASession(checkpoint, plan, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Close(); err != nil {
			t.Errorf("close session: %v", err)
		}
	}()
	var frame0 []float32
	frames := 0
	started := time.Now()
	decodeStats, err := session.Decode(stats, z, latentFrames, latentH, latentW, func(index int, frame []float32, height, width int) error {
		if index == 0 {
			frame0 = append([]float32(nil), frame...)
		}
		frames++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(started)
	if frames != 81 {
		t.Fatalf("decoded %d frames, want 81", frames)
	}
	if wall.Seconds() >= wanOvergoDecodeCeiling {
		t.Fatalf("production CUDA decode wall %.2fs >= %.1fs", wall.Seconds(), wanOvergoDecodeCeiling)
	}
	t.Logf("production cuda decode source=%s frames=%d output=%dx%dx%d wall=%.2fs peak_device=%.1fMB weight_bytes=%d",
		source, frames, decodeStats.OutputChannels, decodeStats.OutputHeight, decodeStats.OutputWidth,
		wall.Seconds(), float64(decodeStats.PeakDeviceBytes)/(1<<20), decodeStats.WeightBytesRead)
	for prefix, elapsed := range session.Profile() {
		t.Logf("profile %-24s %8.3fs", prefix, elapsed.Seconds())
	}
	if _, err := os.Stat(framePath); err == nil && source == latentPath {
		want := loadF32LE(t, framePath, len(frame0))
		requireParity(t, "production cuda frame_00 vs banked host decode", frame0, want, cudaVAEDecodeTolerance)
	}
}
