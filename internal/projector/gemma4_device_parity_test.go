package projector

import (
	"context"
	"image/png"
	"math"
	"os"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
)

// TestGemma4RealDeviceHostParity: real 12B mmproj GGUF, device projector
// forward == host oracle. Bit-exact where achievable; reports max abs diff,
// forward ms, and peak device MiB. Gated on the real model + a real image.
//
//	OVERGO_CUDA_TEST=1 OVERGO_GEMMA4_MMPROJ=... OVERGO_GEMMA4_IMAGE=...
func TestGemma4RealDeviceHostParity(t *testing.T) {
	cudatest.Require(t)
	projectorPath := os.Getenv("OVERGO_GEMMA4_MMPROJ")
	imagePath := os.Getenv("OVERGO_GEMMA4_IMAGE")
	if projectorPath == "" || imagePath == "" {
		t.Skip("set OVERGO_GEMMA4_MMPROJ and OVERGO_GEMMA4_IMAGE")
	}
	handle, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	input, err := png.Decode(handle)
	_ = handle.Close()
	if err != nil {
		t.Fatal(err)
	}

	cpu, err := OpenGemma4(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := OpenGemma4WithOptions(projectorPath, Gemma4OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()

	ctx := context.Background()
	want, err := cpu.EncodeImage(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	rows := got.GridH * got.GridW
	if !got.Embeddings.Shape.Equal(mustShape(uint64(cuda.Spec().Hidden), uint64(rows))) {
		t.Fatalf("device embedding shape = %v, want [%d %d]", got.Embeddings.Shape, cuda.Spec().Hidden, rows)
	}
	if len(got.Embeddings.Data) != len(want.Embeddings.Data) {
		t.Fatalf("device values = %d, want %d", len(got.Embeddings.Data), len(want.Embeddings.Data))
	}

	// Bit-exact device==host holds on tiny fixtures but not on the real model:
	// both paths round to bf16 at every stage boundary, and the wide reductions
	// (patch_dense 6912-wide MulMat, 3840-wide projection, the LN/RMS sums)
	// accumulate in a different order on the CUDA kernels than in the CPU
	// reference, so boundary-straddling values land on adjacent bf16 codes.
	// The adaptive golden probe band for the embeddings is 0.08 absolute
	// (pipeline vs fp32 truth); by the triangle inequality two bf16 pipelines
	// each within 0.08 of truth may sit up to 0.16 apart, so that is the
	// device==host bound. Both paths independently pass the 0.08 golden band
	// (TestGemma4RealFixture).
	const goldenBand = 0.08
	const deviceHostBound = 2 * goldenBand
	var maxAbs float64
	var mismatches, overGolden int
	for i := range want.Embeddings.Data {
		g, w := got.Embeddings.Data[i], want.Embeddings.Data[i]
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
			t.Fatalf("device output[%d] = %g is not finite", i, g)
		}
		if math.Float32bits(g) != math.Float32bits(w) {
			mismatches++
			abs := math.Abs(float64(g - w))
			if abs > maxAbs {
				maxAbs = abs
			}
			if abs > goldenBand {
				overGolden++
			}
		}
	}
	t.Logf("device==host on real 12B mmproj: rows=%d hidden=%d values=%d mismatches=%d (%.2f%%) over-golden-band=%d maxAbsDiff=%.4f",
		rows, cuda.Spec().Hidden, len(want.Embeddings.Data), mismatches,
		100*float64(mismatches)/float64(len(want.Embeddings.Data)), overGolden, maxAbs)
	if maxAbs > deviceHostBound {
		t.Fatalf("device==host maxAbsDiff %.4f exceeds 2x golden band (%.3f)", maxAbs, deviceHostBound)
	}

	// Timing: warm the graph, then average forward wall time.
	if _, err := cuda.EncodeImage(ctx, input); err != nil {
		t.Fatal(err)
	}
	const iters = 30
	start := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := cuda.EncodeImage(ctx, input); err != nil {
			t.Fatal(err)
		}
	}
	perForwardMs := float64(time.Since(start).Microseconds()) / float64(iters) / 1000.0

	stats, err := cuda.cuda.worker.MemoryStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	peakMiB := float64(stats.PeakBytes) / (1024 * 1024)
	currentMiB := float64(stats.CurrentBytes) / (1024 * 1024)
	t.Logf("device projector forward: %.3f ms/forward (%d iters) peak=%.1f MiB current=%.1f MiB",
		perForwardMs, iters, peakMiB, currentMiB)
}
