package latentvideo

import (
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/pytorchzip"
)

// Bit-parity expectations vs adaptive's CURRENT host path, verified
// 2026-08-09: adaptive's own gated run (RELATIVE_POSITION_TEXT_STREAMED_
// ENCODER=1) emits these exact values and fails its committed constants by
// the identical 0.001828022301197052 — those constants predate a bf16
// rounding change and are stale. Values below are on the BF16 grid (the
// encoder's storage dtype), deterministic across worker partitions.
var (
	wantStreamedFirst16 = []float32{0.0014877319, -0.053710938, -0.028442383, 0.064453125, 0.038330078, 0.048095703, 0.032958984, 0.063964844, 0.01159668, 0.01965332, 0.0008010864, -0.14550781, 0.041015625, -0.020629883, -0.045166016, -0.025634766}
	wantStreamedLast16  = []float32{-0.0011749268, -0.015563965, 0.005340576, -0.0077819824, 0.012207031, 0.007019043, -0.0024719238, 0.0036621094, 0.0038146973, -0.0017471313, 0.064941406, -0.00092697144, -2.9921532e-05, 0.00035476685, -0.009277344, 0.008239746}
	wantStreamedSum     = 2.1276999563
	wantStreamedMaxAbs  = 0.80078125
)

// TestEncodeTokensStreamedMatchesReferenceHost: full streamed encoder on the
// real checkpoint vs adaptive's current host path (bit parity — same engine
// class, same arithmetic; tolerance 0). Isolates port fidelity from the
// CUDA-capture accumulation noise the g1 golden gate absorbs.
func TestEncodeTokensStreamedMatchesReferenceHost(t *testing.T) {
	if testing.Short() {
		t.Skip("streams the full encoder checkpoint; skipped in -short")
	}
	modelDir := wanModelDir(t)
	checkpoint := filepath.Join(modelDir, "models_t5_umt5-xxl-enc-bf16.pth")
	metas, err := pytorchzip.ReadTensorMetadata(checkpoint)
	if err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint unreadable: %v", err)
	}
	plan, err := CompileEncoderPlan(metas, referenceEncoderPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.OK {
		t.Fatalf("plan not ok: %+v", plan)
	}
	got, stats, err := EncodeTokensStreamed(checkpoint, plan, []int{154424, 3914, 1}, []int{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Layers != 24 || stats.EmbeddingBytesRead != 24576 || stats.BlockWeightBytes != 9261514752 || stats.MaxBlockWeightBytes != 385896448 || stats.FinalNormBytes != 8192 {
		t.Fatalf("bad streamed stats: %+v", stats)
	}
	if len(got) != 3*4096 {
		t.Fatalf("output len=%d", len(got))
	}
	var sum, maxAbs float64
	for _, v := range got {
		f := float64(v)
		sum += f
		if a := math.Abs(f); a > maxAbs {
			maxAbs = a
		}
	}
	t.Logf("streamed first16=%v", got[:16])
	t.Logf("streamed last16=%v", got[len(got)-16:])
	t.Logf("streamed sum=%.12g max_abs=%.12g wall=%.1fs peak_heap_alloc_bytes=%d", sum, maxAbs, stats.EncoderWallSec, stats.PeakHeapAllocBytes)
	requireFloat32MaxAbs(t, "streamed first16", got[:16], wantStreamedFirst16, 0)
	requireFloat32MaxAbs(t, "streamed last16", got[len(got)-16:], wantStreamedLast16, 0)
	if math.Abs(sum-wantStreamedSum) > 1e-8 {
		t.Fatalf("streamed sum=%.12g want %.12g", sum, wantStreamedSum)
	}
	if math.Abs(maxAbs-wantStreamedMaxAbs) > 0 {
		t.Fatalf("streamed max_abs=%.12g want %.12g", maxAbs, wantStreamedMaxAbs)
	}
}
