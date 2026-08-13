//go:build windows

package latentimage

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// cudaVAERunner binds a fresh worker+executor and returns a GraphRunner plus a
// peak-device-bytes reader. The worker is caller-owned so MemoryStats reports
// the decode's high-water mark.
func cudaVAERunner(t *testing.T) (GraphRunner, func() uint64, func()) {
	t.Helper()
	worker, err := device.New(0)
	if err != nil {
		t.Fatalf("device.New: %v", err)
	}
	exec, err := executor.NewWithWorker(worker)
	if err != nil {
		worker.Close()
		t.Fatalf("executor.NewWithWorker: %v", err)
	}
	_ = worker.Do(context.Background(), func(s *device.State) error { s.Driver.ResetPeakBytes(); return nil })
	run := func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		return exec.Execute(context.Background(), outputs, feeds)
	}
	peak := func() uint64 {
		stats, _ := worker.MemoryStats(context.Background())
		return stats.PeakBytes
	}
	return GraphRunner(run), peak, func() { exec.Close(); worker.Close() }
}

// TestVAEProgramCUDAMatchesReference proves the QwenImage spatial decode graph
// runs on the CUDA generic executor and agrees with the reference backend at
// synthetic scale: every op the decode needs (post_quant pointwise, causal
// Conv2D, channel L2 norm, SiLU, global spatial Attention, nearest-2x RepeatHeads
// upsample, head) executes on the 4090D and matches the host golden. Model-free.
func TestVAEProgramCUDAMatchesReference(t *testing.T) {
	cudatest.Require(t)
	d := syntheticVAEDecoder(t)
	const lh, lw = 8, 8
	z := make([]float32, d.ZDim*lh*lw)
	state := uint64(0x77aa33)
	for i := range z {
		state = state*6364136223846793005 + 1442695040888963407
		z[i] = float32((float64(state>>11)/float64(1<<53) - 0.5) * 2)
	}
	prog, err := CompileVAEProgram(d, lh, lw, dtype.F32)
	if err != nil {
		t.Fatalf("CompileVAEProgram: %v", err)
	}
	want, _, _, err := prog.DecodeGraph(GraphRunner(reference.Execute), d.LatentsMean, d.LatentsStd, z)
	if err != nil {
		t.Fatalf("reference DecodeGraph: %v", err)
	}
	run, peak, closeFn := cudaVAERunner(t)
	defer closeFn()
	got, _, _, err := prog.DecodeGraph(run, d.LatentsMean, d.LatentsStd, z)
	if err != nil {
		t.Fatalf("CUDA DecodeGraph: %v", err)
	}
	maxAbs := 0.0
	u8Diff := 0
	wu8, gu8 := PixelsToU8(want), PixelsToU8(got)
	for i := range want {
		g := float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("CUDA pixel[%d] non-finite", i)
		}
		if a := math.Abs(float64(want[i]) - g); a > maxAbs {
			maxAbs = a
		}
		if wu8[i] != gu8[i] {
			u8Diff++
		}
	}
	t.Logf("CUDA vs reference: max_abs=%.3e u8_diff=%d/%d peak_device=%.1fMB", maxAbs, u8Diff, len(wu8), float64(peak())/(1<<20))
	if maxAbs > 2e-3 {
		t.Fatalf("CUDA/reference divergence max_abs=%.3e exceeds 2e-3", maxAbs)
	}
}

// TestVAEDecodeCUDARealModelMatchesHost decodes a latent through the real
// Krea-2-Turbo VAE on the 4090D and proves the device graph matches the host
// vae.go reference. The exact g3 final latent is a documented NEEDS-HOOK gap
// (fixtures carry the decoded RGB but not the latent that produced it -- see
// TestVAEExactFinalLatentIsHookGap), so this runs a deterministic synthetic
// latent at the g3 geometry ([16,32,32] -> 256x256) and reports peak memory.
func TestVAEDecodeCUDARealModelMatchesHost(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("loads the ~286MB Krea VAE decoder; skipped in -short")
	}
	dir := kreaDirOrSkip(t)
	d, err := loadVAEDecoder(dir, kreaProfileOrSkip(t).Classes.VAE)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}
	const lh, lw = 32, 32 // g3 latent geometry
	z := make([]float32, d.ZDim*lh*lw)
	state := uint64(0x9e3779b97f4a7c15)
	for i := range z {
		state = state*6364136223846793005 + 1442695040888963407
		z[i] = float32((float64(state>>11)/float64(1<<53) - 0.5) * 2)
	}

	hostPix, hoh, how, err := d.DecodeImage(z, lh, lw)
	if err != nil {
		t.Fatalf("host DecodeImage: %v", err)
	}
	prog, err := CompileVAEProgram(d, lh, lw, dtype.F32)
	if err != nil {
		t.Fatalf("CompileVAEProgram: %v", err)
	}
	run, peak, closeFn := cudaVAERunner(t)
	defer closeFn()
	devPix, doh, dow, err := prog.DecodeGraph(run, d.LatentsMean, d.LatentsStd, z)
	if err != nil {
		t.Fatalf("CUDA DecodeGraph: %v", err)
	}
	if doh != hoh || dow != how {
		t.Fatalf("geometry device %dx%d != host %dx%d", dow, doh, how, hoh)
	}
	if doh != 256 || dow != 256 {
		t.Fatalf("expected 256x256 (32*8) got %dx%d", dow, doh)
	}
	maxAbs := 0.0
	u8Diff := 0
	hu8, du8 := PixelsToU8(hostPix), PixelsToU8(devPix)
	for i := range hostPix {
		g := float64(devPix[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("device pixel[%d] non-finite", i)
		}
		if a := math.Abs(float64(hostPix[i]) - g); a > maxAbs {
			maxAbs = a
		}
		if hu8[i] != du8[i] {
			u8Diff++
		}
	}
	t.Logf("REAL Krea VAE device vs host: max_abs=%.3e u8_diff=%d/%d peak_device=%.1fMB weightMB=%.1f (%dx%d RGB, tiling=whole-frame)",
		maxAbs, u8Diff, len(hu8), float64(peak())/(1<<20), float64(d.WeightBytes)/(1<<20), dow, doh)
	if maxAbs > 5e-3 {
		t.Fatalf("device/host divergence max_abs=%.3e exceeds 5e-3", maxAbs)
	}
	if u8Diff*50 > len(hu8) { // device fp32 vs host f64: allow <2% off-by-one u8
		t.Fatalf("device/host u8 mismatch on %d/%d pixels (>2%%)", u8Diff, len(hu8))
	}
}
