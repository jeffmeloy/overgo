//go:build windows

package latentimage

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// cudaVAERunner binds a fresh worker+executor and returns a graph runner plus a
// peak-device-bytes reader. The worker is caller-owned so MemoryStats reports
// the decode's high-water mark.
func cudaVAERunner(t *testing.T) (graphruntime.Runner, func() uint64, func()) {
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
	_ = worker.Do(t.Context(), func(s *device.State) error { s.Driver.ResetPeakBytes(); return nil })
	run := func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		return exec.Execute(context.WithoutCancel(t.Context()), outputs, feeds)
	}
	peak := func() uint64 {
		stats, _ := worker.MemoryStats(t.Context())
		return stats.PeakBytes
	}
	return run, peak, func() { exec.Close(); worker.Close() }
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
	want, _, _, err := prog.DecodeGraph(reference.Execute, d.LatentsMean, d.LatentsStd, z)
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
	wu8, gu8 := media.NormalizedF32ToU8(want), media.NormalizedF32ToU8(got)
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
