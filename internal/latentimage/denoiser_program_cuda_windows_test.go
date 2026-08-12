//go:build windows

package latentimage

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// The Krea2 dual-stream co-attention graph must execute on the CUDA generic
// executor and agree with the reference backend at synthetic scale. This proves
// the device path: every op the 28-block forward needs (gated GQA attention,
// 3-axis interleaved RoPE, zero-centered RMSNorm, SwiGLU, AdaLN modulation,
// concat co-attention) runs on the 4090D and matches the host golden. Model-
// free (synthetic weights); the real 12.82B run rides resident BF16 weights.
func TestDenoiserProgramCUDAMatchesReference(t *testing.T) {
	cudatest.Require(t)
	const gh, gw, textSeq = 2, 2, 3
	imgSeq := gh * gw
	spec := syntheticSpec()
	d, err := NewDenoiser(spec, 1e-5, 1000, syntheticStore(spec))
	if err != nil {
		t.Fatalf("NewDenoiser: %v", err)
	}
	latent, enc := syntheticForwardInputs(spec, textSeq, imgSeq)
	const sigma = 0.9

	prog, err := CompileDenoiserProgram(spec, 1e-5, textSeq, gh, gw, dtype.F32)
	if err != nil {
		t.Fatalf("CompileDenoiserProgram: %v", err)
	}
	want, err := prog.Forward(GraphRunner(reference.Execute), d, latent, enc, sigma)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}

	exec, err := executor.New(0)
	if err != nil {
		t.Fatalf("executor.New: %v", err)
	}
	defer exec.Close()
	cudaRun := func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		return exec.Execute(context.Background(), outputs, feeds)
	}
	got, err := prog.Forward(GraphRunner(cudaRun), d, latent, enc, sigma)
	if err != nil {
		t.Fatalf("CUDA Forward: %v", err)
	}

	maxAbs := 0.0
	for i := range want.Velocity {
		g := float64(got.Velocity[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("CUDA velocity[%d] non-finite", i)
		}
		if abs := math.Abs(float64(want.Velocity[i]) - g); abs > maxAbs {
			maxAbs = abs
		}
	}
	// per-block distribution agreement (the g2 oracle shape).
	for l := range want.BlockHidden {
		wm, ws := meanStd(want.BlockHidden[l])
		gm, gs := meanStd(got.BlockHidden[l])
		t.Logf("block %d: host mean=%.4e std=%.4e | cuda mean=%.4e std=%.4e", l, wm, ws, gm, gs)
		if math.Abs(wm-gm) > 1e-3 || math.Abs(ws-gs) > 1e-3 {
			t.Fatalf("block %d distribution divergence", l)
		}
	}
	t.Logf("CUDA vs reference velocity max_abs=%.3e", maxAbs)
	if maxAbs > 5e-3 {
		t.Fatalf("CUDA/reference velocity divergence max_abs=%.3e exceeds 5e-3", maxAbs)
	}
}

func meanStd(v []float32) (mean, std float64) {
	if len(v) == 0 {
		return 0, 0
	}
	var sum, sumSq float64
	for _, x := range v {
		sum += float64(x)
		sumSq += float64(x) * float64(x)
	}
	n := float64(len(v))
	mean = sum / n
	variance := sumSq/n - mean*mean
	if variance < 0 {
		variance = 0
	}
	return mean, math.Sqrt(variance)
}
