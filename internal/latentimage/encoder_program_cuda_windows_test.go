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

// The selected-layer encoder graph must execute on the CUDA generic executor and
// agree with the reference backend at synthetic scale. This proves the device
// path: every op the Qwen3-VL encoder needs (standard WeightedRMSNorm, per-head
// q/k norm, rotate-half RoPENeoX, causal GQA Attention, SwiGLU) runs on the 4090D
// and matches the host golden. Model-free (synthetic weights); the real 3.5B run
// rides resident BF16 weights (TestEncoderResidentRealCheckpoint).
func TestEncoderProgramCUDAMatchesReference(t *testing.T) {
	cudatest.Require(t)
	e := syntheticEncoderSpec()
	store := encoderStore(e)
	const seq = 5
	embed := f32slice(syntheticEncoderEmbed(seq, e.Hidden))
	weightAt := func(name string) ([]float32, error) { return store[name], nil }

	prog, err := CompileEncoderProgram(e, float32(e.RMSNormEps), seq, dtype.F32)
	if err != nil {
		t.Fatalf("CompileEncoderProgram: %v", err)
	}
	want, err := prog.RunHostFeed(reference.Execute, weightAt, embed)
	if err != nil {
		t.Fatalf("reference RunHostFeed: %v", err)
	}

	exec, err := executor.New(0)
	if err != nil {
		t.Fatalf("executor.New: %v", err)
	}
	defer exec.Close()
	cudaRun := func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		return exec.Execute(context.WithoutCancel(t.Context()), outputs, feeds)
	}
	got, err := prog.RunHostFeed(cudaRun, weightAt, embed)
	if err != nil {
		t.Fatalf("CUDA RunHostFeed: %v", err)
	}

	maxAbs := 0.0
	for i := range want.Data {
		g := got.Data[i]
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("CUDA selected[%d] non-finite", i)
		}
		if abs := math.Abs(want.Data[i] - g); abs > maxAbs {
			maxAbs = abs
		}
	}
	t.Logf("encoder CUDA vs reference: max_abs=%.3e (%d selected values)", maxAbs, len(want.Data))
	if maxAbs > 5e-3 {
		t.Fatalf("encoder CUDA/reference divergence max_abs=%.3e exceeds 5e-3", maxAbs)
	}
}
