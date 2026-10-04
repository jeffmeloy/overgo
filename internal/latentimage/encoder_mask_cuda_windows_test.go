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

// TestEncoderProgramMaskedCUDAMatchesReference proves the MASKED encoder graph
// runs on the CUDA generic executor (the pad-key additive bias flows through the
// attention_online_f32 kernel's new key_bias input) and matches the reference
// backend at synthetic scale. This is the device half of the masked-attention
// capability: the non-breaking key-bias op is exact on the 4090D. Model-free.
func TestEncoderProgramMaskedCUDAMatchesReference(t *testing.T) {
	cudatest.Require(t)
	e := syntheticEncoderSpec()
	store := encoderStore(e)
	const seq = 7
	embed := f32slice(syntheticEncoderEmbed(seq, e.Hidden))
	mask := make([]bool, seq)
	for i := range mask {
		mask[i] = true
	}
	mask[3], mask[4] = false, false // interior pad keys
	weightAt := func(name string) ([]float32, error) { return store[name], nil }

	prog, err := CompileEncoderProgramMasked(e, float32(e.RMSNormEps), seq, dtype.F32, mask)
	if err != nil {
		t.Fatalf("CompileEncoderProgramMasked: %v", err)
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
		if math.IsNaN(got.Data[i]) || math.IsInf(got.Data[i], 0) {
			t.Fatalf("masked CUDA selected[%d] non-finite", i)
		}
		maxAbs = max(maxAbs, math.Abs(want.Data[i]-got.Data[i]))
	}
	t.Logf("masked encoder CUDA vs reference: max_abs=%.3e (%d values)", maxAbs, len(want.Data))
	if maxAbs > 5e-3 {
		t.Fatalf("masked encoder CUDA/reference divergence max_abs=%.3e exceeds 5e-3", maxAbs)
	}
}
