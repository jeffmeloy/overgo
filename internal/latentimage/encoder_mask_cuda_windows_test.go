//go:build windows

package latentimage

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hfbpe"
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
	want, err := prog.RunHostFeed(GraphRunner(reference.Execute), weightAt, embed)
	if err != nil {
		t.Fatalf("reference RunHostFeed: %v", err)
	}

	exec, err := executor.New(0)
	if err != nil {
		t.Fatalf("executor.New: %v", err)
	}
	defer exec.Close()
	cudaRun := func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		return exec.Execute(context.Background(), outputs, feeds)
	}
	got, err := prog.RunHostFeed(GraphRunner(cudaRun), weightAt, embed)
	if err != nil {
		t.Fatalf("CUDA RunHostFeed: %v", err)
	}
	maxAbs := 0.0
	for i := range want.Data {
		if math.IsNaN(got.Data[i]) || math.IsInf(got.Data[i], 0) {
			t.Fatalf("masked CUDA selected[%d] non-finite", i)
		}
		maxAbs = math.Max(maxAbs, math.Abs(want.Data[i]-got.Data[i]))
	}
	t.Logf("masked encoder CUDA vs reference: max_abs=%.3e (%d values)", maxAbs, len(want.Data))
	if maxAbs > 5e-3 {
		t.Fatalf("masked encoder CUDA/reference divergence max_abs=%.3e exceeds 5e-3", maxAbs)
	}
}

// TestEncoderMaskedResidentRealCheckpoint drives the MASKED device bf16 encoder
// over the REAL Krea-2-Turbo checkpoint on the templated [prefix][prompt][pad]
// [suffix] layout (RenderKreaTextInput) and asserts the device selected-hidden
// matches the MASKED f64 host oracle (EncodeSelectedLayersMasked) within the
// bf16 band -- closing the pad-key masking residual dtc-encoder documented. A
// reduced max_prompt_tokens keeps the f64 oracle tractable while still exercising
// a real pad region. Also asserts the mask moves the answer vs maskless device.
func TestEncoderMaskedResidentRealCheckpoint(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("streams the ~3.5B-param text encoder; skipped in -short")
	}
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	e := spec.TextEncoder
	tok, err := hfbpe.Load(dir + `\tokenizer`)
	if err != nil {
		t.Fatalf("load Qwen2 tokenizer: %v", err)
	}

	// Real templated masked input at a tractable pad budget (still 34 prefix + 5
	// suffix + a genuine pad region between prompt and suffix).
	tmpl := KreaChatPromptTemplate()
	tmpl.MaxPromptTokens = 24
	in, err := RenderKreaTextInput(tok, "a red fox eating ice cream, studio photograph", tmpl)
	if err != nil {
		t.Fatalf("RenderKreaTextInput: %v", err)
	}
	attended, pad := 0, 0
	for _, m := range in.Mask {
		if m {
			attended++
		} else {
			pad++
		}
	}
	if pad == 0 {
		t.Fatalf("templated input has no pad rows (seq=%d); mask parity would be vacuous", len(in.IDs))
	}

	host, err := EncodeSelectedLayersMasked(dir, spec, in.IDs, in.Mask)
	if err != nil {
		t.Fatalf("masked host EncodeSelectedLayersMasked: %v", err)
	}
	embed, err := readEmbedRowsF32(dir, e, in.IDs)
	if err != nil {
		t.Fatalf("read embed rows: %v", err)
	}

	prog, err := CompileEncoderProgramMasked(e, float32(e.RMSNormEps), len(in.IDs), dtype.BF16, in.Mask)
	if err != nil {
		t.Fatalf("CompileEncoderProgramMasked: %v", err)
	}
	ctx := context.Background()
	re, err := NewResidentEncoder(ctx, prog, dir, 0)
	if err != nil {
		t.Fatalf("NewResidentEncoder: %v", err)
	}
	defer func() {
		if cerr := re.Close(ctx); cerr != nil {
			t.Errorf("close: %v", cerr)
		}
	}()
	dev, err := re.Encode(ctx, embed)
	if err != nil {
		t.Fatalf("masked resident Encode: %v", err)
	}
	if dev.Seq != host.Seq || dev.LayerCount != host.LayerCount || dev.Hidden != host.Hidden {
		t.Fatalf("device geometry [%d,%d,%d] != host [%d,%d,%d]", dev.Seq, dev.LayerCount, dev.Hidden, host.Seq, host.LayerCount, host.Hidden)
	}
	if !allFinite(dev.Data) {
		t.Fatal("masked device selected-hidden not finite")
	}
	worst := 0.0
	for l := 0; l < host.LayerCount; l++ {
		var maxAbs, sumAbs, sa float64
		for tok := 0; tok < host.Seq; tok++ {
			base := (tok*host.LayerCount + l) * host.Hidden
			for c := 0; c < host.Hidden; c++ {
				hv, dv := host.Data[base+c], dev.Data[base+c]
				abs := math.Abs(hv - dv)
				maxAbs = math.Max(maxAbs, abs)
				sumAbs += abs
				sa += math.Abs(hv)
			}
		}
		rel := sumAbs / (sa + 1e-9)
		worst = math.Max(worst, rel)
		t.Logf("tap %2d (after layer %2d): max_abs=%.3e mean|h-d|/mean|h|=%.3e", l, captureAfter(e)[l], maxAbs, rel)
	}
	t.Logf("MASKED device vs host selected-hidden: worst mean-relative=%.3e over %d rows (%d attended, %d pad), bf16 band",
		worst, len(in.IDs), attended, pad)
	if worst > 2e-1 {
		t.Fatalf("masked device/host selected-hidden mean-relative=%.3e exceeds bf16 band 2e-1", worst)
	}

	// The pad-key mask must move the device answer vs the maskless device path.
	mlProg, err := CompileEncoderProgram(e, float32(e.RMSNormEps), len(in.IDs), dtype.BF16)
	if err != nil {
		t.Fatalf("CompileEncoderProgram (maskless): %v", err)
	}
	mlRe, err := NewResidentEncoder(ctx, mlProg, dir, 0)
	if err != nil {
		t.Fatalf("NewResidentEncoder (maskless): %v", err)
	}
	defer func() {
		if cerr := mlRe.Close(ctx); cerr != nil {
			t.Errorf("close maskless: %v", cerr)
		}
	}()
	ml, err := mlRe.Encode(ctx, embed)
	if err != nil {
		t.Fatalf("maskless resident Encode: %v", err)
	}
	diff := 0.0
	for i := range ml.Data {
		diff = math.Max(diff, math.Abs(ml.Data[i]-dev.Data[i]))
	}
	if diff < 1e-3 {
		t.Fatalf("masked and maskless device encoders agree (max_abs=%.3e); pad-key mask is inert on real weights", diff)
	}
	t.Logf("masked-vs-maskless device delta=%.3e (pad-key mask active on real Krea weights)", diff)
}
