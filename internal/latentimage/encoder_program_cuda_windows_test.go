//go:build windows

package latentimage

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hfbpe"
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

// TestEncoderResidentRealCheckpoint drives the ~3.5B-param Qwen3-VL selected-layer
// encoder on the device (resident BF16 weights, generic executor) over the REAL
// Krea-2-Turbo checkpoint and asserts the device selected-hidden tensors match the
// host f64 reference (textencoder.go EncodeSelectedLayers) within the bf16 band.
// Also exercises the dtc-tokenizer templated ids/mask (RenderKreaTextInput) on the
// device and reports peak device weight residency.
func TestEncoderResidentRealCheckpoint(t *testing.T) {
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

	// --- device==host parity on a raw prompt (host f64 tractable) ---
	prompt := "a red fox eating ice cream, studio photograph"
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("tokenizer produced 0 ids")
	}
	host, err := EncodeSelectedLayers(dir, spec, ids)
	if err != nil {
		t.Fatalf("host EncodeSelectedLayers: %v", err)
	}
	embed, err := readEmbedRowsF32(dir, e, ids)
	if err != nil {
		t.Fatalf("read embed rows: %v", err)
	}

	prog, err := CompileEncoderProgram(e, float32(e.RMSNormEps), len(ids), dtype.BF16)
	if err != nil {
		t.Fatalf("CompileEncoderProgram: %v", err)
	}
	ctx := context.Background()
	re := newResidentFixture(t, ctx, "test encoder", filepath.Join(dir, "text_encoder"), prog.weightInputs, prog.Selected...)
	t.Logf("resident encoder weights: %.3f GiB (%d tapped layers, seq=%d)", float64(re.graph.bytes)/(1<<30), prog.CaptureAfter[len(prog.CaptureAfter)-1]+1, len(ids))
	dev := residentEncode(t, ctx, re, prog, embed)
	if dev.Seq != host.Seq || dev.LayerCount != host.LayerCount || dev.Hidden != host.Hidden {
		t.Fatalf("device geometry [%d,%d,%d] != host [%d,%d,%d]", dev.Seq, dev.LayerCount, dev.Hidden, host.Seq, host.LayerCount, host.Hidden)
	}
	if !allFinite(dev.Data) {
		t.Fatal("device selected-hidden not finite")
	}
	// per-tap bf16-band agreement (relative to the tap's own scale).
	worst := 0.0
	for l := 0; l < host.LayerCount; l++ {
		var maxAbs, sumAbs, sa float64
		n := 0
		for tok := 0; tok < host.Seq; tok++ {
			base := (tok*host.LayerCount + l) * host.Hidden
			for c := 0; c < host.Hidden; c++ {
				hv, dv := host.Data[base+c], dev.Data[base+c]
				abs := math.Abs(hv - dv)
				maxAbs = math.Max(maxAbs, abs)
				sumAbs += abs
				sa += math.Abs(hv)
				n++
			}
		}
		rel := sumAbs / (sa + 1e-9)
		worst = math.Max(worst, rel)
		t.Logf("tap %2d (after layer %2d): max_abs=%.3e mean|h-d|/mean|h|=%.3e", l, captureAfter(e)[l], maxAbs, rel)
	}
	t.Logf("device vs host selected-hidden: worst mean-relative=%.3e (bf16 band)", worst)
	if worst > 2e-1 {
		t.Fatalf("device/host selected-hidden mean-relative=%.3e exceeds bf16 band 2e-1", worst)
	}

	// --- dtc-tokenizer templated ids/mask consumed on the device ---
	in, err := RenderKreaTextInput(tok, prompt, KreaChatPromptTemplate())
	if err != nil {
		t.Fatalf("RenderKreaTextInput: %v", err)
	}
	attended := 0
	for _, m := range in.Mask {
		if m {
			attended++
		}
	}
	tmplEmbed, err := readEmbedRowsF32(dir, e, in.IDs)
	if err != nil {
		t.Fatalf("read templated embed rows: %v", err)
	}
	tprog, err := CompileEncoderProgram(e, float32(e.RMSNormEps), len(in.IDs), dtype.BF16)
	if err != nil {
		t.Fatalf("CompileEncoderProgram(templated): %v", err)
	}
	tre := newResidentFixture(t, ctx, "test templated encoder", filepath.Join(dir, "text_encoder"), tprog.weightInputs, tprog.Selected...)
	tdev := residentEncode(t, ctx, tre, tprog, tmplEmbed)
	if tdev.Seq != len(in.IDs) || tdev.LayerCount != spec.Transformer.TextLayers || tdev.Hidden != spec.Transformer.TextHidden {
		t.Fatalf("templated device geometry [%d,%d,%d]", tdev.Seq, tdev.LayerCount, tdev.Hidden)
	}
	if !allFinite(tdev.Data) {
		t.Fatal("templated device selected-hidden not finite")
	}
	t.Logf("dtc-tokenizer templated: %d rows (%d attended, %d pad), device encoder [%d,%d,%d] finite; peak device weights %.3f GiB. Pad-key masking is the documented residual (telemetry oracle stays until full e2e SHA).",
		len(in.IDs), attended, len(in.IDs)-attended, tdev.Seq, tdev.LayerCount, tdev.Hidden, float64(tre.graph.bytes)/(1<<30))
}
