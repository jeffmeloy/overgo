//go:build windows

package latentimage

import (
	"context"
	"io"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hfbpe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// The fusion graph must execute on the CUDA generic executor and agree with the
// reference backend at synthetic scale. This proves the device path: every op the
// text-fusion stream needs (zero-centered RMSNorm, gated bidirectional GQA
// Attention -- rank-4 batched for the layerwise blocks, rank-3 for the refiner --
// the transpose-free layer projector, SwiGLU, and the gelu-tanh txt_in) runs on
// the 4090D and matches the host golden. Model-free (synthetic weights); the real
// fusion rides resident BF16 weights (TestFusionResidentRealCheckpoint).
func TestFusionProgramCUDAMatchesReference(t *testing.T) {
	cudatest.Require(t)
	const textSeq = 4
	spec := syntheticSpec()
	store := syntheticStore(spec)
	enc := f32slice(syntheticFusionInput(spec, textSeq))
	weightAt := func(name string) ([]float32, error) { return store[name], nil }

	prog, err := CompileFusionProgram(spec, 1e-5, attendedTextMask(textSeq), dtype.F32)
	if err != nil {
		t.Fatalf("CompileFusionProgram: %v", err)
	}
	want, err := prog.RunHostFeed(reference.Execute, weightAt, enc)
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
	got, err := prog.RunHostFeed(cudaRun, weightAt, enc)
	if err != nil {
		t.Fatalf("CUDA RunHostFeed: %v", err)
	}

	maxAbs := 0.0
	for i := range want {
		g := got[i]
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
			t.Fatalf("CUDA fused[%d] non-finite", i)
		}
		if abs := math.Abs(float64(want[i] - g)); abs > maxAbs {
			maxAbs = abs
		}
	}
	t.Logf("fusion CUDA vs reference: max_abs=%.3e (%d fused values)", maxAbs, len(want))
	if maxAbs > 5e-3 {
		t.Fatalf("fusion CUDA/reference divergence max_abs=%.3e exceeds 5e-3", maxAbs)
	}
}

// loadFusionStoreF32 streams only the text-fusion + txt_in tensors from the real
// transformer checkpoint as F32 (a ~1.6 GB subset, NOT the 12.82B-param net), so a
// host f64 textConditioning reference is tractable CPU-only.
func loadFusionStoreF32(t *testing.T, dir string, spec TransformerSpec) map[string][]float32 {
	t.Helper()
	src, err := safetensors.OpenSource(dir + `\transformer`)
	if err != nil {
		t.Fatalf("open transformer: %v", err)
	}
	defer src.Close()
	shapes := FusionTensorShapes(spec)
	store := make(map[string][]float32, len(shapes))
	for name := range shapes {
		tt, ok := src.Tensors[name]
		if !ok {
			t.Fatalf("checkpoint missing fusion tensor %s", name)
		}
		reader, rerr := safetensors.F32Reader(tt)
		if rerr != nil {
			t.Fatalf("F32Reader %s: %v", name, rerr)
		}
		elements := 1
		for _, d := range tt.Shape {
			elements *= int(d)
		}
		raw := make([]byte, elements*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		vals := make([]float32, elements)
		for i := range vals {
			vals[i] = math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
		}
		store[name] = vals
	}
	return store
}

// TestFusionResidentRealCheckpoint drives the text-fusion stream on the device
// (resident BF16 weights, generic executor) over the REAL Krea-2-Turbo checkpoint
// and asserts the device fused conditioning matches the host f64 reference
// (Denoiser.textConditioning) within the bf16 band, on the SAME selected hidden
// states (isolating the fusion's device==host parity). Reports peak device
// weight residency. NOTE: device==host here is exact/clean but NOT yet
// adaptive-golden -- the encoder feeding the selected hiddens still runs maskless
// (dtc-mask is the separate brick); the telemetry oracle stays until the full e2e
// SHA (same caveat as the encoder sibling).
func TestFusionResidentRealCheckpoint(t *testing.T) {
	cudatest.Require(t)
	if testing.Short() {
		t.Skip("streams the ~1.6 GB text-fusion weights; skipped in -short")
	}
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	tspec := spec.Transformer
	if tspec.ModFields == 0 {
		tspec.ModFields = 6
	}
	tok, err := hfbpe.Load(dir + `\tokenizer`)
	if err != nil {
		t.Fatalf("load Qwen2 tokenizer: %v", err)
	}

	// selected hiddens from the real encoder (host f64) -- the fusion input.
	prompt := "a red fox eating ice cream, studio photograph"
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("tokenizer produced 0 ids")
	}
	selected, err := EncodeSelectedLayers(dir, spec, ids)
	if err != nil {
		t.Fatalf("host EncodeSelectedLayers: %v", err)
	}

	// host golden fused conditioning: textConditioning over just the fusion weights.
	fusionStore := loadFusionStoreF32(t, dir, tspec)
	d := &Denoiser{T: tspec, Eps: tspec.NormEps, store: fusionStore}
	hostFused, err := d.textConditioning(append([]float64(nil), selected.Data...), selected.Seq)
	if err != nil {
		t.Fatalf("host textConditioning: %v", err)
	}
	if len(hostFused) != selected.Seq*tspec.Hidden {
		t.Fatalf("host fused len=%d want %d", len(hostFused), selected.Seq*tspec.Hidden)
	}

	// device bf16 fusion over the same selected hiddens.
	prog, err := CompileFusionProgram(tspec, float32(tspec.NormEps), attendedTextMask(selected.Seq), dtype.BF16)
	if err != nil {
		t.Fatalf("CompileFusionProgram: %v", err)
	}
	ctx := t.Context()
	rf := newResidentFixture(t, ctx, "test fusion", filepath.Join(dir, "transformer"), prog.weightInputs, prog.Fused)
	t.Logf("resident fusion weights: %.3f GiB (%d layerwise + %d refiner blocks, textSeq=%d)",
		float64(rf.graph.ProgramBytes())/(1<<30), tspec.LayerwiseTextBlocks, tspec.RefinerTextBlocks, selected.Seq)

	encF32 := make([]float32, len(selected.Data))
	for i, v := range selected.Data {
		encF32[i] = float32(v)
	}
	devFused := residentFuse(t, ctx, rf, prog, encF32)
	if len(devFused) != len(hostFused) {
		t.Fatalf("device fused len=%d want %d", len(devFused), len(hostFused))
	}

	var maxAbs, sumAbs, sa float64
	for i := range hostFused {
		hv, dv := hostFused[i], float64(devFused[i])
		if math.IsNaN(dv) || math.IsInf(dv, 0) {
			t.Fatalf("device fused[%d] non-finite", i)
		}
		abs := math.Abs(hv - dv)
		maxAbs = max(maxAbs, abs)
		sumAbs += abs
		sa += math.Abs(hv)
	}
	rel := sumAbs / (sa + 1e-9)
	t.Logf("device vs host fused conditioning: max_abs=%.3e mean|h-d|/mean|h|=%.3e (bf16 band); peak device weights %.3f GiB",
		maxAbs, rel, float64(rf.graph.ProgramBytes())/(1<<30))
	if rel > 2e-1 {
		t.Fatalf("device/host fused mean-relative=%.3e exceeds bf16 band 2e-1", rel)
	}
	t.Logf("Fusion device==host within bf16 band. Residual: NOT yet adaptive-golden (encoder maskless until dtc-mask lands); telemetry oracle until full e2e SHA.")
}
