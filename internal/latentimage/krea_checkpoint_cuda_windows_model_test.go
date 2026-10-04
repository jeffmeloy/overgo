//go:build windows && modeltest

package latentimage

import (
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
)

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
	tmpl := kreaProfileOrSkip(t).Prompt
	tmpl.MaxTokens = 24
	in, err := renderTextInput(tok, "a red fox eating ice cream, studio photograph", tmpl)
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
	ctx := t.Context()
	re := newResidentFixture(t, ctx, "test masked encoder", filepath.Join(dir, "text_encoder"), prog.weightInputs, prog.Selected...)
	dev := residentEncode(t, ctx, re, prog, embed)
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
				maxAbs = max(maxAbs, abs)
				sumAbs += abs
				sa += math.Abs(hv)
			}
		}
		rel := sumAbs / (sa + 1e-9)
		worst = max(worst, rel)
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
	mlRe := newResidentFixture(t, ctx, "test maskless encoder", filepath.Join(dir, "text_encoder"), mlProg.weightInputs, mlProg.Selected...)
	ml := residentEncode(t, ctx, mlRe, mlProg, embed)
	diff := 0.0
	for i := range ml.Data {
		diff = max(diff, math.Abs(ml.Data[i]-dev.Data[i]))
	}
	if diff < 1e-3 {
		t.Fatalf("masked and maskless device encoders agree (max_abs=%.3e); pad-key mask is inert on real weights", diff)
	}
	t.Logf("masked-vs-maskless device delta=%.3e (pad-key mask active on real Krea weights)", diff)
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
	ctx := t.Context()
	re := newResidentFixture(t, ctx, "test encoder", filepath.Join(dir, "text_encoder"), prog.weightInputs, prog.Selected...)
	t.Logf("resident encoder weights: %.3f GiB (%d tapped layers, seq=%d)", float64(re.graph.ProgramBytes())/(1<<30), prog.CaptureAfter[len(prog.CaptureAfter)-1]+1, len(ids))
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
				maxAbs = max(maxAbs, abs)
				sumAbs += abs
				sa += math.Abs(hv)
				n++
			}
		}
		rel := sumAbs / (sa + 1e-9)
		worst = max(worst, rel)
		t.Logf("tap %2d (after layer %2d): max_abs=%.3e mean|h-d|/mean|h|=%.3e", l, captureAfter(e)[l], maxAbs, rel)
	}
	t.Logf("device vs host selected-hidden: worst mean-relative=%.3e (bf16 band)", worst)
	if worst > 2e-1 {
		t.Fatalf("device/host selected-hidden mean-relative=%.3e exceeds bf16 band 2e-1", worst)
	}

	// --- dtc-tokenizer templated ids/mask consumed on the device ---
	in, err := renderTextInput(tok, prompt, kreaProfileOrSkip(t).Prompt)
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
		len(in.IDs), attended, len(in.IDs)-attended, tdev.Seq, tdev.LayerCount, tdev.Hidden, float64(tre.graph.ProgramBytes())/(1<<30))
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

// TestSeededInitLatentKreaGeometry exercises the real serving geometry: the
// [z_dim,256,256] latent for a 2048x2048 seed-42 Krea serve, with the shape
// DERIVED from the real checkpoint spec (VAE z_dim + spatial scale). It asserts
// the produced latent is exactly the golden init reduction -- bf16(torch.randn)
// of the same seeded draw -- over the full serving-size buffer, and every value
// is finite. Combined with the torch-anchored test above, the serving-shape
// latent is bit-exact to adaptive's mediaMixNoise init contract.
func TestSeededInitLatentKreaGeometry(t *testing.T) {
	cudatest.Require(t)
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	channels, height, width, err := media.DownsampledPlanarGeometry(spec.VAE.ZDim, 2048, 2048, spec.VAE.SpatialScale)
	if err != nil {
		t.Fatalf("LatentShape: %v", err)
	}
	shape := LatentShape{ZDim: channels, Height: height, Width: width}
	t.Logf("krea 2048x2048 latent geometry: [z=%d,h=%d,w=%d] elements=%d",
		shape.ZDim, shape.Height, shape.Width, shape.Elements())

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	latent := make([]float32, shape.Elements())
	reference := make([]float32, shape.Elements())
	err = worker.Do(t.Context(), func(state *device.State) error {
		got, err := SeededInitLatent(torchrng.NewStream(42), state, shape, InitNoiseMix)
		if err != nil {
			return err
		}
		copy(latent, got)
		// Independent draw of the SAME seeded stream, reduced by the pure host
		// core: the device-drawn latent must equal bf16(randn) element-for-element.
		ref := torchrng.NewStream(42)
		defer ref.Close(state)
		if err := ref.FillHost(state, reference); err != nil {
			return err
		}
		SeededLatentFromNoise(reference, InitNoiseMix)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	requireExactF32(t, latent, reference)
	for i, v := range latent {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite latent at %d: %v", i, v)
		}
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
	hu8, du8 := media.NormalizedF32ToU8(hostPix), media.NormalizedF32ToU8(devPix)
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
