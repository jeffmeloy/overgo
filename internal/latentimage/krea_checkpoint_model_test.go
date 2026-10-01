//go:build modeltest

package latentimage

import (
	"crypto/sha256"
	"math"
	"testing"

	"overgo/internal/hfbpe"
	"overgo/internal/media"
)

// End-to-end decode of a synthetic latent: [16,32,32] -> [3,256,256], finite,
// clamped to [-1,1], spatially 8x, and deterministic across runs. (Bit-exact
// g3 parity is NOT asserted here -- see TestVAEExactFinalLatentIsHookGap.)
func TestVAEDecodeSyntheticImage(t *testing.T) {
	requireLongTest(t)
	dir := kreaDirOrSkip(t)
	d, err := loadVAEDecoder(dir, kreaProfileOrSkip(t).Classes.VAE)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}
	const lh, lw = 32, 32
	z := make([]float32, d.ZDim*lh*lw)
	// deterministic pseudo-random latent (unit-normal-ish), no external deps.
	state := uint64(0x9e3779b97f4a7c15)
	for i := range z {
		state = state*6364136223846793005 + 1442695040888963407
		u := float64(state>>11) / float64(1<<53)
		z[i] = float32((u - 0.5) * 2)
	}
	pixels, oh, ow, err := d.DecodeImage(z, lh, lw)
	if err != nil {
		t.Fatalf("DecodeImage: %v", err)
	}
	if oh != lh*8 || ow != lw*8 {
		t.Fatalf("output %dx%d want %dx%d", ow, oh, lw*8, lh*8)
	}
	if len(pixels) != d.OutChannels*oh*ow {
		t.Fatalf("pixel count=%d want %d", len(pixels), d.OutChannels*oh*ow)
	}
	for i, v := range pixels {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("pixel %d non-finite: %g", i, v)
		}
		if v < -1 || v > 1 {
			t.Fatalf("pixel %d out of [-1,1]: %g", i, v)
		}
	}
	u8 := media.NormalizedF32ToU8(pixels)
	sum := sha256.Sum256(u8ToBytes(u8))
	// determinism: a second decode is bit-identical.
	pixels2, _, _, err := d.DecodeImage(z, lh, lw)
	if err != nil {
		t.Fatalf("DecodeImage(2): %v", err)
	}
	for i := range pixels {
		if pixels[i] != pixels2[i] {
			t.Fatalf("nondeterministic decode at %d: %g vs %g", i, pixels[i], pixels2[i])
		}
	}
	t.Logf("synthetic decode ok: %dx%d planar RGB, u8 sha256=%x (regression anchor)", ow, oh, sum[:8])
}

// The full telemetry oracle on the real checkpoint: Qwen2 tokenizer -> streamed
// Qwen3-VL encoder forward -> 12 tapped hidden states -> text_fusion/txt_in ->
// conditioning tensor whose geometry matches the DiT text-stream input. Asserts
// shapes + finiteness (NOT bit-exact; exact parity needs the adaptive dump hook).
func TestEncoderRealCheckpointTelemetry(t *testing.T) {
	requireLongTest(t)
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	tok, err := hfbpe.Load(dir + `\tokenizer`)
	if err != nil {
		t.Fatalf("load Qwen2 tokenizer: %v", err)
	}
	prompt := "a red fox eating ice cream, studio photograph"
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("tokenizer produced 0 ids")
	}
	t.Logf("tokenizer: %q -> %d ids: %v", prompt, len(ids), ids)

	enc, err := EncodeSelectedLayers(dir, spec, ids)
	if err != nil {
		t.Fatalf("EncodeSelectedLayers: %v", err)
	}
	if enc.Seq != len(ids) || enc.LayerCount != spec.Transformer.TextLayers || enc.Hidden != spec.Transformer.TextHidden {
		t.Fatalf("encoder geometry seq=%d layers=%d hidden=%d want %d/%d/%d",
			enc.Seq, enc.LayerCount, enc.Hidden, len(ids), spec.Transformer.TextLayers, spec.Transformer.TextHidden)
	}
	if !allFinite(enc.Data) {
		t.Fatal("real encoder output not finite")
	}
	// per-tap magnitude telemetry (finite, non-degenerate residual stream).
	for l := 0; l < enc.LayerCount; l++ {
		var mn, mx, sa float64 = math.Inf(1), math.Inf(-1), 0
		n := 0
		for tok := 0; tok < enc.Seq; tok++ {
			row := enc.Data[(tok*enc.LayerCount+l)*enc.Hidden : (tok*enc.LayerCount+l)*enc.Hidden+enc.Hidden]
			for _, v := range row {
				mn, mx, sa, n = min(mn, v), max(mx, v), sa+math.Abs(v), n+1
			}
		}
		t.Logf("tap %2d (after layer %2d): min=%+.4f max=%+.4f mean|.|=%.4f", l, spec.TextEncoder.SelectLayers[l]-1, mn, mx, sa/float64(n))
	}

	// fusion geometry on real weights: load only text_fusion.* + txt_in.* from the
	// transformer and run textConditioning -> [seq, Hidden] (the DiT text-stream input).
	store, err := loadFusionStore(dir)
	if err != nil {
		t.Fatalf("load fusion store: %v", err)
	}
	d := &Denoiser{T: spec.Transformer, Eps: spec.Transformer.NormEps, store: store}
	txt, err := d.textConditioning(enc.Data, enc.Seq)
	if err != nil {
		t.Fatalf("textConditioning: %v", err)
	}
	if want := enc.Seq * spec.Transformer.Hidden; len(txt) != want {
		t.Fatalf("fused conditioning len=%d want %d ([%d,%d])", len(txt), want, enc.Seq, spec.Transformer.Hidden)
	}
	if !allFinite(txt) {
		t.Fatal("real fused conditioning not finite")
	}
	t.Logf("TELEMETRY PASS: tokenizer(%d ids) -> Qwen3VL encoder [%d,%d,%d] -> text_fusion/txt_in -> DiT text stream [%d,%d] (== hidden=%d); all finite. Exact values need adaptive dump hook (out of scope).",
		len(ids), enc.Seq, enc.LayerCount, enc.Hidden, enc.Seq, spec.Transformer.Hidden, spec.Transformer.Hidden)
}
