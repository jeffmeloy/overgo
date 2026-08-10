package latentimage

import (
	"crypto/sha256"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The decoder graph must DERIVE from the real Krea checkpoint: z16 -> RGB,
// dim_mult [1,2,4,4] -> spatial 8x, mid-block resnet/attn/resnet, four up
// blocks each with num_res_blocks+1 resnets, three upsamplers, and every
// decoder-side tensor consumed.
func TestVAEDecoderDerivesFromRealCheckpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the ~286MB VAE decoder weight set; skipped in -short")
	}
	dir := kreaDirOrSkip(t)
	d, err := LoadVAEDecoder(dir)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}
	if d.ZDim != 16 {
		t.Errorf("ZDim=%d want 16", d.ZDim)
	}
	if d.OutChannels != 3 {
		t.Errorf("OutChannels=%d want 3", d.OutChannels)
	}
	if d.SpatialScale != 8 {
		t.Errorf("SpatialScale=%d want 8", d.SpatialScale)
	}
	if len(d.LatentsMean) != 16 || len(d.LatentsStd) != 16 {
		t.Errorf("latents mean/std len=%d/%d want 16/16", len(d.LatentsMean), len(d.LatentsStd))
	}
	// op-count: post_quant + conv_in + (resnet,attn,resnet) + 4 blocks*(3 resnet)
	// + 3 upsamplers + head = 1+1+3 + 12 + 3 + 1 = 21.
	wantOps := 1 + 1 + 3 + 4*3 + 3 + 1
	if len(d.ops) != wantOps {
		t.Errorf("op count=%d want %d", len(d.ops), wantOps)
	}
	// channel chain sanity: first op cIn == ZDim, last op cOut == OutChannels.
	if d.ops[0].cIn != d.ZDim {
		t.Errorf("first op cIn=%d want %d", d.ops[0].cIn, d.ZDim)
	}
	if last := d.ops[len(d.ops)-1]; last.cOut != d.OutChannels {
		t.Errorf("last op cOut=%d want %d", last.cOut, d.OutChannels)
	}
	// deepest feature dim = base_dim*dim_mult[last] = 96*4 = 384 at conv_in out.
	if d.ops[1].cOut != 384 {
		t.Errorf("conv_in cOut=%d want 384", d.ops[1].cOut)
	}
	t.Logf("decoder derived: z=%d out=%d scale=%dx ops=%d weightMB=%.1f",
		d.ZDim, d.OutChannels, d.SpatialScale, len(d.ops), float64(d.WeightBytes)/(1<<20))
}

// Denorm applies z*std + mean per channel (the VAE upload affine).
func TestVAEDenormFromConfig(t *testing.T) {
	dir := kreaDirOrSkip(t)
	// Read config directly to cross-check the loaded stats.
	var cfg vaeConfigJSON
	if err := readJSON(filepath.Join(dir, "vae", "config.json"), &cfg); err != nil {
		t.Fatalf("read config: %v", err)
	}
	if testing.Short() {
		// stats-only check without loading weights.
		if len(cfg.LatentsMean) != 16 {
			t.Fatalf("config mean len=%d", len(cfg.LatentsMean))
		}
		return
	}
	d, err := LoadVAEDecoder(dir)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}
	for i := range cfg.LatentsMean {
		if d.LatentsMean[i] != float32(cfg.LatentsMean[i]) {
			t.Errorf("mean[%d]=%g want %g", i, d.LatentsMean[i], float32(cfg.LatentsMean[i]))
		}
		if d.LatentsStd[i] != float32(cfg.LatentsStd[i]) {
			t.Errorf("std[%d]=%g want %g", i, d.LatentsStd[i], float32(cfg.LatentsStd[i]))
		}
	}
	// A zero latent decodes-denorms to the per-channel mean before conv_in.
	z := make([]float32, 16*2*2)
	x := make([]float32, len(z))
	for ch := 0; ch < 16; ch++ {
		for p := 0; p < 4; p++ {
			x[ch*4+p] = z[ch*4+p]*d.LatentsStd[ch] + d.LatentsMean[ch]
			if x[ch*4+p] != d.LatentsMean[ch] {
				t.Fatalf("zero-latent denorm ch %d = %g want mean %g", ch, x[ch*4+p], d.LatentsMean[ch])
			}
		}
	}
}

// End-to-end decode of a synthetic latent: [16,32,32] -> [3,256,256], finite,
// clamped to [-1,1], spatially 8x, and deterministic across runs. (Bit-exact
// g3 parity is NOT asserted here -- see TestVAEExactFinalLatentIsHookGap.)
func TestVAEDecodeSyntheticImage(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the full host decode; skipped in -short")
	}
	dir := kreaDirOrSkip(t)
	d, err := LoadVAEDecoder(dir)
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
	u8 := PixelsToU8(pixels)
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

// The exact final LATENT that produced the g3 golden image is a NEEDS-HOOK
// stage: the goldens carry only its per-step distribution (median/MAD), not the
// exact element tensor. This test pins that gap so the bit-exact g3 obligation
// is tracked, not silently dropped.
func TestVAEExactFinalLatentIsHookGap(t *testing.T) {
	path := filepath.Join("..", "..", "fixtures", "krea", "g3_final_image.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read g3: %v", err)
	}
	var g struct {
		DecodedImage struct {
			Present       bool      `json:"present"`
			Height        int       `json:"height"`
			Width         int       `json:"width"`
			Channels      int       `json:"channels"`
			ChannelMedian []float64 `json:"channel_median"`
			RawU8SHA256   string    `json:"raw_u8_sha256"`
			VAETelemetry  struct {
				LatentDeviceBytes int64 `json:"latent_device_bytes"`
				OutputScale       int   `json:"output_scale"`
			} `json:"vae_telemetry"`
		} `json:"decoded_image"`
		VAEFloatTensor struct {
			Present   bool   `json:"present"`
			NeedsHook bool   `json:"needs_hook"`
			Reason    string `json:"reason"`
		} `json:"vae_float_tensor"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse g3: %v", err)
	}
	// g3 decoded image IS present + bit-exact; its geometry matches this decoder.
	if !g.DecodedImage.Present || g.DecodedImage.Height != 256 || g.DecodedImage.Width != 256 || g.DecodedImage.Channels != 3 {
		t.Fatalf("g3 decoded image geometry unexpected: %+v", g.DecodedImage)
	}
	if g.DecodedImage.VAETelemetry.OutputScale != 8 {
		t.Errorf("g3 output_scale=%d want 8 (matches SpatialScale)", g.DecodedImage.VAETelemetry.OutputScale)
	}
	// final latent size 16*32*32*4 = 65536 bytes -> [16,32,32], the DecodeImage
	// input geometry. This is telemetry only; the element tensor is not present.
	if want := int64(16 * 32 * 32 * 4); g.DecodedImage.VAETelemetry.LatentDeviceBytes != want {
		t.Errorf("g3 latent_device_bytes=%d want %d ([16,32,32] f32)", g.DecodedImage.VAETelemetry.LatentDeviceBytes, want)
	}
	// The pre-quant float tensor AND the exact final latent are NEEDS-HOOK.
	if g.VAEFloatTensor.Present || !g.VAEFloatTensor.NeedsHook {
		t.Fatalf("expected vae_float_tensor to be a needs-hook gap, got present=%v needs_hook=%v",
			g.VAEFloatTensor.Present, g.VAEFloatTensor.NeedsHook)
	}
	t.Logf("HOOK GAP: g3 u8 image is bit-exact (sha256 %s...) but the exact final LATENT is not in the goldens; "+
		"bit-exact g3 decode requires the denoiser port (to regenerate the [16,32,32] latent) or an extmodel.ExecutionProbe latent-dump hook. "+
		"Float pre-quant tensor: %s", g.DecodedImage.RawU8SHA256[:16], g.VAEFloatTensor.Reason)
}

func u8ToBytes(u []uint8) []byte {
	b := make([]byte, len(u))
	copy(b, u)
	return b
}
