package latentimage

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/media"
)

// syntheticSpec builds a tiny but structurally faithful TransformerSpec that
// satisfies every Krea2 constraint (hidden=heads*head_dim, sum(rope axes)=
// head_dim, text_hidden=text_heads*head_dim, GQA divisibility) so the exact
// forward arithmetic can run CPU-cheaply. The real net is 12.82B params.
func syntheticSpec() TransformerSpec {
	return TransformerSpec{
		Layers:              2,
		Heads:               4,
		KVHeads:             2,
		HeadDim:             8,
		Hidden:              32, // 4*8
		KVDim:               16, // 2*8
		InChannels:          16, // z=4, patch=2 -> 4*2*2
		Intermediate:        16,
		RopeAxes:            [3]int{2, 2, 4}, // sum 8 = head_dim
		RopeTheta:           1000,
		TimestepEmbed:       8,
		ModFields:           6,
		TextLayers:          3,
		TextHidden:          24, // 3*8
		TextIntermediate:    16,
		TextHeads:           3,
		TextKVHeads:         3,
		LayerwiseTextBlocks: 2,
		RefinerTextBlocks:   2,
	}
}

// syntheticStore fills every manifest tensor with small deterministic values so
// the forward stays well-conditioned. Norm/table weights are zero-centered in
// the reference, so small values keep (1+w) near 1.
func syntheticStore(t TransformerSpec) map[string][]float32 {
	shapes := DenoiserTensorShapes(t)
	store := make(map[string][]float32, len(shapes))
	state := uint64(0x243f6a8885a308d3)
	next := func() float32 {
		state = state*6364136223846793005 + 1442695040888963407
		u := float64(state>>11) / float64(1<<53)
		return float32((u - 0.5) * 0.1)
	}
	for name, shape := range shapes {
		v := make([]float32, prod(shape))
		for i := range v {
			v[i] = next()
		}
		store[name] = v
	}
	return store
}

func allFinite(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// The forward must run end to end at synthetic scale, produce a velocity of the
// image-sequence shape, and be finite and deterministic. This exercises every
// ported op: text fusion (layerwise+projector+refiner), txt_in, img_in, the
// timestep/AdaLN modulation, GQA gated attention with 3-axis RoPE, SwiGLU, and
// the final adaptive-norm projection.
func TestDenoiserForwardFiniteDeterministic(t *testing.T) {
	spec := syntheticSpec()
	d, err := NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), syntheticStore(spec))
	if err != nil {
		t.Fatalf("NewDenoiser: %v", err)
	}
	const gh, gw, textSeq = 2, 2, 3
	imgSeq := gh * gw
	latent := make([]float64, imgSeq*spec.InChannels)
	for i := range latent {
		latent[i] = math.Sin(float64(i) * 0.37)
	}
	enc := make([]float64, textSeq*spec.TextLayers*spec.TextHidden)
	for i := range enc {
		enc[i] = math.Cos(float64(i) * 0.11)
	}
	vel, err := d.Forward(latent, enc, 0.9, textSeq, gh, gw)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(vel) != imgSeq*spec.InChannels {
		t.Fatalf("velocity len=%d want %d", len(vel), imgSeq*spec.InChannels)
	}
	if !allFinite(vel) {
		t.Fatalf("velocity has non-finite entries")
	}
	vel2, err := d.Forward(latent, enc, 0.9, textSeq, gh, gw)
	if err != nil {
		t.Fatalf("Forward(2): %v", err)
	}
	for i := range vel {
		if vel[i] != vel2[i] {
			t.Fatalf("nondeterministic forward at %d: %g vs %g", i, vel[i], vel2[i])
		}
	}
}

// The denoiser must drive the FlowMatchEuler sampler over all 8 steps and yield
// a finite final latent, proving the denoiser<->sampler contract (velocity =
// model output, x += delta*v) at the schedule from g2's config (mu=1.15).
func TestDenoiserSamplerIntegration8Steps(t *testing.T) {
	spec := syntheticSpec()
	d, err := NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), syntheticStore(spec))
	if err != nil {
		t.Fatalf("NewDenoiser: %v", err)
	}
	sched, err := CompileFlowSchedule(8, 1000, 1.15)
	if err != nil {
		t.Fatalf("CompileFlowSchedule: %v", err)
	}
	const cLat, hLat, wLat, patch = 4, 4, 4, 2 // z=4 -> InChannels 16, grid 2x2
	textSeq := 3
	sample := make([]float64, cLat*hLat*wLat)
	state := uint64(0x9e3779b97f4a7c15)
	for i := range sample {
		state = state*6364136223846793005 + 1442695040888963407
		u := float64(state>>11) / float64(1<<53)
		sample[i] = (u - 0.5) * 2 // unit-ish "noise" seed
	}
	enc := make([]float64, textSeq*spec.TextLayers*spec.TextHidden)
	for i := range enc {
		enc[i] = math.Cos(float64(i) * 0.07)
	}
	for step := 0; step < sched.Steps; step++ {
		sigma := sched.Sigmas[step]
		patches, gh, gw, err := media.PackPlanar(sample, cLat, hLat, wLat, patch, media.PatchChannelsFirst)
		if err != nil {
			t.Fatalf("PackLatent: %v", err)
		}
		velPatches, err := d.Forward(patches, enc, sigma, textSeq, gh, gw)
		if err != nil {
			t.Fatalf("Forward step %d: %v", step, err)
		}
		velocity, err := media.UnpackPlanar(velPatches, cLat, gh, gw, patch, media.PatchChannelsFirst)
		if err != nil {
			t.Fatalf("UnpackLatent: %v", err)
		}
		if err := sched.EulerStep(sample, velocity, step); err != nil {
			t.Fatalf("EulerStep %d: %v", step, err)
		}
	}
	if !allFinite(sample) {
		t.Fatalf("final latent has non-finite entries")
	}
}

// PackLatent/UnpackLatent must round-trip and match the diffusers _pack_latents
// (channel, ph, pw) row order.
func TestPackUnpackLatentRoundTrip(t *testing.T) {
	const c, h, w, patch = 4, 4, 6, 2
	latent := make([]float64, c*h*w)
	for i := range latent {
		latent[i] = float64(i)
	}
	patches, gh, gw, err := media.PackPlanar(latent, c, h, w, patch, media.PatchChannelsFirst)
	if err != nil {
		t.Fatalf("PackLatent: %v", err)
	}
	if gh != h/patch || gw != w/patch {
		t.Fatalf("grid %dx%d want %dx%d", gh, gw, h/patch, w/patch)
	}
	if len(patches) != gh*gw*(c*patch*patch) {
		t.Fatalf("packed len=%d want %d", len(patches), gh*gw*c*patch*patch)
	}
	// explicit element check: patch (0,0), channel 0, (ph,pw)=(0,0) -> latent[0].
	if patches[0] != latent[0] {
		t.Fatalf("pack[0]=%g want %g", patches[0], latent[0])
	}
	back, err := media.UnpackPlanar(patches, c, gh, gw, patch, media.PatchChannelsFirst)
	if err != nil {
		t.Fatalf("UnpackLatent: %v", err)
	}
	for i := range latent {
		if back[i] != latent[i] {
			t.Fatalf("round-trip mismatch at %d: %g vs %g", i, back[i], latent[i])
		}
	}
}

func TestPackPlanarChannelsLastRoundTrip(t *testing.T) {
	const c, h, w, patch = 2, 2, 2, 2
	planar := []float32{0, 1, 2, 3, 10, 11, 12, 13}
	packed, gh, gw, err := media.PackPlanar(planar, c, h, w, patch, media.PatchChannelsLast)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 10, 1, 11, 2, 12, 3, 13}
	if !slices.Equal(packed, want) || gh != 1 || gw != 1 {
		t.Fatalf("packed=%v grid=%dx%d, want=%v grid=1x1", packed, gh, gw, want)
	}
	back, err := media.UnpackPlanar(packed, c, gh, gw, patch, media.PatchChannelsLast)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(back, planar) {
		t.Fatalf("round trip=%v, want=%v", back, planar)
	}
}

// RoPE: text tokens (position origin) must be the identity (cos=1, sin=0); image
// tokens must carry non-trivial rotation on the h/w axes.
func TestRopeTableOriginAndImage(t *testing.T) {
	spec := syntheticSpec()
	d := &Denoiser{T: spec, Eps: 1e-5}
	const gh, gw, textSeq = 2, 2, 1
	cos, sin := d.ropeTable(textSeq, gh, gw)
	hd := spec.HeadDim
	for i := 0; i < hd; i++ {
		if cos[i] != 1 || sin[i] != 0 {
			t.Fatalf("text token rope not identity at ch %d: cos=%g sin=%g", i, cos[i], sin[i])
		}
	}
	// image token index 3 -> grid (h=1,w=1): the h and w axes must rotate.
	tok := textSeq + 3
	var moved bool
	for i := 0; i < hd; i++ {
		if math.Abs(sin[tok*hd+i]) > 1e-9 {
			moved = true
		}
	}
	if !moved {
		t.Fatalf("image token rope produced no rotation")
	}
}

// Zero-centered RMSNorm: with zero weight the output RMS is ~1 (the (1+weight)
// scale is 1), matching Krea2RMSNorm.
func TestRMSNormZeroCentered(t *testing.T) {
	x := []float64{1, -2, 3, -4}
	w := make([]float32, 4)
	out := rmsNormZeroCentered(x, w, 1, 4, 1e-5)
	var ss float64
	for _, v := range out {
		ss += v * v
	}
	rms := math.Sqrt(ss / 4)
	if math.Abs(rms-1) > 1e-3 {
		t.Fatalf("rms=%g want ~1", rms)
	}
}

// The block shapes derived from spec must match the REAL Krea-2-Turbo checkpoint
// exactly, and every transformer tensor must be consumed by the forward
// (capability retention). Headers only -- no 51GB payload read.
func TestDenoiserDerivesFromRealCheckpoint(t *testing.T) {
	dir := kreaDirOrSkip(t)
	w, err := VerifyDenoiserCheckpoint(dir)
	if err != nil {
		if w != nil {
			for _, c := range w.Checks {
				if !c.Equal() {
					t.Errorf("%s", c.String())
				}
			}
		}
		t.Fatalf("VerifyDenoiserCheckpoint: %v", err)
	}
	// derived geometry sanity against the known Krea-2 config.
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	tr := spec.Transformer
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"layers", tr.Layers, 28},
		{"heads", tr.Heads, 48},
		{"kv_heads", tr.KVHeads, 12},
		{"head_dim", tr.HeadDim, 128},
		{"hidden", tr.Hidden, 6144},
		{"kv_dim", tr.KVDim, 1536},
		{"in_channels", tr.InChannels, 64},
		{"intermediate", tr.Intermediate, 16384},
		{"text_hidden", tr.TextHidden, 2560},
		{"text_layers", tr.TextLayers, 12},
	} {
		if tc.got != tc.want {
			t.Errorf("%s=%d want %d", tc.name, tc.got, tc.want)
		}
	}
	t.Logf("denoiser block shapes verified vs real ckpt: %d tensors consumed, 0 unconsumed, mod_fields=%d",
		w.Tensors, w.ModFields)
}

// TestDenoiserExactG3IsHookGapped pins the precise reason the bit-exact g3 image
// is not reproducible from this host port alone, so the obligation is tracked.
func TestDenoiserExactG3IsHookGapped(t *testing.T) {
	// g1 text conditioning: the fused conditioning tensor is a needs-hook gap.
	g1Path := filepath.Join("..", "..", "fixtures", "krea", "g1_text_conditioning.json")
	raw, err := os.ReadFile(g1Path)
	if err != nil {
		t.Fatalf("read g1: %v", err)
	}
	var g1 struct {
		Tensor struct {
			Present   bool   `json:"present"`
			NeedsHook bool   `json:"needs_hook"`
			Reason    string `json:"reason"`
		} `json:"tensor"`
	}
	if err := json.Unmarshal(raw, &g1); err != nil {
		t.Fatalf("parse g1: %v", err)
	}
	if g1.Tensor.Present || !g1.Tensor.NeedsHook {
		t.Fatalf("expected g1 text-cond to be a needs-hook gap, got present=%v needs_hook=%v", g1.Tensor.Present, g1.Tensor.NeedsHook)
	}
	// g2 per-step latent: only the distribution is present (element probes pending).
	g2 := loadG2(t)
	if g2.Schedule.Steps != 8 {
		t.Fatalf("g2 steps=%d want 8", g2.Schedule.Steps)
	}
	t.Logf("HOOK GAP (exact g3 unreachable from host port): "+
		"(1) text conditioning tensor is needs-hook (%s) -- the Qwen3VL 36-layer encoder's 12 selected hidden states "+
		"are now ported (textencoder.go, streamed host forward, telemetry-verified finite on the real ckpt) but their "+
		"exact values are not in the goldens (the adaptive dump hook is out of scope); "+
		"(2) seed-42 init noise is torch randn reproduced natively by adaptive, RNG not matched here; "+
		"(3) the real transformer is 12.82B params (~51GB f32) so a full-scale host forward is not CPU-runnable. "+
		"Achievable bar met: block shapes verified vs real ckpt + exact forward arithmetic runs finite/deterministic "+
		"at synthetic scale + drives the g2 (mu=1.15) FlowMatchEuler schedule for all 8 steps.", g1.Tensor.Reason)
}
