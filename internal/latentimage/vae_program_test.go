package latentimage

import (
	"hash/fnv"
	"math"
	"testing"

	"overgo/internal/media"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// syntheticVAEDecoder builds a tiny but structurally faithful QwenImage spatial
// decoder (z=4, base=8, dim_mult [1,2] -> 2x spatial, 1 res block) with
// deterministic weights, so the decode graph can be validated op-for-op against
// the host vae.go reference with NO real checkpoint (CI path). It exercises
// every op kind: post_quant pointwise, causal conv, resnet (+shortcut path is
// absent here by construction, matching channel-preserving blocks), global
// spatial attention, nearest-2x resize-conv upsample, and the norm+SiLU+conv head.
func syntheticVAEDecoder(t *testing.T) *VAEDecoder {
	t.Helper()
	const z, base, outCh = 4, 8, 3
	dimMult := []int{1, 2}
	deepest := base * dimMult[len(dimMult)-1] // 16

	w := map[string][]float32{}
	put := func(name string, n int) {
		h := fnv.New64a()
		_, _ = h.Write([]byte(name))
		state := h.Sum64() | 1
		gamma := containsSub(name, "gamma")
		vals := make([]float32, n)
		for i := range vals {
			state = state*6364136223846793005 + 1442695040888963407
			u := float64(state>>11) / float64(1<<53) // [0,1)
			if gamma {
				vals[i] = float32(0.5 + u) // ~[0.5,1.5]
			} else {
				vals[i] = float32((u - 0.5) * 0.2) // ~[-0.1,0.1]
			}
		}
		w[name] = vals
	}
	conv := func(prefix string, cOut, cIn int) {
		put(prefix+".weight", cOut*cIn*27)
		put(prefix+".bias", cOut)
	}
	resnet := func(prefix string, cIn, cOut int) {
		put(prefix+".norm1.gamma", cIn)
		conv(prefix+".conv1", cOut, cIn)
		put(prefix+".norm2.gamma", cOut)
		conv(prefix+".conv2", cOut, cOut)
		if cIn != cOut {
			put(prefix+".conv_shortcut.weight", cOut*cIn)
			put(prefix+".conv_shortcut.bias", cOut)
		}
	}

	// post_quant + conv_in
	put("post_quant_conv.weight", z*z)
	put("post_quant_conv.bias", z)
	conv("decoder.conv_in", deepest, z)
	// mid: resnet, attention, resnet at deepest
	resnet("decoder.mid_block.resnets.0", deepest, deepest)
	put("decoder.mid_block.attentions.0.norm.gamma", deepest)
	put("decoder.mid_block.attentions.0.to_qkv.weight", 3*deepest*deepest)
	put("decoder.mid_block.attentions.0.to_qkv.bias", 3*deepest)
	put("decoder.mid_block.attentions.0.proj.weight", deepest*deepest)
	put("decoder.mid_block.attentions.0.proj.bias", deepest)
	resnet("decoder.mid_block.resnets.1", deepest, deepest)
	// up_blocks (dim_mult reversed): b0 blockOut=16 (2 resnets + upsampler 16->8),
	// b1 blockOut=8 (2 resnets, no upsampler).
	ch := deepest
	resnet("decoder.up_blocks.0.resnets.0", ch, 16)
	resnet("decoder.up_blocks.0.resnets.1", 16, 16)
	put("decoder.up_blocks.0.upsamplers.0.resample.1.weight", 8*16*9)
	put("decoder.up_blocks.0.upsamplers.0.resample.1.bias", 8)
	resnet("decoder.up_blocks.1.resnets.0", 8, 8)
	resnet("decoder.up_blocks.1.resnets.1", 8, 8)
	// head
	put("decoder.norm_out.gamma", 8)
	conv("decoder.conv_out", outCh, 8)

	mean := make([]float64, z)
	std := make([]float64, z)
	for i := range mean {
		mean[i] = 0.05 * float64(i-1)
		std[i] = 1.0 + 0.1*float64(i)
	}
	cfg := vaeConfigJSON{
		ClassName: "fixture-vae", BaseDim: base, DimMult: dimMult, ZDim: z,
		NumResBlks: 1, LatentsMean: mean, LatentsStd: std,
	}
	d := &VAEDecoder{ZDim: z, LatentsMean: dtype.Float64SliceToFloat32(mean), LatentsStd: dtype.Float64SliceToFloat32(std)}
	if err := d.build(w, cfg); err != nil {
		t.Fatalf("synthetic build: %v", err)
	}
	if d.SpatialScale != 2 || d.OutChannels != outCh {
		t.Fatalf("synthetic geometry scale=%d out=%d", d.SpatialScale, d.OutChannels)
	}
	return d
}

func containsSub(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// TestVAEProgramReferenceMatchesHost proves the backend-agnostic decode graph
// reproduces the host vae.go reference op-for-op on the reference backend (both
// f64 accumulation). This is the CI oracle: no checkpoint, pure logic.
func TestVAEProgramReferenceMatchesHost(t *testing.T) {
	d := syntheticVAEDecoder(t)
	const lh, lw = 6, 5
	z := make([]float32, d.ZDim*lh*lw)
	state := uint64(0x1234567)
	for i := range z {
		state = state*6364136223846793005 + 1442695040888963407
		z[i] = float32((float64(state>>11)/float64(1<<53) - 0.5) * 2)
	}

	wantPix, woh, wow, err := d.DecodeImage(z, lh, lw)
	if err != nil {
		t.Fatalf("host DecodeImage: %v", err)
	}
	prog, err := CompileVAEProgram(d, lh, lw, dtype.F32)
	if err != nil {
		t.Fatalf("CompileVAEProgram: %v", err)
	}
	gotPix, goh, gow, err := prog.DecodeGraph(reference.Execute, d.LatentsMean, d.LatentsStd, z)
	if err != nil {
		t.Fatalf("DecodeGraph: %v", err)
	}
	if goh != woh || gow != wow {
		t.Fatalf("geometry graph %dx%d != host %dx%d", gow, goh, wow, woh)
	}
	if len(gotPix) != len(wantPix) {
		t.Fatalf("pixel count graph=%d host=%d", len(gotPix), len(wantPix))
	}
	maxAbs := 0.0
	u8Diff := 0
	wu8, gu8 := media.NormalizedF32ToU8(wantPix), media.NormalizedF32ToU8(gotPix)
	for i := range wantPix {
		if a := math.Abs(float64(wantPix[i]) - float64(gotPix[i])); a > maxAbs {
			maxAbs = a
		}
		if wu8[i] != gu8[i] {
			u8Diff++
		}
	}
	t.Logf("graph vs host vae.go: max_abs=%.3e u8_diff=%d/%d (%dx%d RGB)", maxAbs, u8Diff, len(wu8), gow, goh)
	if maxAbs > 1e-4 {
		t.Fatalf("graph/host divergence max_abs=%.3e exceeds 1e-4", maxAbs)
	}
	if u8Diff != 0 {
		t.Fatalf("graph/host u8 mismatch on %d pixels (must be bit-exact)", u8Diff)
	}
}
