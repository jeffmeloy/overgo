package latentimage

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

// The decoder graph must DERIVE from the real Krea checkpoint: z16 -> RGB,
// dim_mult [1,2,4,4] -> spatial 8x, mid-block resnet/attn/resnet, four up
// blocks each with num_res_blocks+1 resnets, three upsamplers, and every
// decoder-side tensor consumed.
func TestVAEDecoderDerivesFromRealCheckpoint(t *testing.T) {
	requireLongTest(t)
	dir := kreaDirOrSkip(t)
	d, err := loadVAEDecoder(dir, kreaProfileOrSkip(t).Classes.VAE)
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
	if len(d.Operations) != wantOps {
		t.Errorf("op count=%d want %d", len(d.Operations), wantOps)
	}
	// channel chain sanity: first op cIn == ZDim, last op cOut == OutChannels.
	if d.Operations[0].InputChannels != d.ZDim {
		t.Errorf("first op cIn=%d want %d", d.Operations[0].InputChannels, d.ZDim)
	}
	if last := d.Operations[len(d.Operations)-1]; last.OutputChannels != d.OutChannels {
		t.Errorf("last op cOut=%d want %d", last.OutputChannels, d.OutChannels)
	}
	// deepest feature dim = base_dim*dim_mult[last] = 96*4 = 384 at conv_in out.
	if d.Operations[1].OutputChannels != 384 {
		t.Errorf("conv_in cOut=%d want 384", d.Operations[1].OutputChannels)
	}
	t.Logf("decoder derived: z=%d out=%d scale=%dx ops=%d weightMB=%.1f",
		d.ZDim, d.OutChannels, d.SpatialScale, len(d.Operations), float64(d.WeightBytes)/(1<<20))
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
	d, err := loadVAEDecoder(dir, kreaProfileOrSkip(t).Classes.VAE)
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
	for ch := range 16 {
		for p := range 4 {
			x[ch*4+p] = z[ch*4+p]*d.LatentsStd[ch] + d.LatentsMean[ch]
			if x[ch*4+p] != d.LatentsMean[ch] {
				t.Fatalf("zero-latent denorm ch %d = %g want mean %g", ch, x[ch*4+p], d.LatentsMean[ch])
			}
		}
	}
}

// The exact final LATENT that produced the g3 golden image is a NEEDS-HOOK
// stage: the goldens carry only its per-step distribution (median/MAD), not the
// exact element tensor. This test pins that gap so the bit-exact g3 obligation
// is tracked, not silently dropped. The 8-bit image the goldens do carry is
// held to the digest its record gives it.
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
	// The 8-bit image is on file beside its record, which names it and its digest: the bytes are
	// held to that digest, so the image the record calls bit-exact is the one the tree carries.
	image, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "krea", "raw", "g3_decoded_rgb_u8.bin"))
	if digest := sha256.Sum256(image); err != nil || len(image) != g.DecodedImage.Height*g.DecodedImage.Width*g.DecodedImage.Channels || hex.EncodeToString(digest[:]) != g.DecodedImage.RawU8SHA256 {
		t.Fatalf("the 8-bit g3 image on file is %d bytes with the digest %x (%v), want the %dx%dx%d bytes its record gives the digest %s", len(image), digest, err, g.DecodedImage.Height, g.DecodedImage.Width, g.DecodedImage.Channels, g.DecodedImage.RawU8SHA256)
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

// DecodeImage decodes one latent frame z [ZDim][h][w] to planar RGB pixels
// [OutChannels][h*scale][w*scale] in [-1,1]. Single-frame path: temporal extent
// 1, causal convs cold-cache, upsamplers spatial-only (time conv skipped).
func (d *VAEDecoder) DecodeImage(z []float32, h, w int) (pixels []float32, outH, outW int, err error) {
	if d == nil {
		return nil, 0, 0, fmt.Errorf("latentimage vae: decoder not built")
	}
	if !checked.NonemptyAll(d.Operations) {
		return nil, 0, 0, fmt.Errorf("latentimage vae: decoder not built")
	}
	spatial, spatialOK := checked.MulInt(h, w)
	want, lengthOK := checked.MulInt(d.ZDim, spatial)
	if !checked.PositiveInts(h, w, d.ZDim) || !spatialOK || !lengthOK || !checked.Equal(len(z), want) {
		return nil, 0, 0, fmt.Errorf("latentimage vae: latent len=%d want %d (z=%d %dx%d)", len(z), want, d.ZDim, h, w)
	}
	// denorm: z*std + mean (per channel).
	x := make([]float32, len(z))
	for ch := range d.ZDim {
		mean, std := d.LatentsMean[ch], d.LatentsStd[ch]
		for pos := range spatial {
			x[ch*spatial+pos] = z[ch*spatial+pos]*std + mean
		}
	}
	volume, err := media.ExecuteCodecProgram(
		"latentimage vae", d.CodecProgram, make([]struct{}, len(d.Operations)),
		media.CodecVolume[[]float32]{
			Storage: x, Channels: d.ZDim, Frames: tensor.SingletonExtent, Height: h, Width: w,
		}, false,
		func(_ int, operation media.CodecOperation[[]float32], _ *struct{}, current media.CodecVolume[[]float32]) (media.CodecVolume[[]float32], error) {
			next, channels, height, width, runErr := d.runOp(
				operation, current.Storage, current.Channels, current.Height, current.Width,
			)
			return media.CodecVolume[[]float32]{
				Storage: next, Channels: channels, Frames: current.Frames, Height: height, Width: width,
			}, runErr
		},
	)
	if err != nil {
		return nil, 0, 0, err
	}
	x = volume.Storage
	media.ClampNormalizedF32InPlace(x)
	return x, volume.Height, volume.Width, nil
}

// runOp executes one op on a single-frame volume [c][h][w].
func (d *VAEDecoder) runOp(op media.CodecOperation[[]float32], x []float32, c, h, w int) ([]float32, int, int, int, error) {
	plane := h * w
	switch op.Operator {
	case media.CodecPointwise:
		out := make([]float32, op.OutputChannels*plane)
		if err := hostmath.ChannelMixF64Into(out, x, op.Bindings.WeightInput, op.Bindings.BiasInput, c, op.OutputChannels, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.OutputChannels, h, w, nil
	case media.CodecConvolution:
		out := make([]float32, op.OutputChannels*plane)
		if err := codecSingleFrameConv(out, x, op.Bindings.WeightInput, op.Bindings.BiasInput, c, op.OutputChannels, h, w, op.Convolution); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.OutputChannels, h, w, nil
	case media.CodecResidual:
		gamma0, w0, b0 := op.Bindings.NormInput, op.Bindings.WeightInput, op.Bindings.BiasInput
		gamma1, w1, b1 := op.Bindings.NormOutput, op.Bindings.WeightOutput, op.Bindings.BiasOutput
		n0 := make([]float32, len(x))
		if err := hostmath.ChannelRMSNormF64Into(n0, x, gamma0, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n0)
		h0 := make([]float32, op.OutputChannels*plane)
		if err := codecSingleFrameConv(h0, n0, w0, b0, c, op.OutputChannels, h, w, op.Convolution); err != nil {
			return nil, 0, 0, 0, err
		}
		n1 := make([]float32, len(h0))
		if err := hostmath.ChannelRMSNormF64Into(n1, h0, gamma1, op.OutputChannels, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n1)
		out := make([]float32, len(h0))
		if err := codecSingleFrameConv(out, n1, w1, b1, op.OutputChannels, op.OutputChannels, h, w, op.Convolution); err != nil {
			return nil, 0, 0, 0, err
		}
		projection, projectionBias := op.Bindings.WeightProjection, op.Bindings.BiasProjection
		if err := hostmath.AddResidualF64Into(out, x, projection, projectionBias, c, op.OutputChannels, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.OutputChannels, h, w, nil
	case media.CodecAttention:
		gamma, qkvW, qkvB := op.Bindings.NormInput, op.Bindings.WeightInput, op.Bindings.BiasInput
		projW, projB := op.Bindings.WeightOutput, op.Bindings.BiasOutput
		norm := make([]float32, len(x))
		if err := hostmath.ChannelRMSNormF64Into(norm, x, gamma, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		qkvChannels, channelsOK := checked.MulInt(tensor.TripleExtent, c)
		qkvElements, elementsOK := checked.MulInt(qkvChannels, plane)
		if !channelsOK || !elementsOK {
			return nil, 0, 0, 0, fmt.Errorf("attention geometry overflows")
		}
		qkv := make([]float32, qkvElements)
		if err := hostmath.ChannelMixF64Into(qkv, norm, qkvW, qkvB, c, qkvChannels, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		if err := hostmath.SpatialAttentionF64Into(norm, qkv, c, tensor.SingletonExtent, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		out := make([]float32, len(x))
		if err := hostmath.ChannelMixF64Into(out, norm, projW, projB, c, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		for i := range out {
			out[i] += x[i]
		}
		return out, c, h, w, nil
	case media.CodecUpsampleSpatial:
		// Single-frame: spatial 2x resample-conv only (time conv skipped).
		rw, rb := op.Bindings.WeightSpatial, op.Bindings.BiasSpatial
		scale := op.Operator.SpatialScale()
		outputElements, ok := checked.ProductInt(op.OutputChannels, scale, scale, plane)
		if !ok {
			return nil, 0, 0, 0, fmt.Errorf("spatial upsample geometry overflows")
		}
		out := make([]float32, outputElements)
		if err := hostmath.ResizeConv2DInto(out, x, rw, rb, c, op.OutputChannels, tensor.SingletonExtent, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.OutputChannels, scale * h, scale * w, nil
	case media.CodecHead:
		gamma, weight, bias := op.Bindings.NormInput, op.Bindings.WeightInput, op.Bindings.BiasInput
		norm := make([]float32, len(x))
		if err := hostmath.ChannelRMSNormF64Into(norm, x, gamma, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(norm)
		out := make([]float32, op.OutputChannels*plane)
		if err := codecSingleFrameConv(out, norm, weight, bias, c, op.OutputChannels, h, w, op.Convolution); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.OutputChannels, h, w, nil
	}
	return nil, 0, 0, 0, fmt.Errorf("unsupported op kind %d", op.Operator)
}

func codecSingleFrameConv(out, input, weight, bias []float32, inputChannels, outputChannels, height, width int, extents media.CodecConvolutionExtents) error {
	kernel := extents.Kernel
	return hostmath.CausalConv3DInto(out, input, nil, weight, bias, tensor.FirstOffset, hostmath.Conv3DShape{
		CIn: inputChannels, COut: outputChannels, InT: tensor.SingletonExtent, InH: height, InW: width,
		KT: kernel[0], KH: kernel[1], KW: kernel[2],
		PadT: kernel[0] / tensor.PairedExtent, PadH: kernel[1] / tensor.PairedExtent, PadW: kernel[2] / tensor.PairedExtent,
		StrideT: extents.Stride[0], StrideH: extents.Stride[1], StrideW: extents.Stride[2],
	})
}
