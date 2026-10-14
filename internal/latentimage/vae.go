// AutoencoderKLQwenImage spatial VAE decode (host port). The Krea/Qwen-Image
// VAE is the Wan-family 3-D causal codec (internal/latentvideo/vae.go) fed a
// single latent frame; for image decode the temporal upsample follows the
// reference first-chunk "Rep" convention (the time conv is skipped, so each
// upsampler is a pure spatial 2x resample-conv). The decode graph is DERIVED
// from the diffusers-named safetensors tensor shapes: post_quant_conv ->
// decoder.conv_in -> mid_block(resnet,attn,resnet) -> up_blocks.N(resnets,
// upsampler) -> norm_out+SiLU -> conv_out. Ops accumulate in f64 on the
// golden-verified hostmath conv3d/resize primitives; the RMS-norm/attention/
// pointwise ops mirror internal/latentvideo verbatim. Output clamps to [-1,1].
//
// NOTE ON VERIFICATION: the g3 golden image is bit-exact, but the exact final
// LATENT that produced it is a NEEDS-HOOK stage (not in the goldens -- only its
// per-step median/MAD distribution is). Bit-exact g3 reproduction therefore
// requires the denoiser port (to regenerate the latent) or a latent-dump hook;
// see DecodeImage / TestVAE* for what is verifiable now.

package latentimage

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// VAEDecoder: a compiled + weight-resident QwenImage spatial decoder.
type VAEDecoder struct {
	media.CodecProgram[[]float32]
	ZDim         int
	OutChannels  int
	SpatialScale int // derived 2^(upsampler count); 8 for dim_mult [1,2,4,4]
	LatentsMean  []float32
	LatentsStd   []float32
	WeightBytes  int64
}

// vaeConfigJSON: the fields DecodeImage derives structure/denorm from.
type vaeConfigJSON struct {
	ClassName   string    `json:"_class_name"`
	BaseDim     int       `json:"base_dim"`
	DimMult     []int     `json:"dim_mult"`
	ZDim        int       `json:"z_dim"`
	NumResBlks  int       `json:"num_res_blocks"`
	LatentsMean []float64 `json:"latents_mean"`
	LatentsStd  []float64 `json:"latents_std"`
}

// loadVAEDecoder reads vae/config.json + vae/*.safetensors under modelDir and
// builds the weight-resident decoder, deriving the op graph from tensor shapes.
// Every decoder-side tensor must be consumed (capability-retention check).
func loadVAEDecoder(modelDir, class string) (*VAEDecoder, error) {
	vaeDir := filepath.Join(modelDir, "vae")
	var cfg vaeConfigJSON
	if err := readJSON(filepath.Join(vaeDir, "config.json"), &cfg); err != nil {
		return nil, err
	}
	if cfg.ClassName != class {
		return nil, fmt.Errorf("latentimage vae: class %q != %q", cfg.ClassName, class)
	}
	if !checked.PositiveInts(cfg.ZDim, cfg.NumResBlks) {
		return nil, fmt.Errorf("latentimage vae: bad config z=%d dim_mult=%v res=%d", cfg.ZDim, cfg.DimMult, cfg.NumResBlks)
	}
	if _, ok := checked.First(cfg.DimMult); !ok {
		return nil, fmt.Errorf("latentimage vae: bad config z=%d dim_mult=%v res=%d", cfg.ZDim, cfg.DimMult, cfg.NumResBlks)
	}
	if err := media.ValidateChannelMoments(cfg.LatentsMean, cfg.LatentsStd, cfg.ZDim); err != nil {
		return nil, fmt.Errorf("latentimage vae: latent normalization: %w", err)
	}
	weights, err := loadVAEWeights(vaeDir)
	if err != nil {
		return nil, err
	}
	d := &VAEDecoder{
		ZDim:        cfg.ZDim,
		LatentsMean: dtype.Float64SliceToFloat32(cfg.LatentsMean),
		LatentsStd:  dtype.Float64SliceToFloat32(cfg.LatentsStd),
	}
	if err := d.build(weights, cfg); err != nil {
		return nil, err
	}
	return d, nil
}

func loadVAEWeights(vaeDir string) (map[string][]float32, error) {
	src, err := safetensors.OpenSource(vaeDir)
	if err != nil {
		return nil, fmt.Errorf("latentimage vae: open weights: %w", err)
	}
	defer src.Close()
	out := make(map[string][]float32, len(src.Tensors))
	for name, tensor := range src.Tensors {
		// Only the decode path needs decoder.* and post_quant_conv.*; skip the
		// encoder and quant_conv (encode side) to keep host residency small.
		if !strings.HasPrefix(name, "decoder.") && !strings.HasPrefix(name, "post_quant_conv.") {
			continue
		}
		reader, err := safetensors.F32Reader(tensor)
		if err != nil {
			return nil, fmt.Errorf("latentimage vae: tensor %q: %w", name, err)
		}
		elements := tensor.Elements()
		raw := make([]byte, elements*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return nil, fmt.Errorf("latentimage vae: tensor %q payload: %w", name, err)
		}
		values := make([]float32, elements)
		for i := range values {
			bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
			values[i] = math.Float32frombits(bits)
		}
		out[name] = values
	}
	return out, nil
}

// build derives the op graph from tensor presence/shapes and binds weights.
func (d *VAEDecoder) build(w map[string][]float32, cfg vaeConfigJSON) error {
	consumed := map[string]bool{}
	need := func(name string) ([]float32, error) {
		v, ok := w[name]
		if !ok {
			return nil, fmt.Errorf("latentimage vae: missing tensor %s", name)
		}
		consumed[name] = true
		return v, nil
	}
	add := func(kind media.CodecOperator, prefix string, cIn, cOut int, names ...string) error {
		op := media.NewCodecOperation[[]float32](kind, prefix, cIn, cOut)
		values := make([][]float32, 0, len(names))
		for _, n := range names {
			v, err := need(n)
			if err != nil {
				return err
			}
			values = append(values, v)
			d.WeightBytes += int64(len(v)) * 4
		}
		var err error
		op.Bindings, err = media.BindCodecWeights(kind, op.RequiresProjection(), values)
		if err != nil {
			return fmt.Errorf("latentimage vae: %s: %w", prefix, err)
		}
		d.Operations = append(d.Operations, op)
		return nil
	}

	// post_quant_conv: pointwise z->z.
	if err := add(media.CodecPointwise, "post_quant_conv", cfg.ZDim, cfg.ZDim,
		"post_quant_conv.weight", "post_quant_conv.bias"); err != nil {
		return err
	}
	// decoder.conv_in: z -> deepest feature dim (base*dim_mult[last]).
	deepestMultiplier, ok := checked.Last(cfg.DimMult)
	if !ok {
		return fmt.Errorf("latentimage vae: empty channel multiplier program")
	}
	deepest, ok := checked.MulInt(cfg.BaseDim, deepestMultiplier)
	if !ok {
		return fmt.Errorf("latentimage vae: deepest channel count overflows")
	}
	if err := add(media.CodecConvolution, "decoder.conv_in", cfg.ZDim, deepest,
		"decoder.conv_in.weight", "decoder.conv_in.bias"); err != nil {
		return err
	}
	channels := deepest

	addResnet := func(prefix string, cOut int) error {
		names := media.ResidualBindings([]string{
			prefix + ".norm1.gamma", prefix + ".conv1.weight", prefix + ".conv1.bias",
			prefix + ".norm2.gamma", prefix + ".conv2.weight", prefix + ".conv2.bias",
		}, channels, cOut, prefix+".conv_shortcut.weight", prefix+".conv_shortcut.bias")
		if err := add(media.CodecResidual, prefix, channels, cOut, names...); err != nil {
			return err
		}
		channels = cOut
		return nil
	}

	// mid_block: resnet, attention, resnet (all at the deepest dim).
	if err := addResnet("decoder.mid_block.resnets.0", channels); err != nil {
		return err
	}
	if err := add(media.CodecAttention, "decoder.mid_block.attentions.0", channels, channels,
		"decoder.mid_block.attentions.0.norm.gamma",
		"decoder.mid_block.attentions.0.to_qkv.weight", "decoder.mid_block.attentions.0.to_qkv.bias",
		"decoder.mid_block.attentions.0.proj.weight", "decoder.mid_block.attentions.0.proj.bias"); err != nil {
		return err
	}
	if err := addResnet("decoder.mid_block.resnets.1", channels); err != nil {
		return err
	}

	// up_blocks: dim_mult reversed. resnets = num_res_blocks+1; an upsampler on
	// all but the last block. The output dim of block b is base*dim_mult[b]
	// (reversed order), matching the encoder mirror.
	nBlocks := len(cfg.DimMult)
	d.SpatialScale = tensor.SingletonExtent
	residualStages, ok := checked.AddInt(cfg.NumResBlks, tensor.SingletonExtent)
	if !ok {
		return fmt.Errorf("latentimage vae: residual stage count overflows")
	}
	for b := range nBlocks {
		mult := cfg.DimMult[nBlocks-tensor.SingletonExtent-b]
		blockOut, ok := checked.MulInt(cfg.BaseDim, mult)
		if !ok {
			return fmt.Errorf("latentimage vae: block channel count overflows")
		}
		for r := range residualStages {
			prefix := fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", b, r)
			if err := addResnet(prefix, blockOut); err != nil {
				return err
			}
		}
		up := fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0", b)
		if _, ok := w[up+".resample.1.weight"]; ok {
			// cOut derived from the resample bias length (== weight[0]).
			cOut := len(w[up+".resample.1.bias"])
			names := []string{up + ".resample.1.weight", up + ".resample.1.bias"}
			if _, hasTime := w[up+".time_conv.weight"]; hasTime {
				for _, name := range []string{up + ".time_conv.weight", up + ".time_conv.bias"} {
					if _, err := need(name); err != nil {
						return err
					}
				}
			}
			if err := add(media.CodecUpsampleSpatial, up, channels, cOut, names...); err != nil {
				return err
			}
			channels = cOut
			d.SpatialScale, ok = checked.MulInt(d.SpatialScale, media.CodecUpsampleSpatial.SpatialScale())
			if !ok {
				return fmt.Errorf("latentimage vae: spatial scale overflows")
			}
		}
	}

	// head: norm_out + SiLU + conv_out. cOut derived from conv_out bias.
	outCh := len(w["decoder.conv_out.bias"])
	if !checked.PositiveInts(outCh) {
		return fmt.Errorf("latentimage vae: missing decoder.conv_out.bias")
	}
	if err := add(media.CodecHead, "decoder.head", channels, outCh,
		"decoder.norm_out.gamma", "decoder.conv_out.weight", "decoder.conv_out.bias"); err != nil {
		return err
	}
	d.OutChannels = outCh

	if err := d.CodecProgram.Validate("latentimage vae"); err != nil {
		return err
	}
	// capability-retention: every decoder tensor consumed.
	var unconsumed []string
	for name := range w {
		if !consumed[name] {
			unconsumed = append(unconsumed, name)
		}
	}
	if len(unconsumed) > 0 {
		return fmt.Errorf("latentimage vae: %d unconsumed decoder tensors, e.g. %s", len(unconsumed), unconsumed[0])
	}
	return nil
}
