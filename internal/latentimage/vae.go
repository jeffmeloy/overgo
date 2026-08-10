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

	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
)

// vaeNormZeroGuard: channel-norm zero-column floor (reference constant, mirrors
// latentvideo channelNormZeroGuard).
const vaeNormZeroGuard = 1e-12

// vaeOp kinds for the derived decode graph.
type vaeOpKind int

const (
	vaePointwise vaeOpKind = iota // 1x1x1 conv (post_quant_conv)
	vaeConv                       // kt=kh=kw=3 causal conv (conv_in)
	vaeResnet                     // norm1,SiLU,conv1,norm2,SiLU,conv2 (+shortcut)
	vaeAttention                  // norm, qkv, spatial self-attn, proj (+residual)
	vaeUpsample                   // spatial 2x resample-conv (time conv skipped, single frame)
	vaeHead                       // norm_out, SiLU, conv_out
)

// vaeOp: one decode operation with resident f32 weights in consume order.
type vaeOp struct {
	kind      vaeOpKind
	prefix    string
	cIn, cOut int
	weights   [][]float32 // op-specific binding order (see build/run)
}

// VAEDecoder: a compiled + weight-resident QwenImage spatial decoder.
type VAEDecoder struct {
	ZDim         int
	OutChannels  int
	SpatialScale int // derived 2^(upsampler count); 8 for dim_mult [1,2,4,4]
	LatentsMean  []float32
	LatentsStd   []float32
	ops          []vaeOp
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

// LoadVAEDecoder reads vae/config.json + vae/*.safetensors under modelDir and
// builds the weight-resident decoder, deriving the op graph from tensor shapes.
// Every decoder-side tensor must be consumed (capability-retention check).
func LoadVAEDecoder(modelDir string) (*VAEDecoder, error) {
	vaeDir := filepath.Join(modelDir, "vae")
	var cfg vaeConfigJSON
	if err := readJSON(filepath.Join(vaeDir, "config.json"), &cfg); err != nil {
		return nil, err
	}
	if cfg.ClassName != VAEClass {
		return nil, fmt.Errorf("latentimage vae: class %q != %q", cfg.ClassName, VAEClass)
	}
	if cfg.ZDim <= 0 || len(cfg.DimMult) == 0 || cfg.NumResBlks <= 0 {
		return nil, fmt.Errorf("latentimage vae: bad config z=%d dim_mult=%v res=%d", cfg.ZDim, cfg.DimMult, cfg.NumResBlks)
	}
	if len(cfg.LatentsMean) != cfg.ZDim || len(cfg.LatentsStd) != cfg.ZDim {
		return nil, fmt.Errorf("latentimage vae: latents_mean/std len (%d/%d) != z_dim %d", len(cfg.LatentsMean), len(cfg.LatentsStd), cfg.ZDim)
	}
	weights, err := loadVAEWeights(vaeDir)
	if err != nil {
		return nil, err
	}
	d := &VAEDecoder{
		ZDim:        cfg.ZDim,
		LatentsMean: f64sToF32(cfg.LatentsMean),
		LatentsStd:  f64sToF32(cfg.LatentsStd),
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
	add := func(kind vaeOpKind, prefix string, cIn, cOut int, names ...string) error {
		op := vaeOp{kind: kind, prefix: prefix, cIn: cIn, cOut: cOut}
		for _, n := range names {
			v, err := need(n)
			if err != nil {
				return err
			}
			op.weights = append(op.weights, v)
			d.WeightBytes += int64(len(v)) * 4
		}
		d.ops = append(d.ops, op)
		return nil
	}

	// post_quant_conv: pointwise z->z.
	if err := add(vaePointwise, "post_quant_conv", cfg.ZDim, cfg.ZDim,
		"post_quant_conv.weight", "post_quant_conv.bias"); err != nil {
		return err
	}
	// decoder.conv_in: z -> deepest feature dim (base*dim_mult[last]).
	deepest := cfg.BaseDim * cfg.DimMult[len(cfg.DimMult)-1]
	if err := add(vaeConv, "decoder.conv_in", cfg.ZDim, deepest,
		"decoder.conv_in.weight", "decoder.conv_in.bias"); err != nil {
		return err
	}
	channels := deepest

	addResnet := func(prefix string, cOut int) error {
		names := []string{
			prefix + ".norm1.gamma", prefix + ".conv1.weight", prefix + ".conv1.bias",
			prefix + ".norm2.gamma", prefix + ".conv2.weight", prefix + ".conv2.bias",
		}
		if channels != cOut {
			names = append(names, prefix+".conv_shortcut.weight", prefix+".conv_shortcut.bias")
		}
		if err := add(vaeResnet, prefix, channels, cOut, names...); err != nil {
			return err
		}
		channels = cOut
		return nil
	}

	// mid_block: resnet, attention, resnet (all at the deepest dim).
	if err := addResnet("decoder.mid_block.resnets.0", channels); err != nil {
		return err
	}
	if err := add(vaeAttention, "decoder.mid_block.attentions.0", channels, channels,
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
	d.SpatialScale = 1
	for b := 0; b < nBlocks; b++ {
		mult := cfg.DimMult[nBlocks-1-b]
		blockOut := cfg.BaseDim * mult
		for r := 0; r <= cfg.NumResBlks; r++ {
			prefix := fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", b, r)
			if err := addResnet(prefix, blockOut); err != nil {
				return err
			}
		}
		up := fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0", b)
		if _, ok := w[up+".resample.1.weight"]; ok {
			// cOut derived from the resample bias length (== weight[0]).
			cOut := len(w[up+".resample.1.bias"])
			// time_conv is unused for single-frame image decode (the reference
			// first-chunk "Rep" convention skips it) but must be marked consumed
			// for the capability-retention check.
			names := []string{up + ".resample.1.weight", up + ".resample.1.bias"}
			if _, hasTime := w[up+".time_conv.weight"]; hasTime {
				names = append(names, up+".time_conv.weight", up+".time_conv.bias")
			}
			if err := add(vaeUpsample, up, channels, cOut, names...); err != nil {
				return err
			}
			channels = cOut
			d.SpatialScale *= 2
		}
	}

	// head: norm_out + SiLU + conv_out. cOut derived from conv_out bias.
	outCh := len(w["decoder.conv_out.bias"])
	if outCh <= 0 {
		return fmt.Errorf("latentimage vae: missing decoder.conv_out.bias")
	}
	if err := add(vaeHead, "decoder.head", channels, outCh,
		"decoder.norm_out.gamma", "decoder.conv_out.weight", "decoder.conv_out.bias"); err != nil {
		return err
	}
	d.OutChannels = outCh

	// chain continuity.
	for i := 1; i < len(d.ops); i++ {
		if d.ops[i].cIn != d.ops[i-1].cOut {
			return fmt.Errorf("latentimage vae: op %d (%s) cIn=%d != prev cOut=%d", i, d.ops[i].prefix, d.ops[i].cIn, d.ops[i-1].cOut)
		}
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

// DecodeImage decodes one latent frame z [ZDim][h][w] to planar RGB pixels
// [OutChannels][h*scale][w*scale] in [-1,1]. Single-frame path: temporal extent
// 1, causal convs cold-cache, upsamplers spatial-only (time conv skipped).
func (d *VAEDecoder) DecodeImage(z []float32, h, w int) (pixels []float32, outH, outW int, err error) {
	if d == nil || len(d.ops) == 0 {
		return nil, 0, 0, fmt.Errorf("latentimage vae: decoder not built")
	}
	spatial := h * w
	if h <= 0 || w <= 0 || len(z) != d.ZDim*spatial {
		return nil, 0, 0, fmt.Errorf("latentimage vae: latent len=%d want %d (z=%d %dx%d)", len(z), d.ZDim*spatial, d.ZDim, h, w)
	}
	// denorm: z*std + mean (per channel).
	x := make([]float32, len(z))
	for ch := 0; ch < d.ZDim; ch++ {
		mean, std := d.LatentsMean[ch], d.LatentsStd[ch]
		for pos := 0; pos < spatial; pos++ {
			x[ch*spatial+pos] = z[ch*spatial+pos]*std + mean
		}
	}
	ch, ch_h, ch_w := d.ZDim, h, w
	for i := range d.ops {
		x, ch, ch_h, ch_w, err = d.runOp(d.ops[i], x, ch, ch_h, ch_w)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("latentimage vae: %s: %w", d.ops[i].prefix, err)
		}
	}
	for i, v := range x {
		switch {
		case v < -1:
			x[i] = -1
		case v > 1:
			x[i] = 1
		}
	}
	return x, ch_h, ch_w, nil
}

// runOp executes one op on a single-frame volume [c][h][w].
func (d *VAEDecoder) runOp(op vaeOp, x []float32, c, h, w int) ([]float32, int, int, int, error) {
	plane := h * w
	switch op.kind {
	case vaePointwise:
		out := make([]float32, op.cOut*plane)
		if err := vaePointwiseInto(out, x, op.weights[0], op.weights[1], c, op.cOut, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.cOut, h, w, nil
	case vaeConv:
		out := make([]float32, op.cOut*plane)
		if err := vaeCausalConv(out, x, op.weights[0], op.weights[1], c, op.cOut, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.cOut, h, w, nil
	case vaeResnet:
		gamma0, w0, b0 := op.weights[0], op.weights[1], op.weights[2]
		gamma1, w1, b1 := op.weights[3], op.weights[4], op.weights[5]
		n0 := make([]float32, len(x))
		if err := vaeChannelRMSNorm(n0, x, gamma0, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n0)
		h0 := make([]float32, op.cOut*plane)
		if err := vaeCausalConv(h0, n0, w0, b0, c, op.cOut, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		n1 := make([]float32, len(h0))
		if err := vaeChannelRMSNorm(n1, h0, gamma1, op.cOut, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n1)
		out := make([]float32, len(h0))
		if err := vaeCausalConv(out, n1, w1, b1, op.cOut, op.cOut, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		if op.cIn == op.cOut {
			for i := range out {
				out[i] += x[i]
			}
			return out, op.cOut, h, w, nil
		}
		shortcut := make([]float32, len(out))
		if err := vaePointwiseInto(shortcut, x, op.weights[6], op.weights[7], c, op.cOut, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		for i := range out {
			out[i] += shortcut[i]
		}
		return out, op.cOut, h, w, nil
	case vaeAttention:
		gamma, qkvW, qkvB := op.weights[0], op.weights[1], op.weights[2]
		projW, projB := op.weights[3], op.weights[4]
		norm := make([]float32, len(x))
		if err := vaeChannelRMSNorm(norm, x, gamma, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		qkv := make([]float32, 3*c*plane)
		if err := vaePointwiseInto(qkv, norm, qkvW, qkvB, c, 3*c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		if err := vaeSpatialAttention(norm, qkv, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		out := make([]float32, len(x))
		if err := vaePointwiseInto(out, norm, projW, projB, c, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		for i := range out {
			out[i] += x[i]
		}
		return out, c, h, w, nil
	case vaeUpsample:
		// Single-frame: spatial 2x resample-conv only (time conv skipped).
		rw, rb := op.weights[0], op.weights[1]
		out := make([]float32, op.cOut*4*plane)
		if err := hostmath.ResizeConv2DInto(out, x, rw, rb, c, op.cOut, 1, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.cOut, 2 * h, 2 * w, nil
	case vaeHead:
		gamma, weight, bias := op.weights[0], op.weights[1], op.weights[2]
		norm := make([]float32, len(x))
		if err := vaeChannelRMSNorm(norm, x, gamma, c, plane); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(norm)
		out := make([]float32, op.cOut*plane)
		if err := vaeCausalConv(out, norm, weight, bias, c, op.cOut, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, op.cOut, h, w, nil
	}
	return nil, 0, 0, 0, fmt.Errorf("unsupported op kind %d", op.kind)
}

// vaeCausalConv: kt=kh=kw=3 causal 3-D conv on a single frame (InT=1, cold
// cache), delegating to the golden-verified hostmath primitive.
func vaeCausalConv(out, x, weight, bias []float32, cIn, cOut, h, w int) error {
	shape := hostmath.Conv3DShape{
		CIn: cIn, COut: cOut, InT: 1, InH: h, InW: w,
		KT: 3, KH: 3, KW: 3, PadT: 1, PadH: 1, PadW: 1,
		StrideT: 1, StrideH: 1, StrideW: 1,
	}
	return hostmath.CausalConv3DInto(out, x, nil, weight, bias, 0, shape)
}

// vaePointwiseInto: 1x1(x1) channel mix at every spatial position.
func vaePointwiseInto(out, x, weight, bias []float32, cIn, cOut, plane int) error {
	if len(x) != cIn*plane || len(out) != cOut*plane {
		return fmt.Errorf("vae pointwise: bad lengths out=%d x=%d", len(out), len(x))
	}
	if len(weight) != cOut*cIn {
		return fmt.Errorf("vae pointwise: weight len=%d want %d", len(weight), cOut*cIn)
	}
	if bias != nil && len(bias) != cOut {
		return fmt.Errorf("vae pointwise: bias len=%d want %d", len(bias), cOut)
	}
	hostmath.ParallelRangeF64(cOut, cIn*plane, func(coLo, coHi int) {
		for co := coLo; co < coHi; co++ {
			for pos := 0; pos < plane; pos++ {
				acc := float64(0)
				if bias != nil {
					acc = float64(bias[co])
				}
				for ci := 0; ci < cIn; ci++ {
					acc += float64(x[ci*plane+pos]) * float64(weight[co*cIn+ci])
				}
				out[co*plane+pos] = float32(acc)
			}
		}
	})
	return nil
}

// vaeChannelRMSNorm: RMS over channels at each position, sqrt(C)-scaled,
// zero-guarded (reference F.normalize(dim=1)*sqrt(C)*gamma).
func vaeChannelRMSNorm(out, x, gamma []float32, c, plane int) error {
	if c <= 0 || plane <= 0 || len(x) != c*plane || len(out) != len(x) || len(gamma) != c {
		return fmt.Errorf("vae rms norm: bad shape c=%d plane=%d len(x)=%d len(gamma)=%d", c, plane, len(x), len(gamma))
	}
	scale := math.Sqrt(float64(c))
	for pos := 0; pos < plane; pos++ {
		var sumSq float64
		for ch := 0; ch < c; ch++ {
			v := float64(x[ch*plane+pos])
			sumSq += v * v
		}
		norm := math.Sqrt(sumSq)
		if norm < vaeNormZeroGuard {
			norm = vaeNormZeroGuard
		}
		for ch := 0; ch < c; ch++ {
			idx := ch*plane + pos
			out[idx] = float32(float64(x[idx]) / norm * scale * float64(gamma[ch]))
		}
	}
	return nil
}

// vaeSpatialAttention: single-frame spatial self-attention over qkv [3c][h*w],
// writing the attended values (residual added by the caller).
func vaeSpatialAttention(out, qkv []float32, c, plane int) error {
	if c <= 0 || plane <= 0 || len(qkv) != 3*c*plane || len(out) != c*plane {
		return fmt.Errorf("vae attention: bad lengths out=%d qkv=%d", len(out), len(qkv))
	}
	scale := 1 / math.Sqrt(float64(c))
	kOff := c * plane
	vOff := 2 * c * plane
	hostmath.ParallelRangeF64(plane, plane*c*2, func(lo, hi int) {
		scores := make([]float32, plane)
		for qi := lo; qi < hi; qi++ {
			for kj := 0; kj < plane; kj++ {
				var dot float64
				for ch := 0; ch < c; ch++ {
					dot += float64(qkv[ch*plane+qi]) * float64(qkv[kOff+ch*plane+kj])
				}
				scores[kj] = float32(dot * scale)
			}
			hostmath.SoftmaxInPlace(scores)
			for ch := 0; ch < c; ch++ {
				var acc float64
				for kj, p := range scores {
					acc += float64(p) * float64(qkv[vOff+ch*plane+kj])
				}
				out[ch*plane+qi] = float32(acc)
			}
		}
	})
	return nil
}

// PixelsToU8 converts a planar [C][H][W] tensor in [-1,1] to the g3 8-bit
// image convention: u8 = round(((x+1)/2 clamped to [0,1]) * 255).
func PixelsToU8(pixels []float32) []uint8 {
	out := make([]uint8, len(pixels))
	for i, v := range pixels {
		p := (float64(v) + 1) * 0.5
		if p < 0 {
			p = 0
		} else if p > 1 {
			p = 1
		}
		out[i] = uint8(math.Round(p * 255))
	}
	return out
}

func f64sToF32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
