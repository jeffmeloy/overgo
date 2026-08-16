// AutoencoderKLQwenImage spatial VAE decode routed through the shared tensor
// graph. The SAME graph definition executes on both the reference backend (host
// golden) and the CUDA generic executor, so THIS file is the device port of the
// VAE decode -- mirroring the host reference VAEDecoder.DecodeImage (vae.go)
// op-for-op, from cataloged ops, with no bespoke CUDA family file. It is the
// direct analogue of denoiser_program.go / encoder_program.go for the media
// codec spine.
//
// Layout: the graph runs channel-INNERMOST (HWC physical) -- the convention the
// cataloged Conv2D and MulMat already use (input [channelsIn, width, height]
// with channels the fastest axis; a torch Linear weight [out,in] declared as
// shape (in,out)). vae.go is channel-OUTERMOST (CHW), so DecodeGraph transposes
// the latent CHW->HWC on the way in and the pixels HWC->CHW on the way out; the
// interior is pure graph.
//
// Op mapping (host vae.go op -> cataloged op), all f64 on the reference backend
// exactly like vae.go, f32 on the CUDA executor (fp32-fma, matching adaptive's
// spatial_vae_*_f32 golden path):
//   - post_quant / qkv / proj / shortcut 1x1  -> MulMat + broadcast Add(bias).
//   - kt=kh=kw=3 single-frame causal conv      -> Conv2D 3x3 pad1: the causal
//     time pad places all 2*PadT padding left, so for InT=1 ONLY the last
//     temporal tap (kt=2) contributes; the graph feeds that pre-sliced 3x3 tap.
//   - channel RMS norm x/max(|x|_2,1e-12)*sqrt(C)*gamma -> L2Norm(eps=1e-12) over
//     Dims[0] (== vaeChannelRMSNorm's exact max-guard form) * sqrt(C) * gamma.
//   - SiLU                                       -> SiLU.
//   - spatial self-attention (single head)       -> GroupSlice qkv + Attention
//     (heads=1, headDim=C, scale=1/sqrt(C), non-causal).
//   - nearest-2x upsample + 3x3 same conv        -> RepeatHeads x2 (a pure
//     data-preserving gather, bit-exact on every backend) + Conv2D 3x3 pad1.
//   - final clamp [-1,1]                          -> Clamp.
package latentimage

import (
	"fmt"
	"math"

	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// VAEProgram: the compiled QwenImage spatial decode graph for one latent
// geometry, plus the static weight feeds prepared (and temporally sliced) at
// compile time. Latent is the per-decode HWC input; Output is planar-in-HWC RGB
// before the CHW transpose DecodeGraph performs.
type VAEProgram struct {
	ZDim         int
	OutChannels  int
	SpatialScale int
	H, W         int // latent extent
	OutH, OutW   int // pixel extent

	Latent *tensor.Tensor
	Output *tensor.Tensor

	feeds      []vaeFeed
	latentMean []float32
	latentStd  []float32
}

// vaeFeed binds a graph Input node to its static weight payload.
type vaeFeed struct {
	node *tensor.Tensor
	data []float32
}

// vaeGraphBuilder threads the builder + accumulating feeds through the op walk.
type vaeGraphBuilder struct {
	b     *tensor.Builder
	feeds *[]vaeFeed
	n     int // unique-name counter
}

// weight declares an F32 Input node bound to data, recording the feed. shape
// dims follow the cataloged convention for the consuming op.
func (g *vaeGraphBuilder) weight(data []float32, dims ...uint64) *tensor.Tensor {
	g.n++
	node := g.b.Input(fmt.Sprintf("vae_w_%d", g.n), dtype.F32, tensor.MustShape(dims...))
	*g.feeds = append(*g.feeds, vaeFeed{node: node, data: data})
	return node
}

// CompileVAEProgram builds the decode graph for a latent of extent h x w from
// an already-derived, weight-resident VAEDecoder (vae.go loadVAEDecoder). The
// op list, channel chain and every weight come from d, so the graph and the
// host reference cannot drift. matmulType selects rank-2 weight storage; only
// dtype.F32 is supported here (the host-feed exact-parity path -- the VAE decode
// is one-shot so there is no BF16 resident-weight variant).
func CompileVAEProgram(d *VAEDecoder, h, w int, matmulType dtype.Type) (*VAEProgram, error) {
	if d == nil || len(d.Operations) == 0 {
		return nil, fmt.Errorf("vae program: decoder not built")
	}
	if h <= 0 || w <= 0 {
		return nil, fmt.Errorf("vae program: bad latent extent %dx%d", w, h)
	}
	if matmulType != dtype.F32 {
		return nil, fmt.Errorf("vae program: matmul weight type %s unsupported (host-feed exact path only)", matmulType)
	}

	p := &VAEProgram{
		ZDim: d.ZDim, OutChannels: d.OutChannels, SpatialScale: d.SpatialScale, H: h, W: w,
		latentMean: append([]float32(nil), d.LatentsMean...),
		latentStd:  append([]float32(nil), d.LatentsStd...),
	}
	b := tensor.NewBuilder()
	g := &vaeGraphBuilder{b: b, feeds: &p.feeds}

	// Latent feed: denormalized HWC columns [ZDim, h*w] (DecodeGraph prepares the
	// z*std+mean affine + CHW->HWC transpose, exactly as vae.go denorms before
	// the op loop).
	p.Latent = b.Input("vae_latent", dtype.F32, tensor.MustShape(uint64(d.ZDim), uint64(h*w)))

	x := p.Latent
	c, ch, cw := d.ZDim, h, w
	for i := range d.Operations {
		var err error
		x, c, ch, cw, err = g.buildOp(d.Operations[i], x, c, ch, cw)
		if err != nil {
			return nil, fmt.Errorf("vae program: op %d (%s): %w", i, d.Operations[i].Name, err)
		}
	}
	p.Output = b.Clamp(x, -1, 1)
	p.OutH, p.OutW = ch, cw

	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("vae program graph: %w", err)
	}
	return p, nil
}

// buildOp emits the graph nodes for one derived op and returns the new
// activation + geometry. x is always [c, ch*cw] HWC columns.
func (g *vaeGraphBuilder) buildOp(op media.CodecOperation[[][]float32], x *tensor.Tensor, c, ch, cw int) (*tensor.Tensor, int, int, int, error) {
	b := g.b
	plane := ch * cw
	switch op.Operator {
	case media.CodecPointwise:
		out := g.pointwise(x, op.Bindings[0], op.Bindings[1], c, op.OutputChannels, plane)
		return out, op.OutputChannels, ch, cw, nil
	case media.CodecConvolution:
		out := g.conv3x3(x, op.Bindings[0], op.Bindings[1], c, op.OutputChannels, ch, cw)
		return out, op.OutputChannels, ch, cw, nil
	case media.CodecResidual:
		gamma0, w0, b0 := op.Bindings[0], op.Bindings[1], op.Bindings[2]
		gamma1, w1, b1 := op.Bindings[3], op.Bindings[4], op.Bindings[5]
		n0 := b.SiLU(g.channelNorm(x, gamma0, c))
		h0 := g.conv3x3(n0, w0, b0, c, op.OutputChannels, ch, cw)
		n1 := b.SiLU(g.channelNorm(h0, gamma1, op.OutputChannels))
		out := g.conv3x3(n1, w1, b1, op.OutputChannels, op.OutputChannels, ch, cw)
		if op.InputChannels == op.OutputChannels {
			return b.Add(out, x), op.OutputChannels, ch, cw, nil
		}
		shortcut := g.pointwise(x, op.Bindings[6], op.Bindings[7], c, op.OutputChannels, plane)
		return b.Add(out, shortcut), op.OutputChannels, ch, cw, nil
	case media.CodecAttention:
		gamma, qkvW, qkvB := op.Bindings[0], op.Bindings[1], op.Bindings[2]
		projW, projB := op.Bindings[3], op.Bindings[4]
		norm := g.channelNorm(x, gamma, c)
		qkv := b.Add(
			b.MulMat(g.weight(qkvW, uint64(c), uint64(3*c)), norm),
			g.weight(qkvB, uint64(3*c)),
		)
		cu, pu := uint64(c), uint64(plane)
		q := b.GroupSlice(qkv, 0, cu, 1, cu)    // [c,1,plane]
		k := b.GroupSlice(qkv, cu, cu, 1, cu)   // [c,1,plane]
		v := b.GroupSlice(qkv, 2*cu, cu, 1, cu) // [c,1,plane]
		scale := float32(1 / math.Sqrt(float64(c)))
		attn := b.Reshape(b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: scale, Causal: false}), cu, pu)
		out := b.Add(b.MulMat(g.weight(projW, cu, cu), attn), g.weight(projB, cu))
		return b.Add(out, x), c, ch, cw, nil
	case media.CodecUpsampleSpatial:
		out := g.upsample(x, op.Bindings[0], op.Bindings[1], c, op.OutputChannels, ch, cw)
		return out, op.OutputChannels, 2 * ch, 2 * cw, nil
	case media.CodecHead:
		gamma, weight, bias := op.Bindings[0], op.Bindings[1], op.Bindings[2]
		norm := b.SiLU(g.channelNorm(x, gamma, c))
		out := g.conv3x3(norm, weight, bias, c, op.OutputChannels, ch, cw)
		return out, op.OutputChannels, ch, cw, nil
	}
	return nil, 0, 0, 0, fmt.Errorf("unsupported op kind %d", op.Operator)
}

// pointwise: 1x1 channel mix out[cOut,plane] = W[cOut,cIn] @ x[cIn,plane] + bias.
// A torch conv1x1 weight [cOut,cIn,1,1,1] is fed flat as MulMat left (in,out).
func (g *vaeGraphBuilder) pointwise(x *tensor.Tensor, weight, bias []float32, cIn, cOut, plane int) *tensor.Tensor {
	b := g.b
	out := b.MulMat(g.weight(weight, uint64(cIn), uint64(cOut)), x)
	if bias != nil {
		out = b.Add(out, g.weight(bias, uint64(cOut)))
	}
	return out
}

// conv3x3: single-frame causal 3x3 conv. The causal InT=1 kernel collapses to
// the last temporal tap (kt=2), which is pre-sliced from the [cOut,cIn,3,3,3]
// torch weight into a [cOut,cIn,3,3] tap fed as the Conv2D weight [kw,kh,cIn,cOut].
func (g *vaeGraphBuilder) conv3x3(x *tensor.Tensor, weight5d, bias []float32, cIn, cOut, ch, cw int) *tensor.Tensor {
	b := g.b
	tap := sliceLastTemporalTap(weight5d, cOut, cIn)
	img := b.Reshape(x, uint64(cIn), uint64(cw), uint64(ch)) // [c,width,height]
	wNode := g.weight(tap, 3, 3, uint64(cIn), uint64(cOut))
	bNode := g.weight(bias, uint64(cOut))
	conv := b.Conv2D(img, wNode, bNode, 1, 1, 1, 1, 1, 1, false) // stride1, pad1 all sides
	return b.Reshape(conv, uint64(cOut), uint64(ch*cw))
}

// channelNorm: x / max(|x|_2 over channels, 1e-12) * sqrt(C) * gamma. L2Norm over
// Dims[0] with eps=1e-12 reproduces vaeChannelRMSNorm's exact max-guard; the
// sqrt(C) scale and per-channel gamma broadcast over the plane.
func (g *vaeGraphBuilder) channelNorm(x *tensor.Tensor, gamma []float32, c int) *tensor.Tensor {
	b := g.b
	n := b.Scale(b.L2Norm(x, vaeNormZeroGuard), float32(math.Sqrt(float64(c))))
	return b.Multiply(n, g.weight(gamma, uint64(c)))
}

// upsample: nearest-2x spatial upsample (RepeatHeads along width then height --
// pure data-preserving gathers) fused with a 3x3 same conv (device_tile_gather
// + tiled_fp32 in adaptive). In HWC [c,plane] the width axis is the inner
// spatial run and the height axis the outer, so RepeatHeads(2) on [c,1,plane]
// doubles width in place, and RepeatHeads(2) on [c*2w,1,h] doubles height.
func (g *vaeGraphBuilder) upsample(x *tensor.Tensor, resampleW, resampleB []float32, cIn, cOut, ch, cw int) *tensor.Tensor {
	b := g.b
	cu := uint64(cIn)
	// double width: [c,plane] -> [c,1,plane] -> [c,2,plane] -> [c,2w,h]
	xw := b.Reshape(b.RepeatHeads(b.Reshape(x, cu, 1, uint64(ch*cw)), 2), cu, uint64(2*cw), uint64(ch))
	// double height: [c,2w,h] -> [c*2w,1,h] -> [c*2w,2,h] -> [c,2w,2h]
	up := b.Reshape(b.RepeatHeads(b.Reshape(xw, cu*uint64(2*cw), 1, uint64(ch)), 2), cu, uint64(2*cw), uint64(2*ch))
	// 3x3 same conv on the [c,2w,2h] nearest-upsampled volume.
	tap := resampleW // resample.1.weight is a plain 2-D conv [cOut,cIn,3,3]
	wNode := g.weight(tap, 3, 3, cu, uint64(cOut))
	bNode := g.weight(resampleB, uint64(cOut))
	conv := b.Conv2D(up, wNode, bNode, 1, 1, 1, 1, 1, 1, false)
	return b.Reshape(conv, uint64(cOut), uint64(2*ch*2*cw))
}

// sliceLastTemporalTap extracts w[:, :, 2, :, :] from a [cOut,cIn,3,3,3] torch
// weight into a [cOut,cIn,3,3] tap (kw fastest), matching the Conv2D weight
// memory order [kw,kh,cIn,cOut].
func sliceLastTemporalTap(weight5d []float32, cOut, cIn int) []float32 {
	const kt, kh, kw = 3, 3, 3
	out := make([]float32, cOut*cIn*kh*kw)
	for co := 0; co < cOut; co++ {
		for ci := 0; ci < cIn; ci++ {
			srcBase := (((co*cIn+ci)*kt + (kt - 1)) * kh) * kw
			dstBase := ((co*cIn + ci) * kh) * kw
			copy(out[dstBase:dstBase+kh*kw], weight5d[srcBase:srcBase+kh*kw])
		}
	}
	return out
}

// DecodeGraph runs the compiled decode through run (reference.Execute for the
// host golden, the CUDA executor closure for the device). z is the CHW latent
// [ZDim][h][w] vae.go consumes; the returned pixels are planar CHW
// [OutChannels][OutH][OutW] in [-1,1], identical in shape to
// VAEDecoder.DecodeImage. mean/std are the per-channel denorm stats.
func (p *VAEProgram) DecodeGraph(run GraphRunner, mean, std []float32, z []float32) (pixels []float32, outH, outW int, err error) {
	plane := p.H * p.W
	if len(z) != p.ZDim*plane {
		return nil, 0, 0, fmt.Errorf("vae program: latent len=%d want %d", len(z), p.ZDim*plane)
	}
	if len(mean) != p.ZDim || len(std) != p.ZDim {
		return nil, 0, 0, fmt.Errorf("vae program: mean/std len=%d/%d want %d", len(mean), len(std), p.ZDim)
	}
	// denorm + CHW->HWC transpose: hwc[pos*ZDim+ch] = z[ch*plane+pos]*std+mean.
	hwc := make([]float32, len(z))
	for ch := 0; ch < p.ZDim; ch++ {
		m, s := mean[ch], std[ch]
		for pos := 0; pos < plane; pos++ {
			hwc[pos*p.ZDim+ch] = z[ch*plane+pos]*s + m
		}
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.feeds)+1)
	for _, f := range p.feeds {
		elements, _ := f.node.Shape.Elements()
		if uint64(len(f.data)) != elements {
			return nil, 0, 0, fmt.Errorf("vae program: feed len=%d want %d", len(f.data), elements)
		}
		feeds[f.node] = reference.Value{Shape: f.node.Shape, Data: f.data}
	}
	feeds[p.Latent] = reference.Value{Shape: p.Latent.Shape, Data: hwc}

	results, err := run([]*tensor.Tensor{p.Output}, feeds)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("vae program: run: %w", err)
	}
	out := results[p.Output].Data
	outPlane := p.OutH * p.OutW
	if len(out) != p.OutChannels*outPlane {
		return nil, 0, 0, fmt.Errorf("vae program: output len=%d want %d", len(out), p.OutChannels*outPlane)
	}
	// HWC->CHW transpose back to the planar [C][H][W] vae.go returns.
	chw := make([]float32, len(out))
	for pos := 0; pos < outPlane; pos++ {
		for ch := 0; ch < p.OutChannels; ch++ {
			chw[ch*outPlane+pos] = out[pos*p.OutChannels+ch]
		}
	}
	return chw, p.OutH, p.OutW, nil
}
