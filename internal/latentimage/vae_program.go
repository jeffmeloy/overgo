// Spatial VAE decode graph. Reference and CUDA share topology.
// Graph storage is channel-innermost; boundaries convert planar layout.

package latentimage

import (
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// VAEProgram holds one decoder geometry and its static weight bindings.
// Latent uses planar packed storage for spatial codecs and HWC storage otherwise.
// Output contains signed RGB in HWC storage.
type VAEProgram struct {
	ZDim         int
	OutChannels  int
	SpatialScale int
	H, W         int // latent extent
	OutH, OutW   int // pixel extent

	Latent     *tensor.Tensor
	Output     *tensor.Tensor
	decoded    *tensor.Tensor
	pixels     *tensor.Tensor
	boundaries map[string]*tensor.Tensor

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
	b          *tensor.Builder
	feeds      *[]vaeFeed
	boundaries map[string]*tensor.Tensor
	winograd   bool
	n          int // unique-name counter
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
	if d == nil {
		return nil, fmt.Errorf("vae program: decoder not built")
	}
	if !checked.NonemptyAll(d.Operations) {
		return nil, fmt.Errorf("vae program: decoder not built")
	}
	if !checked.PositiveInts(h, w) {
		return nil, fmt.Errorf("vae program: bad latent extent %dx%d", w, h)
	}
	inputPlane, ok := checked.MulInt(h, w)
	if !ok {
		return nil, fmt.Errorf("vae program: latent plane overflows")
	}
	if !checked.Equal(matmulType, dtype.F32) {
		return nil, fmt.Errorf("vae program: matmul weight type %s unsupported (host-feed exact path only)", matmulType)
	}

	p := &VAEProgram{
		ZDim: d.ZDim, OutChannels: d.OutChannels, SpatialScale: d.SpatialScale, H: h, W: w,
		latentMean: append([]float32(nil), d.LatentsMean...),
		latentStd:  append([]float32(nil), d.LatentsStd...),
		boundaries: make(map[string]*tensor.Tensor),
	}
	b := tensor.NewBuilder()
	g := &vaeGraphBuilder{b: b, feeds: &p.feeds, boundaries: p.boundaries, winograd: d.convolutionWinograd}

	// Latent feed: denormalized HWC columns [ZDim, h*w] (DecodeGraph prepares the
	// z*std+mean affine + CHW->HWC transpose, exactly as vae.go denorms before
	// the op loop).
	p.Latent = b.Input("vae_latent", dtype.F32, tensor.MustShape(uint64(d.ZDim), uint64(inputPlane)))

	x := p.Latent
	c, ch, cw := d.ZDim, h, w
	if d.patch > 0 {
		packed, err := media.PatchVectorWidth(c, d.patch)
		if err != nil {
			return nil, err
		}
		if err := checked.Length(d.LatentsMean, packed); err != nil {
			return nil, err
		}
		if err := checked.Length(d.latentVariance, packed); err != nil {
			return nil, err
		}
		var ok bool
		ch, ok = checked.MulInt(h, d.patch)
		if !ok {
			return nil, fmt.Errorf("vae program: unpacked height overflows")
		}
		cw, ok = checked.MulInt(w, d.patch)
		if !ok {
			return nil, fmt.Errorf("vae program: unpacked width overflows")
		}
		unpackedPlane, ok := checked.MulInt(ch, cw)
		if !ok {
			return nil, fmt.Errorf("vae program: unpacked plane overflows")
		}
		p.Latent = b.Input("vae_packed_latent", dtype.F32, tensor.MustShape(uint64(inputPlane), uint64(packed)))
		std := make([]float32, packed)
		for i, v := range d.latentVariance {
			std[i] = float32(math.Sqrt(float64(v + d.batchEpsilon)))
		}
		x = b.Add(b.Multiply(b.Transpose2D(p.Latent), g.weight(std, uint64(packed))), g.weight(d.LatentsMean, uint64(packed)))
		p.boundaries["denormalized"] = x
		area := uint64(packed / c)
		var ordered *tensor.Tensor
		for subpixel := range area {
			part := b.Reshape(b.GroupSlice(x, subpixel, tensor.SingletonExtent, uint64(c), area), uint64(c), uint64(inputPlane))
			if ordered == nil {
				ordered = part
			} else {
				ordered = b.Concat(ordered, part, tensor.FirstOffset)
			}
		}
		x = b.Reshape(b.PixelShuffle2D(b.Reshape(ordered, uint64(packed), uint64(w), uint64(h)), uint32(d.patch)), uint64(c), uint64(unpackedPlane))
		p.boundaries["unpacked"] = x
	}
	for i := range d.Operations {
		var err error
		x, c, ch, cw, err = g.buildOp(d.Operations[i], x, c, ch, cw)
		if err != nil {
			return nil, fmt.Errorf("vae program: op %d (%s): %w", i, d.Operations[i].Name, err)
		}
		p.boundaries[d.Operations[i].Name] = x
	}
	p.decoded = x
	lower, upper := media.SignedUnitBounds32()
	p.Output = b.Clamp(x, lower, upper)
	if d.patch > 0 {
		half := tensor.UnitScale / tensor.PairedExtent
		p.pixels = b.Clamp(b.Add(b.Scale(p.Output, half), g.weight([]float32{half}, tensor.SingletonExtent)), 0, tensor.UnitScale)
	}
	p.OutH, p.OutW = ch, cw

	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("vae program graph: %w", err)
	}
	return p, nil
}

// buildOp emits the graph nodes for one derived op and returns the new
// activation + geometry. x is always [c, ch*cw] HWC columns.
func (g *vaeGraphBuilder) buildOp(op media.CodecOperation[[]float32], x *tensor.Tensor, c, ch, cw int) (*tensor.Tensor, int, int, int, error) {
	b := g.b
	plane := ch * cw
	switch op.Operator {
	case media.CodecPointwise:
		out := g.pointwise(x, op.Bindings.WeightInput, op.Bindings.BiasInput, c, op.OutputChannels, plane)
		return out, op.OutputChannels, ch, cw, nil
	case media.CodecConvolution:
		out := g.convolution(x, op.Bindings.WeightInput, op.Bindings.BiasInput, c, op.OutputChannels, ch, cw, op.Convolution)
		return out, op.OutputChannels, ch, cw, nil
	case media.CodecResidual:
		gamma0, w0, b0 := op.Bindings.NormInput, op.Bindings.WeightInput, op.Bindings.BiasInput
		gamma1, w1, b1 := op.Bindings.NormOutput, op.Bindings.WeightOutput, op.Bindings.BiasOutput
		n0 := b.SiLU(g.normalize(op, x, gamma0, op.Bindings.NormInputBias, c))
		h0 := g.convolution(n0, w0, b0, c, op.OutputChannels, ch, cw, op.Convolution)
		n1 := b.SiLU(g.normalize(op, h0, gamma1, op.Bindings.NormOutputBias, op.OutputChannels))
		out := g.convolution(n1, w1, b1, op.OutputChannels, op.OutputChannels, ch, cw, op.Convolution)
		if !op.RequiresProjection() {
			return b.Add(out, x), op.OutputChannels, ch, cw, nil
		}
		shortcut := g.pointwise(x, op.Bindings.WeightProjection, op.Bindings.BiasProjection, c, op.OutputChannels, plane)
		return b.Add(out, shortcut), op.OutputChannels, ch, cw, nil
	case media.CodecAttention:
		gamma, qkvW, qkvB := op.Bindings.NormInput, op.Bindings.WeightInput, op.Bindings.BiasInput
		projW, projB := op.Bindings.WeightOutput, op.Bindings.BiasOutput
		norm := g.normalize(op, x, gamma, op.Bindings.NormInputBias, c)
		qkvChannels, ok := checked.MulInt(tensor.TripleExtent, c)
		if !ok {
			return nil, 0, 0, 0, fmt.Errorf("attention channel geometry overflows")
		}
		qkv := b.Add(
			b.MulMat(g.weight(qkvW, uint64(c), uint64(qkvChannels)), norm),
			g.weight(qkvB, uint64(qkvChannels)),
		)
		cu, pu := uint64(c), uint64(plane)
		q := b.GroupSlice(qkv, tensor.FirstOffset, cu, tensor.SingletonExtent, cu) // [c,1,plane]
		k := b.GroupSlice(qkv, cu, cu, tensor.SingletonExtent, cu)                 // [c,1,plane]
		v := b.GroupSlice(qkv, uint64(tensor.PairedExtent)*cu, cu, tensor.SingletonExtent, cu)
		scale := float32(tensor.SingletonExtent) / float32(math.Sqrt(float64(c)))
		if op.Spatial {
			scale = float32(float64(tensor.SingletonExtent) / math.Sqrt(float64(c)))
		}
		attn := b.Reshape(b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: scale, MemoryEfficientF32: op.Spatial && c > tensor.AttentionF32PanelWidth}), cu, pu)
		out := b.Add(b.MulMat(g.weight(projW, cu, cu), attn), g.weight(projB, cu))
		return b.Add(out, x), c, ch, cw, nil
	case media.CodecUpsampleSpatial:
		out := g.upsample(x, op.Bindings.WeightSpatial, op.Bindings.BiasSpatial, c, op.OutputChannels, ch, cw, op.Convolution)
		scale := op.Operator.SpatialScale()
		return out, op.OutputChannels, scale * ch, scale * cw, nil
	case media.CodecHead:
		gamma, weight, bias := op.Bindings.NormInput, op.Bindings.WeightInput, op.Bindings.BiasInput
		normalized := g.normalize(op, x, gamma, op.Bindings.NormInputBias, c)
		if g.boundaries != nil {
			g.boundaries[op.Name+".norm"] = normalized
		}
		norm := b.SiLU(normalized)
		out := g.convolution(norm, weight, bias, c, op.OutputChannels, ch, cw, op.Convolution)
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

func (g *vaeGraphBuilder) convolution(x *tensor.Tensor, weight, bias []float32, cIn, cOut, ch, cw int, extents media.CodecConvolutionExtents) *tensor.Tensor {
	b := g.b
	kernel := extents.Kernel
	tap := sliceLastTemporalTap(weight, cOut, cIn, kernel)
	img := b.Reshape(x, uint64(cIn), uint64(cw), uint64(ch)) // [c,width,height]
	wNode := g.weight(tap, uint64(kernel[2]), uint64(kernel[1]), uint64(cIn), uint64(cOut))
	bNode := g.weight(bias, uint64(cOut))
	padH, padW := uint32(kernel[1]/tensor.PairedExtent), uint32(kernel[2]/tensor.PairedExtent)
	var conv *tensor.Tensor
	if g.winograd {
		conv = b.Conv2DWinogradF32(img, wNode, bNode)
	} else {
		conv = b.Conv2D(img, wNode, bNode, uint32(extents.Stride[2]), uint32(extents.Stride[1]), padW, padW, padH, padH, false)
	}
	return b.Reshape(conv, uint64(cOut), uint64(ch*cw))
}

// channelNorm: x / max(|x|_2 over channels, 1e-12) * sqrt(C) * gamma. L2Norm over
// Dims[0] with eps=1e-12 reproduces vaeChannelRMSNorm's exact max-guard; the
// sqrt(C) scale and per-channel gamma broadcast over the plane.
func (g *vaeGraphBuilder) channelNorm(x *tensor.Tensor, gamma []float32, c int) *tensor.Tensor {
	b := g.b
	n := b.Scale(b.L2Norm(x, float32(hostmath.ChannelRMSNormZeroGuard())), float32(math.Sqrt(float64(c))))
	return b.Multiply(n, g.weight(gamma, uint64(c)))
}

func (g *vaeGraphBuilder) normalize(op media.CodecOperation[[]float32], x *tensor.Tensor, gamma, bias []float32, channels int) *tensor.Tensor {
	if op.Kernels.Has(media.CodecKernelGroupNorm) {
		return g.b.GroupNorm(x, g.weight(gamma, uint64(channels)), g.weight(bias, uint64(channels)), op.NormGroups, op.NormEpsilon)
	}
	return g.channelNorm(x, gamma, channels)
}

// upsample: nearest-2x spatial upsample (RepeatHeads along width then height --
// pure data-preserving gathers) fused with a 3x3 same conv (device_tile_gather
// + tiled_fp32 in adaptive). In HWC [c,plane] the width axis is the inner
// spatial run and the height axis the outer, so RepeatHeads(2) on [c,1,plane]
// doubles width in place, and RepeatHeads(2) on [c*2w,1,h] doubles height.
func (g *vaeGraphBuilder) upsample(x *tensor.Tensor, resampleW, resampleB []float32, cIn, cOut, ch, cw int, extents media.CodecConvolutionExtents) *tensor.Tensor {
	b := g.b
	cu := uint64(cIn)
	scale := media.CodecUpsampleSpatial.SpatialScale()
	scaledWidth := scale * cw
	scaledHeight := scale * ch
	one := uint64(tensor.SingletonExtent)
	// double width: [c,plane] -> [c,1,plane] -> [c,2,plane] -> [c,2w,h]
	xw := b.Reshape(b.RepeatHeads(b.Reshape(x, cu, one, uint64(ch*cw)), uint32(scale)), cu, uint64(scaledWidth), uint64(ch))
	// double height: [c,2w,h] -> [c*2w,1,h] -> [c*2w,2,h] -> [c,2w,2h]
	up := b.Reshape(b.RepeatHeads(b.Reshape(xw, cu*uint64(scaledWidth), one, uint64(ch)), uint32(scale)), cu, uint64(scaledWidth), uint64(scaledHeight))
	kernel := extents.Kernel
	wNode := g.weight(resampleW, uint64(kernel[2]), uint64(kernel[1]), cu, uint64(cOut))
	bNode := g.weight(resampleB, uint64(cOut))
	padH, padW := uint32(kernel[1]/tensor.PairedExtent), uint32(kernel[2]/tensor.PairedExtent)
	stride := uint32(tensor.SingletonExtent)
	var conv *tensor.Tensor
	if g.winograd {
		conv = b.Conv2DWinogradF32(up, wNode, bNode)
	} else {
		conv = b.Conv2D(up, wNode, bNode, stride, stride, padW, padW, padH, padH, false)
	}
	return b.Reshape(conv, uint64(cOut), uint64(scaledHeight*scaledWidth))
}

// sliceLastTemporalTap extracts w[:, :, 2, :, :] from a [cOut,cIn,3,3,3] torch
// weight into a [cOut,cIn,3,3] tap (kw fastest), matching the Conv2D weight
// memory order [kw,kh,cIn,cOut].
func sliceLastTemporalTap(weight5d []float32, cOut, cIn int, kernel [3]int) []float32 {
	spatialKernel := kernel[1] * kernel[2]
	out := make([]float32, cOut*cIn*spatialKernel)
	for co := range cOut {
		for ci := range cIn {
			srcBase := ((co*cIn+ci)*kernel[0] + (kernel[0] - tensor.SingletonExtent)) * spatialKernel
			dstBase := (co*cIn + ci) * spatialKernel
			copy(out[dstBase:dstBase+spatialKernel], weight5d[srcBase:srcBase+spatialKernel])
		}
	}
	return out
}
