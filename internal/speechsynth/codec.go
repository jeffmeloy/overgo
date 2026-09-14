// Mimi audio codec DECODER (codec-port slice): normalized frame latents ->
// denorm -> quantizer output projection -> depthwise transpose-conv upsample
// -> windowed causal transformer -> SEANet conv stack -> mono PCM.
//
// Geometry derives from tensor shapes plus the config-only facts (heads,
// max_period, attention context, sample/frame rates, SEANet layout); config
// restatements of shape facts are cross-checked. The vendor code facts:
// transpose kernels are 2*stride; SEANet residual blocks have one residual
// layer (dilation 1); the decoder transformer carries LayerScale.
//
// The mimi ENCODER (PCM -> voice latent) is NOT ported: the reference decode
// path never calls it, and the artifact's voice-cloning encoder is amputated
// (all-zero latents are artifact truth). Voice conditioning enters from the
// golden ladder instead.
package speechsynth

import (
	"context"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensorcatalog"
)

// codecConv: one SEANet convolution (weights + geometry).
type codecConv struct {
	w, b               []float32
	cIn, cOut, k, strd int
	transpose          bool
}

// codecAttnLayer: one decoder-transformer layer; LayerScale gains ls1/ls2
// scale the attention/FFN residual updates elementwise.
type codecAttnLayer struct {
	norm1W, norm1B []float32
	norm2W, norm2B []float32
	inProj         []float32 // [3d, d] fused q,k,v
	outProj        []float32 // [d, d]
	lin1           []float32 // [ff, d]
	lin2           []float32 // [d, ff]
	ls1, ls2       []float32 // [d]
}

// CodecDecoder: the latent-to-PCM half of the mimi codec.
type CodecDecoder struct {
	SampleRate int
	FrameRate  float64

	outerDim  int       // quantizer output width == SEANet dimension == d_model
	quantW    []float32 // [outerDim, latentDim] output projection (k=1 conv)
	upsampleW []float32 // depthwise ConvTranspose1d [outerDim][1][k]
	upsampleK int       // kernel; stride = k/2 (vendor code fact)

	layers        []codecAttnLayer
	d, h, hd      int
	ff, window    int
	invFreq       []float64
	scoreScale    float32
	normalization media.NormalizationProgram

	convs   []codecConv    // SEANet chain in execution order
	resnets [][2]codecConv // one residual block after each transpose conv
}

// seanetConvSpec / seanetResnetSpec: config-derived vendor tensor numbering.
type seanetConvSpec struct {
	idx, cIn, cOut, k, strd int
	transpose               bool
}

type seanetResnetSpec struct {
	idx, dim int
}

// seanetDecoderLayout: config-derived [conv, ELU, (convtr, resnet, ELU)*,
// conv]. Stage i uses vendor indices 3i+2 / 3i+3; widths halve per stage;
// transpose kernel = 2*stride.
func seanetDecoderLayout(c seanetConfig) ([]seanetConvSpec, []seanetResnetSpec) {
	nStages := len(c.Ratios)
	mult := tensor.SingletonExtent << nStages
	chain := []seanetConvSpec{{tensor.FirstOffset, c.Dimension, mult * c.NFilters, c.KernelSize, tensor.SingletonExtent, false}}
	var resnets []seanetResnetSpec
	for i, r := range c.Ratios {
		width := (mult >> (i + tensor.SingletonExtent)) * c.NFilters
		chain = append(chain, seanetConvSpec{tensor.TripleExtent*i + tensor.PairedExtent, tensor.PairedExtent * width, width, tensor.PairedExtent * r, r, true})
		resnets = append(resnets, seanetResnetSpec{tensor.TripleExtent*i + tensor.TripleExtent, width})
	}
	chain = append(chain, seanetConvSpec{tensor.TripleExtent*nStages + tensor.PairedExtent, c.NFilters, c.Channels, c.LastKernelSize, tensor.SingletonExtent, false})
	return chain, resnets
}

const (
	quantizerProjName = "mimi.quantizer.output_proj.weight"
	upsampleName      = "mimi.upsample.convtr.convtr.weight"
	codecLayerPrefix  = "mimi.decoder_transformer.transformer.layers."
	codecModelPrefix  = "mimi.decoder.model."
)

// loadCodecDecoder wires the decoder from the shared artifact reader.
func loadCodecDecoder(read func(name string, want ...int) ([]float32, error), shapes map[string][]int, config artifactConfig, latentDim int, normalization media.NormalizationProgram) (*CodecDecoder, error) {
	mimi := config.Mimi
	quantDimensions, err := tensorcatalog.Dimensions(shapes, quantizerProjName, tensor.TripleExtent)
	if err != nil || !checked.Equal(quantDimensions[tensor.SingletonExtent], latentDim) || !checked.Equal(quantDimensions[tensor.PairedExtent], tensor.SingletonExtent) {
		return nil, fmt.Errorf("speechsynth: quantizer output_proj %v incompatible with latent dim %d", quantDimensions, latentDim)
	}
	outerDim := quantDimensions[tensor.FirstOffset]
	upDimensions, err := tensorcatalog.Dimensions(shapes, upsampleName, tensor.TripleExtent)
	if err != nil || !checked.Equal(upDimensions[tensor.FirstOffset], outerDim) || !checked.Equal(upDimensions[tensor.SingletonExtent], tensor.SingletonExtent) {
		return nil, fmt.Errorf("speechsynth: upsample convtr %v is not depthwise [outer,1,k]", upDimensions)
	}
	upsampleK := upDimensions[tensor.PairedExtent]
	if !checked.EvenInt(upsampleK) {
		return nil, fmt.Errorf("speechsynth: upsample kernel %d not 2*stride", upsampleK)
	}

	norm1, err := tensorcatalog.Dimensions(shapes, codecLayerPrefix+"0.norm1.weight", tensor.SingletonExtent)
	if err != nil || !checked.Equal(norm1[tensor.FirstOffset], outerDim) {
		return nil, fmt.Errorf("speechsynth: decoder transformer width %v != outer dim %d", norm1, outerDim)
	}
	d := norm1[tensor.FirstOffset]
	nLayers, err := tensorcatalog.IndexedCount(shapes, codecLayerPrefix, ".norm1.weight")
	if err != nil {
		return nil, err
	}
	lin1, err := tensorcatalog.Dimensions(shapes, codecLayerPrefix+"0.linear1.weight", tensor.PairedExtent)
	if err != nil || !checked.Equal(lin1[tensor.SingletonExtent], d) {
		return nil, fmt.Errorf("speechsynth: decoder transformer linear1 incompatible with d %d", d)
	}
	ff := lin1[tensor.FirstOffset]

	tr := mimi.Transformer
	if !checked.PositiveInts(tr.NumHeads, tr.Context) || !checked.PositiveFinite64(tr.MaxPeriod) {
		return nil, fmt.Errorf("speechsynth: config lacks positive mimi.transformer num_heads/max_period/context")
	}
	if _, ok := checked.DivExactInt(d, tr.NumHeads); !ok {
		return nil, fmt.Errorf("speechsynth: codec heads %d do not partition d %d", tr.NumHeads, d)
	}
	if !checked.PositiveInts(mimi.SampleRate) || !checked.PositiveFinite64(mimi.FrameRate) {
		return nil, fmt.Errorf("speechsynth: config lacks positive mimi sample_rate/frame_rate")
	}
	sn := mimi.Seanet
	if _, ok := checked.First(sn.Ratios); !ok || !checked.PositiveInts(sn.Dimension, sn.NFilters, sn.KernelSize,
		sn.LastKernelSize, sn.Channels, sn.Compress, sn.ResidualKernelSize) {
		return nil, fmt.Errorf("speechsynth: config lacks a complete mimi.seanet layout")
	}
	// Config restatements of shape facts: agree or refuse (0 = unstated).
	for _, check := range []struct {
		name             string
		derived, claimed int
	}{
		{"outer_dim", outerDim, mimi.OuterDim},
		{"quantizer.dimension", latentDim, mimi.Quantizer.Dimension},
		{"quantizer.output_dimension", outerDim, mimi.Quantizer.OutputDimension},
		{"seanet.dimension", outerDim, sn.Dimension},
		{"transformer.d_model", d, tr.DModel},
		{"transformer.num_layers", nLayers, tr.NumLayers},
		{"transformer.dim_feedforward", ff, tr.DimFeedforward},
	} {
		if checked.Nonzero(check.claimed) && !checked.Equal(check.claimed, check.derived) {
			return nil, fmt.Errorf("speechsynth: config mimi.%s=%d contradicts derived %d", check.name, check.claimed, check.derived)
		}
	}

	c := &CodecDecoder{
		SampleRate: mimi.SampleRate, FrameRate: mimi.FrameRate,
		outerDim: outerDim, upsampleK: upsampleK,
		d: d, h: tr.NumHeads, hd: d / tr.NumHeads, ff: ff, window: tr.Context,
		invFreq:       hostmath.RopeInvFreq(tr.MaxPeriod, d/tr.NumHeads),
		scoreScale:    float32(float64(tensor.SingletonExtent) / math.Sqrt(float64(d/tr.NumHeads))),
		normalization: normalization,
	}
	if c.quantW, err = read(quantizerProjName, outerDim, latentDim, tensor.SingletonExtent); err != nil {
		return nil, err
	}
	if c.upsampleW, err = read(upsampleName, outerDim, tensor.SingletonExtent, upsampleK); err != nil {
		return nil, err
	}
	c.layers = make([]codecAttnLayer, nLayers)
	for i := range c.layers {
		p := fmt.Sprintf("%s%d.", codecLayerPrefix, i)
		l := &c.layers[i]
		for _, f := range []struct {
			dst  *[]float32
			name string
			want []int
		}{
			{&l.norm1W, p + "norm1.weight", []int{d}},
			{&l.norm1B, p + "norm1.bias", []int{d}},
			{&l.norm2W, p + "norm2.weight", []int{d}},
			{&l.norm2B, p + "norm2.bias", []int{d}},
			{&l.inProj, p + "self_attn.in_proj.weight", []int{3 * d, d}},
			{&l.outProj, p + "self_attn.out_proj.weight", []int{d, d}},
			{&l.lin1, p + "linear1.weight", []int{ff, d}},
			{&l.lin2, p + "linear2.weight", []int{d, ff}},
			{&l.ls1, p + "layer_scale_1.scale", []int{d}},
			{&l.ls2, p + "layer_scale_2.scale", []int{d}},
		} {
			if *f.dst, err = read(f.name, f.want...); err != nil {
				return nil, err
			}
		}
	}

	chain, resnetStages := seanetDecoderLayout(sn)
	for _, s := range chain {
		kind := "conv"
		if s.transpose {
			kind = "convtr"
		}
		var w, b []float32
		if s.transpose {
			// ConvTranspose1d weight layout [cIn][cOut][k].
			w, err = read(fmt.Sprintf("%s%d.%s.weight", codecModelPrefix, s.idx, kind), s.cIn, s.cOut, s.k)
		} else {
			w, err = read(fmt.Sprintf("%s%d.%s.weight", codecModelPrefix, s.idx, kind), s.cOut, s.cIn, s.k)
		}
		if err != nil {
			return nil, err
		}
		if b, err = read(fmt.Sprintf("%s%d.%s.bias", codecModelPrefix, s.idx, kind), s.cOut); err != nil {
			return nil, err
		}
		c.convs = append(c.convs, codecConv{w: w, b: b, cIn: s.cIn, cOut: s.cOut, k: s.k, strd: s.strd, transpose: s.transpose})
	}
	for _, ri := range resnetStages {
		hidden := ri.dim / sn.Compress
		var blk [tensor.PairedExtent]codecConv
		for bi, bs := range []struct{ sub, cIn, cOut, k int }{
			{tensor.SingletonExtent, ri.dim, hidden, sn.ResidualKernelSize},
			{tensor.TripleExtent, hidden, ri.dim, tensor.SingletonExtent},
		} {
			w, err := read(fmt.Sprintf("%s%d.block.%d.conv.weight", codecModelPrefix, ri.idx, bs.sub), bs.cOut, bs.cIn, bs.k)
			if err != nil {
				return nil, err
			}
			b, err := read(fmt.Sprintf("%s%d.block.%d.conv.bias", codecModelPrefix, ri.idx, bs.sub), bs.cOut)
			if err != nil {
				return nil, err
			}
			blk[bi] = codecConv{w: w, b: b, cIn: bs.cIn, cOut: bs.cOut, k: bs.k, strd: tensor.SingletonExtent}
		}
		c.resnets = append(c.resnets, blk)
	}
	return c, nil
}

// Upsample runs the depthwise transpose conv: [outer][T] -> [outer][T*k/2].
func (c *CodecDecoder) Upsample(latent []float32, T int) []float32 {
	return hostmath.ConvTranspose1dTrim(latent, c.outerDim, T, c.upsampleW, nil, c.outerDim, c.upsampleK, c.upsampleK/2, c.outerDim)
}

// TransformInPlace applies the decoder transformer over channel-major
// [d][T] in place (transpose to time-major, forward, transpose back).
func (c *CodecDecoder) TransformInPlace(ctx context.Context, x []float32, T int) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	d := c.d
	tm := make([]float32, len(x))
	for ch := range d {
		for t := range T {
			tm[t*d+ch] = x[ch*T+t]
		}
	}
	if err := c.forwardTimeMajor(ctx, tm, T); err != nil {
		return err
	}
	for ch := range d {
		for t := range T {
			x[ch*T+t] = tm[t*d+ch]
		}
	}
	return context.Cause(ctx)
}

// forwardTimeMajor: causal decode over [T][d] with the config attention
// window; positions from 0. Same t-major structure as the backbone
// AppendForward, plus LayerScale on both residual updates.
func (c *CodecDecoder) forwardTimeMajor(ctx context.Context, x []float32, T int) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	d, h, hd, ff := c.d, c.h, c.hd, c.ff
	kc := make([][]float32, len(c.layers))
	vc := make([][]float32, len(c.layers))
	for i := range kc {
		kc[i] = make([]float32, 0, T*d)
		vc[i] = make([]float32, 0, T*d)
	}
	qkv := make([]float32, tensor.TripleExtent*d)
	xn := make([]float32, d)
	attn := make([]float32, d)
	proj := make([]float32, d)
	h1 := make([]float32, ff)
	for t := tensor.FirstOffset; t < T; t++ {
		if err := context.Cause(ctx); err != nil {
			return err
		}
		cur := x[t*d : (t+tensor.SingletonExtent)*d]
		lo := t + tensor.SingletonExtent - c.window
		if lo < tensor.FirstOffset {
			lo = tensor.FirstOffset
		}
		for li := range c.layers {
			l := &c.layers[li]
			hostmath.LayerNormInto(xn, cur, l.norm1W, l.norm1B, tensor.SingletonExtent, d, c.normalization.TransformerLayer)
			hostmath.Linear(qkv, xn, l.inProj, tensor.SingletonExtent, d, tensor.TripleExtent*d)
			q, k, v := qkv[:d], qkv[d:tensor.PairedExtent*d], qkv[tensor.PairedExtent*d:]
			for head := range h {
				hostmath.ApplyRotaryInterleaved(q[head*hd:(head+tensor.SingletonExtent)*hd], c.invFreq, t)
				hostmath.ApplyRotaryInterleaved(k[head*hd:(head+tensor.SingletonExtent)*hd], c.invFreq, t)
			}
			kc[li] = append(kc[li], k...)
			vc[li] = append(vc[li], v...)
			for i := range q {
				q[i] *= c.scoreScale
			}
			hostmath.CausalAttentionStep(attn, q, kc[li][lo*d:], vc[li][lo*d:], t+tensor.SingletonExtent-lo, h, h, hd)
			hostmath.Linear(proj, attn, l.outProj, tensor.SingletonExtent, d, d)
			for i := range cur {
				cur[i] += l.ls1[i] * proj[i]
			}
			hostmath.LayerNormInto(xn, cur, l.norm2W, l.norm2B, tensor.SingletonExtent, d, c.normalization.TransformerLayer)
			hostmath.Linear(h1, xn, l.lin1, tensor.SingletonExtent, d, ff)
			hostmath.GELUErfInPlace(h1)
			hostmath.Linear(proj, h1, l.lin2, tensor.SingletonExtent, ff, d)
			for i := range cur {
				cur[i] += l.ls2[i] * proj[i]
			}
		}
	}
	return context.Cause(ctx)
}

// SeanetDecode runs the conv stack: [outer][T] -> [channels][T*prod(ratios)].
// A residual block plus ELU follows each transpose convolution.
func (c *CodecDecoder) SeanetDecode(ctx context.Context, x []float32, T int) ([]float32, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	cur, curT := x, T
	apply := func(cv codecConv) {
		if cv.transpose {
			cur = hostmath.ConvTranspose1dTrim(cur, cv.cIn, curT, cv.w, cv.b, cv.cOut, cv.k, cv.strd, tensor.SingletonExtent)
			curT *= cv.strd
		} else {
			cur = hostmath.CausalConv1d(cur, cv.cIn, curT, cv.w, cv.b, cv.cOut, cv.k, cv.strd)
			curT /= cv.strd
		}
	}
	apply(c.convs[0])
	hostmath.ELUInPlace(cur)
	for i, conv := range c.convs[tensor.SingletonExtent : len(c.convs)-tensor.SingletonExtent] {
		if err := context.Cause(ctx); err != nil {
			return nil, err
		}
		apply(conv)
		cur = seanetResnet(c.resnets[i], cur, curT)
		hostmath.ELUInPlace(cur)
	}
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	apply(c.convs[len(c.convs)-tensor.SingletonExtent])
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	return cur, nil
}

// seanetResnet: identity skip around [ELU -> conv k -> ELU -> conv 1].
func seanetResnet(blk [tensor.PairedExtent]codecConv, x []float32, T int) []float32 {
	v := append([]float32(nil), x...)
	hostmath.ELUInPlace(v)
	v = hostmath.CausalConv1d(v, blk[tensor.FirstOffset].cIn, T, blk[tensor.FirstOffset].w, blk[tensor.FirstOffset].b, blk[tensor.FirstOffset].cOut, blk[tensor.FirstOffset].k, tensor.SingletonExtent)
	hostmath.ELUInPlace(v)
	v = hostmath.CausalConv1d(v, blk[tensor.SingletonExtent].cIn, T, blk[tensor.SingletonExtent].w, blk[tensor.SingletonExtent].b, blk[tensor.SingletonExtent].cOut, blk[tensor.SingletonExtent].k, tensor.SingletonExtent)
	for i := range v {
		v[i] += x[i]
	}
	return v
}

// DecodeFromLatent: outer latent channel-major [outer][T] -> mono PCM
// [T * (k/2) * prod(ratios)].
func (c *CodecDecoder) DecodeFromLatent(ctx context.Context, latent []float32, T int) ([]float32, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	up := c.Upsample(latent, T)
	upT := T * c.upsampleK / tensor.PairedExtent
	if err := c.TransformInPlace(ctx, up, upT); err != nil {
		return nil, err
	}
	return c.SeanetDecode(ctx, up, upT)
}

// LatentsToPCM crosses the codec boundary: normalized frame latents ->
// denorm (emb_std, emb_mean) -> quantizer output projection, written
// channel-major [outer][frames] -> mimi decode. The per-output accumulation
// order matches the reference channel-major projection.
func (m *Model) LatentsToPCM(ctx context.Context, latents LatentBatch) ([]float32, error) {
	if ctx == nil {
		return nil, fmt.Errorf("speechsynth: decoding requires a context")
	}
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	if m.Codec == nil {
		return nil, fmt.Errorf("speechsynth: latent-to-pcm requires the loaded mimi codec decoder")
	}
	ldim := m.Dims.LatentDim
	F := latents.Frames
	elements, ok := checked.MulInt(F, ldim)
	values, valid := checked.Prefix(latents.Values, elements)
	if !checked.Equal(latents.Width, ldim) || !checked.PositiveInts(F) || !ok || !valid {
		return nil, fmt.Errorf("speechsynth: latent batch %dx%d (have %d values) incompatible with latent dim %d", F, latents.Width, len(latents.Values), ldim)
	}
	odim := m.Codec.outerDim
	proj := make([]float32, odim*F)
	if err := hostmath.NormalizeProjectRowsToChannelsF64(proj, values, m.EmbStd, m.EmbMean, m.Codec.quantW, F, ldim, odim); err != nil {
		return nil, fmt.Errorf("speechsynth: latent projection: %w", err)
	}
	return m.Codec.DecodeFromLatent(ctx, proj, F)
}
