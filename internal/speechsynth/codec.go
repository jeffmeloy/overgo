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
	"fmt"
	"math"

	"overgo/internal/hostmath"
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

	layers     []codecAttnLayer
	d, h, hd   int
	ff, window int
	invFreq    []float64
	scoreScale float32

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
	mult := 1 << nStages
	chain := []seanetConvSpec{{0, c.Dimension, mult * c.NFilters, c.KernelSize, 1, false}}
	var resnets []seanetResnetSpec
	for i, r := range c.Ratios {
		width := (mult >> (i + 1)) * c.NFilters
		chain = append(chain, seanetConvSpec{3*i + 2, 2 * width, width, 2 * r, r, true})
		resnets = append(resnets, seanetResnetSpec{3*i + 3, width})
	}
	chain = append(chain, seanetConvSpec{3*nStages + 2, c.NFilters, c.Channels, c.LastKernelSize, 1, false})
	return chain, resnets
}

const (
	quantizerProjName = "mimi.quantizer.output_proj.weight"
	upsampleName      = "mimi.upsample.convtr.convtr.weight"
	codecLayerPrefix  = "mimi.decoder_transformer.transformer.layers."
	codecModelPrefix  = "mimi.decoder.model."
)

// loadCodecDecoder wires the decoder from the shared artifact reader.
func loadCodecDecoder(read func(name string, want ...int) ([]float32, error), shapes map[string][]int, config artifactConfig, latentDim int) (*CodecDecoder, error) {
	mimi := config.Mimi
	quantShape, ok := shapes[quantizerProjName]
	if !ok || len(quantShape) != 3 || quantShape[1] != latentDim || quantShape[2] != 1 {
		return nil, fmt.Errorf("speechsynth: quantizer output_proj %v incompatible with latent dim %d", quantShape, latentDim)
	}
	outerDim := quantShape[0]
	upShape, ok := shapes[upsampleName]
	if !ok || len(upShape) != 3 || upShape[0] != outerDim || upShape[1] != 1 {
		return nil, fmt.Errorf("speechsynth: upsample convtr %v is not depthwise [outer,1,k]", upShape)
	}
	upsampleK := upShape[2]
	if upsampleK%2 != 0 {
		return nil, fmt.Errorf("speechsynth: upsample kernel %d not 2*stride", upsampleK)
	}

	norm1, ok := shapes[codecLayerPrefix+"0.norm1.weight"]
	if !ok || len(norm1) != 1 || norm1[0] != outerDim {
		return nil, fmt.Errorf("speechsynth: decoder transformer width %v != outer dim %d", norm1, outerDim)
	}
	d := norm1[0]
	nLayers, err := layerCount(shapes, codecLayerPrefix, ".norm1.weight")
	if err != nil {
		return nil, err
	}
	lin1, ok := shapes[codecLayerPrefix+"0.linear1.weight"]
	if !ok || len(lin1) != 2 || lin1[1] != d {
		return nil, fmt.Errorf("speechsynth: decoder transformer linear1 incompatible with d %d", d)
	}
	ff := lin1[0]

	tr := mimi.Transformer
	if tr.NumHeads <= 0 || tr.MaxPeriod <= 0 || tr.Context <= 0 {
		return nil, fmt.Errorf("speechsynth: config lacks positive mimi.transformer num_heads/max_period/context")
	}
	if d%tr.NumHeads != 0 {
		return nil, fmt.Errorf("speechsynth: codec heads %d do not partition d %d", tr.NumHeads, d)
	}
	if mimi.SampleRate <= 0 || mimi.FrameRate <= 0 {
		return nil, fmt.Errorf("speechsynth: config lacks positive mimi sample_rate/frame_rate")
	}
	sn := mimi.Seanet
	if len(sn.Ratios) == 0 || sn.Dimension <= 0 || sn.NFilters <= 0 || sn.KernelSize <= 0 ||
		sn.LastKernelSize <= 0 || sn.Channels <= 0 || sn.Compress <= 0 || sn.ResidualKernelSize <= 0 {
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
		if check.claimed != 0 && check.claimed != check.derived {
			return nil, fmt.Errorf("speechsynth: config mimi.%s=%d contradicts derived %d", check.name, check.claimed, check.derived)
		}
	}

	c := &CodecDecoder{
		SampleRate: mimi.SampleRate, FrameRate: mimi.FrameRate,
		outerDim: outerDim, upsampleK: upsampleK,
		d: d, h: tr.NumHeads, hd: d / tr.NumHeads, ff: ff, window: tr.Context,
		invFreq:    hostmath.RopeInvFreq(tr.MaxPeriod, d/tr.NumHeads),
		scoreScale: float32(1 / math.Sqrt(float64(d/tr.NumHeads))),
	}
	if c.quantW, err = read(quantizerProjName, outerDim, latentDim, 1); err != nil {
		return nil, err
	}
	if c.upsampleW, err = read(upsampleName, outerDim, 1, upsampleK); err != nil {
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
		var blk [2]codecConv
		for bi, bs := range []struct{ sub, cIn, cOut, k int }{
			{1, ri.dim, hidden, sn.ResidualKernelSize}, {3, hidden, ri.dim, 1},
		} {
			w, err := read(fmt.Sprintf("%s%d.block.%d.conv.weight", codecModelPrefix, ri.idx, bs.sub), bs.cOut, bs.cIn, bs.k)
			if err != nil {
				return nil, err
			}
			b, err := read(fmt.Sprintf("%s%d.block.%d.conv.bias", codecModelPrefix, ri.idx, bs.sub), bs.cOut)
			if err != nil {
				return nil, err
			}
			blk[bi] = codecConv{w: w, b: b, cIn: bs.cIn, cOut: bs.cOut, k: bs.k, strd: 1}
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
func (c *CodecDecoder) TransformInPlace(x []float32, T int) {
	d := c.d
	tm := make([]float32, len(x))
	for ch := 0; ch < d; ch++ {
		for t := 0; t < T; t++ {
			tm[t*d+ch] = x[ch*T+t]
		}
	}
	c.forwardTimeMajor(tm, T)
	for ch := 0; ch < d; ch++ {
		for t := 0; t < T; t++ {
			x[ch*T+t] = tm[t*d+ch]
		}
	}
}

// forwardTimeMajor: causal decode over [T][d] with the config attention
// window; positions from 0. Same t-major structure as the backbone
// AppendForward, plus LayerScale on both residual updates.
func (c *CodecDecoder) forwardTimeMajor(x []float32, T int) {
	d, h, hd, ff := c.d, c.h, c.hd, c.ff
	kc := make([][]float32, len(c.layers))
	vc := make([][]float32, len(c.layers))
	for i := range kc {
		kc[i] = make([]float32, 0, T*d)
		vc[i] = make([]float32, 0, T*d)
	}
	qkv := make([]float32, 3*d)
	xn := make([]float32, d)
	attn := make([]float32, d)
	proj := make([]float32, d)
	h1 := make([]float32, ff)
	for t := 0; t < T; t++ {
		cur := x[t*d : (t+1)*d]
		lo := t + 1 - c.window
		if lo < 0 {
			lo = 0
		}
		for li := range c.layers {
			l := &c.layers[li]
			hostmath.LayerNormInto(xn, cur, l.norm1W, l.norm1B, 1, d, transformerLayerNormEps)
			hostmath.Linear(qkv, xn, l.inProj, 1, d, 3*d)
			q, k, v := qkv[:d], qkv[d:2*d], qkv[2*d:]
			for head := 0; head < h; head++ {
				hostmath.ApplyRotaryInterleaved(q[head*hd:(head+1)*hd], c.invFreq, t)
				hostmath.ApplyRotaryInterleaved(k[head*hd:(head+1)*hd], c.invFreq, t)
			}
			kc[li] = append(kc[li], k...)
			vc[li] = append(vc[li], v...)
			for i := range q {
				q[i] *= c.scoreScale
			}
			hostmath.CausalAttentionStep(attn, q, kc[li][lo*d:], vc[li][lo*d:], t+1-lo, h, h, hd)
			hostmath.Linear(proj, attn, l.outProj, 1, d, d)
			for i := range cur {
				cur[i] += l.ls1[i] * proj[i]
			}
			hostmath.LayerNormInto(xn, cur, l.norm2W, l.norm2B, 1, d, transformerLayerNormEps)
			hostmath.Linear(h1, xn, l.lin1, 1, d, ff)
			hostmath.GELUErfInPlace(h1)
			hostmath.Linear(proj, h1, l.lin2, 1, ff, d)
			for i := range cur {
				cur[i] += l.ls2[i] * proj[i]
			}
		}
	}
}

// SeanetDecode runs the conv stack: [outer][T] -> [channels][T*prod(ratios)].
// A residual block plus ELU follows each transpose convolution.
func (c *CodecDecoder) SeanetDecode(x []float32, T int) []float32 {
	cur, curT := x, T
	apply := func(cv codecConv) {
		if cv.transpose {
			cur = hostmath.ConvTranspose1dTrim(cur, cv.cIn, curT, cv.w, cv.b, cv.cOut, cv.k, cv.strd, 1)
			curT *= cv.strd
		} else {
			cur = hostmath.CausalConv1d(cur, cv.cIn, curT, cv.w, cv.b, cv.cOut, cv.k, cv.strd)
			curT /= cv.strd
		}
	}
	apply(c.convs[0])
	hostmath.ELUInPlace(cur)
	for i, conv := range c.convs[1 : len(c.convs)-1] {
		apply(conv)
		cur = seanetResnet(c.resnets[i], cur, curT)
		hostmath.ELUInPlace(cur)
	}
	apply(c.convs[len(c.convs)-1])
	return cur
}

// seanetResnet: identity skip around [ELU -> conv k -> ELU -> conv 1].
func seanetResnet(blk [2]codecConv, x []float32, T int) []float32 {
	v := append([]float32(nil), x...)
	hostmath.ELUInPlace(v)
	v = hostmath.CausalConv1d(v, blk[0].cIn, T, blk[0].w, blk[0].b, blk[0].cOut, blk[0].k, 1)
	hostmath.ELUInPlace(v)
	v = hostmath.CausalConv1d(v, blk[1].cIn, T, blk[1].w, blk[1].b, blk[1].cOut, blk[1].k, 1)
	for i := range v {
		v[i] += x[i]
	}
	return v
}

// DecodeFromLatent: outer latent channel-major [outer][T] -> mono PCM
// [T * (k/2) * prod(ratios)].
func (c *CodecDecoder) DecodeFromLatent(latent []float32, T int) []float32 {
	up := c.Upsample(latent, T)
	upT := T * c.upsampleK / 2
	c.TransformInPlace(up, upT)
	return c.SeanetDecode(up, upT)
}

// LatentsToPCM crosses the codec boundary: normalized frame latents ->
// denorm (emb_std, emb_mean) -> quantizer output projection, written
// channel-major [outer][frames] -> mimi decode. The per-output accumulation
// order matches the reference channel-major projection.
func (m *Model) LatentsToPCM(latents LatentBatch) ([]float32, error) {
	if m.Codec == nil {
		return nil, fmt.Errorf("speechsynth: latent-to-pcm requires the loaded mimi codec decoder")
	}
	ldim := m.Dims.LatentDim
	F := latents.Frames
	if latents.Width != ldim || F <= 0 || len(latents.Values) < F*ldim {
		return nil, fmt.Errorf("speechsynth: latent batch %dx%d (have %d values) incompatible with latent dim %d", F, latents.Width, len(latents.Values), ldim)
	}
	odim := m.Codec.outerDim
	proj := make([]float32, odim*F)
	for f := 0; f < F; f++ {
		row := latents.Values[f*ldim : (f+1)*ldim]
		for o := 0; o < odim; o++ {
			wr := m.Codec.quantW[o*ldim : (o+1)*ldim]
			var acc float64
			for i := 0; i < ldim; i++ {
				acc += float64(row[i]*m.EmbStd[i]+m.EmbMean[i]) * float64(wr[i])
			}
			proj[o*F+f] = float32(acc)
		}
	}
	return m.Codec.DecodeFromLatent(proj, F), nil
}
