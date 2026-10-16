// Mimi audio codec DECODER (codec-port slice): normalized frame latents ->
// denorm -> quantizer output projection -> depthwise transpose-conv upsample
// -> windowed causal transformer -> SEANet conv stack -> mono PCM.
//
// Geometry derives from tensor shapes plus the config-only facts (heads,
// max_period, attention context, sample/frame rates, SEANet layout); config
// restatements of shape facts are cross-checked. The vendor code facts:
// transpose kernels are 2*stride; SEANet residual blocks have one residual
// layer (dilation 1); the decoder transformer carries LayerScale.
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
	codecTransformer
	SampleRate int
	FrameRate  float64

	outerDim  int       // quantizer output width == SEANet dimension == d_model
	quantW    []float32 // [outerDim, latentDim] output projection (k=1 conv)
	upsampleW []float32 // depthwise ConvTranspose1d [outerDim][1][k]
	upsampleK int       // kernel; stride = k/2 (vendor code fact)

	convs   []codecConv
	resnets [][2]codecConv
}

// codecTransformer owns the shared causal attention geometry and weights of both Mimi directions.
type codecTransformer struct {
	layers        []codecAttnLayer
	d, h, hd      int
	ff, window    int
	invFreq       []float32
	scoreScale    float32
	normalization media.NormalizationProgram
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

	transform, err := loadCodecTransformer(read, shapes, config, codecLayerPrefix, outerDim, normalization)
	if err != nil {
		return nil, err
	}
	if !checked.PositiveInts(mimi.SampleRate) || !checked.PositiveFinite64(mimi.FrameRate) {
		return nil, fmt.Errorf("speechsynth: config lacks positive mimi sample_rate/frame_rate")
	}
	sn := mimi.Seanet
	if _, ok := checked.First(sn.Ratios); !ok || !checked.PositiveInts(sn.Dimension, sn.NFilters, sn.KernelSize,
		sn.LastKernelSize, sn.Channels, sn.Compress, sn.ResidualKernelSize) {
		return nil, fmt.Errorf("speechsynth: config lacks a complete mimi.seanet layout")
	}
	for _, check := range []struct {
		name             string
		derived, claimed int
	}{
		{"outer_dim", outerDim, mimi.OuterDim},
		{"quantizer.dimension", latentDim, mimi.Quantizer.Dimension},
		{"quantizer.output_dimension", outerDim, mimi.Quantizer.OutputDimension},
		{"seanet.dimension", outerDim, sn.Dimension},
	} {
		if checked.Nonzero(check.claimed) && !checked.Equal(check.claimed, check.derived) {
			return nil, fmt.Errorf("speechsynth: config mimi.%s=%d contradicts derived %d", check.name, check.claimed, check.derived)
		}
	}
	c := &CodecDecoder{codecTransformer: *transform, SampleRate: mimi.SampleRate, FrameRate: mimi.FrameRate,
		outerDim: outerDim, upsampleK: upsampleK}
	if c.quantW, err = read(quantizerProjName, outerDim, latentDim, tensor.SingletonExtent); err != nil {
		return nil, err
	}
	if c.upsampleW, err = read(upsampleName, outerDim, tensor.SingletonExtent, upsampleK); err != nil {
		return nil, err
	}
	chain, resnetStages := seanetDecoderLayout(sn)
	for _, s := range chain {
		kind := "conv"
		if s.transpose {
			kind = "convtr"
		}
		shape := []int{s.cOut, s.cIn, s.k}
		if s.transpose {
			shape[0], shape[1] = shape[1], shape[0]
		}
		w, err := read(fmt.Sprintf("%s%d.%s.weight", codecModelPrefix, s.idx, kind), shape...)
		if err != nil {
			return nil, err
		}
		b, err := read(fmt.Sprintf("%s%d.%s.bias", codecModelPrefix, s.idx, kind), s.cOut)
		if err != nil {
			return nil, err
		}
		c.convs = append(c.convs, codecConv{w: w, b: b, cIn: s.cIn, cOut: s.cOut, k: s.k, strd: s.strd, transpose: s.transpose})
	}
	for _, ri := range resnetStages {
		block, err := loadCodecResidual(read, codecModelPrefix, ri.idx, ri.dim, sn)
		if err != nil {
			return nil, err
		}
		c.resnets = append(c.resnets, block)
	}
	return c, nil
}

func loadCodecTransformer(read func(string, ...int) ([]float32, error), shapes map[string][]int, config artifactConfig, prefix string, outerDim int, normalization media.NormalizationProgram) (*codecTransformer, error) {
	norm1, err := tensorcatalog.Dimensions(shapes, prefix+"0.norm1.weight", tensor.SingletonExtent)
	if err != nil || !checked.Equal(norm1[tensor.FirstOffset], outerDim) {
		return nil, fmt.Errorf("speechsynth: codec transformer width %v != outer dim %d", norm1, outerDim)
	}
	d := norm1[tensor.FirstOffset]
	nLayers, err := tensorcatalog.IndexedCount(shapes, prefix, ".norm1.weight")
	if err != nil {
		return nil, err
	}
	lin1, err := tensorcatalog.Dimensions(shapes, prefix+"0.linear1.weight", tensor.PairedExtent)
	if err != nil || !checked.Equal(lin1[tensor.SingletonExtent], d) {
		return nil, fmt.Errorf("speechsynth: codec transformer linear1 incompatible with d %d", d)
	}
	ff := lin1[tensor.FirstOffset]

	tr := config.Mimi.Transformer
	if !checked.PositiveInts(tr.NumHeads, tr.Context) || !checked.PositiveFinite64(tr.MaxPeriod) {
		return nil, fmt.Errorf("speechsynth: config lacks positive mimi.transformer num_heads/max_period/context")
	}
	if _, ok := checked.DivExactInt(d, tr.NumHeads); !ok {
		return nil, fmt.Errorf("speechsynth: codec heads %d do not partition d %d", tr.NumHeads, d)
	}
	// Config restatements of shape facts: agree or refuse (0 = unstated).
	for _, check := range []struct {
		name             string
		derived, claimed int
	}{
		{"transformer.d_model", d, tr.DModel},
		{"transformer.num_layers", nLayers, tr.NumLayers},
		{"transformer.dim_feedforward", ff, tr.DimFeedforward},
	} {
		if checked.Nonzero(check.claimed) && !checked.Equal(check.claimed, check.derived) {
			return nil, fmt.Errorf("speechsynth: config mimi.%s=%d contradicts derived %d", check.name, check.claimed, check.derived)
		}
	}

	c := &codecTransformer{
		d: d, h: tr.NumHeads, hd: d / tr.NumHeads, ff: ff, window: tr.Context,
		invFreq:       hostmath.RopeInvFreqF32(tr.MaxPeriod, d/tr.NumHeads),
		scoreScale:    float32(float64(tensor.SingletonExtent) / math.Sqrt(float64(d/tr.NumHeads))),
		normalization: normalization,
	}
	c.layers = make([]codecAttnLayer, nLayers)
	for i := range c.layers {
		p := fmt.Sprintf("%s%d.", prefix, i)
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

	return c, nil
}

func loadCodecResidual(read func(string, ...int) ([]float32, error), prefix string, index, width int, sn seanetConfig) ([tensor.PairedExtent]codecConv, error) {
	hidden := width / sn.Compress
	var blk [tensor.PairedExtent]codecConv
	for bi, bs := range []struct{ sub, cIn, cOut, k int }{
		{tensor.SingletonExtent, width, hidden, sn.ResidualKernelSize},
		{tensor.TripleExtent, hidden, width, tensor.SingletonExtent},
	} {
		w, err := read(fmt.Sprintf("%s%d.block.%d.conv.weight", prefix, index, bs.sub), bs.cOut, bs.cIn, bs.k)
		if err != nil {
			return blk, err
		}
		b, err := read(fmt.Sprintf("%s%d.block.%d.conv.bias", prefix, index, bs.sub), bs.cOut)
		if err != nil {
			return blk, err
		}
		blk[bi] = codecConv{w: w, b: b, cIn: bs.cIn, cOut: bs.cOut, k: bs.k, strd: tensor.SingletonExtent}
	}
	return blk, nil
}

// LatentsToPCM crosses the codec boundary: normalized frame latents ->
// denorm (emb_std, emb_mean) -> quantizer output projection, written
// channel-major [outer][frames] -> mimi decode. The per-output accumulation
// order matches the reference channel-major projection.
func (m *Model) LatentsToPCM(ctx context.Context, latents LatentBatch) ([]float32, error) {
	if m.Codec == nil {
		return nil, fmt.Errorf("speechsynth: latent-to-pcm requires the loaded mimi codec decoder")
	}
	return m.decodeLatents(ctx, newCodecState(m.Codec), latents)
}

func (m *Model) decodeLatents(ctx context.Context, state *codecState, latents LatentBatch) ([]float32, error) {
	if ctx == nil {
		return nil, fmt.Errorf("speechsynth: decoding requires a context")
	}
	if err := state.cause(ctx); err != nil {
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
	return state.decode(ctx, proj, F)
}
