package latentimage

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// VAEEncoder is the encode side of a spatial VAE: the operations from pixels
// to moments, and the batch statistics that normalize a packed latent.
type VAEEncoder struct {
	media.CodecProgram[[]float32]
	ZDim, InChannels, SpatialScale int
	patch                          int
	latentMean, latentVariance     []float32
	batchEpsilon                   float32
}

func loadSpatialVAEEncoder(root string) (*VAEEncoder, error) {
	var config spatialVAEConfig
	if err := readJSON(filepath.Join(root, "vae", "config.json"), &config); err != nil {
		return nil, err
	}
	weights, err := loadVAETensors(filepath.Join(root, "vae"), func(name string) bool {
		return strings.HasPrefix(name, "encoder.") || strings.HasPrefix(name, "quant_conv.") || name == "bn.running_mean" || name == "bn.running_var"
	})
	if err != nil {
		return nil, err
	}
	return buildSpatialVAEEncoder(config, weights)
}

// buildSpatialVAEEncoder walks diffusers' Encoder: the first convolution, each
// down block's residual blocks and its stride-two convolution but the last's,
// the middle block, the head, and the convolution that makes the moments.
func buildSpatialVAEEncoder(config spatialVAEConfig, weights map[string][]float32) (*VAEEncoder, error) {
	if config.ClassName != "AutoencoderKLFlux2" || config.Activation != "silu" || !checked.PositiveInts(config.Latent, config.InChannels, config.Layers) || config.Groups == 0 || len(config.EncoderChannels) == 0 || len(config.DownBlocks) != len(config.EncoderChannels) || len(config.Patch) != tensor.PairedExtent || config.Patch[0] <= 0 || config.Patch[0] != config.Patch[1] || !checked.Finite32(config.BatchEpsilon) || config.BatchEpsilon <= 0 || !config.QuantConv {
		return nil, fmt.Errorf("spatial VAE encoder: unsupported configuration")
	}
	for i, c := range config.EncoderChannels {
		if c <= 0 || c%int(config.Groups) != 0 || config.DownBlocks[i] != "DownEncoderBlock2D" {
			return nil, fmt.Errorf("spatial VAE encoder: invalid down block %d", i)
		}
	}
	packed, err := media.PatchVectorWidth(config.Latent, config.Patch[0])
	if err != nil {
		return nil, err
	}
	codec := newSpatialCodecBuilder(weights, config.Groups)
	e := &VAEEncoder{ZDim: config.Latent, InChannels: config.InChannels, SpatialScale: tensor.SingletonExtent, patch: config.Patch[0], batchEpsilon: config.BatchEpsilon}
	if e.latentMean, err = codec.need("bn.running_mean", packed); err != nil {
		return nil, err
	}
	if e.latentVariance, err = codec.need("bn.running_var", packed); err != nil {
		return nil, err
	}
	first := config.EncoderChannels[0]
	if err := codec.conv(media.CodecConvolution, "encoder.conv_in", config.InChannels, first, tensor.TripleExtent); err != nil {
		return nil, err
	}
	codec.channels = first
	for block, output := range config.EncoderChannels {
		for layer := range config.Layers {
			if err := codec.residual(fmt.Sprintf("encoder.down_blocks.%d.resnets.%d", block, layer), output); err != nil {
				return nil, err
			}
		}
		if block < len(config.EncoderChannels)-1 {
			if err := codec.conv(media.CodecDownsampleSpatial, fmt.Sprintf("encoder.down_blocks.%d.downsamplers.0.conv", block), output, output, tensor.TripleExtent); err != nil {
				return nil, err
			}
			scale, ok := checked.MulInt(e.SpatialScale, tensor.PairedExtent)
			if !ok {
				return nil, fmt.Errorf("spatial VAE encoder: scale overflows")
			}
			e.SpatialScale = scale
		}
	}
	if err := codec.residual("encoder.mid_block.resnets.0", codec.channels); err != nil {
		return nil, err
	}
	if config.Attention {
		if err := codec.attention("encoder.mid_block.attentions.0"); err != nil {
			return nil, err
		}
	}
	if err := codec.residual("encoder.mid_block.resnets.1", codec.channels); err != nil {
		return nil, err
	}
	moments, ok := checked.MulInt(tensor.PairedExtent, config.Latent)
	if !ok {
		return nil, fmt.Errorf("spatial VAE encoder: moments overflow")
	}
	c := codec.channels
	if err := codec.add(media.CodecHead, "encoder.conv_out", c, moments, []string{"encoder.conv_norm_out.weight", "encoder.conv_out.weight", "encoder.conv_out.bias"}, [][]int{{c}, {moments, c, tensor.TripleExtent, tensor.TripleExtent}, {moments}}, []string{"encoder.conv_norm_out.bias"}); err != nil {
		return nil, err
	}
	if err := codec.conv(media.CodecPointwise, "quant_conv", moments, moments, tensor.SingletonExtent); err != nil {
		return nil, err
	}
	if err := codec.finish("spatial VAE encoder"); err != nil {
		return nil, err
	}
	e.CodecProgram = codec.program
	return e, nil
}

// VAEEncoderProgram is the graph from a picture to the mean of its moments.
type VAEEncoderProgram struct {
	Pixels, Mean *tensor.Tensor
	ZDim, H, W   int
	feeds        []vaeFeed
	boundaries   map[string]*tensor.Tensor
}

// compileVAEEncoderProgram builds the encode graph of a picture of height by
// width, read as [channels, height*width] rows in the signed unit range.
func compileVAEEncoderProgram(e *VAEEncoder, height, width int) (*VAEEncoderProgram, error) {
	if e == nil || !checked.NonemptyAll(e.Operations) {
		return nil, fmt.Errorf("vae encoder program: encoder not built")
	}
	if !checked.PositiveInts(height, width) || height%e.SpatialScale != 0 || width%e.SpatialScale != 0 {
		return nil, fmt.Errorf("vae encoder program: picture %dx%d is no multiple of %d", width, height, e.SpatialScale)
	}
	plane, ok := checked.MulInt(height, width)
	if !ok {
		return nil, fmt.Errorf("vae encoder program: picture plane overflows")
	}
	shape, err := tensor.NewShape(uint64(e.InChannels), uint64(plane))
	if err != nil {
		return nil, fmt.Errorf("vae encoder program: picture extent %dx%d: %w", width, height, err)
	}
	p := &VAEEncoderProgram{ZDim: e.ZDim, boundaries: make(map[string]*tensor.Tensor)}
	b := tensor.NewBuilder()
	g := &vaeGraphBuilder{b: b, feeds: &p.feeds, boundaries: p.boundaries, winograd: true}
	p.Pixels = b.Input("vae_pixels", dtype.F32, shape)
	x, c, ch, cw := p.Pixels, e.InChannels, height, width
	for i := range e.Operations {
		var err error
		if x, c, ch, cw, err = g.buildOp(e.Operations[i], x, c, ch, cw); err != nil {
			return nil, fmt.Errorf("vae encoder program: op %d (%s): %w", i, e.Operations[i].Name, err)
		}
	}
	p.Mean = b.Reshape(b.GroupSlice(x, tensor.FirstOffset, uint64(e.ZDim), tensor.SingletonExtent, uint64(e.ZDim)), uint64(e.ZDim), uint64(ch*cw))
	p.H, p.W = ch, cw
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("vae encoder program graph: %w", err)
	}
	return p, nil
}

// normalizedLatent turns the mean of an encoded picture, [channels, h*w] rows,
// into the transformer's rows: patched as the pinned pipeline's decode inverts
// and divided by the batch statistics the decoder multiplies back.
func (e *VAEEncoder) normalizedLatent(mean []float32, h, w int) ([]float32, int, int, error) {
	planar := make([]float32, len(mean))
	for position := range h * w {
		for channel := range e.ZDim {
			planar[channel*h*w+position] = mean[position*e.ZDim+channel]
		}
	}
	packed, gridHeight, gridWidth, err := media.PackPlanar(planar, e.ZDim, h, w, e.patch, media.PatchChannelsFirst)
	if err != nil {
		return nil, 0, 0, err
	}
	width := len(e.latentMean)
	for token := range gridHeight * gridWidth {
		row := packed[token*width : (token+1)*width]
		for index := range row {
			row[index] = float32(float64(row[index]-e.latentMean[index]) / math.Sqrt(float64(e.latentVariance[index]+e.batchEpsilon)))
		}
	}
	return packed, gridHeight, gridWidth, nil
}
