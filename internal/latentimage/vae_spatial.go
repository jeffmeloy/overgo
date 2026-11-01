package latentimage

import (
	"fmt"
	"path/filepath"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// spatialVAENormEpsilon belongs to diffusers Decoder and ResnetBlock2D.
// Reconsider it when admitting a source with configurable decoder epsilon.
const spatialVAENormEpsilon = 1e-6

type spatialVAEConfig struct {
	ClassName       string   `json:"_class_name"`
	Activation      string   `json:"act_fn"`
	Channels        []int    `json:"decoder_block_out_channels"`
	Blocks          []string `json:"up_block_types"`
	EncoderChannels []int    `json:"block_out_channels"`
	DownBlocks      []string `json:"down_block_types"`
	Layers          int      `json:"layers_per_block"`
	Latent          int      `json:"latent_channels"`
	InChannels      int      `json:"in_channels"`
	Output          int      `json:"out_channels"`
	Groups          uint32   `json:"norm_num_groups"`
	Patch           []int    `json:"patch_size"`
	BatchEpsilon    float32  `json:"batch_norm_eps"`
	QuantConv       bool     `json:"use_quant_conv"`
	PostQuant       bool     `json:"use_post_quant_conv"`
	Attention       bool     `json:"mid_block_add_attention"`
}

func loadSpatialVAEDecoder(root string) (*VAEDecoder, error) {
	var config spatialVAEConfig
	if err := readJSON(filepath.Join(root, "vae", "config.json"), &config); err != nil {
		return nil, err
	}
	weights, err := loadVAEWeights(filepath.Join(root, "vae"))
	if err != nil {
		return nil, err
	}
	return buildSpatialVAEDecoder(config, weights)
}

// spatialCodecBuilder gathers one side of a spatial VAE's operations from its
// tensors, checking each tensor's extent and values. channels follows the
// activation through the operations a side adds in order.
type spatialCodecBuilder struct {
	weights     map[string][]float32
	consumed    map[string]bool
	groups      uint32
	program     media.CodecProgram[[]float32]
	weightBytes int64
	channels    int
}

func newSpatialCodecBuilder(weights map[string][]float32, groups uint32) *spatialCodecBuilder {
	return &spatialCodecBuilder{weights: weights, consumed: map[string]bool{}, groups: groups}
}

func (s *spatialCodecBuilder) need(name string, extent ...int) ([]float32, error) {
	v, ok := s.weights[name]
	if !ok {
		return nil, fmt.Errorf("spatial VAE: missing tensor %s", name)
	}
	if err := checked.Length(v, extent...); err != nil {
		return nil, fmt.Errorf("spatial VAE: tensor %s: %w", name, err)
	}
	for _, x := range v {
		if !checked.Finite32(x) {
			return nil, fmt.Errorf("spatial VAE: nonfinite tensor %s", name)
		}
	}
	s.consumed[name] = true
	bytes, _ := dtype.F32.ScalarBytes()
	s.weightBytes += int64(len(v)) * int64(bytes)
	return v, nil
}

func (s *spatialCodecBuilder) add(kind media.CodecOperator, name string, ci, co int, bindings []string, extents [][]int, normNames []string) error {
	op := media.NewSpatialCodecOperation[[]float32](kind, name, ci, co, s.groups, spatialVAENormEpsilon)
	values := make([][]float32, len(bindings))
	for i, key := range bindings {
		var e error
		values[i], e = s.need(key, extents[i]...)
		if e != nil {
			return e
		}
	}
	var e error
	op.Bindings, e = media.BindCodecWeights(kind, op.RequiresProjection(), values)
	if e != nil {
		return e
	}
	if len(normNames) > 0 {
		op.Bindings.NormInputBias, e = s.need(normNames[0], ci)
		if e != nil {
			return e
		}
	}
	if len(normNames) > 1 {
		op.Bindings.NormOutputBias, e = s.need(normNames[1], co)
		if e != nil {
			return e
		}
	}
	s.program.Operations = append(s.program.Operations, op)
	return nil
}

func (s *spatialCodecBuilder) conv(kind media.CodecOperator, name string, ci, co, k int) error {
	return s.add(kind, name, ci, co, []string{name + ".weight", name + ".bias"}, [][]int{{co, ci, k, k}, {co}}, nil)
}

func (s *spatialCodecBuilder) residual(name string, co int) error {
	c := s.channels
	names := []string{name + ".norm1.weight", name + ".conv1.weight", name + ".conv1.bias", name + ".norm2.weight", name + ".conv2.weight", name + ".conv2.bias"}
	shapes := [][]int{{c}, {co, c, tensor.TripleExtent, tensor.TripleExtent}, {co}, {co}, {co, co, tensor.TripleExtent, tensor.TripleExtent}, {co}}
	if c != co {
		names = append(names, name+".conv_shortcut.weight", name+".conv_shortcut.bias")
		shapes = append(shapes, []int{co, c}, []int{co})
	}
	if err := s.add(media.CodecResidual, name, c, co, names, shapes, []string{name + ".norm1.bias", name + ".norm2.bias"}); err != nil {
		return err
	}
	s.channels = co
	return nil
}

func (s *spatialCodecBuilder) attention(name string) error {
	c := s.channels
	// The shared codec stores Q, K and V in one concatenated projection.
	var qkvWeight, qkvBias []float32
	for _, part := range []string{"to_q", "to_k", "to_v"} {
		w, e := s.need(name+"."+part+".weight", c, c)
		if e != nil {
			return e
		}
		b, e := s.need(name+"."+part+".bias", c)
		if e != nil {
			return e
		}
		qkvWeight = append(qkvWeight, w...)
		qkvBias = append(qkvBias, b...)
	}
	op := media.NewSpatialCodecOperation[[]float32](media.CodecAttention, name, c, c, s.groups, spatialVAENormEpsilon)
	gamma, e := s.need(name+".group_norm.weight", c)
	if e != nil {
		return e
	}
	w, e := s.need(name+".to_out.0.weight", c, c)
	if e != nil {
		return e
	}
	bias, e := s.need(name+".to_out.0.bias", c)
	if e != nil {
		return e
	}
	op.Bindings, e = media.BindCodecWeights(media.CodecAttention, false, [][]float32{gamma, qkvWeight, qkvBias, w, bias})
	if e != nil {
		return e
	}
	op.Bindings.NormInputBias, e = s.need(name+".group_norm.bias", c)
	if e != nil {
		return e
	}
	s.program.Operations = append(s.program.Operations, op)
	return nil
}

// finish refuses a tensor no operation read and validates the operations.
func (s *spatialCodecBuilder) finish(scope string) error {
	for name := range s.weights {
		if !s.consumed[name] {
			return fmt.Errorf("%s: unconsumed tensor %s", scope, name)
		}
	}
	return s.program.Validate(scope)
}

func buildSpatialVAEDecoder(config spatialVAEConfig, weights map[string][]float32) (*VAEDecoder, error) {
	if config.ClassName != "AutoencoderKLFlux2" || config.Activation != "silu" || !checked.PositiveInts(config.Latent, config.Output, config.Layers) || config.Groups == 0 || len(config.Channels) == 0 || len(config.Blocks) != len(config.Channels) || len(config.Patch) != tensor.PairedExtent || config.Patch[0] <= 0 || config.Patch[0] != config.Patch[1] || !checked.Finite32(config.BatchEpsilon) || config.BatchEpsilon <= 0 {
		return nil, fmt.Errorf("spatial VAE: unsupported configuration")
	}
	for i, c := range config.Channels {
		if c <= 0 || c%int(config.Groups) != 0 || config.Blocks[i] != "UpDecoderBlock2D" {
			return nil, fmt.Errorf("spatial VAE: invalid decoder block %d", i)
		}
	}
	packed, err := media.PatchVectorWidth(config.Latent, config.Patch[0])
	if err != nil {
		return nil, err
	}
	d := &VAEDecoder{ZDim: config.Latent, OutChannels: config.Output, SpatialScale: tensor.SingletonExtent, patch: config.Patch[0], batchEpsilon: config.BatchEpsilon, convolutionWinograd: true}
	codec := newSpatialCodecBuilder(weights, config.Groups)
	d.LatentsMean, err = codec.need("bn.running_mean", packed)
	if err != nil {
		return nil, err
	}
	d.latentVariance, err = codec.need("bn.running_var", packed)
	if err != nil {
		return nil, err
	}
	for _, v := range d.latentVariance {
		if v < 0 {
			return nil, fmt.Errorf("spatial VAE: negative running variance")
		}
	}
	c := config.Latent
	if config.PostQuant {
		if err := codec.conv(media.CodecPointwise, "post_quant_conv", c, c, tensor.SingletonExtent); err != nil {
			return nil, err
		}
	}
	deepest := config.Channels[len(config.Channels)-1]
	if err := codec.conv(media.CodecConvolution, "decoder.conv_in", c, deepest, tensor.TripleExtent); err != nil {
		return nil, err
	}
	codec.channels = deepest
	if err := codec.residual("decoder.mid_block.resnets.0", codec.channels); err != nil {
		return nil, err
	}
	if config.Attention {
		if err := codec.attention("decoder.mid_block.attentions.0"); err != nil {
			return nil, err
		}
	}
	if err := codec.residual("decoder.mid_block.resnets.1", codec.channels); err != nil {
		return nil, err
	}
	for block := range config.Channels {
		co := config.Channels[len(config.Channels)-1-block]
		for layer := range config.Layers + 1 {
			if err := codec.residual(fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", block, layer), co); err != nil {
				return nil, err
			}
		}
		if block < len(config.Channels)-1 {
			if err := codec.conv(media.CodecUpsampleSpatial, fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0.conv", block), co, co, tensor.TripleExtent); err != nil {
				return nil, err
			}
			scale, ok := checked.MulInt(d.SpatialScale, tensor.PairedExtent)
			if !ok {
				return nil, fmt.Errorf("spatial VAE: scale overflows")
			}
			d.SpatialScale = scale
		}
	}
	c = codec.channels
	if err := codec.add(media.CodecHead, "decoder.conv_out", c, config.Output, []string{"decoder.conv_norm_out.weight", "decoder.conv_out.weight", "decoder.conv_out.bias"}, [][]int{{c}, {config.Output, c, tensor.TripleExtent, tensor.TripleExtent}, {config.Output}}, []string{"decoder.conv_norm_out.bias"}); err != nil {
		return nil, err
	}
	if err := codec.finish("spatial VAE"); err != nil {
		return nil, err
	}
	d.CodecProgram, d.WeightBytes = codec.program, codec.weightBytes
	return d, nil
}
