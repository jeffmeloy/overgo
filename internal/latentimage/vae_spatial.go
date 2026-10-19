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
	ClassName    string   `json:"_class_name"`
	Activation   string   `json:"act_fn"`
	Channels     []int    `json:"decoder_block_out_channels"`
	Blocks       []string `json:"up_block_types"`
	Layers       int      `json:"layers_per_block"`
	Latent       int      `json:"latent_channels"`
	Output       int      `json:"out_channels"`
	Groups       uint32   `json:"norm_num_groups"`
	Patch        []int    `json:"patch_size"`
	BatchEpsilon float32  `json:"batch_norm_eps"`
	PostQuant    bool     `json:"use_post_quant_conv"`
	Attention    bool     `json:"mid_block_add_attention"`
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
	consumed := map[string]bool{}
	need := func(name string, extent ...int) ([]float32, error) {
		v, ok := weights[name]
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
		consumed[name] = true
		bytes, _ := dtype.F32.ScalarBytes()
		d.WeightBytes += int64(len(v)) * int64(bytes)
		return v, nil
	}
	d.LatentsMean, err = need("bn.running_mean", packed)
	if err != nil {
		return nil, err
	}
	d.latentVariance, err = need("bn.running_var", packed)
	if err != nil {
		return nil, err
	}
	for _, v := range d.latentVariance {
		if v < 0 {
			return nil, fmt.Errorf("spatial VAE: negative running variance")
		}
	}
	add := func(kind media.CodecOperator, name string, ci, co int, bindings []string, extents [][]int, normNames []string) error {
		op := media.NewSpatialCodecOperation[[]float32](kind, name, ci, co, config.Groups, spatialVAENormEpsilon)
		values := make([][]float32, len(bindings))
		for i, key := range bindings {
			var e error
			values[i], e = need(key, extents[i]...)
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
			op.Bindings.NormInputBias, e = need(normNames[0], ci)
			if e != nil {
				return e
			}
		}
		if len(normNames) > 1 {
			op.Bindings.NormOutputBias, e = need(normNames[1], co)
			if e != nil {
				return e
			}
		}
		d.Operations = append(d.Operations, op)
		return nil
	}
	conv := func(kind media.CodecOperator, name string, ci, co, k int) error {
		return add(kind, name, ci, co, []string{name + ".weight", name + ".bias"}, [][]int{{co, ci, k, k}, {co}}, nil)
	}
	c := config.Latent
	if config.PostQuant {
		if err := conv(media.CodecPointwise, "post_quant_conv", c, c, tensor.SingletonExtent); err != nil {
			return nil, err
		}
	}
	deepest := config.Channels[len(config.Channels)-1]
	if err := conv(media.CodecConvolution, "decoder.conv_in", c, deepest, tensor.TripleExtent); err != nil {
		return nil, err
	}
	c = deepest
	res := func(name string, co int) error {
		names := []string{name + ".norm1.weight", name + ".conv1.weight", name + ".conv1.bias", name + ".norm2.weight", name + ".conv2.weight", name + ".conv2.bias"}
		shapes := [][]int{{c}, {co, c, tensor.TripleExtent, tensor.TripleExtent}, {co}, {co}, {co, co, tensor.TripleExtent, tensor.TripleExtent}, {co}}
		if c != co {
			names = append(names, name+".conv_shortcut.weight", name+".conv_shortcut.bias")
			shapes = append(shapes, []int{co, c}, []int{co})
		}
		if err := add(media.CodecResidual, name, c, co, names, shapes, []string{name + ".norm1.bias", name + ".norm2.bias"}); err != nil {
			return err
		}
		c = co
		return nil
	}
	if err := res("decoder.mid_block.resnets.0", c); err != nil {
		return nil, err
	}
	if config.Attention {
		name := "decoder.mid_block.attentions.0"
		// The shared codec stores Q, K and V in one concatenated projection.
		var qkvWeight, qkvBias []float32
		for _, part := range []string{"to_q", "to_k", "to_v"} {
			w, e := need(name+"."+part+".weight", c, c)
			if e != nil {
				return nil, e
			}
			b, e := need(name+"."+part+".bias", c)
			if e != nil {
				return nil, e
			}
			qkvWeight = append(qkvWeight, w...)
			qkvBias = append(qkvBias, b...)
		}
		op := media.NewSpatialCodecOperation[[]float32](media.CodecAttention, name, c, c, config.Groups, spatialVAENormEpsilon)
		gamma, e := need(name+".group_norm.weight", c)
		if e != nil {
			return nil, e
		}
		w, e := need(name+".to_out.0.weight", c, c)
		if e != nil {
			return nil, e
		}
		bias, e := need(name+".to_out.0.bias", c)
		if e != nil {
			return nil, e
		}
		op.Bindings, e = media.BindCodecWeights(media.CodecAttention, false, [][]float32{gamma, qkvWeight, qkvBias, w, bias})
		if e != nil {
			return nil, e
		}
		op.Bindings.NormInputBias, e = need(name+".group_norm.bias", c)
		if e != nil {
			return nil, e
		}
		d.Operations = append(d.Operations, op)
	}
	if err := res("decoder.mid_block.resnets.1", c); err != nil {
		return nil, err
	}
	for block := range config.Channels {
		co := config.Channels[len(config.Channels)-1-block]
		for layer := range config.Layers + 1 {
			if err := res(fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", block, layer), co); err != nil {
				return nil, err
			}
		}
		if block < len(config.Channels)-1 {
			if err := conv(media.CodecUpsampleSpatial, fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0.conv", block), c, c, tensor.TripleExtent); err != nil {
				return nil, err
			}
			scale, ok := checked.MulInt(d.SpatialScale, tensor.PairedExtent)
			if !ok {
				return nil, fmt.Errorf("spatial VAE: scale overflows")
			}
			d.SpatialScale = scale
		}
	}
	if err := add(media.CodecHead, "decoder.conv_out", c, config.Output, []string{"decoder.conv_norm_out.weight", "decoder.conv_out.weight", "decoder.conv_out.bias"}, [][]int{{c}, {config.Output, c, tensor.TripleExtent, tensor.TripleExtent}, {config.Output}}, []string{"decoder.conv_norm_out.bias"}); err != nil {
		return nil, err
	}
	for name := range weights {
		if !consumed[name] {
			return nil, fmt.Errorf("spatial VAE: unconsumed decoder tensor %s", name)
		}
	}
	if err := d.CodecProgram.Validate("spatial VAE"); err != nil {
		return nil, err
	}
	return d, nil
}
