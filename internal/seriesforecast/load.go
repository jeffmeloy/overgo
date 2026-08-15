// Package seriesforecast owns the patched time-series forecasting capability
// (ladder rung 1: the first non-token task). A capability port from
// adaptive_new expressed fresh against overgo's owners — behavior and
// evidence carried, no substrate copied; parity is judged against the
// imported forward golden and the ten per-component gradient goldens.
//
// Every dimension is DERIVED from the artifact's own tensor shapes and
// config declarations; nothing here asserts a model constant.
package seriesforecast

import (
	"fmt"
	"io"
	"math"
	"path/filepath"

	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
	"overgo/internal/tensorcatalog"
)

// patchInputStreams: each patch token carries values plus their mask.
const patchInputStreams = 2

// qkvProjections: fused query/key/value projection rows per hidden column.
const qkvProjections = 3

// Dims: model geometry, derived from tensor shapes and config.
type Dims struct {
	PatchLen  int
	Hidden    int
	Layers    int
	Heads     int
	HeadDim   int
	Horizon   int
	Quantiles int
	RopeTheta float64
	RMSEps    float64
}

// Model: loaded weights, derived dims, and quantile levels.
type Model struct {
	Dims    Dims
	Weights map[string][]float32
	Shapes  map[string][]int
	Levels  []float64
}

type artifactConfig struct {
	ModelType      string    `json:"model_type"`
	Quantiles      []float64 `json:"quantiles"`
	RMSNormEps     float64   `json:"rms_norm_eps"`
	RopeParameters struct {
		RopeTheta float64 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

// Load opens the safetensors artifact and materializes every tensor as f32.
func Load(directory string) (*Model, error) {
	config, err := loadConfig(filepath.Join(directory, "config.json"))
	if err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()

	shapes := make(map[string][]int, len(source.Tensors))
	for name, tensor := range source.Tensors {
		dims := make([]int, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			if dim == 0 || dim > 1<<31 {
				return nil, fmt.Errorf("seriesforecast: tensor %q dimension %d out of range", name, dim)
			}
			dims[i] = int(dim)
		}
		shapes[name] = dims
	}
	weights := make(map[string][]float32, len(source.Tensors))
	for name, tensor := range source.Tensors {
		reader, err := safetensors.F32Reader(tensor)
		if err != nil {
			return nil, fmt.Errorf("seriesforecast: tensor %q: %w", name, err)
		}
		elements := tensor.Elements()
		raw := make([]byte, elements*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return nil, fmt.Errorf("seriesforecast: tensor %q payload: %w", name, err)
		}
		values := make([]float32, elements)
		for i := range values {
			bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
			values[i] = math.Float32frombits(bits)
		}
		weights[name] = values
	}
	weights, shapes, err = canonicalize(weights, shapes)
	if err != nil {
		return nil, err
	}
	dims, err := dimsFromShapes(shapes, 1+len(config.Quantiles))
	if err != nil {
		return nil, err
	}
	dims.RopeTheta = config.RopeParameters.RopeTheta
	dims.RMSEps = config.RMSNormEps
	return &Model{Dims: dims, Weights: weights, Shapes: shapes, Levels: config.Quantiles}, nil
}

// canonicalize maps the transformers checkpoint layout onto the canonical
// tensor names (the naming contract shared with the goldens): input_ff_layer
// -> tokenizer, model.layers.N -> stacked_xf.N with per-tensor renames, and
// the separate q/k/v projections CONCATENATED row-wise into one qkv tensor.
// A checkpoint already in canonical form passes through untouched.
func canonicalize(
	weights map[string][]float32,
	shapes map[string][]int,
) (map[string][]float32, map[string][]int, error) {
	if _, transformers := shapes["model.input_ff_layer.input_layer.weight"]; !transformers {
		return weights, shapes, nil
	}
	renames := map[string]string{}
	for _, suffix := range []string{".weight", ".bias"} {
		renames["model.input_ff_layer.input_layer"+suffix] = "tokenizer.hidden_layer" + suffix
		renames["model.input_ff_layer.output_layer"+suffix] = "tokenizer.output_layer" + suffix
		renames["model.input_ff_layer.residual_layer"+suffix] = "tokenizer.residual_layer" + suffix
		for _, head := range []string{"output_projection_point", "output_projection_quantiles"} {
			renames[head+".input_layer"+suffix] = head + ".hidden_layer" + suffix
		}
	}
	layerRenames := [][2]string{
		{"input_layernorm.weight", "pre_attn_ln.scale"},
		{"post_attention_layernorm.weight", "post_attn_ln.scale"},
		{"pre_feedforward_layernorm.weight", "pre_ff_ln.scale"},
		{"post_feedforward_layernorm.weight", "post_ff_ln.scale"},
		{"mlp.ff0.weight", "ff0.weight"},
		{"mlp.ff1.weight", "ff1.weight"},
		{"self_attn.q_norm.weight", "attn.query_ln.scale"},
		{"self_attn.k_norm.weight", "attn.key_ln.scale"},
		{"self_attn.scaling", "attn.per_dim_scale.per_dim_scale"},
		{"self_attn.o_proj.weight", "attn.out.weight"},
	}
	type concatSpec struct {
		target  string
		sources []string
	}
	var concats []concatSpec
	for layer := 0; ; layer++ {
		source := fmt.Sprintf("model.layers.%d.", layer)
		if _, ok := shapes[source+"input_layernorm.weight"]; !ok {
			if layer == 0 {
				return nil, nil, fmt.Errorf("seriesforecast: transformers layout has no model.layers.0")
			}
			break
		}
		destination := fmt.Sprintf("stacked_xf.%d.", layer)
		for _, rename := range layerRenames {
			renames[source+rename[0]] = destination + rename[1]
		}
		concats = append(concats, concatSpec{
			target: destination + "attn.qkv_proj.weight",
			sources: []string{
				source + "self_attn.q_proj.weight",
				source + "self_attn.k_proj.weight",
				source + "self_attn.v_proj.weight",
			},
		})
	}
	outWeights := make(map[string][]float32, len(weights))
	outShapes := make(map[string][]int, len(shapes))
	consumed := map[string]bool{}
	for _, spec := range concats {
		var joined []float32
		var rows int
		var cols int
		for _, name := range spec.sources {
			shape, ok := shapes[name]
			if !ok || len(shape) != 2 {
				return nil, nil, fmt.Errorf("seriesforecast: concat source %q missing or non-matrix", name)
			}
			if cols == 0 {
				cols = shape[1]
			} else if shape[1] != cols {
				return nil, nil, fmt.Errorf("seriesforecast: concat source %q column mismatch", name)
			}
			rows += shape[0]
			joined = append(joined, weights[name]...)
			consumed[name] = true
		}
		outWeights[spec.target] = joined
		outShapes[spec.target] = []int{rows, cols}
	}
	for name, values := range weights {
		if consumed[name] {
			continue
		}
		target := name
		if renamed, ok := renames[name]; ok {
			target = renamed
		}
		if _, exists := outWeights[target]; exists {
			return nil, nil, fmt.Errorf("seriesforecast: duplicate canonical tensor %q", target)
		}
		outWeights[target] = values
		outShapes[target] = shapes[name]
	}
	return outWeights, outShapes, nil
}

func loadConfig(path string) (artifactConfig, error) {
	var config artifactConfig
	if err := jsonfile.Decode(path, &config); err != nil {
		return artifactConfig{}, fmt.Errorf("seriesforecast: parse config.json: %w", err)
	}
	if len(config.Quantiles) == 0 {
		return artifactConfig{}, fmt.Errorf("seriesforecast: config.json quantiles required")
	}
	for i, level := range config.Quantiles {
		if level <= 0 || level >= 1 || i > 0 && level <= config.Quantiles[i-1] {
			return artifactConfig{}, fmt.Errorf("seriesforecast: quantiles must be strictly increasing in (0,1): %v", config.Quantiles)
		}
	}
	if config.RopeParameters.RopeTheta <= 0 {
		return artifactConfig{}, fmt.Errorf("seriesforecast: config.json lacks positive rope_theta")
	}
	if config.RMSNormEps <= 0 {
		return artifactConfig{}, fmt.Errorf("seriesforecast: config.json lacks positive rms_norm_eps")
	}
	return config, nil
}

// dimsFromShapes derives every geometric fact from canonical tensor shapes;
// outputDim is 1+len(levels) (point column plus quantile columns).
func dimsFromShapes(shapes map[string][]int, outputDim int) (Dims, error) {
	var d Dims
	if outputDim < 2 {
		return d, fmt.Errorf("seriesforecast: output dimension %d must hold point and quantile columns", outputDim)
	}
	tokenHidden, err := tensorcatalog.Shape(shapes, "tokenizer.hidden_layer.weight", 2)
	if err != nil {
		return d, err
	}
	if tokenHidden[1]%patchInputStreams != 0 {
		return d, fmt.Errorf("seriesforecast: tokenizer input width %d is not values+mask pairs", tokenHidden[1])
	}
	d.PatchLen = tokenHidden[1] / patchInputStreams

	tokenOutput, err := tensorcatalog.Shape(shapes, "tokenizer.output_layer.weight", 2)
	if err != nil {
		return d, err
	}
	if tokenOutput[1] != tokenHidden[0] {
		return d, fmt.Errorf("seriesforecast: tokenizer hidden width %d != output input %d", tokenHidden[0], tokenOutput[1])
	}
	d.Hidden = tokenOutput[0]

	d.Layers, err = tensorcatalog.IndexedCount(shapes, "stacked_xf.", ".attn.qkv_proj.weight")
	if err != nil {
		return d, err
	}
	for layer := 0; layer < d.Layers; layer++ {
		prefix := fmt.Sprintf("stacked_xf.%d.attn.", layer)
		qkv, err := tensorcatalog.Shape(shapes, prefix+"qkv_proj.weight", 2)
		if err != nil {
			return d, err
		}
		if qkv[0] != qkvProjections*d.Hidden || qkv[1] != d.Hidden {
			return d, fmt.Errorf("seriesforecast: layer %d qkv shape %v incompatible with hidden %d", layer, qkv, d.Hidden)
		}
		queryNorm, err := tensorcatalog.Shape(shapes, prefix+"query_ln.scale", 1)
		if err != nil {
			return d, err
		}
		if layer == 0 {
			d.HeadDim = queryNorm[0]
			if d.Hidden%d.HeadDim != 0 {
				return d, fmt.Errorf("seriesforecast: hidden %d not divisible by head dim %d", d.Hidden, d.HeadDim)
			}
			d.Heads = d.Hidden / d.HeadDim
		} else if queryNorm[0] != d.HeadDim {
			return d, fmt.Errorf("seriesforecast: layer %d head dim %d != layer 0 head dim %d", layer, queryNorm[0], d.HeadDim)
		}
	}

	pointOutput, err := tensorcatalog.Shape(shapes, "output_projection_point.output_layer.weight", 2)
	if err != nil {
		return d, err
	}
	d.Quantiles = outputDim
	if pointOutput[0]%d.Quantiles != 0 {
		return d, fmt.Errorf("seriesforecast: point-head output %d not divisible by output width %d", pointOutput[0], d.Quantiles)
	}
	d.Horizon = pointOutput[0] / d.Quantiles
	return d, nil
}
