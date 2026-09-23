// Package vitencoder runs a vision transformer encoder -- patch embedding, a
// class token, pre-norm attention blocks with layer scale and a gated
// feed-forward, and a final norm -- from a checkpoint whose geometry and
// image processor its own configuration files state and whose tensors a
// reviewed declaration binds, so a new checkpoint of the family is data, not
// code.
package vitencoder

import (
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/jsonfile"
	"overgo/internal/patchtower"
	"overgo/internal/safetensors"
)

// LayerTensors names one block's tensors; each name holds %d for the layer.
// An empty layer-scale name means the block has none.
type LayerTensors struct {
	Norm1Weight string `json:"norm1_weight"`
	Norm1Bias   string `json:"norm1_bias"`
	Query       string `json:"query"`
	QueryBias   string `json:"query_bias"`
	Key         string `json:"key"`
	KeyBias     string `json:"key_bias"`
	Value       string `json:"value"`
	ValueBias   string `json:"value_bias"`
	Output      string `json:"output"`
	OutputBias  string `json:"output_bias"`
	Scale1      string `json:"scale1,omitzero"`
	Norm2Weight string `json:"norm2_weight"`
	Norm2Bias   string `json:"norm2_bias"`
	GateUp      string `json:"gate_up"`
	GateUpBias  string `json:"gate_up_bias"`
	Down        string `json:"down"`
	DownBias    string `json:"down_bias"`
	Scale2      string `json:"scale2,omitzero"`
}

// Declaration is what a checkpoint does not say about itself: where each of
// its tensors lives.
type Declaration struct {
	ModelType string `json:"model_type"`
	Tensors   struct {
		ClassToken         string       `json:"class_token"`
		PositionEmbeddings string       `json:"position_embeddings"`
		PatchWeight        string       `json:"patch_weight"`
		PatchBias          string       `json:"patch_bias"`
		FinalNormWeight    string       `json:"final_norm_weight"`
		FinalNormBias      string       `json:"final_norm_bias"`
		Layer              LayerTensors `json:"layer"`
	} `json:"tensors"`
}

// geometry is what config.json states of the encoder's shape.
type geometry struct {
	ModelType string  `json:"model_type"`
	Hidden    int     `json:"hidden_size"`
	Heads     int     `json:"num_attention_heads"`
	Layers    int     `json:"num_hidden_layers"`
	PatchSize int     `json:"patch_size"`
	ImageSize int     `json:"image_size"`
	Epsilon   float64 `json:"layer_norm_eps"`
}

type layer struct {
	norm1Weight, norm1Bias, query, queryBias, key, keyBias, value, valueBias []float32
	output, outputBias, scale1, norm2Weight, norm2Bias                       []float32
	gateUp, gateUpBias, down, downBias, scale2                               []float32
	intermediate                                                             int
}

// Encoder is a loaded checkpoint.
type Encoder struct {
	geometry                     geometry
	preprocess                   preprocess
	classToken, position         []float32
	patchWeight, patchBias       []float32
	finalWeight, finalBias       []float32
	layers                       []layer
	grid, tokens, patchInputSize int
	pixelCount                   int
}

// Load reads a checkpoint directory: its configuration names the family and
// states the geometry, its image processor configuration states the
// preprocessing, and the family's reviewed declaration binds the tensors. A
// checkpoint no reviewed declaration answers for is refused.
func Load(directory string) (*Encoder, error) {
	var g geometry
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &g); err != nil {
		return nil, fmt.Errorf("vitencoder: checkpoint configuration: %w", err)
	}
	declaration, known, err := declarationFor(g.ModelType)
	if err != nil {
		return nil, err
	}
	if !known {
		return nil, fmt.Errorf("vitencoder: no reviewed declaration answers for model type %q", g.ModelType)
	}
	p, err := loadPreprocess(filepath.Join(directory, "preprocessor_config.json"))
	if err != nil {
		return nil, err
	}
	if g.PatchSize <= 0 || g.Hidden <= 0 || g.Heads <= 0 || g.Hidden%g.Heads != 0 || g.Layers <= 0 || g.Epsilon <= 0 ||
		p.crop != g.ImageSize || p.crop%g.PatchSize != 0 {
		return nil, errors.New("vitencoder: checkpoint geometry is incomplete or disagrees with its image processor")
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	read := func(name string, size int) ([]float32, error) {
		tensor, found := source.Tensors[name]
		if !found {
			return nil, fmt.Errorf("vitencoder: tensor %s is absent", name)
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil {
			return nil, fmt.Errorf("vitencoder: tensor %s: %w", name, err)
		}
		if size != checked.UnknownCount() && len(values) != size {
			return nil, fmt.Errorf("vitencoder: tensor %s holds %d values, want %d", name, len(values), size)
		}
		return values, nil
	}
	e := &Encoder{geometry: g, preprocess: p, grid: p.crop / g.PatchSize, patchInputSize: channels * g.PatchSize * g.PatchSize}
	e.tokens = 1 + e.grid*e.grid
	e.pixelCount = channels * p.crop * p.crop
	t := declaration.Tensors
	for _, binding := range []struct {
		target *[]float32
		name   string
		size   int
	}{
		{&e.classToken, t.ClassToken, g.Hidden},
		{&e.position, t.PositionEmbeddings, e.tokens * g.Hidden},
		{&e.patchWeight, t.PatchWeight, g.Hidden * e.patchInputSize},
		{&e.patchBias, t.PatchBias, g.Hidden},
		{&e.finalWeight, t.FinalNormWeight, g.Hidden},
		{&e.finalBias, t.FinalNormBias, g.Hidden},
	} {
		if *binding.target, err = read(binding.name, binding.size); err != nil {
			return nil, err
		}
	}
	square := g.Hidden * g.Hidden
	for index := range g.Layers {
		names := t.Layer
		var l layer
		for _, binding := range []struct {
			target *[]float32
			name   string
			size   int
		}{
			{&l.norm1Weight, names.Norm1Weight, g.Hidden}, {&l.norm1Bias, names.Norm1Bias, g.Hidden},
			{&l.query, names.Query, square}, {&l.queryBias, names.QueryBias, g.Hidden},
			{&l.key, names.Key, square}, {&l.keyBias, names.KeyBias, g.Hidden},
			{&l.value, names.Value, square}, {&l.valueBias, names.ValueBias, g.Hidden},
			{&l.output, names.Output, square}, {&l.outputBias, names.OutputBias, g.Hidden},
			{&l.scale1, names.Scale1, g.Hidden},
			{&l.norm2Weight, names.Norm2Weight, g.Hidden}, {&l.norm2Bias, names.Norm2Bias, g.Hidden},
			{&l.gateUp, names.GateUp, checked.UnknownCount()}, {&l.gateUpBias, names.GateUpBias, checked.UnknownCount()},
			{&l.down, names.Down, checked.UnknownCount()}, {&l.downBias, names.DownBias, g.Hidden},
			{&l.scale2, names.Scale2, g.Hidden},
		} {
			if binding.name == "" {
				continue
			}
			if *binding.target, err = read(fmt.Sprintf(binding.name, index), binding.size); err != nil {
				return nil, err
			}
		}
		l.intermediate = len(l.gateUpBias) / 2
		if l.intermediate == 0 || len(l.gateUp) != 2*l.intermediate*g.Hidden || len(l.down) != g.Hidden*l.intermediate {
			return nil, fmt.Errorf("vitencoder: layer %d gated feed-forward shapes disagree", index)
		}
		e.layers = append(e.layers, l)
	}
	return e, nil
}

// Embed decodes one PNG or JPEG image, prepares it as the checkpoint's image
// processor does and returns the class token after the final norm, the
// checkpoint's pooled image embedding.
func (e *Encoder) Embed(image []byte) ([]float32, error) {
	rgb, height, width, err := patchtower.DecodeImageBytesRGB(image)
	if err != nil {
		return nil, fmt.Errorf("vitencoder: decode image: %w", err)
	}
	pixels, err := e.preprocess.pixels(rgb, height, width)
	if err != nil {
		return nil, err
	}
	tokens, err := e.encode(pixels)
	if err != nil {
		return nil, err
	}
	return tokens[:e.geometry.Hidden:e.geometry.Hidden], nil
}

// encode runs normalized channel-first pixels through the encoder and returns
// every token after the final norm, the class token first.
func (e *Encoder) encode(pixels []float32) ([]float32, error) {
	g := e.geometry
	crop := e.preprocess.crop
	if !checked.Equal(len(pixels), e.pixelCount) {
		return nil, fmt.Errorf("vitencoder: %d pixels, want %d", len(pixels), e.pixelCount)
	}
	patches := make([]float32, e.grid*e.grid*e.patchInputSize)
	for row := range e.grid {
		for column := range e.grid {
			patch := patches[(row*e.grid+column)*e.patchInputSize:]
			for c := range channels {
				for y := range g.PatchSize {
					for x := range g.PatchSize {
						patch[(c*g.PatchSize+y)*g.PatchSize+x] = pixels[(c*crop+row*g.PatchSize+y)*crop+column*g.PatchSize+x]
					}
				}
			}
		}
	}
	hidden := make([]float32, e.tokens*g.Hidden)
	copy(hidden, e.classToken)
	linear(hidden[g.Hidden:], patches, e.patchWeight, e.patchBias, e.grid*e.grid, e.patchInputSize, g.Hidden)
	for index, value := range e.position {
		hidden[index] += value
	}
	normed := make([]float32, len(hidden))
	q, k, v := make([]float32, len(hidden)), make([]float32, len(hidden)), make([]float32, len(hidden))
	projected := make([]float32, len(hidden))
	for _, l := range e.layers {
		hostmath.LayerNormInto(normed, hidden, l.norm1Weight, l.norm1Bias, e.tokens, g.Hidden, g.Epsilon)
		linear(q, normed, l.query, l.queryBias, e.tokens, g.Hidden, g.Hidden)
		linear(k, normed, l.key, l.keyBias, e.tokens, g.Hidden, g.Hidden)
		linear(v, normed, l.value, l.valueBias, e.tokens, g.Hidden, g.Hidden)
		attended := hostmath.BatchedAttentionF32(q, k, v, len(q)/(e.tokens*g.Hidden), e.tokens, e.tokens, g.Hidden, g.Heads)
		linear(projected, attended, l.output, l.outputBias, e.tokens, g.Hidden, g.Hidden)
		addScaled(hidden, projected, l.scale1, g.Hidden)
		hostmath.LayerNormInto(normed, hidden, l.norm2Weight, l.norm2Bias, e.tokens, g.Hidden, g.Epsilon)
		gated := make([]float32, e.tokens*2*l.intermediate)
		linear(gated, normed, l.gateUp, l.gateUpBias, e.tokens, g.Hidden, 2*l.intermediate)
		inner := make([]float32, e.tokens*l.intermediate)
		for token := range e.tokens {
			row := gated[token*2*l.intermediate:]
			for index := range l.intermediate {
				inner[token*l.intermediate+index] = float32(hostmath.SiLU(float64(row[index]))) * row[l.intermediate+index]
			}
		}
		linear(projected, inner, l.down, l.downBias, e.tokens, l.intermediate, g.Hidden)
		addScaled(hidden, projected, l.scale2, g.Hidden)
	}
	hostmath.LayerNormInto(hidden, hidden, e.finalWeight, e.finalBias, e.tokens, g.Hidden, g.Epsilon)
	for _, value := range hidden {
		if !checked.Finite32(value) {
			return nil, errors.New("vitencoder: the encoder produced a non-finite value")
		}
	}
	return hidden, nil
}

// linear is dst = x * w^T + bias over rows.
func linear(dst, x, w, bias []float32, rows, in, out int) {
	hostmath.Linear(dst[:rows*out], x[:rows*in], w, rows, in, out)
	for row := range rows {
		for column, value := range bias {
			dst[row*out+column] += value
		}
	}
}

// addScaled adds a branch to the residual stream, per channel scaled when the
// block declares a layer scale.
func addScaled(residual, branch, scale []float32, width int) {
	for index, value := range branch {
		if scale != nil {
			value *= scale[index%width]
		}
		residual[index] += value
	}
}
