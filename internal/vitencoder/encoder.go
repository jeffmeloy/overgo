// Package vitencoder runs a vision transformer encoder -- patch embedding, a
// class token and any register tokens, pre-norm attention blocks with layer
// scale and a gated or a plain feed-forward, and a final norm -- from a
// checkpoint whose geometry and image processor its own configuration files
// state and whose tensors a reviewed declaration binds, so a new checkpoint
// of a family is data, not code.
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
// An empty layer-scale name means the block has none. A block names either
// the gated feed-forward's fused input (gate_up) or the plain one's (up).
type LayerTensors struct {
	Norm1Weight string `json:"norm1_weight"`
	Norm1Bias   string `json:"norm1_bias"`
	Query       string `json:"query"`
	QueryBias   string `json:"query_bias"`
	Key         string `json:"key"`
	KeyBias     string `json:"key_bias"`
	Value       string `json:"value"`
	ValueBias   string `json:"value_bias"`
	// QKV and QKVBias name the query, key and value fused in one projection, in that order, for a checkpoint that
	// fuses them, in place of Query, Key and Value.
	QKV         string `json:"qkv,omitzero"`
	QKVBias     string `json:"qkv_bias,omitzero"`
	Output      string `json:"output"`
	OutputBias  string `json:"output_bias"`
	Scale1      string `json:"scale1,omitzero"`
	Norm2Weight string `json:"norm2_weight"`
	Norm2Bias   string `json:"norm2_bias"`
	GateUp      string `json:"gate_up,omitzero"`
	GateUpBias  string `json:"gate_up_bias,omitzero"`
	Up          string `json:"up,omitzero"`
	UpBias      string `json:"up_bias,omitzero"`
	Down        string `json:"down"`
	DownBias    string `json:"down_bias"`
	Scale2      string `json:"scale2,omitzero"`
}

// Declaration is what a checkpoint does not say about itself: where each of
// its tensors lives.
type Declaration struct {
	ModelType string `json:"model_type"`
	// Geometry is the encoder's shape, for a checkpoint that states none of its own (a PyTorch file).
	Geometry *geometry `json:"geometry,omitzero"`
	// Depth is a depth checkpoint's head and how it reads the encoder (DepthModel).
	Depth   *depthSpec `json:"depth,omitzero"`
	Tensors struct {
		ClassToken         string       `json:"class_token"`
		RegisterTokens     string       `json:"register_tokens,omitzero"`
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
	Registers int     `json:"num_register_tokens"`
	Epsilon   float64 `json:"layer_norm_eps"`
}

type layer struct {
	norm1Weight, norm1Bias, query, queryBias, key, keyBias, value, valueBias []float32
	output, outputBias, scale1, norm2Weight, norm2Bias                       []float32
	gateUp, gateUpBias, up, upBias, down, downBias, scale2                   []float32
	intermediate                                                             int
}

// Encoder is a loaded checkpoint.
type Encoder struct {
	geometry                     geometry
	preprocess                   preprocess
	classToken, position         []float32
	registers                    []float32
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
	t := declaration.Tensors
	if g.PatchSize <= 0 || g.Hidden <= 0 || g.Heads <= 0 || g.Hidden%g.Heads != 0 || g.Layers <= 0 || g.Epsilon <= 0 || g.Registers < 0 ||
		p.crop <= 0 || p.crop%g.PatchSize != 0 || g.ImageSize <= 0 || g.ImageSize%g.PatchSize != 0 || (t.RegisterTokens == "") != (g.Registers == 0) {
		return nil, errors.New("vitencoder: checkpoint geometry is incomplete or disagrees with its image processor or declaration")
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
	e.tokens = 1 + g.Registers + e.grid*e.grid
	e.pixelCount = channels * p.crop * p.crop
	// The checkpoint holds a position for the class token and for each patch
	// of the grid it was trained at.
	trained := g.ImageSize / g.PatchSize
	for _, binding := range []struct {
		target *[]float32
		name   string
		size   int
	}{
		{&e.classToken, t.ClassToken, g.Hidden},
		{&e.position, t.PositionEmbeddings, (1 + trained*trained) * g.Hidden},
		{&e.registers, t.RegisterTokens, g.Registers * g.Hidden},
		{&e.patchWeight, t.PatchWeight, g.Hidden * e.patchInputSize},
		{&e.patchBias, t.PatchBias, g.Hidden},
		{&e.finalWeight, t.FinalNormWeight, g.Hidden},
		{&e.finalBias, t.FinalNormBias, g.Hidden},
	} {
		if binding.name == "" {
			continue
		}
		if *binding.target, err = read(binding.name, binding.size); err != nil {
			return nil, err
		}
	}
	e.position = fitPositions(e.position, trained, e.grid, g.Hidden)
	if e.layers, err = bindLayers(read, g, t.Layer); err != nil {
		return nil, err
	}
	return e, nil
}

// bindLayers reads each block's tensors by the declaration's names through read, which refuses a tensor of another
// size than it is given (checked.UnknownCount() for any). A declaration that fuses the query, key and value in one
// projection (qkv) has it split into its thirds, in that order, rows and biases alike.
func bindLayers(read func(name string, size int) ([]float32, error), g geometry, names LayerTensors) ([]layer, error) {
	square := g.Hidden * g.Hidden
	if (names.QKV == "") == (names.Query == "") {
		return nil, errors.New("vitencoder: a declaration names its query, key and value, or their fused projection, not both")
	}
	var layers []layer
	for index := range g.Layers {
		var l layer
		var fused, fusedBias []float32
		for _, binding := range []struct {
			target *[]float32
			name   string
			size   int
		}{
			{&l.norm1Weight, names.Norm1Weight, g.Hidden}, {&l.norm1Bias, names.Norm1Bias, g.Hidden},
			{&l.query, names.Query, square}, {&l.queryBias, names.QueryBias, g.Hidden},
			{&l.key, names.Key, square}, {&l.keyBias, names.KeyBias, g.Hidden},
			{&l.value, names.Value, square}, {&l.valueBias, names.ValueBias, g.Hidden},
			{&fused, names.QKV, 3 * square}, {&fusedBias, names.QKVBias, 3 * g.Hidden},
			{&l.output, names.Output, square}, {&l.outputBias, names.OutputBias, g.Hidden},
			{&l.scale1, names.Scale1, g.Hidden},
			{&l.norm2Weight, names.Norm2Weight, g.Hidden}, {&l.norm2Bias, names.Norm2Bias, g.Hidden},
			{&l.gateUp, names.GateUp, checked.UnknownCount()}, {&l.gateUpBias, names.GateUpBias, checked.UnknownCount()},
			{&l.up, names.Up, checked.UnknownCount()}, {&l.upBias, names.UpBias, checked.UnknownCount()},
			{&l.down, names.Down, checked.UnknownCount()}, {&l.downBias, names.DownBias, g.Hidden},
			{&l.scale2, names.Scale2, g.Hidden},
		} {
			if binding.name == "" {
				continue
			}
			var err error
			if *binding.target, err = read(fmt.Sprintf(binding.name, index), binding.size); err != nil {
				return nil, err
			}
		}
		if fused != nil {
			l.query, l.key, l.value = fused[:square], fused[square:2*square], fused[2*square:]
			if fusedBias != nil {
				l.queryBias, l.keyBias, l.valueBias = fusedBias[:g.Hidden], fusedBias[g.Hidden:2*g.Hidden], fusedBias[2*g.Hidden:]
			}
		}
		// The gated feed-forward fuses its gate and its value in one input
		// of twice the width; the plain one's input is the width itself.
		l.intermediate = len(l.gateUpBias)/2 + len(l.upBias)
		if l.intermediate == 0 || (l.gateUp == nil) == (l.up == nil) || len(l.gateUp)+len(l.up) != (len(l.gateUpBias)+len(l.upBias))*g.Hidden ||
			len(l.down) != g.Hidden*l.intermediate {
			return nil, fmt.Errorf("vitencoder: layer %d feed-forward shapes disagree", index)
		}
		layers = append(layers, l)
	}
	return layers, nil
}

// Embed decodes one PNG or JPEG image, prepares it as the checkpoint's image
// processor does and returns the class token after the final norm, the
// checkpoint's pooled image embedding.
func (e *Encoder) Embed(image []byte) ([]float32, error) {
	tokens, err := e.Tokens(image)
	if err != nil {
		return nil, err
	}
	return tokens[:e.geometry.Hidden:e.geometry.Hidden], nil
}

// Tokens decodes one PNG or JPEG image as Embed does and returns every token
// after the final norm, each of the encoder's width: the class token, any
// registers, then the patches, the output an encoder's seam reads.
func (e *Encoder) Tokens(image []byte) ([]float32, error) {
	pixels, err := e.imagePixels(image)
	if err != nil {
		return nil, err
	}
	return e.encode(pixels, true)
}

// imagePixels decodes one PNG or JPEG image and prepares it as the checkpoint's image processor does: normalized
// channel-first pixels of the encoder's input size.
func (e *Encoder) imagePixels(image []byte) ([]float32, error) {
	rgb, height, width, err := patchtower.DecodeImageBytesRGB(image)
	if err != nil {
		return nil, fmt.Errorf("vitencoder: decode image: %w", err)
	}
	return e.preprocess.pixels(rgb, height, width)
}

// Patches runs normalized channel-first pixels of the encoder's input size
// through the encoder and returns its patch tokens, row by row over the grid,
// without the class and register tokens. affine keeps the final norm's
// learned scale and shift; a representation autoencoder reads the tokens of
// the norm without them.
func (e *Encoder) Patches(pixels []float32, affine bool) ([]float32, error) {
	tokens, err := e.encode(pixels, affine)
	if err != nil {
		return nil, err
	}
	return tokens[(1+e.geometry.Registers)*e.geometry.Hidden:], nil
}

// Shape is the side of the square input the encoder takes, the side of its
// patch grid, and the width of each token.
func (e *Encoder) Shape() (input, grid, hidden int) {
	return e.preprocess.crop, e.grid, e.geometry.Hidden
}

// encode runs normalized channel-first pixels through the encoder and returns
// every token after the final norm: the class token, any registers, then the
// patches.
func (e *Encoder) encode(pixels []float32, affine bool) ([]float32, error) {
	g := e.geometry
	hidden, err := e.squareEmbed(pixels)
	if err != nil {
		return nil, err
	}
	e.blocks(hidden, e.tokens, nil)
	finalWeight, finalBias := e.finalWeight, e.finalBias
	if !affine {
		finalWeight, finalBias = nil, nil
	}
	hostmath.LayerNormInto(hidden, hidden, finalWeight, finalBias, e.tokens, g.Hidden, g.Epsilon)
	return finite(hidden)
}

// blocks runs the encoder's blocks over tokens rows of hidden in place, calling after, where given, with each block's
// index once the block has run.
func (e *Encoder) blocks(hidden []float32, tokens int, after func(index int)) {
	g := e.geometry
	normed := make([]float32, len(hidden))
	q, k, v := make([]float32, len(hidden)), make([]float32, len(hidden)), make([]float32, len(hidden))
	projected := make([]float32, len(hidden))
	for index, l := range e.layers {
		hostmath.LayerNormInto(normed, hidden, l.norm1Weight, l.norm1Bias, tokens, g.Hidden, g.Epsilon)
		hostmath.LinearBias(q, normed, l.query, l.queryBias, tokens, g.Hidden, g.Hidden)
		hostmath.LinearBias(k, normed, l.key, l.keyBias, tokens, g.Hidden, g.Hidden)
		hostmath.LinearBias(v, normed, l.value, l.valueBias, tokens, g.Hidden, g.Hidden)
		attended := hostmath.BatchedAttentionF32(q, k, v, len(q)/(tokens*g.Hidden), tokens, tokens, g.Hidden, g.Heads)
		hostmath.LinearBias(projected, attended, l.output, l.outputBias, tokens, g.Hidden, g.Hidden)
		addScaled(hidden, projected, l.scale1, g.Hidden)
		hostmath.LayerNormInto(normed, hidden, l.norm2Weight, l.norm2Bias, tokens, g.Hidden, g.Epsilon)
		inner := make([]float32, tokens*l.intermediate)
		if l.up != nil {
			hostmath.LinearBias(inner, normed, l.up, l.upBias, tokens, g.Hidden, l.intermediate)
			hostmath.GELUErfInPlace(inner)
		} else {
			gated := make([]float32, tokens*2*l.intermediate)
			hostmath.LinearBias(gated, normed, l.gateUp, l.gateUpBias, tokens, g.Hidden, 2*l.intermediate)
			for token := range tokens {
				row := gated[token*2*l.intermediate:]
				for index := range l.intermediate {
					inner[token*l.intermediate+index] = float32(hostmath.SiLU(float64(row[index]))) * row[l.intermediate+index]
				}
			}
		}
		hostmath.LinearBias(projected, inner, l.down, l.downBias, tokens, l.intermediate, g.Hidden)
		addScaled(hidden, projected, l.scale2, g.Hidden)
		if after != nil {
			after(index)
		}
	}
}

// squareEmbed is the encoder's input rows for normalized channel-first pixels of its square input size.
func (e *Encoder) squareEmbed(pixels []float32) ([]float32, error) {
	if !checked.Equal(len(pixels), e.pixelCount) {
		return nil, fmt.Errorf("vitencoder: %d pixels, want %d", len(pixels), e.pixelCount)
	}
	return e.embed(pixels, e.grid, e.grid, e.position), nil
}

// embed is the encoder's input rows for normalized channel-first pixels of rows by columns patches: the class token,
// any registers, then the patches, row by row, the class token and the patches with position's positions.
func (e *Encoder) embed(pixels []float32, rows, columns int, position []float32) []float32 {
	g := e.geometry
	width := columns * g.PatchSize
	height := rows * g.PatchSize
	patches := make([]float32, rows*columns*e.patchInputSize)
	for row := range rows {
		for column := range columns {
			patch := patches[(row*columns+column)*e.patchInputSize:]
			for c := range channels {
				for y := range g.PatchSize {
					for x := range g.PatchSize {
						patch[(c*g.PatchSize+y)*g.PatchSize+x] = pixels[(c*height+row*g.PatchSize+y)*width+column*g.PatchSize+x]
					}
				}
			}
		}
	}
	// The class token and the patches take their positions; the registers
	// sit between them and take none.
	hidden := make([]float32, (1+g.Registers+rows*columns)*g.Hidden)
	patched := hidden[(1+g.Registers)*g.Hidden:]
	hostmath.LinearBias(patched, patches, e.patchWeight, e.patchBias, rows*columns, e.patchInputSize, g.Hidden)
	for index, value := range position[g.Hidden:] {
		patched[index] += value
	}
	for index, value := range e.classToken {
		hidden[index] = value + position[index]
	}
	copy(hidden[g.Hidden:], e.registers)
	return hidden
}

// finite is tokens, refused when one holds a non-finite value.
func finite(tokens []float32) ([]float32, error) {
	for _, value := range tokens {
		if !checked.Finite32(value) {
			return nil, errors.New("vitencoder: the encoder produced a non-finite value")
		}
	}
	return tokens, nil
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
