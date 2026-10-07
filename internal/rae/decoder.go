// Package rae runs a representation autoencoder (RAE, bytetriper/RAE): the
// latent grid of a frozen representation encoder, decoded to pixels by a
// trained ViT decoder -- the space a stage-2 diffusion transformer generates
// in. A decoder's geometry is read from its checkpoint's own tensors; what
// they cannot state (attention heads, the norm epsilon) is the reviewed
// declaration of the reference's decoder configurations. A collection is
// served through the image-gen capability with the reviewed declaration of
// the reference configuration it samples with: an ImageNet class and a seed
// in, the decoded image out.
package rae

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/jsonfile"
	"overgo/internal/pytorchzip"
	"overgo/internal/strictjson"
)

//go:embed recipes/decoders.json
var decoderDeclarations []byte

// decoderDeclaration is what a decoder checkpoint does not state of itself.
type decoderDeclaration struct {
	Hidden  int     `json:"decoder_hidden_size"`
	Heads   int     `json:"decoder_num_attention_heads"`
	Epsilon float64 `json:"layer_norm_eps"`
}

// statisticsEpsilon keeps the latent scale positive where a channel's
// variance is zero, as the reference's decode does.
const statisticsEpsilon = 1e-5

// processorConfig is the file in an encoder's repository that holds its
// image processor: the pixel mean and deviation it normalizes by.
const processorConfig = "preprocessor_config.json"

// layerPrefix names one decoder layer's tensors in the checkpoint.
const layerPrefix = "decoder_layers."

// The ranks the decoder's tensors carry: a weight matrix is [out, in] and the
// position embedding is [batch, sequence, hidden].
const (
	matrixRank   = 2
	sequenceRank = 3
)

// Each decoder layer's tensors, after its prefix, in the order layer reads
// them.
var layerTensorNames = []string{
	"layernorm_before.weight", "layernorm_before.bias",
	"attention.attention.query.weight", "attention.attention.query.bias",
	"attention.attention.key.weight", "attention.attention.key.bias",
	"attention.attention.value.weight", "attention.attention.value.bias",
	"attention.output.dense.weight", "attention.output.dense.bias",
	"layernorm_after.weight", "layernorm_after.bias",
	"intermediate.dense.weight", "intermediate.dense.bias",
	"output.dense.weight", "output.dense.bias",
}

// The decoder's own tensors, in the order loadDecoder reads them.
var decoderTensorNames = []string{
	"decoder_embed.weight", "decoder_embed.bias", "trainable_cls_token", "decoder_pos_embed",
	"decoder_norm.weight", "decoder_norm.bias", "decoder_pred.weight", "decoder_pred.bias",
}

// layer is one pre-norm transformer block: attention, then a GELU MLP.
type layer struct {
	norm1Weight, norm1Bias, query, queryBias, key, keyBias, value, valueBias []float32
	output, outputBias, norm2Weight, norm2Bias, up, upBias, down, downBias   []float32
}

// vitDecoder is a loaded RAE decoder with the latent statistics its latents
// are normalized by and the pixel normalization of the encoder that made
// them.
type vitDecoder struct {
	latent, hidden, heads, intermediate int
	grid, patch                         int
	epsilon                             float64
	embedWeight, embedBias              []float32
	class, position                     []float32
	normWeight, normBias                []float32
	predWeight, predBias                []float32
	layers                              []layer
	scale, shift                        []float32
	pixelMean, pixelStd                 []float32
}

// loadDecoder reads a decoder checkpoint, the latent statistics its latents
// were normalized by, and the pixel mean and deviation of the encoder whose
// image processor configuration lives in encoder.
func loadDecoder(checkpoint, statistics, encoder string) (*vitDecoder, error) {
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		return nil, fmt.Errorf("rae: decoder checkpoint: %w", err)
	}
	shapes := map[string][]int64{}
	layers := 0
	for _, tensor := range catalog.Tensors {
		shapes[tensor.Name] = tensor.Shape
		if rest, ok := strings.CutPrefix(tensor.Name, layerPrefix); ok {
			index, _, _ := strings.Cut(rest, ".")
			if value, err := strconv.Atoi(index); err == nil {
				layers = max(layers, value+1)
			}
		}
	}
	embed, position, pred := shapes["decoder_embed.weight"], shapes["decoder_pos_embed"], shapes["decoder_pred.weight"]
	up := shapes[layerPrefix+"0.intermediate.dense.weight"]
	if len(embed) != matrixRank || len(position) != sequenceRank || len(pred) != matrixRank || len(up) != matrixRank || layers == 0 {
		return nil, errors.New("rae: the decoder checkpoint lacks its embedding, position, prediction or layer tensors")
	}
	d := &vitDecoder{latent: int(embed[1]), hidden: int(embed[0]), intermediate: int(up[0])}
	declaration, err := declarationFor(d.hidden)
	if err != nil {
		return nil, err
	}
	d.heads, d.epsilon = declaration.Heads, declaration.Epsilon
	var preprocess struct {
		Mean []float32 `json:"image_mean"`
		Std  []float32 `json:"image_std"`
	}
	if err := jsonfile.Decode(filepath.Join(encoder, processorConfig), &preprocess); err != nil {
		return nil, fmt.Errorf("rae: encoder image processor: %w", err)
	}
	d.pixelMean, d.pixelStd = preprocess.Mean, preprocess.Std
	channels := len(d.pixelMean)
	patches := int(position[1]) - 1
	d.grid = int(math.Sqrt(float64(patches)))
	d.patch = int(math.Sqrt(float64(int(pred[0]) / max(channels, 1))))
	if channels == 0 || len(d.pixelStd) != channels || d.grid*d.grid != patches || d.patch*d.patch*channels != int(pred[0]) {
		return nil, errors.New("rae: the decoder's patch grid or pixel channels disagree with its tensors")
	}
	names := slices.Clone(decoderTensorNames)
	for index := range layers {
		for _, name := range layerTensorNames {
			names = append(names, layerPrefix+strconv.Itoa(index)+"."+name)
		}
	}
	values, err := readTensors(checkpoint, catalog, names)
	if err != nil {
		return nil, err
	}
	fill(d.tensors(), values)
	for index := range layers {
		var l layer
		fill(l.tensors(), values[len(decoderTensorNames)+index*len(layerTensorNames):])
		d.layers = append(d.layers, l)
	}
	return d, d.loadStatistics(statistics)
}

// tensors are the decoder's own weights in the order decoderTensorNames
// names them.
func (d *vitDecoder) tensors() []*[]float32 {
	return []*[]float32{&d.embedWeight, &d.embedBias, &d.class, &d.position, &d.normWeight, &d.normBias, &d.predWeight, &d.predBias}
}

// tensors are the layer's weights in the order layerTensorNames names them.
func (l *layer) tensors() []*[]float32 {
	return []*[]float32{
		&l.norm1Weight, &l.norm1Bias, &l.query, &l.queryBias, &l.key, &l.keyBias, &l.value, &l.valueBias,
		&l.output, &l.outputBias, &l.norm2Weight, &l.norm2Bias, &l.up, &l.upBias, &l.down, &l.downBias,
	}
}

// fill binds values, read in the same order, to the weights they name.
func fill(targets []*[]float32, values [][]float32) {
	for index, target := range targets {
		*target = values[index]
	}
}

// loadStatistics reads the latent mean and variance the latents were
// normalized by; an absent mean is zero, as the reference reads it.
func (d *vitDecoder) loadStatistics(statistics string) error {
	catalog, err := pytorchzip.ReadCatalog(statistics)
	if err != nil {
		return fmt.Errorf("rae: latent statistics: %w", err)
	}
	size := d.latent * d.grid * d.grid
	names := []string{"var"}
	if slices.ContainsFunc(catalog.Tensors, func(tensor pytorchzip.TensorMeta) bool { return tensor.Name == "mean" }) {
		names = append(names, "mean")
	}
	values, err := readTensors(statistics, catalog, names)
	if err != nil {
		return err
	}
	if len(values[0]) != size {
		return fmt.Errorf("rae: latent variance holds %d values, want %d", len(values[0]), size)
	}
	d.scale, d.shift = make([]float32, size), make([]float32, size)
	for index, variance := range values[0] {
		d.scale[index] = float32(math.Sqrt(float64(variance) + statisticsEpsilon))
	}
	if len(values) > 1 {
		if len(values[1]) != size {
			return fmt.Errorf("rae: latent mean holds %d values, want %d", len(values[1]), size)
		}
		copy(d.shift, values[1])
	}
	return nil
}

// decode turns one normalized latent, channel-first over the latent grid,
// into channel-first pixels in the encoder's input range.
func (d *vitDecoder) decode(latent []float32) ([]float32, error) {
	tokens := d.grid * d.grid
	if !checked.Equal(len(latent), d.latent*tokens) {
		return nil, fmt.Errorf("rae: latent holds %d values, want %d", len(latent), d.latent*tokens)
	}
	rows := make([]float32, tokens*d.latent)
	for channel := range d.latent {
		for token := range tokens {
			index := channel*tokens + token
			rows[token*d.latent+channel] = latent[index]*d.scale[index] + d.shift[index]
		}
	}
	return d.decodeTokens(rows)
}

// decodeTokens turns the encoder's own tokens, one row of latent channels
// for each patch of the grid, into channel-first pixels in the encoder's
// input range: what decode reads once it has undone a latent's normalization.
func (d *vitDecoder) decodeTokens(rows []float32) ([]float32, error) {
	tokens := d.grid * d.grid
	if !checked.Equal(len(rows), d.latent*tokens) {
		return nil, fmt.Errorf("rae: %d token values, want %d", len(rows), d.latent*tokens)
	}
	sequence := tokens + 1
	hidden := make([]float32, sequence*d.hidden)
	copy(hidden, d.class)
	linear(hidden[d.hidden:], rows, d.embedWeight, d.embedBias, tokens, d.latent, d.hidden)
	for index, value := range d.position {
		hidden[index] += value
	}
	normed := make([]float32, len(hidden))
	q, k, v := make([]float32, len(hidden)), make([]float32, len(hidden)), make([]float32, len(hidden))
	projected := make([]float32, len(hidden))
	inner := make([]float32, sequence*d.intermediate)
	for _, l := range d.layers {
		hostmath.LayerNormInto(normed, hidden, l.norm1Weight, l.norm1Bias, sequence, d.hidden, d.epsilon)
		linear(q, normed, l.query, l.queryBias, sequence, d.hidden, d.hidden)
		linear(k, normed, l.key, l.keyBias, sequence, d.hidden, d.hidden)
		linear(v, normed, l.value, l.valueBias, sequence, d.hidden, d.hidden)
		attended := hostmath.BatchedAttentionF32(q, k, v, 1, sequence, sequence, d.hidden, d.heads)
		linear(projected, attended, l.output, l.outputBias, sequence, d.hidden, d.hidden)
		add(hidden, projected)
		hostmath.LayerNormInto(normed, hidden, l.norm2Weight, l.norm2Bias, sequence, d.hidden, d.epsilon)
		linear(inner, normed, l.up, l.upBias, sequence, d.hidden, d.intermediate)
		hostmath.GELUErfInPlace(inner)
		linear(projected, inner, l.down, l.downBias, sequence, d.intermediate, d.hidden)
		add(hidden, projected)
	}
	hostmath.LayerNormInto(hidden, hidden, d.normWeight, d.normBias, sequence, d.hidden, d.epsilon)
	channels := len(d.pixelMean)
	width := d.patch * channels
	predicted := make([]float32, tokens*d.patch*width)
	linear(predicted, hidden[d.hidden:], d.predWeight, d.predBias, tokens, d.hidden, d.patch*width)
	side := d.grid * d.patch
	pixels := make([]float32, channels*side*side)
	for row := range d.grid {
		for column := range d.grid {
			patch := predicted[(row*d.grid+column)*d.patch*width:]
			for y := range d.patch {
				for x := range d.patch {
					for channel := range channels {
						value := patch[(y*d.patch+x)*channels+channel]
						pixels[(channel*side+row*d.patch+y)*side+column*d.patch+x] = value*d.pixelStd[channel] + d.pixelMean[channel]
					}
				}
			}
		}
	}
	for _, value := range pixels {
		if !checked.Finite32(value) {
			return nil, errors.New("rae: the decoder produced a non-finite value")
		}
	}
	return pixels, nil
}

// declarationFor returns the reviewed configuration of the decoder whose
// hidden width is hidden.
func declarationFor(hidden int) (decoderDeclaration, error) {
	var document struct {
		Doc      string               `json:"doc"`
		Decoders []decoderDeclaration `json:"decoders"`
	}
	if err := strictjson.DecodeBytes(decoderDeclarations, &document); err != nil {
		return decoderDeclaration{}, fmt.Errorf("rae: decoder declarations: %w", err)
	}
	for _, declaration := range document.Decoders {
		if declaration.Hidden == hidden && declaration.Heads > 0 && hidden%declaration.Heads == 0 {
			return declaration, nil
		}
	}
	return decoderDeclaration{}, fmt.Errorf("rae: no reviewed decoder configuration answers for hidden width %d", hidden)
}

// readTensors reads the named tensors of a checkpoint as float32, in order.
func readTensors(checkpoint string, catalog pytorchzip.Catalog, names []string) ([][]float32, error) {
	bindings, err := pytorchzip.CompileBindings(catalog.Tensors, names)
	if err != nil {
		return nil, fmt.Errorf("rae: %s: %w", filepath.Base(checkpoint), err)
	}
	reader, err := pytorchzip.Open(checkpoint)
	if err != nil {
		return nil, err
	}
	values, err := reader.ReadBindingValues(bindings)
	return values, errors.Join(err, reader.Close())
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

// add adds a branch to the residual stream.
func add(residual, branch []float32) {
	for index, value := range branch {
		residual[index] += value
	}
}
