package rae

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/pytorchzip"
)

// rmsEpsilon is the reference RMSNorm's default epsilon, which every DiT-DH
// norm uses.
const rmsEpsilon = 1e-6

// The DiT-DH block's adaptive modulation: shift, scale and gate for the
// attention and for the MLP, in the reference's chunk order; the final
// layer's modulation carries a shift and a scale.
const (
	blockModulations = 6
	finalModulations = 2
)

// A fused projection holds its parts one after another along the output:
// the attention's query, key and value; the SwiGLU's gate and value.
const (
	attentionParts = 3
	gatedParts     = 2
)

// fourierTurn scales a time into the Gaussian Fourier features' radians.
const fourierTurn = 2 * math.Pi

// The Gaussian Fourier features are a sine and a cosine of each frequency;
// the rotary embedding turns channels in adjacent pairs.
const (
	fourierParts = 2
	rotaryPair   = 2
)

// blockPrefix names one DiT-DH block's tensors in the checkpoint.
const blockPrefix = "blocks."

// Each block's tensors, after its prefix, in the order block reads them.
var blockTensorNames = []string{
	"norm1.weight", "norm2.weight", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
	"mlp.w12.weight", "mlp.w12.bias", "mlp.w3.weight", "mlp.w3.bias",
	"adaLN_modulation.1.weight", "adaLN_modulation.1.bias",
}

// The model's own tensors, in the order loadDiT reads them.
var ditTensorNames = []string{
	"pos_embed", "s_embedder.proj.weight", "s_embedder.proj.bias", "x_embedder.proj.weight", "x_embedder.proj.bias",
	"t_embedder.W", "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias",
	"y_embedder.embedding_table.weight",
	"final_layer.norm_final.weight", "final_layer.linear.weight", "final_layer.linear.bias",
	"final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.bias",
	"enc_feat_rope.freqs_cos", "enc_feat_rope.freqs_sin", "dec_feat_rope.freqs_cos", "dec_feat_rope.freqs_sin",
}

// The projection from the encoder's width to the head's, which a head wider
// than its encoder always has.
var projectorTensorNames = []string{"s_projector.weight", "s_projector.bias"}

// block is one DiT-DH block: RMS-normed attention with 2-D rotary positions
// and a SwiGLU MLP, each modulated and gated by its condition.
type block struct {
	width, heads, gated                    int
	norm1, norm2, qkv, qkvBias, proj, bias []float32
	up, upBias, down, downBias             []float32
	modulation, modulationBias             []float32
}

// tensors are the block's weights in the order blockTensorNames names them.
func (b *block) tensors() []*[]float32 {
	return []*[]float32{
		&b.norm1, &b.norm2, &b.qkv, &b.qkvBias, &b.proj, &b.bias,
		&b.up, &b.upBias, &b.down, &b.downBias, &b.modulation, &b.modulationBias,
	}
}

// diffusionTransformer is a loaded stage-2 DiT with a DDT head: an encoder
// stack conditioned on time and class, then a wide head stack conditioned
// token by token on the encoder's output, predicting the flow velocity.
type diffusionTransformer struct {
	latent, grid, encoderWidth, headWidth, classes int
	position, sWeight, sBias, xWeight, xBias       []float32
	fourier, time1, time1Bias, time2, time2Bias    []float32
	labels                                         []float32
	finalNorm, finalWeight, finalBias              []float32
	finalModulation, finalModulationBias           []float32
	encoderCos, encoderSin, headCos, headSin       []float32
	projector, projectorBias                       []float32
	encoder, head                                  []block
}

// tensors are the model's weights in the order ditTensorNames names them.
func (m *diffusionTransformer) tensors() []*[]float32 {
	return []*[]float32{
		&m.position, &m.sWeight, &m.sBias, &m.xWeight, &m.xBias,
		&m.fourier, &m.time1, &m.time1Bias, &m.time2, &m.time2Bias, &m.labels,
		&m.finalNorm, &m.finalWeight, &m.finalBias, &m.finalModulation, &m.finalModulationBias,
		&m.encoderCos, &m.encoderSin, &m.headCos, &m.headSin,
	}
}

// loadDiT reads a DiT-DH checkpoint. Its geometry is its tensors': the
// latent and the two widths from the embedders, the class count from the
// label table, the heads from the rotary tables' width, which is one head's.
func loadDiT(checkpoint string) (*diffusionTransformer, error) {
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		return nil, fmt.Errorf("rae: DiT checkpoint: %w", err)
	}
	shapes := map[string][]int64{}
	blocks := 0
	for _, tensor := range catalog.Tensors {
		shapes[tensor.Name] = tensor.Shape
		if rest, ok := strings.CutPrefix(tensor.Name, blockPrefix); ok {
			index, _, _ := strings.Cut(rest, ".")
			if value, err := strconv.Atoi(index); err == nil {
				blocks = max(blocks, value+1)
			}
		}
	}
	s, x, labels := shapes["s_embedder.proj.weight"], shapes["x_embedder.proj.weight"], shapes["y_embedder.embedding_table.weight"]
	position, encoderRope, headRope := shapes["pos_embed"], shapes["enc_feat_rope.freqs_cos"], shapes["dec_feat_rope.freqs_cos"]
	if len(s) < matrixRank || len(x) < matrixRank || len(labels) != matrixRank || len(position) != sequenceRank ||
		len(encoderRope) != matrixRank || len(headRope) != matrixRank || blocks == 0 {
		return nil, errors.New("rae: the DiT checkpoint lacks its embedders, label table, positions or rotary tables")
	}
	m := &diffusionTransformer{
		latent: int(s[1]), encoderWidth: int(s[0]), headWidth: int(x[0]), classes: int(labels[0]) - 1,
		grid: int(math.Sqrt(float64(position[1]))),
	}
	// The encoder stack ends where the block width changes to the head's; a
	// head as wide as the encoder leaves the split unstated.
	if m.encoderWidth == m.headWidth {
		return nil, errors.New("rae: the DiT's encoder and head share a width, so its blocks cannot be split by their tensors")
	}
	names := append(append([]string{}, ditTensorNames...), projectorTensorNames...)
	for index := range blocks {
		for _, name := range blockTensorNames {
			names = append(names, blockPrefix+strconv.Itoa(index)+"."+name)
		}
	}
	values, err := readTensors(checkpoint, catalog, names)
	if err != nil {
		return nil, err
	}
	fill(m.tensors(), values)
	rest := values[len(ditTensorNames):]
	fill([]*[]float32{&m.projector, &m.projectorBias}, rest)
	rest = rest[len(projectorTensorNames):]
	for index := range blocks {
		var b block
		fill(b.tensors(), rest[index*len(blockTensorNames):])
		b.width = len(b.norm1)
		b.gated = len(b.upBias) / gatedParts
		if b.width == m.encoderWidth {
			b.heads = b.width / int(encoderRope[1])
			m.encoder = append(m.encoder, b)
			continue
		}
		b.heads = b.width / int(headRope[1])
		m.head = append(m.head, b)
	}
	if m.grid*m.grid != int(position[1]) || len(m.head) == 0 || len(m.encoder) == 0 {
		return nil, errors.New("rae: the DiT's position grid or block stacks disagree with its tensors")
	}
	return m, nil
}

// velocity predicts the flow velocity of one channel-first latent at time t
// for a class label.
func (m *diffusionTransformer) velocity(latent []float32, t float32, label int) ([]float32, error) {
	tokens := m.grid * m.grid
	if !checked.Equal(len(latent), m.latent*tokens) || label < 0 || label >= m.classes {
		return nil, fmt.Errorf("rae: latent of %d values or label %d does not fit the DiT", len(latent), label)
	}
	rows := make([]float32, tokens*m.latent)
	for channel := range m.latent {
		for token := range tokens {
			rows[token*m.latent+channel] = latent[channel*tokens+token]
		}
	}
	time := m.timeEmbedding(t)
	condition := make([]float32, m.encoderWidth)
	for index := range condition {
		condition[index] = silu(time[index] + m.labels[label*m.encoderWidth+index])
	}
	s := make([]float32, tokens*m.encoderWidth)
	linear(s, rows, m.sWeight, m.sBias, tokens, m.latent, m.encoderWidth)
	add(s, m.position)
	for _, b := range m.encoder {
		b.forward(s, condition, tokens, m.encoderCos, m.encoderSin)
	}
	for token := range tokens {
		for index := range m.encoderWidth {
			s[token*m.encoderWidth+index] = silu(s[token*m.encoderWidth+index] + time[index])
		}
	}
	projected := make([]float32, tokens*m.headWidth)
	linear(projected, s, m.projector, m.projectorBias, tokens, m.encoderWidth, m.headWidth)
	s = projected
	x := make([]float32, tokens*m.headWidth)
	linear(x, rows, m.xWeight, m.xBias, tokens, m.latent, m.headWidth)
	for _, b := range m.head {
		b.forward(x, s, tokens, m.headCos, m.headSin)
	}
	modulation := modulate(s, m.finalModulation, m.finalModulationBias, tokens, m.headWidth, finalModulations)
	normed := rmsNorm(x, m.finalNorm, tokens, m.headWidth)
	shift, scale := modulation[0], modulation[1]
	for index := range normed {
		normed[index] = normed[index]*(1+scale[index]) + shift[index]
	}
	predicted := make([]float32, tokens*m.latent)
	linear(predicted, normed, m.finalWeight, m.finalBias, tokens, m.headWidth, m.latent)
	out := make([]float32, len(predicted))
	for token := range tokens {
		for channel := range m.latent {
			out[channel*tokens+token] = predicted[token*m.latent+channel]
		}
	}
	for _, value := range out {
		if !checked.Finite32(value) {
			return nil, errors.New("rae: the DiT produced a non-finite velocity")
		}
	}
	return out, nil
}

// timeEmbedding is the Gaussian Fourier embedding of t through its MLP.
func (m *diffusionTransformer) timeEmbedding(t float32) []float32 {
	features := make([]float32, len(m.fourier)*fourierParts)
	for index, frequency := range m.fourier {
		angle := float64(t) * float64(frequency) * fourierTurn
		features[index] = float32(math.Sin(angle))
		features[len(m.fourier)+index] = float32(math.Cos(angle))
	}
	hidden := make([]float32, m.encoderWidth)
	linear(hidden, features, m.time1, m.time1Bias, 1, len(features), m.encoderWidth)
	for index, value := range hidden {
		hidden[index] = silu(value)
	}
	out := make([]float32, m.encoderWidth)
	linear(out, hidden, m.time2, m.time2Bias, 1, m.encoderWidth, m.encoderWidth)
	return out
}

// forward runs the block over x in place under condition, one row for every
// token or one row shared by all.
func (b *block) forward(x, condition []float32, tokens int, cos, sin []float32) {
	var shiftAttention, scaleAttention, gateAttention, shiftMLP, scaleMLP, gateMLP []float32
	fill([]*[]float32{&shiftAttention, &scaleAttention, &gateAttention, &shiftMLP, &scaleMLP, &gateMLP},
		modulate(condition, b.modulation, b.modulationBias, len(condition)/b.width, b.width, blockModulations))
	shared := len(condition) == b.width
	at := func(values []float32, token, index int) float32 {
		if shared {
			return values[index]
		}
		return values[token*b.width+index]
	}
	normed := rmsNorm(x, b.norm1, tokens, b.width)
	for token := range tokens {
		for index := range b.width {
			normed[token*b.width+index] = normed[token*b.width+index]*(1+at(scaleAttention, token, index)) + at(shiftAttention, token, index)
		}
	}
	qkv := make([]float32, tokens*attentionParts*b.width)
	linear(qkv, normed, b.qkv, b.qkvBias, tokens, b.width, attentionParts*b.width)
	q, k, v := make([]float32, tokens*b.width), make([]float32, tokens*b.width), make([]float32, tokens*b.width)
	for token := range tokens {
		row := qkv[token*attentionParts*b.width:]
		for part, into := range [][]float32{q, k, v} {
			copy(into[token*b.width:(token+1)*b.width], row[part*b.width:])
		}
	}
	rotate(q, cos, sin, tokens, b.width, b.heads)
	rotate(k, cos, sin, tokens, b.width, b.heads)
	attended := hostmath.BatchedAttentionF32(q, k, v, 1, tokens, tokens, b.width, b.heads)
	projected := make([]float32, tokens*b.width)
	linear(projected, attended, b.proj, b.bias, tokens, b.width, b.width)
	for token := range tokens {
		for index := range b.width {
			x[token*b.width+index] += at(gateAttention, token, index) * projected[token*b.width+index]
		}
	}
	normed = rmsNorm(x, b.norm2, tokens, b.width)
	for token := range tokens {
		for index := range b.width {
			normed[token*b.width+index] = normed[token*b.width+index]*(1+at(scaleMLP, token, index)) + at(shiftMLP, token, index)
		}
	}
	up := make([]float32, tokens*gatedParts*b.gated)
	linear(up, normed, b.up, b.upBias, tokens, b.width, gatedParts*b.gated)
	inner := make([]float32, tokens*b.gated)
	for token := range tokens {
		row := up[token*gatedParts*b.gated:]
		for index := range b.gated {
			inner[token*b.gated+index] = silu(row[index]) * row[b.gated+index]
		}
	}
	linear(projected, inner, b.down, b.downBias, tokens, b.gated, b.width)
	for token := range tokens {
		for index := range b.width {
			x[token*b.width+index] += at(gateMLP, token, index) * projected[token*b.width+index]
		}
	}
}

// modulate is the adaptive modulation of a condition: SiLU, then a linear
// map whose output is split into parts of width each, per condition row.
func modulate(condition, weight, bias []float32, rows, width, parts int) [][]float32 {
	activated := make([]float32, len(condition))
	for index, value := range condition {
		activated[index] = silu(value)
	}
	out := make([]float32, rows*parts*width)
	linear(out, activated, weight, bias, rows, width, parts*width)
	split := make([][]float32, parts)
	for part := range parts {
		split[part] = make([]float32, rows*width)
		for row := range rows {
			copy(split[part][row*width:(row+1)*width], out[(row*parts+part)*width:])
		}
	}
	return split
}

// rotate applies the 2-D rotary position embedding to every head of x in
// place: each head's pairs of adjacent channels turn by their token's angle.
func rotate(x, cos, sin []float32, tokens, width, heads int) {
	headWidth := width / heads
	for token := range tokens {
		angleCos, angleSin := cos[token*headWidth:], sin[token*headWidth:]
		for head := range heads {
			row := x[token*width+head*headWidth : token*width+(head+1)*headWidth]
			for pair := 0; pair < headWidth; pair += rotaryPair {
				even, odd := row[pair], row[pair+1]
				row[pair] = even*angleCos[pair] - odd*angleSin[pair]
				row[pair+1] = odd*angleCos[pair+1] + even*angleSin[pair+1]
			}
		}
	}
}

// rmsNorm scales each row to unit root mean square, then by weight.
func rmsNorm(x, weight []float32, rows, width int) []float32 {
	out := make([]float32, len(x))
	for row := range rows {
		values := x[row*width : (row+1)*width]
		var squares float64
		for _, value := range values {
			squares += float64(value) * float64(value)
		}
		inverse := 1 / math.Sqrt(squares/float64(width)+rmsEpsilon)
		for index, value := range values {
			out[row*width+index] = float32(float64(value)*inverse) * weight[index]
		}
	}
	return out
}

func silu(value float32) float32 { return float32(hostmath.SiLU(float64(value))) }
