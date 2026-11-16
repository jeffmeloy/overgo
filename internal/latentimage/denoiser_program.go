// Dual-stream denoiser graph. Reference and CUDA share topology.
// Activations: [feature,tokens]. Host feeds text and timestep boundaries.

package latentimage

import (
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/model"
	"overgo/internal/representation"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// DenoiserProgram: the compiled step graph for one image geometry. Weight
// inputs are named exactly as DenoiserTensorShapes; rank-2 projection weights
// take matmulType storage (BF16 for the device tensor-core path), everything
// else stays F32.
type DenoiserProgram struct {
	T          TransformerSpec
	Eps        float32
	TextSeq    int
	GH, GW     int
	ImgSeq     int
	Seq        int
	MatmulType dtype.Type

	InLatent  *tensor.Tensor // (InChannels, imgSeq)   image patch columns
	InText    *tensor.Tensor // (Hidden, textSeq)      host text conditioning
	InTemb    *tensor.Tensor // (Hidden)               host timestep embedding
	InTembMod *tensor.Tensor // (ModFields*Hidden)      host AdaLN-single vector
	InDelta   *tensor.Tensor // (1)                     flow Euler delta
	// InNegative is the negative prompt's text conditioning (Hidden, textSeq) where the program guides by it
	// (NegativeGuide), nil otherwise; negativeBias masks its padding as keyBias the prompt's, and negativeScale weighs
	// its rows' values as valueScale the prompt's.
	InNegative        *tensor.Tensor
	negativeBias      *tensor.Tensor
	negativeData      []float32
	negativeScale     *tensor.Tensor
	negativeScaleData []float32
	keyBias           *tensor.Tensor
	keyData           []float32
	// valueScale scales each row's attention value by the prompt's weights (CompileDenoiserProgram's textWeights),
	// none for a prompt that weights nothing.
	valueScale *tensor.Tensor
	valueData  []float32

	weightInputs tensor.WeightBindings

	// BlockOutputs[layer] is the co-attention hidden state after block layer
	// ([Hidden, seq]); the g2 distribution oracle taps these. Velocity is the
	// [InChannels, seq] flow-matching output (image columns sliced host-side).
	BlockOutputs []*tensor.Tensor
	Velocity     *tensor.Tensor
	NextLatent   *tensor.Tensor
}

// CompileDenoiserProgram builds the Krea2 step graph for one text/image
// geometry. matmulType selects rank-2 weight storage (dtype.F32 exact, dtype.BF16
// device path). imgSeq must equal gh*gw.
// textWeights are each text row's attention-value weight (weightedSpans): the co-attention scales the values of the
// text rows by them, the image rows' by one. None compile the graph as it was.
// A negative guide, at most one, guides the graph by a negative prompt (NegativeGuide); none compiles it unguided.
func CompileDenoiserProgram(t TransformerSpec, eps float32, textMask []bool, textWeights []float32, gh, gw int, matmulType dtype.Type, guides ...NegativeGuide) (*DenoiserProgram, error) {
	textSeq := len(textMask)
	if len(guides) > 1 {
		return nil, fmt.Errorf("denoiser program: %d negative prompts, where a graph guides by one", len(guides))
	}
	var guide NegativeGuide
	if len(guides) == 1 {
		guide = guides[0]
	}
	guided := guide.Mask != nil && guide.Attention.On()
	if guided && len(guide.Mask) != textSeq {
		return nil, fmt.Errorf("denoiser program: %d negative prompt rows for %d prompt rows", len(guide.Mask), textSeq)
	}
	if guided && guide.Weights != nil && len(guide.Weights) != textSeq {
		return nil, fmt.Errorf("denoiser program: %d negative prompt weights for %d rows", len(guide.Weights), textSeq)
	}
	if err := guide.Attention.Validate(); err != nil {
		return nil, fmt.Errorf("denoiser program: %w", err)
	}
	if textWeights != nil && len(textWeights) != textSeq {
		return nil, fmt.Errorf("denoiser program: %d text weights for %d text rows", len(textWeights), textSeq)
	}
	if !checked.PositiveFinite32(eps) {
		return nil, fmt.Errorf("denoiser program: eps must be positive, got %g", eps)
	}
	if !checked.PositiveInts(textSeq, gh, gw) {
		return nil, fmt.Errorf("denoiser program: textSeq/gh/gw must be positive (%d/%d/%d)", textSeq, gh, gw)
	}
	if !slices.Contains([]dtype.Type{dtype.F32, dtype.BF16}, matmulType) {
		return nil, fmt.Errorf("denoiser program: matmul weight type %s unsupported", matmulType)
	}
	t.ModFields = t.ModFieldsOr6()
	h := uint64(t.Hidden)
	headDim := uint64(t.HeadDim)
	heads := uint64(t.Heads)
	kvHeads := uint64(t.KVHeads)
	qDim := heads * headDim
	kvDim := kvHeads * headDim
	inter := uint64(t.Intermediate)
	inCh := uint64(t.InChannels)
	imgSeq, ok := checked.MulInt(gh, gw)
	if !ok {
		return nil, fmt.Errorf("denoiser program: image grid %dx%d overflows", gw, gh)
	}
	seq, ok := checked.AddInt(textSeq, imgSeq)
	if !ok {
		return nil, fmt.Errorf("denoiser program: sequence of %d image tokens overflows", imgSeq)
	}
	latentShape, err := tensor.NewShape(inCh, uint64(imgSeq))
	if err != nil {
		return nil, fmt.Errorf("denoiser program: latent extent %dx%d: %w", gw, gh, err)
	}

	p := &DenoiserProgram{
		T: t, Eps: eps, TextSeq: textSeq, GH: gh, GW: gw, ImgSeq: imgSeq, Seq: seq,
		MatmulType: matmulType,
	}
	b := tensor.NewBuilder()
	setBuilderMatmulCompute(b, matmulType)
	bind := tensor.WeightInputs{Builder: b, Bindings: &p.weightInputs, MatrixType: matmulType}

	p.InLatent = b.Input("latent_patches", dtype.F32, latentShape)
	p.InText = b.Input("text_conditioning", dtype.F32, tensor.MustShape(h, uint64(textSeq)))
	p.InTemb = b.Input("timestep_embed", dtype.F32, tensor.MustShape(h))
	p.InTembMod = b.Input("timestep_mod", dtype.F32, tensor.MustShape(uint64(t.ModFields)*h))
	p.InDelta = b.Input("flow_delta", dtype.F32, tensor.MustShape(tensor.SingletonExtent))
	valueShape := tensor.MustShape(tensor.SingletonExtent, tensor.SingletonExtent, uint64(seq))
	if p.valueData = rowScales(textWeights, seq); p.valueData != nil {
		p.valueScale = b.Input("denoiser_value_scale", dtype.F32, valueShape)
	}
	if p.keyData = paddingBias(textMask, seq); p.keyData != nil {
		p.keyBias = b.Input("denoiser_key_bias", dtype.F32, tensor.MustShape(uint64(seq)))
	}
	var negative *tensor.Tensor
	if guided {
		p.InNegative = b.Input("negative_text_conditioning", dtype.F32, p.InText.Shape)
		negative = p.InNegative
		if p.negativeData = paddingBias(guide.Mask, seq); p.negativeData != nil {
			p.negativeBias = b.Input("denoiser_negative_key_bias", dtype.F32, tensor.MustShape(uint64(seq)))
		}
		if p.negativeScaleData = rowScales(guide.Weights, seq); p.negativeScaleData != nil {
			p.negativeScale = b.Input("denoiser_negative_value_scale", dtype.F32, valueShape)
		}
	}

	// img_in: (InChannels->Hidden) + bias, over the image tokens only.
	img := b.Add(
		b.MulMat(bind.Input("img_in.weight", inCh, h), p.InLatent),
		bind.Input("img_in.bias", h),
	)
	// [text, image] token concatenation -> the co-attention sequence.
	hidden := b.Concat(p.InText, img, tensor.SingletonExtent) // (Hidden, seq)

	// 3-axis interleaved RoPE positions: text tokens at the origin (identity),
	// image token n at (0, n/gw, n%gw). Axis 0 (temporal) is always 0.
	positions := representation.AxisGridPositions(textSeq, gh, gw)
	axes := [tensor.TripleExtent]uint64{
		uint64(t.RopeAxes[tensor.FirstOffset]),
		uint64(t.RopeAxes[tensor.SingletonExtent]),
		uint64(t.RopeAxes[tensor.PairedExtent]),
	}
	theta := float32(t.RopeTheta)
	scale := float32(float64(tensor.SingletonExtent) / math.Sqrt(float64(headDim)))

	for layer := range t.Layers {
		prefix := fmt.Sprintf("transformer_blocks.%d.", layer)

		// AdaLN-single: shared timestep vector + per-block learned table.
		mod := b.Add(p.InTembMod, bind.Input(prefix+"scale_shift_table", uint64(t.ModFields)*h))
		chunk := func(i uint64) *tensor.Tensor { return b.FlatSlice(mod, i*h, h) }
		offsets := media.PairedShiftScaleGateOffsets()
		preScale, preShift, preGate := chunk(uint64(offsets.PreScale)), chunk(uint64(offsets.PreShift)), chunk(uint64(offsets.PreGate))
		postScale, postShift, postGate := chunk(uint64(offsets.PostScale)), chunk(uint64(offsets.PostShift)), chunk(uint64(offsets.PostGate))

		norm1 := bind.Input(prefix+"norm1.weight", h)
		toQ, toK, toV := bind.Input(prefix+"attn.to_q.weight", h, qDim), bind.Input(prefix+"attn.to_k.weight", h, kvDim), bind.Input(prefix+"attn.to_v.weight", h, kvDim)
		toGate := bind.Input(prefix+"attn.to_gate.weight", h, h)
		normQ, normK := bind.Input(prefix+"attn.norm_q.weight", headDim), bind.Input(prefix+"attn.norm_k.weight", headDim)
		toOut, norm2 := bind.Input(prefix+"attn.to_out.0.weight", h, h), bind.Input(prefix+"norm2.weight", h)
		ffGate, ffUp, ffDown := bind.Input(prefix+"ff.gate.weight", h, inter), bind.Input(prefix+"ff.up.weight", h, inter), bind.Input(prefix+"ff.down.weight", inter, h)

		// --- gated GQA co-attention ---
		modulated := func(rows *tensor.Tensor) *tensor.Tensor {
			return b.AdaptiveShiftScale(zeroCenteredRMSNorm(b, rows, norm1, eps), preShift, preScale)
		}
		// attend is the co-attention of a sequence's projected rows, (Hidden, seq).
		attend := func(q, k, v, valueScale, keyBias *tensor.Tensor) *tensor.Tensor {
			q = b.Reshape(q, headDim, heads, uint64(seq))
			k = b.Reshape(k, headDim, kvHeads, uint64(seq))
			v = b.Reshape(v, headDim, kvHeads, uint64(seq))
			if valueScale != nil {
				v = b.Multiply(v, valueScale)
			}
			q = buildInterleavedRoPE(b, zeroCenteredRMSNorm(b, q, normQ, eps), axes, positions, theta)
			k = buildInterleavedRoPE(b, zeroCenteredRMSNorm(b, k, normK, eps), axes, positions, theta)
			q, k, v = roundAttentionForStorage(b, matmulType, q, k, v)
			return b.Reshape(b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{KeyBias: keyBias, Scale: scale, Causal: false}), h, uint64(seq))
		}
		// output is the attention's sigmoid-gated projection, as its residual adds it.
		output := func(attn, gate *tensor.Tensor) *tensor.Tensor {
			return b.Multiply(b.MulMat(toOut, b.Multiply(attn, b.Sigmoid(gate))), preGate)
		}
		// --- SwiGLU feed-forward, as its residual adds it ---
		feedForward := func(rows *tensor.Tensor) *tensor.Tensor {
			n2 := b.AdaptiveShiftScale(zeroCenteredRMSNorm(b, rows, norm2, eps), postShift, postScale)
			return b.Multiply(b.MulMat(ffDown, b.SwiGLU(b.MulMat(ffGate, n2), b.MulMat(ffUp, n2))), postGate)
		}
		n1 := modulated(hidden)
		q, k, v := b.MulMat(toQ, n1), b.MulMat(toK, n1), b.MulMat(toV, n1)
		attn := attend(q, k, v, p.valueScale, p.keyBias)
		if negative != nil {
			text := func(x *tensor.Tensor) *tensor.Tensor {
				return b.FlatSlice(x, tensor.FirstOffset, x.Shape.Dims[tensor.FirstOffset], uint64(textSeq))
			}
			image := func(x *tensor.Tensor) *tensor.Tensor {
				rows := x.Shape.Dims[tensor.FirstOffset]
				return b.FlatSlice(x, uint64(textSeq)*rows, rows, uint64(imgSeq))
			}
			// The negative prompt's rows beside the image stream's own.
			beside := func(own, shared *tensor.Tensor) *tensor.Tensor {
				return b.Concat(own, image(shared), tensor.SingletonExtent)
			}
			m1 := modulated(negative)
			against := attend(beside(b.MulMat(toQ, m1), q), beside(b.MulMat(toK, m1), k), beside(b.MulMat(toV, m1), v), p.negativeScale, p.negativeBias)
			attn = b.Concat(text(attn), model.BuildNormalizedAttention(b, image(attn), image(against), guide.Attention), tensor.SingletonExtent)
			negative = b.Add(negative, output(text(against), b.MulMat(toGate, m1)))
			negative = b.Add(negative, feedForward(negative))
		}
		hidden = b.Add(hidden, output(attn, b.MulMat(toGate, n1)))
		hidden = b.Add(hidden, feedForward(hidden))

		p.BlockOutputs = append(p.BlockOutputs, hidden)
	}

	// final adaptive-norm + projection to the flow-matching velocity. The
	// modulation is temb + the [2,Hidden] table; run over the full sequence
	// (image columns are sliced host-side in Forward).
	table := bind.Input("final_layer.scale_shift_table", tensor.PairedExtent*h)
	finScale := b.Add(p.InTemb, b.FlatSlice(table, tensor.FirstOffset, h))
	finShift := b.Add(p.InTemb, b.FlatSlice(table, h, h))
	fn := b.AdaptiveShiftScale(
		zeroCenteredRMSNorm(b, hidden, bind.Input("final_layer.norm.weight", h), eps),
		finShift, finScale)
	p.Velocity = b.Add(
		b.MulMat(bind.Input("final_layer.linear.weight", h, inCh), fn),
		bind.Input("final_layer.linear.bias", inCh),
	)
	imageVelocity := b.FlatSlice(
		p.Velocity, uint64(textSeq)*inCh, inCh, uint64(imgSeq),
	)
	p.NextLatent = b.Add(p.InLatent, b.Multiply(imageVelocity, p.InDelta))

	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("denoiser program graph: %w", err)
	}
	return p, nil
}

// ForwardResult: the graph outputs for one denoise step. Velocity holds the
// image tokens only ([imgSeq*InChannels], token-major); BlockHidden[layer] is
// the full co-attention hidden state after block layer ([seq*Hidden]) for the
// g2 per-block distribution oracle.
type ForwardResult struct {
	Velocity    []float32
	BlockHidden [][]float32
}

// Forward runs one step of the step graph through run. Host boundaries
// (timestep sinusoid/MLP, text-fusion stream) are computed with the exact host
// reference arithmetic on d and fed in; the 28-block co-attention + final
// projection run on run's backend. latentPatches is [imgSeq*InChannels] and
// encoderHidden is [textSeq*TextLayers*TextHidden] (both token-major, host
// f64). This host-feed path requires F32 weight storage (BF16 weights ride
// resident device feeds, not host values).
func (p *DenoiserProgram) Forward(run graphruntime.Runner, d *Denoiser, latentPatches, encoderHidden []float64, sigma float64) (ForwardResult, error) {
	feeds, err := p.hostFeeds(d, latentPatches, encoderHidden, sigma)
	if err != nil {
		return ForwardResult{}, err
	}
	outputs := append(append([]*tensor.Tensor(nil), p.BlockOutputs...), p.Velocity)
	results, err := run(outputs, feeds)
	if err != nil {
		return ForwardResult{}, fmt.Errorf("denoiser forward: %w", err)
	}
	full := results[p.Velocity].Data // (InChannels, seq) token-major [seq][InChannels]
	fullWant, fullOK := checked.MulInt(p.Seq, p.T.InChannels)
	velocityStart, startOK := checked.MulInt(p.TextSeq, p.T.InChannels)
	if !fullOK || !startOK || len(full) != fullWant {
		return ForwardResult{}, fmt.Errorf("denoiser forward: velocity len=%d want %d", len(full), fullWant)
	}
	res := ForwardResult{
		Velocity:    append([]float32(nil), full[velocityStart:]...),
		BlockHidden: make([][]float32, len(p.BlockOutputs)),
	}
	for i, node := range p.BlockOutputs {
		res.BlockHidden[i] = results[node].Data
	}
	return res, nil
}

func (p *DenoiserProgram) hostFeeds(
	d *Denoiser,
	latentPatches, encoderHidden []float64,
	sigma float64,
) (map[*tensor.Tensor]reference.Value, error) {
	if p.MatmulType != dtype.F32 {
		return nil, fmt.Errorf("denoiser forward: host-feed path needs F32 weights, program compiled %s", p.MatmulType)
	}
	latentWant, latentOK := checked.MulInt(p.ImgSeq, p.T.InChannels)
	if !latentOK || len(latentPatches) != latentWant {
		return nil, fmt.Errorf("denoiser forward: latent patches len=%d want %d", len(latentPatches), latentWant)
	}
	temb, tembMod, err := d.timestepConditioning(sigma)
	if err != nil {
		return nil, err
	}
	txt, err := d.textConditioning(encoderHidden, p.TextSeq)
	if err != nil {
		return nil, err
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.weightInputs)+6)
	if err := graphruntime.AddHostWeights(feeds, p.weightInputs, func(name string) ([]float32, error) {
		return d.w(name), nil
	}); err != nil {
		return nil, fmt.Errorf("denoiser forward: %w", err)
	}
	feeds[p.InLatent] = reference.Value{Shape: p.InLatent.Shape, Data: dtype.Float64SliceToFloat32(latentPatches)}
	feeds[p.InText] = reference.Value{Shape: p.InText.Shape, Data: dtype.Float64SliceToFloat32(txt)}
	feeds[p.InTemb] = reference.Value{Shape: p.InTemb.Shape, Data: dtype.Float64SliceToFloat32(temb)}
	feeds[p.InTembMod] = reference.Value{Shape: p.InTembMod.Shape, Data: dtype.Float64SliceToFloat32(tembMod)}
	feeds[p.InDelta] = reference.Value{Shape: p.InDelta.Shape, Data: make([]float32, tensor.SingletonExtent)}
	for _, row := range p.staticRows() {
		feeds[row.node] = reference.Value{Shape: row.node.Shape, Data: row.data}
	}
	return feeds, nil
}

// staticRow is an input of a program whose rows are fixed when it compiles: a prompt's padding mask as a key bias, or
// its weights as a value scale.
type staticRow struct {
	name string
	node *tensor.Tensor
	data []float32
}

// staticRows are the fixed rows a program reads, the prompt's and the negative prompt's, those it holds.
func (p *DenoiserProgram) staticRows() []staticRow {
	rows := []staticRow{
		{"key bias", p.keyBias, p.keyData}, {"value scale", p.valueScale, p.valueData},
		{"negative key bias", p.negativeBias, p.negativeData}, {"negative value scale", p.negativeScale, p.negativeScaleData},
	}
	return slices.DeleteFunc(rows, func(row staticRow) bool { return row.node == nil })
}

// rowScales are a prompt's weights as a value scale over a sequence of rows, the prompt's first: the image rows weigh
// one, so only a weighted prompt's rows move. A prompt that weights nothing has none.
func rowScales(weights []float32, rows int) []float32 {
	if weights == nil {
		return nil
	}
	scales := make([]float32, rows)
	for index := range scales {
		scales[index] = unweighted
	}
	copy(scales, weights)
	return scales
}

// zeroCenteredRMSNorm mirrors Krea2RMSNorm: x/sqrt(mean(x^2)+eps) * (1+weight),
// composed as RMSNorm(x) + RMSNorm(x)*weight so the (1+w) scale needs no
// constant. weight ([Dims0]) broadcasts over every trailing axis.
func zeroCenteredRMSNorm(b *tensor.Builder, x, weight *tensor.Tensor, eps float32) *tensor.Tensor {
	n := b.RMSNorm(x, eps)
	return b.Add(n, b.Multiply(n, weight))
}

// buildInterleavedRoPE applies the 3-axis interleaved (adjacent-pair) RoPE over
// contiguous per-axis channel spans of x ([head_dim, heads, tokens]): axis a of
// width axes[a] rotates pair j by position*theta^(-2j/axes[a]). Slice, rotate,
// reassemble -- every stage is a cataloged op, matching the host ropeTable.
func buildInterleavedRoPE(b *tensor.Builder, x *tensor.Tensor, axes [tensor.TripleExtent]uint64, positions [tensor.TripleExtent][]uint32, theta float32) *tensor.Tensor {
	heads := x.Shape.Dims[tensor.SingletonExtent]
	tokens := x.Shape.Dims[tensor.PairedExtent]
	var joined *tensor.Tensor
	offset := uint64(tensor.FirstOffset)
	for axis := range axes {
		span := axes[axis]
		part := b.Reshape(b.GroupSlice(x, offset, span, tensor.SingletonExtent, span), span, heads, tokens)
		rotated := b.RoPEWithOptions(part, tensor.RoPEOptions{Layout: tensor.RoPELayoutNormal, Positions: positions[axis], RotaryDimensions: uint32(span), FrequencyBase: theta, FrequencyScale: tensor.SingletonExtent})
		if joined == nil {
			joined = rotated
		} else {
			joined = b.Concat(joined, rotated, tensor.FirstOffset)
		}
		offset += span
	}
	return joined
}

// NegativeGuide is a negative prompt a step graph guides by through NAG (arXiv 2505.21179), its rows' padding mask and
// weights and the attention guidance that weighs it, as guidance-ops/nag-krea reads the paper for joint attention: the
// negative prompt's rows are a stream of their own through every block, as the prompt's are, attending to themselves
// and the image rows, whose queries, keys and values are the one image stream's; NAG replaces the image rows' attention
// output, before its gate, with its combination of theirs against the prompt and against the negative prompt
// (model.BuildNormalizedAttention). Weights scale the negative prompt's rows' values as the prompt's weights scale its
// own (weightedSpans), none for a negative prompt that weights nothing. A guide with no mask, or its attention guidance
// off, guides by nothing.
type NegativeGuide struct {
	Mask      []bool
	Weights   []float32
	Attention sampling.NormalizedAttention
}

// paddingBias is a text mask's key bias over a sequence of rows, the text's first: padKeyBias on each row the mask
// leaves unattended, none where it attends every row.
func paddingBias(mask []bool, rows int) []float32 {
	if !slices.Contains(mask, false) {
		return nil
	}
	bias := make([]float32, rows)
	for index, attended := range mask {
		if !attended {
			bias[index] = padKeyBias
		}
	}
	return bias
}
