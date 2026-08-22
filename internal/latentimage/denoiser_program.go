// Krea2 dual-stream MMDiT denoiser routed through the shared tensor graph. The
// SAME graph definition executes on both the reference backend (host golden)
// and the CUDA generic executor, so this file IS the device port -- the block
// arithmetic is defined once, from cataloged ops, and mirrors the host
// reference Denoiser.Forward (denoiser.go) exactly.
//
// Layout convention (matches MulMat: left[in,out] row-major torch [out,in],
// right[in,tokens] token-major): every activation is [feature, tokens] with
// token-major data, so RMSNorm over Dims[0] is per-token, and a q/k tensor
// reshaped to [head_dim, heads, tokens] gets per-head normalization for free.
//
// The heavy compute -- the 28-block [text,image] co-attention plus the final
// modulated projection, i.e. the 12.82B-param forward -- lives in the STEP
// graph here. The text-fusion stream (text_fusion + txt_in) and the timestep
// sinusoid/MLP are small once-per-prompt / once-per-step boundaries computed on
// the host (Denoiser.textConditioning / timestepConditioning) and fed in, the
// same split latentvideo uses for its context projection and timestep math.

package latentimage

import (
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// GraphRunner: one backend executing a tensor graph. reference.Execute
// satisfies it directly; the CUDA executor satisfies it through a closure.
type GraphRunner func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error)

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
	keyBias   *tensor.Tensor
	keyData   []float32

	weightInputs map[string]*tensor.Tensor

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
func CompileDenoiserProgram(t TransformerSpec, eps float32, textMask []bool, gh, gw int, matmulType dtype.Type) (*DenoiserProgram, error) {
	textSeq := len(textMask)
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
	imgSeq := gh * gw
	seq := textSeq + imgSeq

	p := &DenoiserProgram{
		T: t, Eps: eps, TextSeq: textSeq, GH: gh, GW: gw, ImgSeq: imgSeq, Seq: seq,
		MatmulType:   matmulType,
		weightInputs: make(map[string]*tensor.Tensor),
	}
	b := tensor.NewBuilder()
	setBuilderMatmulCompute(b, matmulType)
	bind := tensor.WeightInputs{Builder: b, Inputs: p.weightInputs, MatrixType: matmulType}

	p.InLatent = b.Input("latent_patches", dtype.F32, tensor.MustShape(inCh, uint64(imgSeq)))
	p.InText = b.Input("text_conditioning", dtype.F32, tensor.MustShape(h, uint64(textSeq)))
	p.InTemb = b.Input("timestep_embed", dtype.F32, tensor.MustShape(h))
	p.InTembMod = b.Input("timestep_mod", dtype.F32, tensor.MustShape(uint64(t.ModFields)*h))
	p.InDelta = b.Input("flow_delta", dtype.F32, tensor.MustShape(tensor.SingletonExtent))
	for _, attended := range textMask {
		if !attended {
			p.keyBias = b.Input("denoiser_key_bias", dtype.F32, tensor.MustShape(uint64(seq)))
			p.keyData = make([]float32, seq)
			for index, active := range textMask {
				if !active {
					p.keyData[index] = padKeyBias
				}
			}
			break
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

		// --- gated GQA co-attention ---
		n1 := b.AdaptiveShiftScale(zeroCenteredRMSNorm(b, hidden, bind.Input(prefix+"norm1.weight", h), eps), preShift, preScale)
		q := b.MulMat(bind.Input(prefix+"attn.to_q.weight", h, qDim), n1)
		k := b.MulMat(bind.Input(prefix+"attn.to_k.weight", h, kvDim), n1)
		v := b.MulMat(bind.Input(prefix+"attn.to_v.weight", h, kvDim), n1)
		gate := b.MulMat(bind.Input(prefix+"attn.to_gate.weight", h, h), n1)

		q = b.Reshape(q, headDim, heads, uint64(seq))
		k = b.Reshape(k, headDim, kvHeads, uint64(seq))
		v = b.Reshape(v, headDim, kvHeads, uint64(seq))
		q = zeroCenteredRMSNorm(b, q, bind.Input(prefix+"attn.norm_q.weight", headDim), eps)
		k = zeroCenteredRMSNorm(b, k, bind.Input(prefix+"attn.norm_k.weight", headDim), eps)
		q = buildInterleavedRoPE(b, q, axes, positions, theta)
		k = buildInterleavedRoPE(b, k, axes, positions, theta)
		q, k, v = roundAttentionForStorage(b, matmulType, q, k, v)
		var attn *tensor.Tensor
		if p.keyBias != nil {
			attn = b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{KeyBias: p.keyBias, Scale: scale, Causal: false})
		} else {
			attn = b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: scale, Causal: false})
		}
		attn = b.Reshape(attn, h, uint64(seq))   // (Hidden, seq)
		attn = b.Multiply(attn, b.Sigmoid(gate)) // sigmoid output gate
		attn = b.MulMat(bind.Input(prefix+"attn.to_out.0.weight", h, h), attn)
		hidden = b.Add(hidden, b.Multiply(attn, preGate))

		// --- SwiGLU feed-forward ---
		n2 := b.AdaptiveShiftScale(zeroCenteredRMSNorm(b, hidden, bind.Input(prefix+"norm2.weight", h), eps), postShift, postScale)
		g := b.MulMat(bind.Input(prefix+"ff.gate.weight", h, inter), n2)
		u := b.MulMat(bind.Input(prefix+"ff.up.weight", h, inter), n2)
		ff := b.MulMat(bind.Input(prefix+"ff.down.weight", inter, h), b.SwiGLU(g, u))
		hidden = b.Add(hidden, b.Multiply(ff, postGate))

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
func (p *DenoiserProgram) Forward(run GraphRunner, d *Denoiser, latentPatches, encoderHidden []float64, sigma float64) (ForwardResult, error) {
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
	for name, node := range p.weightInputs {
		data := d.w(name)
		elements, _ := node.Shape.Elements()
		if uint64(len(data)) != elements {
			return nil, fmt.Errorf("denoiser forward: weight %s len=%d want %d", name, len(data), elements)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
	}
	feeds[p.InLatent] = reference.Value{Shape: p.InLatent.Shape, Data: dtype.Float64SliceToFloat32(latentPatches)}
	feeds[p.InText] = reference.Value{Shape: p.InText.Shape, Data: dtype.Float64SliceToFloat32(txt)}
	feeds[p.InTemb] = reference.Value{Shape: p.InTemb.Shape, Data: dtype.Float64SliceToFloat32(temb)}
	feeds[p.InTembMod] = reference.Value{Shape: p.InTembMod.Shape, Data: dtype.Float64SliceToFloat32(tembMod)}
	feeds[p.InDelta] = reference.Value{Shape: p.InDelta.Shape, Data: make([]float32, tensor.SingletonExtent)}
	if p.keyBias != nil {
		feeds[p.keyBias] = reference.Value{Shape: p.keyBias.Shape, Data: p.keyData}
	}
	return feeds, nil
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
