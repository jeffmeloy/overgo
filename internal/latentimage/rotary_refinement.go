package latentimage

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// The pinned MOSS _situ contract bounds gate and hidden magnitudes independently.
// These constants belong to that arithmetic and change only with its source contract.
const (
	refinementGateBound   = 4
	refinementHiddenBound = 25
)

// compileRotaryRefinementProgram compiles biased text projection and rotary SITU blocks.
// Inputs contain one unpadded sequence and its supplied sine/cosine coefficients.
// BF16 weights retain F32 activation storage with explicit source rounding boundaries.
func compileRotaryRefinementProgram(t TransformerSpec, epsilon float32, tokens int, storage dtype.Type) (*FusionProgram, error) {
	if !checked.PositiveInts(t.Hidden, t.Heads, t.HeadDim, t.Intermediate, t.TextHidden, t.RefinerTextBlocks, tokens) || t.Hidden%t.Heads != 0 || t.Hidden/t.Heads != t.HeadDim || t.HeadDim%tensor.PairedExtent != 0 || !checked.PositiveFinite32(epsilon) || (storage != dtype.F32 && storage != dtype.BF16) {
		return nil, errors.New("rotary refinement: invalid geometry, epsilon or storage")
	}
	t.TextLayers = tensor.SingletonExtent
	p := &FusionProgram{T: t, Eps: epsilon, TextSeq: tokens, MatmulType: storage}
	b := tensor.NewBuilder()
	setBuilderMatmulCompute(b, storage)
	bind := tensor.WeightInputs{Builder: b, Bindings: &p.weightInputs, MatrixType: storage}
	width, heads, headWidth, sequence, intermediate := uint64(t.Hidden), uint64(t.Heads), uint64(t.HeadDim), uint64(tokens), uint64(t.Intermediate)
	p.InEncoder = b.Input("encoder_hidden", dtype.F32, tensor.MustShape(uint64(t.TextHidden), sequence))
	p.InRoPE = b.Input("text_rotary", dtype.F32, tensor.MustShape(headWidth, sequence))
	round := func(x *tensor.Tensor) *tensor.Tensor {
		if storage == dtype.BF16 {
			return b.BF16Round(x)
		}
		return x
	}
	linear := func(name string, x *tensor.Tensor, in, out uint64) *tensor.Tensor {
		return round(b.Add(b.MulMat(bind.Input(name+".weight", in, out), x), round(bind.Input(name+".bias", out))))
	}
	norm := func(name string, x *tensor.Tensor, channels uint64) *tensor.Tensor {
		return round(b.Multiply(round(b.RMSNorm(x, epsilon)), bind.Input(name+".weight", channels)))
	}
	half := headWidth / tensor.PairedExtent
	rope := round(p.InRoPE)
	part := func(x *tensor.Tensor, start, count uint64) *tensor.Tensor {
		dimensions := x.Shape.Slice()
		dimensions[0] = count
		return b.Reshape(b.GroupSlice(x, start, count, tensor.SingletonExtent, count), dimensions...)
	}
	sine := b.Reshape(part(rope, tensor.FirstOffset, half), half, tensor.SingletonExtent, sequence)
	cosine := b.Reshape(part(rope, half, half), half, tensor.SingletonExtent, sequence)
	sine = b.Concat(sine, sine, tensor.FirstOffset)
	cosine = b.Concat(cosine, cosine, tensor.FirstOffset)
	rotate := func(x *tensor.Tensor) *tensor.Tensor {
		a, c := part(x, tensor.FirstOffset, half), part(x, half, half)
		turned := b.Concat(b.Scale(c, -tensor.UnitScale), a, tensor.FirstOffset)
		return round(b.Add(round(b.Multiply(x, cosine)), round(b.Multiply(turned, sine))))
	}
	hidden := linear("text_embedder.proj", round(p.InEncoder), uint64(t.TextHidden), width)
	for index := range t.RefinerTextBlocks {
		prefix := fmt.Sprintf("text_embedder.blocks.%d.", index)
		qkv := linear(prefix+"qkv", norm(prefix+"norm1", hidden, width), width, tensor.TripleExtent*width)
		q := norm(prefix+"q_norm", b.Reshape(part(qkv, tensor.FirstOffset, width), headWidth, heads, sequence), headWidth)
		k := norm(prefix+"k_norm", b.Reshape(part(qkv, width, width), headWidth, heads, sequence), headWidth)
		v := b.Reshape(part(qkv, tensor.PairedExtent*width, width), headWidth, heads, sequence)
		q, k, v = roundAttentionForStorage(b, storage, rotate(q), rotate(k), v)
		attention := round(b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: float32(float64(tensor.UnitScale) / math.Sqrt(float64(headWidth)))}))
		hidden = round(b.Add(hidden, linear(prefix+"proj", b.Reshape(attention, width, sequence), width, width)))
		y := norm(prefix+"norm2", hidden, width)
		up := linear(prefix+"ffn_up", y, width, intermediate)
		gate := linear(prefix+"ffn_gate", y, width, intermediate)
		boundedGate := round(b.Scale(round(b.Tanh(round(b.Scale(gate, tensor.UnitScale/refinementGateBound)))), refinementGateBound))
		boundedHidden := round(b.Scale(round(b.Tanh(round(b.Scale(up, tensor.UnitScale/refinementHiddenBound)))), refinementHiddenBound))
		activated := round(b.Multiply(round(b.Multiply(boundedGate, round(b.Sigmoid(gate)))), boundedHidden))
		hidden = round(b.Add(hidden, linear(prefix+"ffn_down", activated, intermediate, width)))
		p.RefinementOutputs = append(p.RefinementOutputs, hidden)
	}
	p.Fused = hidden
	if e := b.Err(); e != nil {
		return nil, fmt.Errorf("rotary refinement: %w", e)
	}
	return p, nil
}
