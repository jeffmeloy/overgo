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

type rotaryGraph struct {
	b                                               *tensor.Builder
	bind                                            tensor.WeightInputs
	storage                                         dtype.Type
	epsilon                                         float32
	width, heads, headWidth, sequence, intermediate uint64
	sine, cosine                                    *tensor.Tensor
}

func newRotaryGraph(t TransformerSpec, epsilon float32, tokens int, storage dtype.Type, weights *tensor.WeightBindings) (*rotaryGraph, error) {
	if !checked.PositiveInts(t.Hidden, t.Heads, t.HeadDim, t.Intermediate, tokens) || t.Hidden%t.Heads != 0 || t.Hidden/t.Heads != t.HeadDim || t.HeadDim%tensor.PairedExtent != 0 || !checked.PositiveFinite32(epsilon) || (storage != dtype.F32 && storage != dtype.BF16) {
		return nil, errors.New("rotary graph: invalid geometry, epsilon or storage")
	}
	b := tensor.NewBuilder()
	setBuilderMatmulCompute(b, storage)
	return &rotaryGraph{b: b, bind: tensor.WeightInputs{Builder: b, Bindings: weights, MatrixType: storage}, storage: storage, epsilon: epsilon, width: uint64(t.Hidden), heads: uint64(t.Heads), headWidth: uint64(t.HeadDim), sequence: uint64(tokens), intermediate: uint64(t.Intermediate)}, nil
}
func (g *rotaryGraph) round(x *tensor.Tensor) *tensor.Tensor {
	if g.storage == dtype.BF16 {
		return g.b.BF16RoundNative(x)
	}
	return x
}
func (g *rotaryGraph) linear(name string, x *tensor.Tensor, in, out uint64, bias bool) *tensor.Tensor {
	y := g.b.MulMat(g.bind.Input(name+".weight", in, out), x)
	if bias {
		y = g.b.Add(y, g.round(g.bind.Input(name+".bias", out)))
	}
	return g.round(y)
}
func (g *rotaryGraph) norm(name string, x *tensor.Tensor, channels uint64) *tensor.Tensor {
	return g.round(g.b.Multiply(g.round(g.b.RMSNorm(x, g.epsilon)), g.bind.Input(name+".weight", channels)))
}
func (g *rotaryGraph) part(x *tensor.Tensor, start, count uint64) *tensor.Tensor {
	dimensions := x.Shape.Slice()
	dimensions[0] = count
	return g.b.Reshape(g.b.GroupSlice(x, start, count, tensor.SingletonExtent, count), dimensions...)
}
func (g *rotaryGraph) rotaryInput(name string) *tensor.Tensor {
	x := g.b.Input(name, dtype.F32, tensor.MustShape(g.headWidth, g.sequence))
	rope := g.round(x)
	half := g.headWidth / tensor.PairedExtent
	sine := g.b.Reshape(g.part(rope, tensor.FirstOffset, half), half, tensor.SingletonExtent, g.sequence)
	cosine := g.b.Reshape(g.part(rope, half, half), half, tensor.SingletonExtent, g.sequence)
	g.sine = g.b.Concat(sine, sine, tensor.FirstOffset)
	g.cosine = g.b.Concat(cosine, cosine, tensor.FirstOffset)
	return x
}
func (g *rotaryGraph) rotate(x *tensor.Tensor) *tensor.Tensor {
	half := g.headWidth / tensor.PairedExtent
	a, c := g.part(x, tensor.FirstOffset, half), g.part(x, half, half)
	turned := g.b.Concat(g.b.Scale(c, -tensor.UnitScale), a, tensor.FirstOffset)
	return g.round(g.b.Add(g.round(g.b.Multiply(x, g.cosine)), g.round(g.b.Multiply(turned, g.sine))))
}
func (g *rotaryGraph) attention(q, k, v *tensor.Tensor) *tensor.Tensor {
	q, k, v = g.round(g.rotate(q)), g.round(g.rotate(k)), g.round(v)
	return g.b.Reshape(g.round(g.b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: float32(float64(tensor.UnitScale) / math.Sqrt(float64(g.headWidth)))})), g.width, g.sequence)
}
func (g *rotaryGraph) situ(up, gate *tensor.Tensor) *tensor.Tensor {
	boundedGate := g.round(g.b.Scale(g.round(g.b.Tanh(g.round(g.b.Scale(gate, tensor.UnitScale/refinementGateBound)))), refinementGateBound))
	boundedHidden := g.round(g.b.Scale(g.round(g.b.Tanh(g.round(g.b.Scale(up, tensor.UnitScale/refinementHiddenBound)))), refinementHiddenBound))
	return g.round(g.b.Multiply(g.round(g.b.Multiply(boundedGate, g.round(g.b.Sigmoid(gate)))), boundedHidden))
}

// compileRotaryRefinementProgram compiles biased text projection and rotary SITU blocks.
// Inputs contain one unpadded sequence and its supplied sine/cosine coefficients.
// BF16 weights retain F32 activation storage with explicit source rounding boundaries.
func compileRotaryRefinementProgram(t TransformerSpec, epsilon float32, tokens int, storage dtype.Type) (*FusionProgram, error) {
	if !checked.PositiveInts(t.TextHidden, t.RefinerTextBlocks) {
		return nil, errors.New("rotary refinement: invalid text geometry")
	}
	t.TextLayers = tensor.SingletonExtent
	p := &FusionProgram{T: t, Eps: epsilon, TextSeq: tokens, MatmulType: storage}
	g, err := newRotaryGraph(t, epsilon, tokens, storage, &p.weightInputs)
	if err != nil {
		return nil, err
	}
	b := g.b
	p.InEncoder = b.Input("encoder_hidden", dtype.F32, tensor.MustShape(uint64(t.TextHidden), g.sequence))
	p.InRoPE = g.rotaryInput("text_rotary")
	p.Fused, p.RefinementOutputs = g.refine(t, p.InEncoder)
	if e := b.Err(); e != nil {
		return nil, fmt.Errorf("rotary refinement: %w", e)
	}
	return p, nil
}

func (g *rotaryGraph) refine(t TransformerSpec, input *tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor) {
	b := g.b
	hidden := g.linear("text_embedder.proj", g.round(input), uint64(t.TextHidden), g.width, true)
	var outputs []*tensor.Tensor
	for index := range t.RefinerTextBlocks {
		prefix := fmt.Sprintf("text_embedder.blocks.%d.", index)
		qkv := g.linear(prefix+"qkv", g.norm(prefix+"norm1", hidden, g.width), g.width, tensor.TripleExtent*g.width, true)
		q := g.norm(prefix+"q_norm", b.Reshape(g.part(qkv, tensor.FirstOffset, g.width), g.headWidth, g.heads, g.sequence), g.headWidth)
		k := g.norm(prefix+"k_norm", b.Reshape(g.part(qkv, g.width, g.width), g.headWidth, g.heads, g.sequence), g.headWidth)
		v := b.Reshape(g.part(qkv, tensor.PairedExtent*g.width, g.width), g.headWidth, g.heads, g.sequence)
		hidden = g.round(b.Add(hidden, g.linear(prefix+"proj", g.attention(q, k, v), g.width, g.width, true)))
		y := g.norm(prefix+"norm2", hidden, g.width)
		up := g.linear(prefix+"ffn_up", y, g.width, g.intermediate, true)
		gate := g.linear(prefix+"ffn_gate", y, g.width, g.intermediate, true)
		hidden = g.round(b.Add(hidden, g.linear(prefix+"ffn_down", g.situ(up, gate), g.intermediate, g.width, true)))
		outputs = append(outputs, hidden)
	}
	return hidden, outputs
}
