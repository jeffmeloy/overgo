package latentimage

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"

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
	// valueScale scales each row's attention value in the joint blocks (a weighted prompt's rows), none for a graph
	// that weights nothing.
	valueScale *tensor.Tensor
	// eightBit computes the attention in 8 bits (tensor.AttentionAttributes.EightBit), from F32 heads with no storage
	// round before it.
	eightBit bool
	// registers keeps the BF16 attention's scores in registers (tensor.AttentionAttributes.ScoresInRegisters).
	registers bool
	// int8 holds the joint blocks' projections as the int8 ConvRot file stores them and multiplies them as
	// comfy_kitchen's int8_linear does (dtype.I8ConvRot); off, they are decoded to the graph's storage.
	int8 bool
	// adapters patch the graph's projections and weights unmerged (mediaAdapter), each its tensors bound from its file
	// and scaled by a strength of its own fed each run, in the order a request names them; none for a graph without one.
	adapters []graphAdapter
}

// rotaryOptions are what a rotary program computes beside its weights: its attention in 8 bits, or in BF16 with its
// scores in registers, its joint projections in int8 as an int8 ConvRot file holds them, and its adapters, in the order
// a request names them.
type rotaryOptions struct {
	eightBit, registers, int8 bool
	adapters                  []*mediaAdapter
}

// graphAdapter is one adapter a rotary graph applies: its patches, its tensors' inputs, bound from its file, the scalar
// fed each run as its strength, and the place its inputs' names carry (none for the first, adapterPlace and its number
// for each after it), since two files name the same projection's tensors alike.
type graphAdapter struct {
	adapter  *mediaAdapter
	inputs   tensor.WeightInputs
	strength *tensor.Tensor
	place    string
}

// adapterStrengthInput names the scalar fed each run as an adapter's strength, its place after it.
const adapterStrengthInput = "adapter_strength"

// adapterPlace opens an adapter's place in the name of each of its inputs after the first adapter's.
const adapterPlace = "@"

// use takes the options into the graph, binding each adapter's tensors into its bindings, and gives each adapter's
// strength input.
func (g *rotaryGraph) use(options rotaryOptions, bindings []tensor.WeightBindings) []*tensor.Tensor {
	g.eightBit, g.registers, g.int8 = options.eightBit, options.registers, options.int8
	strengths := make([]*tensor.Tensor, len(options.adapters))
	for index, adapter := range options.adapters {
		place := ""
		if index > 0 {
			place = adapterPlace + strconv.Itoa(index)
		}
		strengths[index] = g.b.Input(adapterStrengthInput+place, dtype.F32, tensor.MustShape(tensor.SingletonExtent))
		g.adapters = append(g.adapters, graphAdapter{adapter: adapter, place: place, strength: strengths[index],
			inputs: tensor.WeightInputs{Builder: g.b, Bindings: &bindings[index], MatrixType: dtype.F32}})
	}
	return strengths
}

// patched adds to a weight or bias each adapter's full difference of it times its strength, where it has one.
func (g *rotaryGraph) patched(name string, weight *tensor.Tensor, dimensions ...uint64) *tensor.Tensor {
	for _, applied := range g.adapters {
		if diff, held := applied.adapter.diffs[name]; held {
			weight = g.b.Add(weight, g.b.Multiply(applied.input(diff, dimensions...), applied.strength))
		}
	}
	return weight
}

// input is the graph input of the adapter's tensor named name in its file, at the adapter's place, one for every
// projection that reads it (a fused pair's down).
func (a graphAdapter) input(name string, dimensions ...uint64) *tensor.Tensor {
	name += a.place
	if held := a.inputs.Bindings.Node(name); held != nil {
		return held
	}
	return a.inputs.Input(name, dimensions...)
}

// lowRank is the adapters' branches of a projection reading in values summed, each s * scale * (x down) up at F32, or
// none. A branch reading another width than a weight an adapter replaced at in is left out, as ComfyUI leaves a LoRA
// whose difference cannot take its weight's shape (comfy/weight_adapter/lora.py calculate_weight logs the error and
// keeps the weight): the SFT adapter's patch of the patch embedder reads the latent's 128 channels where moss-control,
// before it in the depth Turbo section's stack, widened the embedder to 384 (moss-eight-step).
func (g *rotaryGraph) lowRank(name string, x *tensor.Tensor, in uint64) *tensor.Tensor {
	var sum *tensor.Tensor
	for _, applied := range g.adapters {
		patch, held := applied.adapter.lowRank[name]
		if !held || patch.in != in && g.replacedWidth(name+".weight") == in {
			continue
		}
		down := applied.input(patch.down, patch.in, patch.rank)
		up := applied.input(patch.upInputName(), patch.rank, patch.rows)
		branch := g.b.MulMatWithCompute(up, g.b.MulMatWithCompute(down, x, tensor.MulMatComputeExact), tensor.MulMatComputeExact)
		branch = g.b.Multiply(g.b.Scale(branch, patch.scale), applied.strength)
		if sum == nil {
			sum = branch
		} else {
			sum = g.b.Add(sum, branch)
		}
	}
	return sum
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

// projection is a joint block's projection with no bias, which an int8 ConvRot file holds in int8: under the int8
// option, where no adapter patches it, its weight is held as the file holds it and multiplied as comfy_kitchen's
// int8_linear multiplies it (dtype.I8ConvRot, moss-int8), else as linear holds it. ComfyUI merges a LoRA into the
// dequantized weight and quantizes the sum again with a seeded stochastic rounding (comfy/ops.py set_weight), which
// overgo does not reproduce; with the int8 product beside its low-rank branch, the depth control section, whose
// moss-control adapter patches every joint projection, kept 13.14 dB of latent where the BF16 weight keeps 14.37.
func (g *rotaryGraph) projection(name string, x *tensor.Tensor, in, out uint64) *tensor.Tensor {
	weight := name + ".weight"
	if !g.int8 || slices.ContainsFunc(g.adapters, func(applied graphAdapter) bool {
		_, replaced := applied.adapter.replaced[weight]
		_, patched := applied.adapter.lowRank[name]
		return replaced || patched
	}) {
		return g.linear(name, x, in, out, false)
	}
	return g.round(g.b.MulMat(g.bind.InputOf(dtype.I8ConvRot, weight, in, out), x))
}

func (g *rotaryGraph) linear(name string, x *tensor.Tensor, in, out uint64, bias bool) *tensor.Tensor {
	y := g.b.MulMat(g.weight(name+".weight", in, out), x)
	if branch := g.lowRank(name, x, in); branch != nil {
		y = g.b.Add(y, branch)
	}
	if bias {
		y = g.b.Add(y, g.round(g.patched(name+".bias", g.weight(name+".bias", out), out)))
	}
	return g.round(y)
}

// weight is the graph input of a weight or bias: the replacement of the adapter that names one, read from its file in
// the model's own storage, since it stands in the model's weight's place, else the model's. At most one adapter
// replaces a weight (replacedTwice).
func (g *rotaryGraph) weight(name string, dimensions ...uint64) *tensor.Tensor {
	for _, applied := range g.adapters {
		if replaced, held := applied.adapter.replaced[name]; held {
			stored := replaced.tensor + applied.place
			if node := applied.inputs.Bindings.Node(stored); node != nil {
				return node
			}
			return tensor.WeightInputs{Builder: g.b, Bindings: applied.inputs.Bindings, MatrixType: g.bind.MatrixType}.Input(stored, dimensions...)
		}
	}
	return g.bind.Input(name, dimensions...)
}

// replacedWidth is the width of the input a weight an adapter replaces reads, 0 where none replaces it.
func (g *rotaryGraph) replacedWidth(name string) uint64 {
	for _, applied := range g.adapters {
		if replaced, held := applied.adapter.replaced[name]; held && len(replaced.shape) == tensor.PairedExtent {
			return replaced.shape[1]
		}
	}
	return 0
}

// replacedTwice is a weight two of adapters replace, which no graph can read from both; empty where none is.
func replacedTwice(adapters []*mediaAdapter) string {
	replacing := map[string]bool{}
	for _, adapter := range adapters {
		for name := range adapter.replaced {
			if replacing[name] {
				return name
			}
			replacing[name] = true
		}
	}
	return ""
}

func (g *rotaryGraph) norm(name string, x *tensor.Tensor, channels uint64) *tensor.Tensor {
	weight := g.weight(name+".weight", channels)
	if patched := g.patched(name+".weight", weight, channels); patched != weight {
		weight = g.round(patched)
	}
	return g.round(g.b.Multiply(g.round(g.b.RMSNormNativeMean(x, g.epsilon)), weight))
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
	if g.eightBit {
		q, k = g.rotate(q), g.rotate(k)
	} else {
		q, k, v = g.round(g.rotate(q)), g.round(g.rotate(k)), g.round(v)
	}
	options := tensor.AttentionOptions{Scale: float32(float64(tensor.UnitScale) / math.Sqrt(float64(g.headWidth))), EightBit: g.eightBit, ScoresInRegisters: g.registers && !g.eightBit}
	return g.b.Reshape(g.round(g.b.AttentionWithOptions(q, k, v, options)), g.width, g.sequence)
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
