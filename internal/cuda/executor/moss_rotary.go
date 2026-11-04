package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
)

// mossRotaryRank is the rank of the heads MOSS's rotary embedding turns, [width, heads, tokens], by these axes.
const (
	mossRotaryRank      = 3
	mossRotaryWidthAxis = 0
	mossRotaryHeadAxis  = 1
	mossRotaryTokenAxis = 2
)

// mossRotaryOperands are a folded rotary embedding's operands in order: the heads, the cosine and the sine.
const (
	mossRotaryHeads = iota
	mossRotaryCosine
	mossRotarySine
)

// mossRotaryNegation is the scale that negates a head's second half.
const mossRotaryNegation = -tensor.UnitScale

// applyMOSSRotaryRewrite folds MOSS's rounded rotary embedding (latentimage rotaryGraph.rotate): the heads times the
// cosine, plus the half-turned heads times the sine, each product and the sum rounded, and the rounding the attention
// takes of the result again, into one kernel that reads the heads and the tables once and writes the rotated heads
// once. Its slices, negation, concatenation, products, sum and second rounding took about 100 ms of a 1.279 s step.
func applyMOSSRotaryRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if _, skipped := compiled.skipped[node]; skipped || node.Op != tensor.OpBF16Round || len(node.Inputs) != 1 || compiled.fusions[node] != nil {
			continue
		}
		// A chain rounded again, as its attention takes it, folds into that rounding, which the order meets later, unless
		// the BF16 attention's fusion reads past it.
		_, retained := context.outputSet[node]
		if consumers := context.consumers[node]; !retained && len(consumers) == 1 && consumers[0].Op == tensor.OpBF16Round {
			if _, skipped := compiled.skipped[consumers[0]]; !skipped && compiled.fusions[consumers[0]] == nil {
				continue
			}
		}
		chain := &roundedChain{context: context, ok: true}
		last := node
		if inner := node.Inputs[0]; inner.Op == tensor.OpBF16Round {
			chain.claim(inner)
			last = inner
		}
		sum := chain.rounded(last, tensor.OpAdd)
		cosined := chain.rounded(chain.operand(sum, 0), tensor.OpMultiply)
		sined := chain.rounded(chain.operand(sum, 1), tensor.OpMultiply)
		turned := chain.operand(sined, 0)
		negated := chain.operand(turned, 0)
		second := chain.operand(chain.operand(negated, 0), 0)
		first := chain.operand(chain.operand(turned, 1), 0)
		if !chain.ok || !mossRotaryHalves(cosined.Inputs[0], turned, negated, first, second) {
			continue
		}
		heads, cosine, sine := cosined.Inputs[0], cosined.Inputs[1], sined.Inputs[1]
		table := tensor.MustShape(heads.Shape.Dims[mossRotaryWidthAxis], tensor.SingletonExtent, heads.Shape.Dims[mossRotaryTokenAxis])
		if !node.Shape.Equal(heads.Shape) || !cosine.Shape.Equal(table) || !sine.Shape.Equal(table) {
			continue
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionMOSSRotary, operands: []*tensor.Tensor{heads, cosine, sine}})
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		for _, part := range chain.parts {
			compiled.skipped[part] = struct{}{}
		}
	}
}

// mossRotaryHalves reports whether turned is the half-turned heads: the concatenation, along the channels, of the
// heads' second half negated and their first half, each half a group slice of the heads seen through a reshape.
func mossRotaryHalves(heads, turned, negated, first, second *tensor.Tensor) bool {
	if heads.Shape.Rank != mossRotaryRank || heads.Shape.Dims[mossRotaryWidthAxis]%tensor.PairedExtent != 0 {
		return false
	}
	half := heads.Shape.Dims[mossRotaryWidthAxis] / tensor.PairedExtent
	concat, concatenated := turned.Attrs.(tensor.ConcatAttributes)
	scale, scaled := negated.Attrs.(tensor.ScaleAttributes)
	slice := func(node *tensor.Tensor, offset uint64) bool {
		attributes, ok := node.Attrs.(tensor.GroupSliceAttributes)
		return node.Op == tensor.OpGroupSlice && ok && node.Inputs[0] == heads &&
			attributes == tensor.GroupSliceAttributes{Offset: offset, Width: half, Groups: tensor.SingletonExtent, Stride: half}
	}
	return turned.Op == tensor.OpConcat && concatenated && concat.Axis == tensor.FirstOffset &&
		negated.Op == tensor.OpScale && scaled && scale.Value == mossRotaryNegation &&
		negated.Inputs[0].Op == tensor.OpReshape && turned.Inputs[1].Op == tensor.OpReshape &&
		slice(second, half) && slice(first, tensor.FirstOffset)
}

// launchMOSSRotary turns a folded rotary embedding's heads by its tables into its last rounding's output.
func launchMOSSRotary(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	count := shapeCount32(node.Shape)
	width, heads, tokens := uint32(node.Shape.Dims[mossRotaryWidthAxis]), uint32(node.Shape.Dims[mossRotaryHeadAxis]), uint32(node.Shape.Dims[mossRotaryTokenAxis])
	input, cosine, sine := pointers.input(mossRotaryHeads), pointers.input(mossRotaryCosine), pointers.input(mossRotarySine)
	output := pointers.output()
	return launch1DABI(
		state, functions[kernelMossRotateBf16RoundF32], count,
		&input, &cosine, &sine, &output, &width, &heads, &tokens,
	)
}
