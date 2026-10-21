package executor

import (
	"math"
	"slices"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type weightedRMSFusion struct {
	normalization *tensor.Tensor
	weight        *tensor.Tensor
	addLeft       *tensor.Tensor
	addRight      *tensor.Tensor
	// weightRows are the rows of weight: input row r takes weight row r mod weightRows.
	weightRows uint64
}

type activatedGateKind uint32

const (
	activatedGateSiLU activatedGateKind = iota + 1
	activatedGateSigmoid
)

type activatedGateFusion struct {
	gate       *tensor.Tensor
	up         *tensor.Tensor
	activation *tensor.Tensor
	kind       activatedGateKind
}

type weightedRMSGateFusion struct {
	weightedRMSFusion
	gate *tensor.Tensor
	kind activatedGateKind
}

// geluTanhFusion: the exact tanh-GELU elementwise chain
// (x -> half + half*tanh((x + c*x^3)*s), half = h*x) collapsed into one
// kernel pass; scale factors come from the graph's Scale attributes. bias
// non-nil additionally folds the preceding rank-1 broadcast bias add.
type geluTanhFusion struct {
	input            *tensor.Tensor
	bias             *tensor.Tensor
	cubicCoefficient float32
	innerScale       float32
	halfScale        float32
}

// layerNormModulateFusion: LayerNorm with the per-channel modulation
// epilogue folded into its write-back pass. adaptive selects
// n + n*scale + shift (adaptive shift-scale) over n*scale + shift (affine).
type layerNormModulateFusion struct {
	normalization *tensor.Tensor
	scaleVector   *tensor.Tensor
	shiftVector   *tensor.Tensor
	adaptive      bool
}

// streamSumFusion: a token's streams, consecutive slices of width values of
// values, summed in stream order and scaled, in one pass; gate, where set,
// is an activated gate of kind folded into each value.
type streamSumFusion struct {
	gate, values   *tensor.Tensor
	kind           activatedGateKind
	width, streams uint64
	scale          float32
}

// scaledActivationFusion: the SiLU or sigmoid of input times scale, in one
// pass.
type scaledActivationFusion struct {
	input *tensor.Tensor
	kind  activatedGateKind
	scale float32
}

// streamCombineFusion: residual + current*gate, current a token's row
// repeated over its streams and gate one value a stream and token, in one
// pass; scaled folds outer*sigmoid(inner*gate) into the gate.
type streamCombineFusion struct {
	residual, current, gate *tensor.Tensor
	scaled                  bool
	inner, outer            float32
}

// bf16ProjAddFusion: one-token BF16 matvec with the residual add folded into
// the epilogue.
type bf16ProjAddFusion struct {
	projection *tensor.Tensor
	addend     *tensor.Tensor
}

// bf16GateFusion: one-token BF16 gate/up matvec pair with the activation
// product folded into the epilogue.
type bf16GateFusion struct {
	gate *tensor.Tensor
	up   *tensor.Tensor
	kind activatedGateKind
}

// ropeAppendFusion: rope_normal rotated directly into its cache-append slot.
type ropeAppendFusion struct {
	rope           *tensor.Tensor
	attributeIndex int
}

// bf16AppendFusion: one-token BF16 matvec written directly into its
// cache-append slot.
type bf16AppendFusion struct {
	projection *tensor.Tensor
}

// bf16AttentionFusion: Attention(BF16Round(q), BF16Round(k), BF16Round(v))
// collapsed into the width-128 tensor-core flash kernel. The kernel rounds
// the pre-round F32 inputs on stage-in (round-to-nearest-even, identical to
// the elided bf16_round launches), accumulates products in F32, and keeps
// the online softmax exact F32; probabilities round to BF16 for the value
// product (declared kernel semantics, measured by the differential fixture).
type bf16AttentionFusion struct {
	query, key, value, keyBias *tensor.Tensor
}

type rewriteContext struct {
	compiled   *CompiledGraph
	order      []*tensor.Tensor
	uses       map[*tensor.Tensor]int
	consumers  map[*tensor.Tensor][]*tensor.Tensor
	outputSet  map[*tensor.Tensor]struct{}
	dependency map[*tensor.Tensor][]*tensor.Tensor
}

type graphRewrite struct {
	name  string
	apply func(*rewriteContext)
}

func (c *CompiledGraph) setFusion(node *tensor.Tensor, fusion *compiledFusion) {
	if c.fusions == nil {
		c.fusions = make(map[*tensor.Tensor]*compiledFusion)
	}
	c.fusions[node] = fusion
}

func (c *CompiledGraph) hasFusion(node *tensor.Tensor, kinds ...compiledFusionKind) bool {
	fusion := c.fusions[node]
	if fusion == nil {
		return false
	}
	return slices.Contains(kinds, fusion.kind)
}

var graphRewriteCatalog = [...]graphRewrite{
	{name: "weighted-rms", apply: applyWeightedRMSRewrite},
	{name: "activated-gate", apply: applyActivatedGateRewrite},
	{name: "stream-sum", apply: applyStreamSumRewrite},
	{name: "stream-combine", apply: applyStreamCombineRewrite},
	{name: "scaled-activation", apply: applyScaledActivationRewrite},
	{name: "gelu-tanh", apply: applyGELUTanhRewrite},
	{name: "layer-norm-modulate", apply: applyLayerNormModulateRewrite},
	{name: "q8-emission", apply: applyQ8EmissionRewrite},
	{name: "q8-argmax", apply: applyQ8ArgmaxRewrite},
	{name: "bf16-gate", apply: applyBF16GateRewrite},
	{name: "bf16-projection-add", apply: applyBF16ProjAddRewrite},
	{name: "rope-append", apply: applyRopeAppendRewrite},
	{name: "bf16-append", apply: applyBF16AppendRewrite},
	{name: "bf16-attention", apply: applyBF16AttentionRewrite},
	{name: "bf16-argmax", apply: applyBF16ArgmaxRewrite},
}

func compileGraphRewrites(
	compiled *CompiledGraph,
	order []*tensor.Tensor,
	uses map[*tensor.Tensor]int,
	consumers map[*tensor.Tensor][]*tensor.Tensor,
	outputSet map[*tensor.Tensor]struct{},
) map[*tensor.Tensor][]*tensor.Tensor {
	context := &rewriteContext{
		compiled:   compiled,
		order:      order,
		uses:       uses,
		consumers:  consumers,
		outputSet:  outputSet,
		dependency: make(map[*tensor.Tensor][]*tensor.Tensor),
	}
	for _, rewrite := range graphRewriteCatalog {
		rewrite.apply(context)
	}
	context.compileDependencies()
	return context.dependency
}

func applyWeightedRMSRewrite(context *rewriteContext) {
	for _, node := range context.order {
		if node.Op != tensor.OpMultiply || len(node.Inputs) != 2 {
			continue
		}
		normalization, weight := node.Inputs[0], node.Inputs[1]
		if normalization.Op != tensor.OpRMSNorm {
			normalization, weight = weight, normalization
		}
		if normalization.Op != tensor.OpRMSNorm || context.uses[normalization] != 1 {
			continue
		}
		weightRows, compatible := rmsWeightRows(normalization, weight)
		if _, retained := context.outputSet[normalization]; retained ||
			!compatible {
			continue
		}
		compiled := context.compiled
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		fusion := weightedRMSFusion{normalization: normalization, weight: weight, weightRows: weightRows}
		// The fused add kernel reads both operands at full row extent; a
		// broadcast add (e.g. a rank-1 bias) must stay unfused.
		if source := normalization.Inputs[0]; weightRows == 1 && source.Op == tensor.OpAdd && context.uses[source] == 1 &&
			source.Inputs[0].Shape.Equal(source.Shape) && source.Inputs[1].Shape.Equal(source.Shape) {
			if _, retained := context.outputSet[source]; !retained {
				fusion.addLeft, fusion.addRight = source.Inputs[0], source.Inputs[1]
				compiled.skipped[source] = struct{}{}
			}
		}
		operands := []*tensor.Tensor{fusion.normalization.Inputs[0], fusion.weight}
		if fusion.addLeft != nil {
			operands = append(operands, fusion.addLeft, fusion.addRight)
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionWeightedRMS, operands: operands, weightedRMS: fusion})
		compiled.skipped[normalization] = struct{}{}
	}
}

func applyActivatedGateRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpMultiply || len(node.Inputs) != 2 {
			continue
		}
		activation, up := node.Inputs[0], node.Inputs[1]
		kind, ok := activatedGateKindFor(activation)
		if !ok {
			activation, up = up, activation
			kind, ok = activatedGateKindFor(activation)
		}
		if !ok || context.uses[activation] != 1 ||
			!activation.Shape.Equal(up.Shape) || !node.Shape.Equal(up.Shape) {
			continue
		}
		if _, retained := context.outputSet[activation]; retained {
			continue
		}
		if prior := compiled.fusions[up]; prior != nil && prior.kind == compiledFusionWeightedRMS && prior.weightedRMS.weightRows == 1 && context.uses[up] == 1 {
			if _, retained := context.outputSet[up]; !retained {
				weighted := prior.weightedRMS
				operands := []*tensor.Tensor{weighted.normalization.Inputs[0], activation.Inputs[0], weighted.weight}
				if weighted.addLeft != nil {
					operands = append(operands, weighted.addLeft, weighted.addRight)
				}
				compiled.setFusion(node, &compiledFusion{kind: compiledFusionWeightedRMSGate, operands: operands, weightedGate: weightedRMSGateFusion{
					weightedRMSFusion: weighted,
					gate:              activation.Inputs[0],
					kind:              kind,
				}})
				delete(compiled.fusions, up)
				compiled.skipped[up] = struct{}{}
				compiled.skipped[activation] = struct{}{}
				continue
			}
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionActivatedGate, operands: []*tensor.Tensor{activation.Inputs[0], up}, activatedGate: activatedGateFusion{
			gate: activation.Inputs[0], up: up, activation: activation, kind: kind,
		}})
		compiled.skipped[activation] = struct{}{}
	}
}

// applyStreamSumRewrite: the reduction of a gated hyper-connection mix,
// Scale(((p0 + p1) + p2) + ...) where each p_s reshapes
// GroupSlice(x, s*width, width, 1, ·) of one [width*streams, tokens] x,
// becomes one kernel: Flash-Next's decode ran its group slices, adds and
// scale as eight kernels a mix, two mixes a layer. Where x is an activated
// gate read by the slices alone, the gate folds in and x is never written.
func applyStreamSumRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		scale, ok := scaleFactorFor(node)
		if !ok || node.Type != dtype.F32 || node.Shape.Rank != tensor.PairedExtent {
			continue
		}
		var parts, interior []*tensor.Tensor
		sum := node.Inputs[0]
		for sum.Op == tensor.OpAdd && len(sum.Inputs) == tensor.PairedExtent && sum.Shape.Equal(node.Shape) && context.fusableIntermediate(sum, 1) {
			interior = append(interior, sum)
			parts = append(parts, sum.Inputs[1])
			sum = sum.Inputs[0]
		}
		parts = append(parts, sum)
		slices.Reverse(parts)
		source, ok := streamSlices(context, parts, node.Shape)
		if !ok {
			continue
		}
		fusion := streamSumFusion{values: source, width: node.Shape.Dims[0], streams: uint64(len(parts)), scale: scale}
		operands := []*tensor.Tensor{source}
		if prior := compiled.fusions[source]; prior != nil && prior.kind == compiledFusionActivatedGate && context.fusableIntermediate(source, len(parts)) {
			gate := prior.activatedGate
			fusion.gate, fusion.values, fusion.kind = gate.gate, gate.up, gate.kind
			operands = []*tensor.Tensor{gate.gate, gate.up}
			delete(compiled.fusions, source)
			interior = append(interior, source)
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		for _, part := range parts {
			interior = append(interior, part, part.Inputs[0])
		}
		for _, item := range interior {
			compiled.skipped[item] = struct{}{}
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionStreamSum, operands: operands, streamSum: fusion})
	}
}

// applyStreamCombineRewrite: a hyper-connection combine,
// Add(residual, Multiply(RepeatHeads(Reshape(current)), gate)) with gate a
// [1, streams, tokens] value a stream and token, becomes one kernel; a gate
// that reshapes Scale(Sigmoid(Scale(x))) read by the combine alone folds in.
// Flash-Next's decode ran the repeat, product and add, and the gate's scale,
// sigmoid and scale, as six kernels a combine, two combines a layer.
func applyStreamCombineRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpAdd || len(node.Inputs) != tensor.PairedExtent || node.Type != dtype.F32 ||
			node.Shape.Rank != tensor.TripleExtent || compiled.fusions[node] != nil {
			continue
		}
		if _, skipped := compiled.skipped[node]; skipped {
			continue
		}
		residual, product := node.Inputs[0], node.Inputs[1]
		if product.Op != tensor.OpMultiply {
			residual, product = product, residual
		}
		if product.Op != tensor.OpMultiply || len(product.Inputs) != tensor.PairedExtent || !residual.Shape.Equal(node.Shape) ||
			!product.Shape.Equal(node.Shape) || compiled.fusions[product] != nil || !context.fusableIntermediate(product, 1) {
			continue
		}
		repeated, gate := product.Inputs[0], product.Inputs[1]
		if repeated.Op != tensor.OpRepeatHeads {
			repeated, gate = gate, repeated
		}
		streams, tokens := node.Shape.Dims[1], node.Shape.Dims[tensor.PairedExtent]
		if repeated.Op != tensor.OpRepeatHeads || !context.fusableIntermediate(repeated, 1) || gate.Type != dtype.F32 ||
			gate.Shape.Rank != tensor.TripleExtent || gate.Shape.Dims[0] != 1 || gate.Shape.Dims[1] != streams || gate.Shape.Dims[tensor.PairedExtent] != tokens {
			continue
		}
		interior := []*tensor.Tensor{product, repeated}
		current := repeated.Inputs[0]
		if current.Op == tensor.OpReshape && context.fusableIntermediate(current, 1) {
			interior, current = append(interior, current), current.Inputs[0]
		}
		fusion := streamCombineFusion{residual: residual, current: current, gate: gate}
		if chain, ok := scaledSigmoid(context, gate); ok {
			innerScale := chain[len(chain)-1]
			inner, _ := scaleFactorFor(innerScale)
			outer, _ := scaleFactorFor(gate.Inputs[0])
			fusion.gate, fusion.scaled, fusion.inner, fusion.outer = innerScale.Inputs[0], true, inner, outer
			interior = append(interior, chain...)
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		for _, item := range interior {
			compiled.skipped[item] = struct{}{}
		}
		compiled.setFusion(node, &compiledFusion{
			kind: compiledFusionStreamCombine, operands: []*tensor.Tensor{residual, current, fusion.gate}, streamCombine: fusion,
		})
	}
}

// applyScaledActivationRewrite: SiLU or Sigmoid of Scale(x), the scale read
// by the activation alone, becomes one kernel, which writes the Q8_0 blocks
// of a product reading it: a gated hyper-connection mix's low-rank gate ran
// scale, silu and a quantization as three kernels.
func applyScaledActivationRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		kind, ok := activatedGateKindFor(node)
		if !ok || node.Type != dtype.F32 || compiled.fusions[node] != nil {
			continue
		}
		if _, skipped := compiled.skipped[node]; skipped {
			continue
		}
		scaled := node.Inputs[0]
		scale, ok := scaleFactorFor(scaled)
		if !ok || scaled.Type != dtype.F32 || !context.fusableIntermediate(scaled, 1) {
			continue
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		compiled.skipped[scaled] = struct{}{}
		compiled.setFusion(node, &compiledFusion{
			kind: compiledFusionScaledActivation, operands: []*tensor.Tensor{scaled.Inputs[0]},
			scaledActivation: scaledActivationFusion{input: scaled.Inputs[0], kind: kind, scale: scale},
		})
	}
}

// scaledSigmoid returns, where view reshapes Scale(Sigmoid(Scale(x))) and
// each is read once and by nothing else, the chain from view to the inner
// scale: view, the outer scale, the sigmoid and the inner scale.
func scaledSigmoid(context *rewriteContext, view *tensor.Tensor) ([]*tensor.Tensor, bool) {
	if view.Op != tensor.OpReshape || !context.fusableIntermediate(view, 1) {
		return nil, false
	}
	chain := []*tensor.Tensor{view}
	for _, op := range []tensor.Op{tensor.OpScale, tensor.OpSigmoid, tensor.OpScale} {
		next := chain[len(chain)-1].Inputs[0]
		if next.Op != op || len(next.Inputs) != 1 || next.Type != dtype.F32 || !context.fusableIntermediate(next, 1) {
			return nil, false
		}
		chain = append(chain, next)
	}
	return chain, true
}

// streamSlices returns the one tensor parts slice in stream order: each part
// reshapes, to shape, GroupSlice(source, s*width, width, 1, ·) of a
// [width*len(parts), tokens] F32 source, and nothing else reads either.
func streamSlices(context *rewriteContext, parts []*tensor.Tensor, shape tensor.Shape) (*tensor.Tensor, bool) {
	width := shape.Dims[0]
	var source *tensor.Tensor
	for stream, part := range parts {
		if part.Op != tensor.OpReshape || !part.Shape.Equal(shape) || !context.fusableIntermediate(part, 1) {
			return nil, false
		}
		slice := part.Inputs[0]
		attributes, ok := slice.Attrs.(tensor.GroupSliceAttributes)
		if slice.Op != tensor.OpGroupSlice || !ok || !context.fusableIntermediate(slice, 1) ||
			attributes.Offset != uint64(stream)*width || attributes.Width != width || attributes.Groups != 1 ||
			source != nil && slice.Inputs[0] != source {
			return nil, false
		}
		source = slice.Inputs[0]
	}
	if len(parts) < tensor.PairedExtent || source.Type != dtype.F32 || source.Shape.Rank != tensor.PairedExtent ||
		source.Shape.Dims[0] != width*uint64(len(parts)) || source.Shape.Dims[1] != shape.Dims[1] {
		return nil, false
	}
	return source, true
}

func scaleFactorFor(node *tensor.Tensor) (float32, bool) {
	if node == nil || node.Op != tensor.OpScale || len(node.Inputs) != 1 {
		return 0, false
	}
	attributes, ok := node.Attrs.(tensor.ScaleAttributes)
	return attributes.Value, ok
}

// fusableIntermediate: single-consumer interior node available for elision.
func (context *rewriteContext) fusableIntermediate(node *tensor.Tensor, uses int) bool {
	if node == nil || context.uses[node] != uses {
		return false
	}
	if _, skipped := context.compiled.skipped[node]; skipped {
		return false
	}
	_, retained := context.outputSet[node]
	return !retained
}

// applyGELUTanhRewrite: the GELUTanhExact builder chain
// (squared = x*x; cubic = squared*x; shifted = x + Scale(cubic, c);
// inner = Scale(shifted, s); half = Scale(x, h); out = half + half*Tanh(inner))
// collapses into one elementwise kernel that replays the identical
// per-element rounding sequence. When x is itself a single-purpose rank-1
// broadcast bias add (a biased projection), the bias add folds in too.
func applyGELUTanhRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpAdd || len(node.Inputs) != 2 {
			continue
		}
		if _, skipped := compiled.skipped[node]; skipped {
			continue
		}
		half, product := node.Inputs[0], node.Inputs[1]
		if half.Op != tensor.OpScale {
			half, product = product, half
		}
		if half.Op != tensor.OpScale || product == nil ||
			product.Op != tensor.OpMultiply || len(product.Inputs) != 2 {
			continue
		}
		hyperbolic := product.Inputs[0]
		if hyperbolic == half {
			hyperbolic = product.Inputs[1]
		}
		if (product.Inputs[0] != half && product.Inputs[1] != half) ||
			hyperbolic.Op != tensor.OpTanh || len(hyperbolic.Inputs) != 1 {
			continue
		}
		inner := hyperbolic.Inputs[0]
		innerScale, innerOK := scaleFactorFor(inner)
		if !innerOK {
			continue
		}
		shifted := inner.Inputs[0]
		if shifted.Op != tensor.OpAdd || len(shifted.Inputs) != 2 {
			continue
		}
		x := half.Inputs[0]
		scaledCubic := shifted.Inputs[0]
		if scaledCubic == x {
			scaledCubic = shifted.Inputs[1]
		}
		if shifted.Inputs[0] != x && shifted.Inputs[1] != x {
			continue
		}
		cubicCoefficient, cubicOK := scaleFactorFor(scaledCubic)
		if !cubicOK {
			continue
		}
		halfScale, halfOK := scaleFactorFor(half)
		if !halfOK {
			continue
		}
		cubic := scaledCubic.Inputs[0]
		if cubic.Op != tensor.OpMultiply || len(cubic.Inputs) != 2 {
			continue
		}
		squared := cubic.Inputs[0]
		if squared == x {
			squared = cubic.Inputs[1]
		}
		if (cubic.Inputs[0] != x && cubic.Inputs[1] != x) ||
			squared.Op != tensor.OpMultiply || len(squared.Inputs) != 2 ||
			squared.Inputs[0] != x || squared.Inputs[1] != x {
			continue
		}
		if !x.Shape.Equal(node.Shape) {
			continue
		}
		if !context.fusableIntermediate(half, 2) ||
			!context.fusableIntermediate(product, 1) ||
			!context.fusableIntermediate(hyperbolic, 1) ||
			!context.fusableIntermediate(inner, 1) ||
			!context.fusableIntermediate(shifted, 1) ||
			!context.fusableIntermediate(scaledCubic, 1) ||
			!context.fusableIntermediate(cubic, 1) ||
			!context.fusableIntermediate(squared, 1) {
			continue
		}
		fusion := geluTanhFusion{
			input:            x,
			cubicCoefficient: cubicCoefficient,
			innerScale:       innerScale,
			halfScale:        halfScale,
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		// Bias fold: x = biased projection whose every use is this chain
		// (squared twice, cubic, shifted, half).
		if x.Op == tensor.OpAdd && len(x.Inputs) == 2 && context.fusableIntermediate(x, 5) {
			projection, bias := x.Inputs[0], x.Inputs[1]
			if projection.Shape.Rank == 1 {
				projection, bias = bias, projection
			}
			if bias.Shape.Rank == 1 && x.Shape.Rank == 2 &&
				projection.Shape.Equal(x.Shape) && bias.Shape.Dims[0] == x.Shape.Dims[0] {
				fusion.input = projection
				fusion.bias = bias
				compiled.skipped[x] = struct{}{}
			}
		}
		operands := []*tensor.Tensor{fusion.input}
		if fusion.bias != nil {
			operands = append(operands, fusion.bias)
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionGELUTanh, operands: operands, geluTanh: fusion})
		for _, interior := range [...]*tensor.Tensor{
			half, product, hyperbolic, inner, shifted, scaledCubic, cubic, squared,
		} {
			compiled.skipped[interior] = struct{}{}
		}
	}
}

// modulationVectorFor: rank-1 per-channel vector broadcastable over the
// rank-2 node's rows.
func modulationVectorFor(node, vector *tensor.Tensor) bool {
	return vector != nil && vector.Shape.Rank == 1 && node.Shape.Rank == 2 &&
		vector.Shape.Dims[0] == node.Shape.Dims[0]
}

// applyLayerNormModulateRewrite: the adaptive shift-scale chain
// Add(Add(ln, Multiply(ln, scale)), shift) and the affine norm chain
// Add(Multiply(ln, weight), bias) fold into the layer-norm write-back pass.
func applyLayerNormModulateRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpAdd || len(node.Inputs) != 2 {
			continue
		}
		if _, skipped := compiled.skipped[node]; skipped {
			continue
		}
		if compiled.hasFusion(node, compiledFusionGELUTanh) {
			continue
		}
		inner, shift := node.Inputs[0], node.Inputs[1]
		if !modulationVectorFor(node, shift) {
			inner, shift = shift, inner
		}
		if !modulationVectorFor(node, shift) || !inner.Shape.Equal(node.Shape) {
			continue
		}
		var fusion layerNormModulateFusion
		var interior []*tensor.Tensor
		if inner.Op == tensor.OpAdd && len(inner.Inputs) == 2 {
			// Adaptive: inner = Add(ln, Multiply(ln, scale)).
			normalization, product := inner.Inputs[0], inner.Inputs[1]
			if product.Op != tensor.OpMultiply {
				normalization, product = product, normalization
			}
			if normalization.Op != tensor.OpLayerNorm || product.Op != tensor.OpMultiply ||
				len(product.Inputs) != 2 {
				continue
			}
			scale := product.Inputs[0]
			if scale == normalization {
				scale = product.Inputs[1]
			}
			if (product.Inputs[0] != normalization && product.Inputs[1] != normalization) ||
				!modulationVectorFor(node, scale) {
				continue
			}
			if !context.fusableIntermediate(normalization, 2) ||
				!context.fusableIntermediate(product, 1) ||
				!context.fusableIntermediate(inner, 1) {
				continue
			}
			fusion = layerNormModulateFusion{
				normalization: normalization, scaleVector: scale, shiftVector: shift, adaptive: true,
			}
			interior = []*tensor.Tensor{normalization, product, inner}
		} else if inner.Op == tensor.OpMultiply && len(inner.Inputs) == 2 {
			// Affine: inner = Multiply(ln, weight).
			normalization, weight := inner.Inputs[0], inner.Inputs[1]
			if normalization.Op != tensor.OpLayerNorm {
				normalization, weight = weight, normalization
			}
			if normalization.Op != tensor.OpLayerNorm || !modulationVectorFor(node, weight) {
				continue
			}
			if !context.fusableIntermediate(normalization, 1) ||
				!context.fusableIntermediate(inner, 1) {
				continue
			}
			fusion = layerNormModulateFusion{
				normalization: normalization, scaleVector: weight, shiftVector: shift, adaptive: false,
			}
			interior = []*tensor.Tensor{normalization, inner}
		} else {
			continue
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		compiled.setFusion(node, &compiledFusion{
			kind:      compiledFusionLayerNormModulate,
			operands:  []*tensor.Tensor{fusion.normalization.Inputs[0], fusion.scaleVector, fusion.shiftVector},
			layerNorm: fusion,
		})
		for _, item := range interior {
			compiled.skipped[item] = struct{}{}
		}
	}
}

// applyQ8EmissionRewrite has a Q8_0 product's fused producer write the product's Q8_0 input beside its F32 output,
// for a decode step's column and a verify span's columns alike: the producers' kernels write each column's blocks where
// the separate quantization would (a span's two columns ran weighted_rms_norm_f32 and quantize_q8_0_input_f32 apart).
func applyQ8EmissionRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpMulMat || node.Inputs[0].Type != dtype.Q8_0 ||
			node.Inputs[1].Shape.Rank != 2 || node.Inputs[1].Shape.Dims[1] > q8InputSpanColumns {
			continue
		}
		// A producer read through a reshape serves the product's blocks where its rows hold whole blocks.
		producer := q8Source(node.Inputs[1])
		if !compiled.hasFusion(
			producer, compiledFusionWeightedRMS, compiledFusionActivatedGate, compiledFusionWeightedRMSGate, compiledFusionStreamSum,
			compiledFusionScaledActivation,
		) || producer.Shape.Dims[0]%q8InputTraits.BlockSize != 0 {
			continue
		}
		if compiled.q8Emit == nil {
			compiled.q8Emit = make(map[*tensor.Tensor]struct{})
		}
		compiled.q8Emit[producer] = struct{}{}
	}
}

func applyQ8ArgmaxRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, projection := range context.order {
		if projection.Op != tensor.OpMulMat || projection.Inputs[0].Type != dtype.Q8_0 ||
			projection.Inputs[1].Shape.Rank != 2 || projection.Inputs[1].Shape.Dims[1] != 1 ||
			context.uses[projection] != 1 {
			continue
		}
		if _, retained := context.outputSet[projection]; retained {
			continue
		}
		selection := context.consumers[projection][0]
		attributes, ok := selection.Attrs.(tensor.TopKAttributes)
		rows := projection.Shape.Dims[0]
		partials, partialsOK := q8ArgmaxPartialCount(rows)
		if selection.Op != tensor.OpTopK || !ok || attributes.K != 1 || rows < 2 ||
			!partialsOK || q8ArgmaxPartialValues*uint64(partials) > rows {
			continue
		}
		compiled.setFusion(projection, &compiledFusion{kind: compiledFusionQ8ArgmaxPartials, operands: projection.Inputs, peer: selection})
		compiled.setFusion(selection, &compiledFusion{kind: compiledFusionQ8ArgmaxReduction, operands: []*tensor.Tensor{projection}, peer: projection})
	}
}

// bf16DecodeProjection: one-token BF16 matvec candidate for epilogue fusion.
func (context *rewriteContext) bf16DecodeProjection(node *tensor.Tensor) bool {
	if node == nil || node.Op != tensor.OpMulMat || len(node.Inputs) != 2 {
		return false
	}
	left, right := node.Inputs[0], node.Inputs[1]
	if left.Type != dtype.BF16 || right.Type != dtype.F32 ||
		right.Shape.Rank != 2 || right.Shape.Dims[1] != 1 || left.Shape.Dims[0]%2 != 0 {
		return false
	}
	if context.uses[node] != 1 {
		return false
	}
	if _, skipped := context.compiled.skipped[node]; skipped {
		return false
	}
	_, retained := context.outputSet[node]
	return !retained
}

func applyBF16GateRewrite(context *rewriteContext) {
	compiled := context.compiled
	for node, descriptor := range compiled.fusions {
		if descriptor.kind != compiledFusionActivatedGate {
			continue
		}
		fusion := descriptor.activatedGate
		if _, emit := compiled.q8Emit[node]; emit {
			continue
		}
		gate, up := fusion.gate, fusion.up
		if !context.bf16DecodeProjection(gate) || !context.bf16DecodeProjection(up) ||
			gate.Inputs[1] != up.Inputs[1] || gate == up {
			continue
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionBF16Gate, operands: []*tensor.Tensor{gate.Inputs[0], up.Inputs[0], gate.Inputs[1]}, bf16Gate: bf16GateFusion{
			gate: gate, up: up, kind: fusion.kind,
		}})
		compiled.skipped[gate] = struct{}{}
		compiled.skipped[up] = struct{}{}
	}
}

func applyBF16ProjAddRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpAdd || len(node.Inputs) != 2 {
			continue
		}
		if _, skipped := compiled.skipped[node]; skipped {
			continue
		}
		projection, addend := node.Inputs[0], node.Inputs[1]
		if !context.bf16DecodeProjection(projection) {
			projection, addend = addend, projection
		}
		if !context.bf16DecodeProjection(projection) || projection == addend ||
			!node.Shape.Equal(projection.Shape) || !addendEpilogueCompatible(node, addend) {
			continue
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionBF16ProjAdd, operands: []*tensor.Tensor{projection.Inputs[0], projection.Inputs[1], addend}, bf16ProjAdd: bf16ProjAddFusion{
			projection: projection, addend: addend,
		}})
		compiled.skipped[projection] = struct{}{}
	}
}

// addendEpilogueCompatible: residual (same shape) or rank-1 bias vector
// broadcast over the single decode column; the epilogue kernel indexes
// addend[row] identically in both layouts.
func addendEpilogueCompatible(node, addend *tensor.Tensor) bool {
	if addend.Shape.Equal(node.Shape) {
		return true
	}
	return addend.Shape.Rank == 1 && node.Shape.Rank == 2 &&
		node.Shape.Dims[1] == 1 && addend.Shape.Dims[0] == node.Shape.Dims[0]
}

// applyBF16AppendRewrite: cache_append of a reshaped single-use one-token
// BF16 matvec writes its rows directly into the cache slot. The projection is
// elided from launching only; its buffer stays planned for alias resolution.
func applyBF16AppendRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpCacheAppend || len(node.Inputs) != 2 {
			continue
		}
		if compiled.hasFusion(node, compiledFusionRopeAppend) {
			continue
		}
		view := node.Inputs[1]
		if view.Op != tensor.OpReshape || len(view.Inputs) != 1 || context.uses[view] != 1 {
			continue
		}
		if _, retained := context.outputSet[view]; retained {
			continue
		}
		projection := view.Inputs[0]
		if !context.bf16DecodeProjection(projection) {
			continue
		}
		if compiled.hasFusion(projection, compiledFusionBF16ProjAdd) {
			continue
		}
		if compiled.elided == nil {
			compiled.elided = make(map[*tensor.Tensor]struct{})
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionBF16Append, operands: []*tensor.Tensor{node.Inputs[0], projection.Inputs[0], projection.Inputs[1]}, bf16Append: bf16AppendFusion{
			projection: projection,
		}})
		compiled.elided[projection] = struct{}{}
	}
}

// applyBF16ArgmaxRewrite: greedy selection over a one-token BF16 projection
// computes block-level argmax partials in the projection kernel; logits never
// materialize. Partials reuse the projection allocation.
func applyBF16ArgmaxRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, projection := range context.order {
		if !context.bf16DecodeProjection(projection) {
			continue
		}
		if compiled.hasFusion(projection, compiledFusionBF16ProjAdd) {
			continue
		}
		if _, elided := compiled.elided[projection]; elided {
			continue
		}
		selection := context.consumers[projection][0]
		attributes, ok := selection.Attrs.(tensor.TopKAttributes)
		rows := projection.Shape.Dims[0]
		partials, partialsOK := q8ArgmaxPartialCount(rows)
		if selection.Op != tensor.OpTopK || !ok || attributes.K != 1 || rows < 2 ||
			!partialsOK || q8ArgmaxPartialValues*uint64(partials) > rows {
			continue
		}
		compiled.setFusion(projection, &compiledFusion{kind: compiledFusionBF16ArgmaxPartials, operands: projection.Inputs, peer: selection})
		compiled.setFusion(selection, &compiledFusion{kind: compiledFusionBF16ArgmaxReduction, operands: []*tensor.Tensor{projection}, peer: projection})
	}
}

// applyRopeAppendRewrite: cache_append of a single-use rope_normal/rope_neox
// rotates directly into the cache slot.
func applyRopeAppendRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpCacheAppend || len(node.Inputs) != 2 {
			continue
		}
		rope := node.Inputs[1]
		if (rope.Op != tensor.OpRoPENormal && rope.Op != tensor.OpRoPENeoX) ||
			context.uses[rope] != 1 {
			continue
		}
		if _, skipped := compiled.skipped[rope]; skipped {
			continue
		}
		if _, retained := context.outputSet[rope]; retained {
			continue
		}
		if compiled.elided == nil {
			compiled.elided = make(map[*tensor.Tensor]struct{})
		}
		// launch-only elision: the retained cache output's alias validation
		// still addresses the rope buffer
		operands := append([]*tensor.Tensor{node.Inputs[0]}, rope.Inputs...)
		compiled.setFusion(node, &compiledFusion{
			kind: compiledFusionRopeAppend, operands: operands,
			ropeAppend: ropeAppendFusion{rope: rope, attributeIndex: compiled.orderIndexes[rope]},
		})
		compiled.elided[rope] = struct{}{}
	}
}

// applyBF16AttentionRewrite: featureless non-causal width-128 attention whose
// three inputs are single-use BF16Round nodes fuses into the tensor-core
// flash kernel; the rounds elide entirely (the kernel rounds on stage-in).
func applyBF16AttentionRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpAttention || (len(node.Inputs) != 3 && len(node.Inputs) != 4) {
			continue
		}
		attributes, ok := node.Attrs.(tensor.AttentionAttributes)
		if !ok || attributes.Causal || attributes.HasSinks || attributes.HasBlockMask ||
			attributes.SymmetricWindow || attributes.ChunkedWindow ||
			attributes.Softcap != 0 || attributes.MaxALiBiBias != 0 ||
			attributes.QueryStart != 0 || attributes.KeyValueTokens != 0 ||
			attributes.Window != 0 || attributes.RelativeBuckets != 0 {
			continue
		}
		if len(node.Inputs) == 4 && !attributes.HasKeyBias {
			continue
		}
		query, key, value := node.Inputs[0], node.Inputs[1], node.Inputs[2]
		if query == key || query == value || key == value {
			continue
		}
		fusable := true
		for _, round := range [...]*tensor.Tensor{query, key, value} {
			if round.Op != tensor.OpBF16Round || context.uses[round] != 1 ||
				round.Shape.Rank < attentionMinimumRank || round.Shape.Dims[0] != uint64(attentionBF16Width) {
				fusable = false
				break
			}
			if _, retained := context.outputSet[round]; retained {
				fusable = false
				break
			}
			if _, alreadySkipped := compiled.skipped[round]; alreadySkipped {
				fusable = false
				break
			}
		}
		if !fusable {
			continue
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		fusion := bf16AttentionFusion{
			query: query.Inputs[0], key: key.Inputs[0], value: value.Inputs[0],
		}
		if attributes.HasKeyBias {
			fusion = bf16AttentionFusion{
				query: query.Inputs[0], key: key.Inputs[0], value: value.Inputs[0], keyBias: node.Inputs[3],
			}
		}
		operands := []*tensor.Tensor{fusion.query, fusion.key, fusion.value}
		if fusion.keyBias != nil {
			operands = append(operands, fusion.keyBias)
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionBF16Attention, operands: operands, bf16Attention: fusion})
		compiled.skipped[query] = struct{}{}
		compiled.skipped[key] = struct{}{}
		compiled.skipped[value] = struct{}{}
	}
}

func (context *rewriteContext) compileDependencies() {
	for node, descriptor := range context.compiled.fusions {
		for _, operand := range descriptor.operands {
			if _, skipped := context.compiled.skipped[operand]; skipped {
				continue
			}
			if _, elided := context.compiled.elided[operand]; elided {
				continue
			}
			context.dependency[node] = append(context.dependency[node], operand)
		}
	}
}

func activatedGateKindFor(node *tensor.Tensor) (activatedGateKind, bool) {
	if node == nil || len(node.Inputs) != 1 {
		return 0, false
	}
	switch node.Op {
	case tensor.OpSiLU:
		return activatedGateSiLU, true
	case tensor.OpSigmoid:
		return activatedGateSigmoid, true
	default:
		return 0, false
	}
}

// rmsWeightRows reports the rows of an F32 weight the normalization's rows
// take in turn: one row of its width, or [width, k] over a
// [width, k, tokens] normalization, row r taking weight row r mod k, as a
// gated hyper-connection mix weighs each of its streams.
func rmsWeightRows(normalization, weight *tensor.Tensor) (uint64, bool) {
	if normalization == nil || weight == nil || len(normalization.Inputs) != 1 || weight.Type != dtype.F32 {
		return 0, false
	}
	width := normalization.Shape.Dims[0]
	elements, err := weight.Shape.Elements()
	if err != nil || weight.Shape.Dims[0] != width {
		return 0, false
	}
	if elements == width {
		return 1, true
	}
	shape := normalization.Shape
	return weight.Shape.Dims[1], weight.Shape.Rank == tensor.PairedExtent && shape.Rank == tensor.TripleExtent &&
		weight.Shape.Dims[1] == shape.Dims[1] && weight.Shape.Dims[1] <= math.MaxUint32
}

// q8Source is the tensor whose Q8_0 input blocks serve node's: node, or the
// tensor its reshapes view, whose values they hold in the same order.
func q8Source(node *tensor.Tensor) *tensor.Tensor {
	for node != nil && node.Op == tensor.OpReshape && len(node.Inputs) == 1 {
		node = node.Inputs[0]
	}
	return node
}
