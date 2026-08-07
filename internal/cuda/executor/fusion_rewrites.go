package executor

import (
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type weightedRMSFusion struct {
	normalization *tensor.Tensor
	weight        *tensor.Tensor
	addLeft       *tensor.Tensor
	addRight      *tensor.Tensor
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

var graphRewriteCatalog = [...]graphRewrite{
	{name: "weighted-rms", apply: applyWeightedRMSRewrite},
	{name: "activated-gate", apply: applyActivatedGateRewrite},
	{name: "q8-emission", apply: applyQ8EmissionRewrite},
	{name: "q8-argmax", apply: applyQ8ArgmaxRewrite},
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
		if _, retained := context.outputSet[normalization]; retained ||
			!rmsWeightCompatible(normalization, weight) {
			continue
		}
		compiled := context.compiled
		if compiled.weightedRMS == nil {
			compiled.weightedRMS = make(map[*tensor.Tensor]weightedRMSFusion)
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		fusion := weightedRMSFusion{normalization: normalization, weight: weight}
		if source := normalization.Inputs[0]; source.Op == tensor.OpAdd && context.uses[source] == 1 {
			if _, retained := context.outputSet[source]; !retained {
				fusion.addLeft, fusion.addRight = source.Inputs[0], source.Inputs[1]
				compiled.skipped[source] = struct{}{}
			}
		}
		compiled.weightedRMS[node] = fusion
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
		if weighted, fused := compiled.weightedRMS[up]; fused && context.uses[up] == 1 {
			if _, retained := context.outputSet[up]; !retained {
				if compiled.weightedRMSGate == nil {
					compiled.weightedRMSGate = make(map[*tensor.Tensor]weightedRMSGateFusion)
				}
				compiled.weightedRMSGate[node] = weightedRMSGateFusion{
					weightedRMSFusion: weighted,
					gate:              activation.Inputs[0],
					kind:              kind,
				}
				delete(compiled.weightedRMS, up)
				compiled.skipped[up] = struct{}{}
				compiled.skipped[activation] = struct{}{}
				continue
			}
		}
		if compiled.activatedGate == nil {
			compiled.activatedGate = make(map[*tensor.Tensor]activatedGateFusion)
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		compiled.activatedGate[node] = activatedGateFusion{
			gate: activation.Inputs[0], up: up, activation: activation, kind: kind,
		}
		compiled.skipped[activation] = struct{}{}
	}
}

func applyQ8EmissionRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpMulMat || node.Inputs[0].Type != dtype.Q8_0 ||
			node.Inputs[1].Shape.Rank != 2 || node.Inputs[1].Shape.Dims[1] != 1 {
			continue
		}
		producer := node.Inputs[1]
		if _, weighted := compiled.weightedRMS[producer]; !weighted {
			if _, activated := compiled.activatedGate[producer]; !activated {
				if _, gatedNorm := compiled.weightedRMSGate[producer]; !gatedNorm {
					continue
				}
			}
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
		if compiled.q8Argmax == nil {
			compiled.q8Argmax = make(map[*tensor.Tensor]*tensor.Tensor)
		}
		compiled.q8Argmax[projection] = selection
		compiled.q8Argmax[selection] = projection
	}
}

func (context *rewriteContext) compileDependencies() {
	for node, fusion := range context.compiled.weightedRMS {
		if fusion.addLeft != nil {
			context.dependency[node] = append(context.dependency[node], fusion.addLeft, fusion.addRight)
		} else {
			context.dependency[node] = append(context.dependency[node], fusion.normalization.Inputs[0])
		}
	}
	for node, fusion := range context.compiled.activatedGate {
		context.dependency[node] = append(context.dependency[node], fusion.gate)
	}
	for node, fusion := range context.compiled.weightedRMSGate {
		context.dependency[node] = append(context.dependency[node], fusion.gate, fusion.weight)
		if fusion.addLeft != nil {
			context.dependency[node] = append(context.dependency[node], fusion.addLeft, fusion.addRight)
		} else {
			context.dependency[node] = append(context.dependency[node], fusion.normalization.Inputs[0])
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

func rmsWeightCompatible(normalization, weight *tensor.Tensor) bool {
	if normalization == nil || weight == nil || len(normalization.Inputs) != 1 || weight.Type != dtype.F32 {
		return false
	}
	width := normalization.Shape.Dims[0]
	elements, err := weight.Shape.Elements()
	return err == nil && elements == width && weight.Shape.Dims[0] == width
}
