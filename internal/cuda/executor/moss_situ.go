package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
)

// mossSituFusion is a rounded SITU activation's bounds: the scales by each bound's inverse and by the bound, the gate's
// and the hidden value's.
type mossSituFusion struct {
	gateInverse, gateBound, hiddenInverse, hiddenBound float32
}

// situChain walks a rounded SITU activation backward from its last rounding: each step answers the operation a native
// BF16 rounding rounds, and every node it passes is read by the next alone, is no output and belongs to no fusion.
type situChain struct {
	context *rewriteContext
	parts   []*tensor.Tensor
	ok      bool
}

// claim adds node to the chain's parts where the next node is its only reader and nothing else holds it.
func (c *situChain) claim(node *tensor.Tensor) {
	_, retained := c.context.outputSet[node]
	_, skipped := c.context.compiled.skipped[node]
	if node == nil || retained || skipped || c.context.uses[node] != 1 || c.context.compiled.fusions[node] != nil {
		c.ok = false
		return
	}
	c.parts = append(c.parts, node)
}

// rounded answers the operation of op a rounding rounds, claiming the operation.
func (c *situChain) rounded(round *tensor.Tensor, op tensor.Op) *tensor.Tensor {
	if !c.ok || round == nil || round.Op != tensor.OpBF16Round || len(round.Inputs) != 1 || round.Inputs[0].Op != op {
		c.ok = false
		return nil
	}
	operation := round.Inputs[0]
	if _, native := operation.Attrs.(tensor.NativeMathAttributes); native {
		c.ok = false
		return nil
	}
	c.claim(operation)
	return operation
}

// operand is input index of operation, claimed as a rounding the chain passes through.
func (c *situChain) operand(operation *tensor.Tensor, index int) *tensor.Tensor {
	if !c.ok || operation == nil || index >= len(operation.Inputs) {
		c.ok = false
		return nil
	}
	c.claim(operation.Inputs[index])
	return operation.Inputs[index]
}

// bounded walks bounded(x), a rounded scale of a rounded tanh of a rounded scale of x, from its last rounding, and
// answers x and the two scales.
func (c *situChain) bounded(round *tensor.Tensor) (*tensor.Tensor, float32, float32) {
	outer := c.rounded(round, tensor.OpScale)
	tanh := c.rounded(c.operand(outer, 0), tensor.OpTanh)
	inner := c.rounded(c.operand(tanh, 0), tensor.OpScale)
	if !c.ok {
		return nil, 0, 0
	}
	return inner.Inputs[0], inner.Attrs.(tensor.ScaleAttributes).Value, outer.Attrs.(tensor.ScaleAttributes).Value
}

// applyMOSSSituRewrite folds MOSS's rounded SITU activation, bounded(gate) times sigmoid(gate), times bounded(up), each
// operation rounded to BF16 (latentimage rotaryGraph.situ), into one kernel that reads the gate and the hidden values
// once and writes the result once. Its nine launches took 238 ms of a 1.449 s denoising step.
func applyMOSSSituRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpBF16Round || len(node.Inputs) != 1 || compiled.fusions[node] != nil {
			continue
		}
		chain := &situChain{context: context, ok: true}
		outer := chain.rounded(node, tensor.OpMultiply)
		gated := chain.rounded(chain.operand(outer, 0), tensor.OpMultiply)
		gate, gateInverse, gateBound := chain.bounded(chain.operand(gated, 0))
		sigmoid := chain.rounded(chain.operand(gated, 1), tensor.OpSigmoid)
		up, hiddenInverse, hiddenBound := chain.bounded(chain.operand(outer, 1))
		if !chain.ok || sigmoid.Inputs[0] != gate || !gate.Shape.Equal(node.Shape) || !up.Shape.Equal(node.Shape) {
			continue
		}
		compiled.setFusion(node, &compiledFusion{
			kind: compiledFusionMOSSSitu, operands: []*tensor.Tensor{gate, up},
			situ: mossSituFusion{gateInverse: gateInverse, gateBound: gateBound, hiddenInverse: hiddenInverse, hiddenBound: hiddenBound},
		})
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		for _, part := range chain.parts {
			compiled.skipped[part] = struct{}{}
		}
	}
}

// launchMOSSSitu computes a folded SITU activation from its gate and hidden values into its last rounding's output.
func launchMOSSSitu(state *device.State, functions functionSet, node *tensor.Tensor, situ mossSituFusion, pointers launchPointerFrame) error {
	count := shapeCount32(node.Shape)
	gate, up, output := pointers.input(0), pointers.input(1), pointers.output()
	return launch1DABI(
		state, functions[kernelMossSituBf16RoundF32], count,
		&gate, &up, &output, &situ.gateInverse, &situ.gateBound, &situ.hiddenInverse, &situ.hiddenBound, &count,
	)
}
