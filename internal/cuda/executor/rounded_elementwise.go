package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
)

// roundedElementwiseKernels are the elementwise kernels that store each value's nearest BF16, by the F32 kernel each
// stands in for: the same arithmetic, rounded as bf16_round_f32 rounds it.
var roundedElementwiseKernels = map[kernelFunctionID]kernelFunctionID{
	kernelAddF32:               kernelAddBf16RoundF32,
	kernelMultiplyF32:          kernelMultiplyBf16RoundF32,
	kernelBroadcastAddF32:      kernelBroadcastAddBf16RoundF32,
	kernelBroadcastMultiplyF32: kernelBroadcastMultiplyBf16RoundF32,
	kernelScaleF32:             kernelScaleBf16RoundF32,
	kernelSigmoidF32:           kernelSigmoidBf16RoundF32,
	kernelTanhF32:              kernelTanhBf16RoundF32,
	kernelRmsNormMeanF32:       kernelRmsNormMeanBf16RoundF32,
}

// launchesRoundedKernel reports whether a product's launch is one kernel of roundedElementwiseKernels writing its
// output, as launchMathVision and launchLinearLayout choose it: a sum or product, a scale, a sigmoid or tanh of the
// ordinary arithmetic, or an RMS norm by the native mean.
func launchesRoundedKernel(product *tensor.Tensor) bool {
	switch product.Op {
	case tensor.OpAdd, tensor.OpMultiply, tensor.OpScale:
		return true
	case tensor.OpSigmoid, tensor.OpTanh:
		_, native := product.Attrs.(tensor.NativeMathAttributes)
		return !native
	case tensor.OpRMSNorm:
		attributes, ok := product.Attrs.(tensor.RMSNormAttributes)
		return ok && attributes.NativeMean && !attributes.Native
	}
	return false
}

// applyRoundedElementwiseRewrite folds a BF16 rounding into the elementwise product it rounds: the product's own launch
// stores each value's nearest BF16 into the rounding's output, and the product launches nothing of its own. MOSS's
// denoiser rounded such a product 1,672 times a forward, which took 0.37 s of a 1.77 s step.
func applyRoundedElementwiseRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if node.Op != tensor.OpBF16Round || len(node.Inputs) != 1 || compiled.fusions[node] != nil {
			continue
		}
		product := node.Inputs[0]
		_, roundSkipped := compiled.skipped[node]
		_, productSkipped := compiled.skipped[product]
		_, retained := context.outputSet[product]
		if roundSkipped || productSkipped || retained || context.uses[product] != 1 || compiled.fusions[product] != nil ||
			!launchesRoundedKernel(product) {
			continue
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionRoundedElementwise, operands: product.Inputs, rounded: product})
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		compiled.skipped[product] = struct{}{}
	}
}

// launchRoundedElementwise launches product as its own launch does, its kernel the one of roundedElementwiseKernels it
// stands for, so the rounding's output receives the product's values at their nearest BF16.
func launchRoundedElementwise(
	state *device.State,
	functions functionSet,
	blas *blasState,
	q8Input *q8InputState,
	product *tensor.Tensor,
	pointers launchPointerFrame,
) error {
	for f32, rounded := range roundedElementwiseKernels {
		functions[f32] = functions[rounded]
	}
	if product.Op == tensor.OpRMSNorm {
		return launchLinearLayout(state, functions, blas, q8Input, product, product.Attrs, pointers)
	}
	return launchMathVision(state, functions, blas, q8Input, product, product.Attrs, pointers)
}
