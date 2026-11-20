package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
)

// applyInt8PrologueRewrite folds a MOSS norm and its steps (applyMOSSNormRewrite), or a SITU activation over one
// gate-and-up product's halves (applyMOSSSituRewrite), whose result int8 ConvRot products alone read, into the
// quantization each product begins with, as comfy_kitchen quantizes an int8 product's activation with its input
// activation applied: the result is computed in the quantizer and never stored (quantize_int8_convrot_norm_columns_f32,
// quantize_int8_convrot_situ_columns_f32), each product taking the result's operands beside its weight, and the
// products that read one result quantize it once, as they read one activation once.
func applyInt8PrologueRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, head := range context.order {
		source := compiled.fusions[head]
		if source == nil || !(source.kind == compiledFusionMOSSNorm || source.kind == compiledFusionMOSSSitu && source.situ.halves != 0) {
			continue
		}
		readers := context.consumers[head]
		if _, retained := context.outputSet[head]; retained || len(readers) == 0 || len(readers) != context.uses[head] {
			continue
		}
		products := true
		for _, reader := range readers {
			_, skipped := compiled.skipped[reader]
			if skipped || compiled.fusions[reader] != nil || reader.Op != tensor.OpMulMat || len(reader.Inputs) != 2 ||
				reader.Inputs[1] != head || reader.Inputs[0] == head || !int8ConvRotMulMat(reader) {
				products = false
				break
			}
		}
		if !products {
			continue
		}
		for _, reader := range readers {
			compiled.setFusion(reader, &compiledFusion{
				kind: compiledFusionInt8Prologue, operands: append([]*tensor.Tensor{reader.Inputs[0]}, source.operands...), prologue: source,
			})
		}
		delete(compiled.fusions, head)
		skip(compiled, []*tensor.Tensor{head})
	}
}

// applyInt8GatedAddRewrite folds an int8 ConvRot product that a gated residual sum alone reads (applyGatedAddRewrite),
// as a MOSS block's output and down projections are, into the sum: the sum's node runs the product, its activation
// quantized plainly or by the product's prologue (applyInt8PrologueRewrite), and the product's epilogue writes the sum
// (scale_int8_product_gated_add_bf16_f32), so the product is never stored.
func applyInt8GatedAddRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		sum := compiled.fusions[node]
		if sum == nil || sum.kind != compiledFusionGatedAdd {
			continue
		}
		residual, product, gate := sum.operands[0], sum.operands[1], sum.operands[2]
		_, retained := context.outputSet[product]
		_, skipped := compiled.skipped[product]
		folded := compiled.fusions[product]
		if retained || skipped || context.uses[product] != 1 || product.Op != tensor.OpMulMat || len(product.Inputs) != 2 ||
			!int8ConvRotMulMat(product) || folded != nil && folded.kind != compiledFusionInt8Prologue || product.Shape.Dims[0]%fp8QuadWidth != 0 {
			continue
		}
		operands := []*tensor.Tensor{product.Inputs[0], product.Inputs[1]}
		var prologue *compiledFusion
		if folded != nil {
			operands, prologue = folded.operands, folded.prologue
		}
		compiled.setFusion(node, &compiledFusion{
			kind: compiledFusionInt8GatedAdd, operands: append(operands, residual, gate), product: product, prologue: prologue,
		})
		delete(compiled.fusions, product)
		skip(compiled, []*tensor.Tensor{product})
	}
}

// int8PrologueQuantizer quantizes an int8 ConvRot product's activation as its folded norm or SITU activation computes
// it from its operands, the frame's inputs from first on, rows token columns of inner values.
func int8PrologueQuantizer(state *device.State, functions functionSet, source *compiledFusion, pointers launchPointerFrame, first int, inner, rows uint32) (int8ConvRotQuantizer, error) {
	grid, block := kernel.Grid1D(int(rows)), kernel.DefaultBlock1D()
	switch source.kind {
	case compiledFusionMOSSNorm:
		input := pointers.input(first)
		var steps [rmsNormSteps]driver.DevicePtr
		for step := range source.norm.steps {
			steps[step] = pointers.input(first + 1 + step)
		}
		norm := source.norm
		return func(activation, columnScale driver.DevicePtr) error {
			return launchGridABI(state, functions[kernelQuantizeInt8ConvrotNormColumnsF32], grid, block,
				&input, &steps[0], &steps[1], &steps[2], &norm.kinds, &norm.epsilon, &activation, &columnScale, &inner, &rows)
		}, nil
	case compiledFusionMOSSSitu:
		product, situ := pointers.input(first), source.situ
		return func(activation, columnScale driver.DevicePtr) error {
			return launchGridABI(state, functions[kernelQuantizeInt8ConvrotSituColumnsF32], grid, block,
				&product, &situ.gateInverse, &situ.gateBound, &situ.hiddenInverse, &situ.hiddenBound, &activation, &columnScale, &inner, &rows)
		}, nil
	}
	return nil, errors.New("int8 ConvRot prologue is no fused norm or SITU activation")
}

// launchInt8Prologue multiplies an int8 ConvRot weight by an activation its quantization computes from the folded
// norm's or SITU activation's operands (applyInt8PrologueRewrite).
func launchInt8Prologue(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, fusion *compiledFusion, pointers launchPointerFrame) error {
	leftNode, rightNode := node.Inputs[0], node.Inputs[1]
	inner, leftRows, rightRows := uint32(leftNode.Shape.Dims[0]), uint32(leftNode.Shape.Dims[1]), uint32(rightNode.Shape.Dims[1])
	left := pointers.input(0)
	quantize, err := int8PrologueQuantizer(state, functions, fusion.prologue, pointers, 1, inner, rightRows)
	if err != nil {
		return err
	}
	return launchInt8ConvRotMulMat(state, functions, blas, rightNode, left, left+driver.DevicePtr(uint64(inner)*uint64(leftRows)),
		pointers.output(), quantize, int8ConvRotScale(state, functions), inner, leftRows, rightRows)
}

// launchInt8GatedAdd runs a folded int8 ConvRot product into the gated residual sum that reads it
// (applyInt8GatedAddRewrite), its epilogue adding the product times its gate to the residual.
func launchInt8GatedAdd(state *device.State, functions functionSet, blas *blasState, fusion *compiledFusion, pointers launchPointerFrame) error {
	leftNode, rightNode := fusion.product.Inputs[0], fusion.product.Inputs[1]
	inner, leftRows, rightRows := uint32(leftNode.Shape.Dims[0]), uint32(leftNode.Shape.Dims[1]), uint32(rightNode.Shape.Dims[1])
	// The fusion's slots are its node's and then its operands' (compileFusion), the residual and gate last.
	left, count := pointers.input(0), fusion.operandCount-1
	residual, gate := pointers.input(count-2), pointers.input(count-1)
	quantize := int8ConvRotColumns(state, functions, pointers.input(1), inner, rightRows)
	if fusion.prologue != nil {
		var err error
		if quantize, err = int8PrologueQuantizer(state, functions, fusion.prologue, pointers, 1, inner, rightRows); err != nil {
			return err
		}
	}
	epilogue := func(output, weightScale, columnScale driver.DevicePtr, rows, quads uint32) error {
		return launch1DABI(state, functions[kernelScaleInt8ProductGatedAddBf16F32], quads,
			&output, &weightScale, &columnScale, &residual, &gate, &rows, &quads)
	}
	return launchInt8ConvRotMulMat(state, functions, blas, rightNode, left, left+driver.DevicePtr(uint64(inner)*uint64(leftRows)),
		pointers.output(), quantize, epilogue, inner, leftRows, rightRows)
}

// mossHeadWidth and mossHeadRows mirror MOSS_HEAD_WIDTH and MOSS_HEAD_ROWS: the head scale_int8_product_norm_rotate_bf16_f32
// holds, a warp's, and the heads' rows a block of it takes, a warp a row.
const mossHeadWidth, mossHeadRows = 128, 4

// applyInt8NormRotaryRewrite folds an int8 ConvRot product whose heads a MOSS norm and rotation alone read
// (applyMOSSNormRotaryRewrite), as a MOSS block's q and k projections are, into the rotation: the rotation's node runs
// the product, its activation quantized plainly or by the product's prologue (applyInt8PrologueRewrite), and the
// product's epilogue scales, normalizes, steps and rotates each head before it writes it
// (scale_int8_product_norm_rotate_bf16_f32), so the product's heads are never stored to be read back.
func applyInt8NormRotaryRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		rotary := compiled.fusions[node]
		if rotary == nil || rotary.kind != compiledFusionMOSSNormRotary {
			continue
		}
		heads := rotary.operands[mossRotaryHeads]
		if heads.Op != tensor.OpReshape || len(heads.Inputs) != 1 || context.uses[heads] != 1 {
			continue
		}
		product := heads.Inputs[0]
		_, headsRetained := context.outputSet[heads]
		_, retained := context.outputSet[product]
		_, skipped := compiled.skipped[product]
		folded := compiled.fusions[product]
		if headsRetained || retained || skipped || context.uses[product] != 1 || !int8ConvRotMulMat(product) ||
			folded != nil && folded.kind != compiledFusionInt8Prologue || node.Shape.Dims[mossRotaryWidthAxis] != mossHeadWidth ||
			product.Shape.Dims[0] != mossHeadWidth*node.Shape.Dims[mossRotaryHeadAxis] {
			continue
		}
		operands := []*tensor.Tensor{product.Inputs[0], product.Inputs[1]}
		var prologue *compiledFusion
		if folded != nil {
			operands, prologue = folded.operands, folded.prologue
		}
		// The rotation's operands after its heads: the norm's steps, then the cosine and sine tables.
		compiled.setFusion(node, &compiledFusion{
			kind: compiledFusionInt8NormRotary, operands: append(operands, rotary.operands[1:]...), product: product, prologue: prologue, norm: rotary.norm,
		})
		delete(compiled.fusions, product)
		skip(compiled, []*tensor.Tensor{product, heads})
	}
}

// launchInt8NormRotary runs a folded int8 ConvRot product into the MOSS norm and rotation that read its heads
// (applyInt8NormRotaryRewrite), its epilogue normalizing, stepping and rotating each head.
func launchInt8NormRotary(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, fusion *compiledFusion, pointers launchPointerFrame) error {
	leftNode, rightNode := fusion.product.Inputs[0], fusion.product.Inputs[1]
	inner, leftRows, rightRows := uint32(leftNode.Shape.Dims[0]), uint32(leftNode.Shape.Dims[1]), uint32(rightNode.Shape.Dims[1])
	width, heads, tokens := uint32(node.Shape.Dims[mossRotaryWidthAxis]), uint32(node.Shape.Dims[mossRotaryHeadAxis]), uint32(node.Shape.Dims[mossRotaryTokenAxis])
	// The fusion's slots are its node's and then its operands' (compileFusion): the norm's steps, the cosine and the sine
	// last.
	left, count, norm := pointers.input(0), fusion.operandCount-1, fusion.norm
	cosine, sine := pointers.input(count-2), pointers.input(count-1)
	var steps [rmsNormSteps]driver.DevicePtr
	for step := range norm.steps {
		steps[step] = pointers.input(count - 2 - norm.steps + step)
	}
	quantize := int8ConvRotColumns(state, functions, pointers.input(1), inner, rightRows)
	if fusion.prologue != nil {
		var err error
		if quantize, err = int8PrologueQuantizer(state, functions, fusion.prologue, pointers, 1, inner, rightRows); err != nil {
			return err
		}
	}
	epilogue := func(output, weightScale, columnScale driver.DevicePtr, _, _ uint32) error {
		rows := heads * tokens
		return launchGridABI(state, functions[kernelScaleInt8ProductNormRotateBf16F32],
			kernel.Grid1D(int((rows+mossHeadRows-1)/mossHeadRows)), kernel.Grid1D(mossHeadRows*kernel.WarpThreads()),
			&output, &weightScale, &columnScale, &steps[0], &steps[1], &steps[2], &norm.kinds, &norm.epsilon, &cosine, &sine, &width, &heads, &tokens)
	}
	return launchInt8ConvRotMulMat(state, functions, blas, rightNode, left, left+driver.DevicePtr(uint64(inner)*uint64(leftRows)),
		pointers.output(), quantize, epilogue, inner, leftRows, rightRows)
}
