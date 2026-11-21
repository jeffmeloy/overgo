package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
)

// applyInt8PrologueRewrite folds a MOSS norm and its steps (applyMOSSNormRewrite), or a MOSS SiTU of a gate-and-up
// product's halves as comfy_kitchen computes it (tensor.OpMOSSSituHalves), whose result int8 ConvRot products alone
// read, into the quantization each product begins with, as comfy_kitchen quantizes an int8 product's activation with
// its input activation applied: the result is computed in the quantizer and never stored
// (quantize_int8_convrot_norm_columns_f32, quantize_int8_convrot_situ_columns_f32), each product taking the result's
// operands beside its weight, and the products that read one result quantize it once, as they read one activation
// once. The SiTU the quantizer computes is the operation's own (moss_situ_halves_value), so the fold changes no bit;
// MOSS's rounded chain (applyMOSSSituRewrite) is a BF16 module's and stays apart, as ComfyUI's int8 linear never
// reads it.
func applyInt8PrologueRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, head := range context.order {
		source := compiled.fusions[head]
		if source == nil && head.Op == tensor.OpMOSSSituHalves && len(head.Inputs) == 1 {
			source = &compiledFusion{kind: compiledFusionMOSSSituHalves, operands: []*tensor.Tensor{head.Inputs[0]}}
		}
		if source == nil || !(source.kind == compiledFusionMOSSNorm || source.kind == compiledFusionMOSSSituHalves) {
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
// (int8_convrot_product_gated_add_bf16_f32), so the product is never stored.
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
	grid, block := kernel.Grid1D(int(rows)), int8QuantizeBlock(inner)
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
	case compiledFusionMOSSSituHalves:
		product := pointers.input(first)
		return func(activation, columnScale driver.DevicePtr) error {
			return launchGridABI(state, functions[kernelQuantizeInt8ConvrotSituColumnsF32], grid, block,
				&product, &activation, &columnScale, &inner, &rows)
		}, nil
	}
	return nil, errors.New("int8 ConvRot prologue is no fused norm or SiTU of a product's halves")
}

// launchInt8Prologue multiplies an int8 ConvRot weight by an activation its quantization computes from the folded
// norm's or SITU activation's operands (applyInt8PrologueRewrite).
func launchInt8Prologue(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, fusion *compiledFusion, pointers launchPointerFrame) error {
	leftNode, rightNode := node.Inputs[0], node.Inputs[1]
	inner, leftRows, rightRows := uint32(leftNode.Shape.Dims[0]), uint32(leftNode.Shape.Dims[1]), uint32(rightNode.Shape.Dims[1])
	left, output := pointers.input(0), pointers.output()
	quantize, err := int8PrologueQuantizer(state, functions, fusion.prologue, pointers, 1, inner, rightRows)
	if err != nil {
		return err
	}
	return launchInt8ConvRotMulMat(blas, rightNode, left, output, quantize, int8ConvRotScaled(state, functions, left, output, inner, leftRows, rightRows), inner, rightRows)
}

// int8FoldedQuantizer quantizes a folded int8 ConvRot product's activation, plainly from the frame's second input or by
// the product's prologue (applyInt8PrologueRewrite).
func int8FoldedQuantizer(state *device.State, functions functionSet, fusion *compiledFusion, pointers launchPointerFrame, inner, rightRows uint32) (int8ConvRotQuantizer, error) {
	if fusion.prologue != nil {
		return int8PrologueQuantizer(state, functions, fusion.prologue, pointers, 1, inner, rightRows)
	}
	return int8ConvRotColumns(state, functions, pointers.input(1), inner, rightRows), nil
}

// launchInt8GatedAdd runs a folded int8 ConvRot product into the gated residual sum that reads it
// (applyInt8GatedAddRewrite), its epilogue adding the product times its gate to the residual
// (int8_convrot_product_gated_add_bf16_f32).
func launchInt8GatedAdd(state *device.State, functions functionSet, blas *blasState, fusion *compiledFusion, pointers launchPointerFrame) error {
	leftNode, rightNode := fusion.product.Inputs[0], fusion.product.Inputs[1]
	inner, leftRows, rightRows := uint32(leftNode.Shape.Dims[0]), uint32(leftNode.Shape.Dims[1]), uint32(rightNode.Shape.Dims[1])
	// The fusion's slots are its node's and then its operands' (compileFusion), the residual and gate last.
	left, output, count := pointers.input(0), pointers.output(), fusion.operandCount-1
	weightScale := left + driver.DevicePtr(uint64(inner)*uint64(leftRows))
	residual, gate := pointers.input(count-2), pointers.input(count-1)
	quantize, err := int8FoldedQuantizer(state, functions, fusion, pointers, inner, rightRows)
	if err != nil {
		return err
	}
	product := func(activation, columnScale driver.DevicePtr) error {
		return launchInt8ConvRotProduct(state, functions[kernelInt8ConvrotProductGatedAddBf16F32], leftRows, rightRows,
			&activation, &columnScale, &left, &weightScale, &residual, &gate, &output, &inner, &leftRows, &rightRows)
	}
	return launchInt8ConvRotMulMat(blas, rightNode, left, output, quantize, product, inner, rightRows)
}

// mossHeadWidth mirrors MOSS_HEAD_WIDTH: the head int8_convrot_product_norm_rotate_bf16_f32 normalizes and rotates, a
// warp's.
const mossHeadWidth = 128

// applyInt8NormRotaryRewrite folds an int8 ConvRot product whose heads a MOSS norm and rotation alone read
// (applyMOSSNormRotaryRewrite), as a MOSS block's q and k projections are, into the rotation: the rotation's node runs
// the product, its activation quantized plainly or by the product's prologue (applyInt8PrologueRewrite), and the
// product's epilogue scales, normalizes, steps and rotates each head before it writes it
// (int8_convrot_product_norm_rotate_bf16_f32), so the product's heads are never stored to be read back.
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
// (applyInt8NormRotaryRewrite), its epilogue normalizing, stepping and rotating each head
// (int8_convrot_product_norm_rotate_bf16_f32).
func launchInt8NormRotary(state *device.State, functions functionSet, blas *blasState, fusion *compiledFusion, pointers launchPointerFrame) error {
	leftNode, rightNode := fusion.product.Inputs[0], fusion.product.Inputs[1]
	inner, leftRows, rightRows := uint32(leftNode.Shape.Dims[0]), uint32(leftNode.Shape.Dims[1]), uint32(rightNode.Shape.Dims[1])
	// The fusion's slots are its node's and then its operands' (compileFusion): the norm's steps, the cosine and the sine
	// last.
	left, output, count, norm := pointers.input(0), pointers.output(), fusion.operandCount-1, fusion.norm
	weightScale := left + driver.DevicePtr(uint64(inner)*uint64(leftRows))
	cosine, sine := pointers.input(count-2), pointers.input(count-1)
	var steps [rmsNormSteps]driver.DevicePtr
	for step := range norm.steps {
		steps[step] = pointers.input(count - 2 - norm.steps + step)
	}
	quantize, err := int8FoldedQuantizer(state, functions, fusion, pointers, inner, rightRows)
	if err != nil {
		return err
	}
	product := func(activation, columnScale driver.DevicePtr) error {
		return launchInt8ConvRotProduct(state, functions[kernelInt8ConvrotProductNormRotateBf16F32], leftRows, rightRows,
			&activation, &columnScale, &left, &weightScale, &steps[0], &steps[1], &steps[2], &norm.kinds, &norm.epsilon,
			&cosine, &sine, &output, &inner, &leftRows, &rightRows)
	}
	return launchInt8ConvRotMulMat(blas, rightNode, left, output, quantize, product, inner, rightRows)
}
