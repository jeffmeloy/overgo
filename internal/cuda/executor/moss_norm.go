package executor

import (
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// mossNormFusion is a native-mean RMS norm rounded to BF16 with the per-channel steps that follow it, each a multiply or
// an add by a vector rounded to BF16: how many steps, their kinds packed as rms_norm_mean_affine_bf16_round_f32 reads
// them, and the norm's epsilon.
type mossNormFusion struct {
	steps   int
	kinds   uint32
	epsilon float32
}

// The kernel's RMS_NORM_* constants: at most rmsNormSteps steps, each rmsNormStepBits bits of the kinds, the first
// step's lowest, a multiply or an add.
const (
	rmsNormSteps        = 3
	rmsNormStepBits     = 2
	rmsNormStepMultiply = 1
	rmsNormStepAdd      = 2
)

// applyMOSSNormRewrite folds MOSS's rounded native-mean RMS norm and the rounded per-channel steps after it, its weight,
// its modulation's scale and its shift (latentimage rotaryGraph.norm and modulate), into one kernel that reads each row
// once and writes it once (rms_norm_mean_affine_bf16_round_f32): from each rounding back through one to three rounded
// multiplies or adds of a vector the row's width wide to a rounded native-mean RMS norm, each node it passes read by
// the next alone. The roundings are walked from the graph's end, so a chain's last rounding takes the whole of it
// before a rounding inside it is met. A norm with no step after it stays the rounded norm rounded-elementwise launches.
func applyMOSSNormRewrite(context *rewriteContext) {
	compiled := context.compiled
	for at := len(context.order) - 1; at >= 0; at-- {
		node := context.order[at]
		if _, skipped := compiled.skipped[node]; skipped || node.Op != tensor.OpBF16Round || len(node.Inputs) != 1 || compiled.fusions[node] != nil || node.Shape.Rank < 1 {
			continue
		}
		width := node.Shape.Dims[0]
		vector := func(candidate *tensor.Tensor) bool {
			return candidate.Shape.Rank == 1 && candidate.Shape.Dims[0] == width
		}
		chain := &roundedChain{context: context, ok: true}
		var vectors []*tensor.Tensor
		var kinds []uint32
		current := node
		for chain.ok && current.Inputs[0].Op != tensor.OpRMSNorm {
			operation := current.Inputs[0]
			kind := map[tensor.Op]uint32{tensor.OpMultiply: rmsNormStepMultiply, tensor.OpAdd: rmsNormStepAdd}[operation.Op]
			if _, native := operation.Attrs.(tensor.NativeMathAttributes); native || kind == 0 || len(operation.Inputs) != 2 || len(vectors) == rmsNormSteps {
				chain.ok = false
				break
			}
			stream, scale := operation.Inputs[0], operation.Inputs[1]
			if !vector(scale) {
				stream, scale = scale, stream
			}
			if !vector(scale) || stream.Op != tensor.OpBF16Round || len(stream.Inputs) != 1 || !stream.Shape.Equal(node.Shape) {
				chain.ok = false
				break
			}
			if current != node {
				chain.claim(current)
			}
			chain.claim(operation)
			vectors, kinds = append(vectors, scale), append(kinds, kind)
			current = stream
		}
		if !chain.ok || len(vectors) == 0 {
			continue
		}
		norm := current.Inputs[0]
		attributes, held := norm.Attrs.(tensor.RMSNormAttributes)
		if !held || !attributes.NativeMean || attributes.Native || len(norm.Inputs) != 1 {
			continue
		}
		chain.claim(current)
		chain.claim(norm)
		if !chain.ok {
			continue
		}
		fusion := mossNormFusion{steps: len(vectors), epsilon: attributes.Epsilon}
		operands := []*tensor.Tensor{norm.Inputs[0]}
		for step := range vectors {
			// The walk met the steps last first.
			at := len(vectors) - 1 - step
			operands = append(operands, vectors[at])
			fusion.kinds |= kinds[at] << (rmsNormStepBits * step)
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionMOSSNorm, operands: operands, norm: fusion})
		skip(compiled, chain.parts)
	}
}

// applyGatedAddRewrite folds a gated residual sum, a rounded add of a residual and the rounded product of an update and
// a per-channel gate vector (latentimage rotaryGraph.joint), into one kernel that reads each value once
// (gated_add_bf16_round_f32), the product read by the add alone.
func applyGatedAddRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		if _, skipped := compiled.skipped[node]; skipped || node.Op != tensor.OpBF16Round || len(node.Inputs) != 1 || compiled.fusions[node] != nil || node.Shape.Rank < 1 {
			continue
		}
		width := node.Shape.Dims[0]
		chain := &roundedChain{context: context, ok: true}
		add := chain.rounded(node, tensor.OpAdd)
		if !chain.ok || len(add.Inputs) != 2 {
			continue
		}
		var operands []*tensor.Tensor
		for _, order := range [2][2]int{{0, 1}, {1, 0}} {
			residual, gated := add.Inputs[order[0]], add.Inputs[order[1]]
			if gated.Op != tensor.OpBF16Round || len(gated.Inputs) != 1 || gated.Inputs[0].Op != tensor.OpMultiply || !residual.Shape.Equal(node.Shape) {
				continue
			}
			product := gated.Inputs[0]
			if _, native := product.Attrs.(tensor.NativeMathAttributes); native || len(product.Inputs) != 2 {
				continue
			}
			update, gate := product.Inputs[0], product.Inputs[1]
			if gate.Shape.Rank != 1 {
				update, gate = gate, update
			}
			if gate.Shape.Rank != 1 || gate.Shape.Dims[0] != width || !update.Shape.Equal(node.Shape) {
				continue
			}
			chain.claim(gated)
			chain.claim(product)
			operands = []*tensor.Tensor{residual, update, gate}
			break
		}
		if !chain.ok || operands == nil {
			continue
		}
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionGatedAdd, operands: operands})
		skip(compiled, chain.parts)
	}
}

// skip marks the parts a fusion took as nodes no launch computes.
func skip(compiled *CompiledGraph, parts []*tensor.Tensor) {
	if compiled.skipped == nil {
		compiled.skipped = make(map[*tensor.Tensor]struct{})
	}
	for _, part := range parts {
		compiled.skipped[part] = struct{}{}
	}
}

// launchMOSSNorm computes a folded norm and its steps from its input and step vectors into its last rounding's output.
func launchMOSSNorm(state *device.State, functions functionSet, node *tensor.Tensor, norm mossNormFusion, pointers launchPointerFrame) error {
	width, rows := rowWords(node.Shape)
	input, output := pointers.input(0), pointers.output()
	var steps [rmsNormSteps]driver.DevicePtr
	for step := range norm.steps {
		steps[step] = pointers.input(1 + step)
	}
	return launchNormalizationABI(state, functions[kernelRmsNormMeanAffineBf16RoundF32], rows,
		&input, &output, &steps[0], &steps[1], &steps[2], &norm.kinds, &width, &rows, &norm.epsilon)
}

// launchGatedAdd computes a folded gated residual sum from its residual, update and gate into its rounding's output.
func launchGatedAdd(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	width, count := uint32(node.Shape.Dims[0]), shapeCount32(node.Shape)
	residual, update, gate, output := pointers.input(0), pointers.input(1), pointers.input(2), pointers.output()
	return launch1DABI(state, functions[kernelGatedAddBf16RoundF32], count, &residual, &update, &gate, &output, &width, &count)
}

// mossRotateMaxWidth mirrors MOSS_ROTATE_MAX_WIDTH: the widest head moss_norm_rotate_bf16_round_f32 holds.
const mossRotateMaxWidth = 256

// applyMOSSNormRotaryRewrite folds a MOSS rotary embedding (applyMOSSRotaryRewrite) whose heads are a folded norm and
// its steps (applyMOSSNormRewrite) that the embedding alone reads, as the denoiser's q and k norms are, into one kernel
// that normalizes, steps and rotates each head's row (moss_norm_rotate_bf16_round_f32), the heads never stored.
func applyMOSSNormRotaryRewrite(context *rewriteContext) {
	compiled := context.compiled
	for _, node := range context.order {
		rotary := compiled.fusions[node]
		if rotary == nil || rotary.kind != compiledFusionMOSSRotary {
			continue
		}
		heads := rotary.operands[mossRotaryHeads]
		norm := compiled.fusions[heads]
		_, retained := context.outputSet[heads]
		if norm == nil || norm.kind != compiledFusionMOSSNorm || retained || heads.Shape.Dims[mossRotaryWidthAxis] > mossRotateMaxWidth {
			continue
		}
		absorbed := true
		for _, reader := range context.consumers[heads] {
			if _, skipped := compiled.skipped[reader]; !skipped {
				absorbed = false
			}
		}
		if !absorbed {
			continue
		}
		operands := append(slices.Clone(norm.operands), rotary.operands[mossRotaryCosine], rotary.operands[mossRotarySine])
		compiled.setFusion(node, &compiledFusion{kind: compiledFusionMOSSNormRotary, operands: operands, norm: norm.norm})
		delete(compiled.fusions, heads)
		skip(compiled, []*tensor.Tensor{heads})
	}
}

// launchMOSSNormRotary normalizes, steps and rotates heads from a folded norm's operands and the rotation's tables.
func launchMOSSNormRotary(state *device.State, functions functionSet, node *tensor.Tensor, norm mossNormFusion, pointers launchPointerFrame) error {
	width, heads, tokens := uint32(node.Shape.Dims[mossRotaryWidthAxis]), uint32(node.Shape.Dims[mossRotaryHeadAxis]), uint32(node.Shape.Dims[mossRotaryTokenAxis])
	input, output := pointers.input(0), pointers.output()
	var steps [rmsNormSteps]driver.DevicePtr
	for step := range norm.steps {
		steps[step] = pointers.input(1 + step)
	}
	cosine, sine := pointers.input(1+norm.steps), pointers.input(2+norm.steps)
	return launchNormalizationABI(state, functions[kernelMossNormRotateBf16RoundF32], heads*tokens,
		&input, &steps[0], &steps[1], &steps[2], &norm.kinds, &norm.epsilon, &cosine, &sine, &output, &width, &heads, &tokens)
}
