package executor

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// q8PairFusion: a gated layer's gate and up Q8_0 products of one input in one launch; the later product in the
// graph's order launches both and writes the earlier one's output, whose own launch is empty.
type q8PairFusion struct {
	gate, up *tensor.Tensor
}

// Operands of the fused Q8_0 launches, after the node's own slot.
const (
	// q8PairGate, q8PairUp, q8PairInput and q8PairEarlier: the gate and up weights, their input and the earlier
	// product's output.
	q8PairGate = iota
	q8PairUp
	q8PairInput
	q8PairEarlier
)

const (
	// q8GatedAddWeight, q8GatedAddInput, q8GatedAddGate and q8GatedAddAddend: the product's weight and input, the
	// gate a column and the value the gated product adds to.
	q8GatedAddWeight = iota
	q8GatedAddInput
	q8GatedAddGate
	q8GatedAddAddend
)

// q8ProductLaunch stages a decode product's input as a Q8_0 product's launch does (one quantization unless its
// producer staged it) and returns the product's inner width, rows, columns, the warps a row takes and its threads.
func q8ProductLaunch(
	state *device.State, functions functionSet, q8Input *q8InputState, product *tensor.Tensor, right driver.DevicePtr,
) (inner, rows, columns, rowWarps, threads uint32, err error) {
	leftNode, rightNode := product.Inputs[0], product.Inputs[1]
	if q8Input == nil || q8Input.staging == 0 {
		return 0, 0, 0, 0, 0, errors.New("Q8_0 fused product input workspace is unavailable")
	}
	if leftNode.Shape.Dims[0] > math.MaxUint32 || leftNode.Shape.Dims[1] > math.MaxUint32 || rightNode.Shape.Dims[1] > q8InputSpanColumns {
		return 0, 0, 0, 0, 0, errors.New("Q8_0 fused product shape is outside its kernel")
	}
	inner, rows, columns = uint32(leftNode.Shape.Dims[0]), uint32(leftNode.Shape.Dims[1]), uint32(rightNode.Shape.Dims[1])
	if q8Input.stagedNode != rightNode && q8Input.stagedNode != q8Source(rightNode) {
		blocks := uint64(inner) * uint64(columns) / q8InputTraits.BlockSize
		if err := launchQ8InputQuantization(state, functions[kernelQuantizeQ80InputF32], right, q8Input.staging, uint32(blocks)); err != nil {
			return 0, 0, 0, 0, 0, err
		}
		q8Input.stagedNode = rightNode
	}
	launchCount, err := q8InputMulMatLaunchCount(rows, 1)
	if err != nil {
		return 0, 0, 0, 0, 0, err
	}
	rowWarps = q8_0RowWarps(inner, rows)
	return inner, rows, columns, rowWarps, launchCount * rowWarps, nil
}

// launchQ8Pair runs mul_mat_q8_0_input_pair_f32 for the pair's later product, writing the earlier one's output too.
func launchQ8Pair(
	state *device.State, functions functionSet, q8Input *q8InputState, node *tensor.Tensor, fusion q8PairFusion,
	pointers launchPointerFrame,
) error {
	inner, rows, columns, rowWarps, threads, err := q8ProductLaunch(state, functions, q8Input, node, pointers.input(q8PairInput))
	if err != nil {
		return err
	}
	gate, up := pointers.input(q8PairGate), pointers.input(q8PairUp)
	gateOutput, upOutput := pointers.input(q8PairEarlier), pointers.output()
	if node == fusion.gate {
		gateOutput, upOutput = upOutput, gateOutput
	}
	return launch1DABI(
		state, functions[kernelMulMatQ80InputPairF32], threads,
		&gate, &up, &q8Input.staging, &gateOutput, &upOutput, &inner, &rows, &columns, &rowWarps,
	)
}

// launchQ8GatedAdd runs mul_mat_q8_0_input_gated_add_f32: the product scaled by its column's sigmoid gate and added.
func launchQ8GatedAdd(
	state *device.State, functions functionSet, q8Input *q8InputState, node *tensor.Tensor, pointers launchPointerFrame,
) error {
	product := node.Inputs[0]
	if product.Op != tensor.OpMultiply {
		product = node.Inputs[1]
	}
	matrix := product.Inputs[0]
	if matrix.Op != tensor.OpMulMat {
		matrix = product.Inputs[1]
	}
	inner, rows, columns, rowWarps, threads, err := q8ProductLaunch(state, functions, q8Input, matrix, pointers.input(q8GatedAddInput))
	if err != nil {
		return err
	}
	weight, gate, addend, output := pointers.input(q8GatedAddWeight), pointers.input(q8GatedAddGate), pointers.input(q8GatedAddAddend), pointers.output()
	return launch1DABI(
		state, functions[kernelMulMatQ80InputGatedAddF32], threads,
		&weight, &q8Input.staging, &gate, &addend, &output, &inner, &rows, &columns, &rowWarps,
	)
}
