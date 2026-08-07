package executor

import (
	"errors"
	"fmt"
	"math"
	"runtime"

	"llamacpp2go/internal/cuda/cublas"
	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

func launchLinearLayout(
	state *device.State,
	functions functionSet,
	blas *blasState,
	q8Input *q8InputState,
	node *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	output := pointers[node]
	switch node.Op {
	case tensor.OpRepeatHeads:
		attributes, ok := node.Attrs.(tensor.RepeatHeadsAttributes)
		if !ok || attributes.Heads == 0 || len(node.Inputs) != 1 {
			return errors.New("invalid RepeatHeads attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "RepeatHeads width")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[2], "RepeatHeads token count")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		heads := attributes.Heads
		return launch1DABI(
			state, functions.repeatHeads, count,
			&input, &output, &width, &heads, &tokens, &count,
		)
	case tensor.OpTranspose2D:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "Transpose2D width")
		if err != nil {
			return err
		}
		rows, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "Transpose2D rows")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		return launch1DABI(state, functions.transpose2D, count, &input, &output, &width, &rows, &count)
	case tensor.OpGroupSlice:
		attributes, ok := node.Attrs.(tensor.GroupSliceAttributes)
		if !ok {
			return errors.New("invalid GroupSlice attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		inputWidth, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "GroupSlice input width")
		if err != nil {
			return err
		}
		offset, err := uint32Checked(attributes.Offset, "GroupSlice offset")
		if err != nil {
			return err
		}
		width, err := uint32Checked(attributes.Width, "GroupSlice width")
		if err != nil {
			return err
		}
		groups, err := uint32Checked(attributes.Groups, "GroupSlice groups")
		if err != nil {
			return err
		}
		stride, err := uint32Checked(attributes.Stride, "GroupSlice stride")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		return launch1DABI(
			state, functions.groupSlice, count,
			&input, &output, &inputWidth, &offset, &width, &groups, &stride, &count,
		)
	case tensor.OpFlatSlice:
		attributes, ok := node.Attrs.(tensor.FlatSliceAttributes)
		if !ok {
			return errors.New("invalid FlatSlice attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		offset, err := uint32Checked(attributes.Offset, "FlatSlice offset")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		return launch1DABI(state, functions.flatSlice, count, &input, &output, &offset, &count)
	case tensor.OpRMSNorm:
		attributes, ok := node.Attrs.(tensor.RMSNormAttributes)
		if !ok {
			return errors.New("invalid RMSNorm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		epsilon := attributes.Epsilon
		return launchNormalizationABI(state, functions.rmsNorm, rows, &input, &output, &width, &rows, &epsilon)
	case tensor.OpLayerNorm:
		attributes, ok := node.Attrs.(tensor.LayerNormAttributes)
		if !ok {
			return errors.New("invalid LayerNorm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		epsilon := attributes.Epsilon
		return launchNormalizationABI(state, functions.layerNorm, rows, &input, &output, &width, &rows, &epsilon)
	case tensor.OpSoftmax:
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		return launch1DABI(state, functions.softmax, rows, &input, &output, &width, &rows)
	case tensor.OpMulMat:
		leftNode := node.Inputs[0]
		rightNode := node.Inputs[1]
		inner, err := uint32Checked(leftNode.Shape.Dims[0], "mul_mat inner dimension")
		if err != nil {
			return err
		}
		leftRows, err := uint32Checked(leftNode.Shape.Dims[1], "mul_mat left rows")
		if err != nil {
			return err
		}
		rightRows, err := uint32Checked(rightNode.Shape.Dims[1], "mul_mat right rows")
		if err != nil {
			return err
		}
		left := pointers[leftNode]
		right := pointers[rightNode]
		if leftNode.Type == dtype.BF16 {
			if blas == nil || blas.staging == 0 {
				return errors.New("cuBLAS BF16 workspace is unavailable")
			}
			elements := uint64(inner) * uint64(rightRows)
			if elements > math.MaxUint32 || elements*2 > blas.stagingBytes {
				return errors.New("BF16 mul_mat input exceeds workspace")
			}
			if blas.stagedNode != rightNode {
				count := uint32(elements)
				if err := launch1DABI(state, functions.f32ToBF16, count, &right, &blas.staging, &count); err != nil {
					return err
				}
				blas.stagedNode = rightNode
			}
			return blas.library.GEMMEx(
				blas.handle, cublas.OperationTranspose, cublas.OperationNone,
				int32(leftRows), int32(rightRows), int32(inner), 1,
				left, cublas.DataBF16, int32(inner),
				blas.staging, cublas.DataBF16, int32(inner), 0,
				output, cublas.DataF32, int32(leftRows), cublas.ComputeF32, cublas.GemmDefault,
			)
		}
		if nativeQuantizedType(leftNode.Type) {
			if rightNode.Type != dtype.F32 {
				return fmt.Errorf("%s mul_mat right input has type %s", leftNode.Type, rightNode.Type)
			}
			traits, _ := leftNode.Type.Traits()
			if uint64(inner)%traits.BlockSize != 0 {
				return fmt.Errorf("%s mul_mat inner dimension is not block aligned", leftNode.Type)
			}
			if uint64(leftRows)*uint64(rightRows) > math.MaxUint32 {
				return fmt.Errorf("%s mul_mat output element count exceeds uint32", leftNode.Type)
			}
			function := quantKernels[leftNode.Type].mulMat(functions)
			quantizedRight := right
			if leftNode.Type == dtype.Q8_0 && rightRows == 1 {
				if q8Input == nil || q8Input.staging == 0 {
					return errors.New("Q8_0 mul_mat input workspace is unavailable")
				}
				if q8Input.stagedNode != rightNode {
					blocks := uint64(inner) * uint64(rightRows) / q8InputBlockWidth
					if blocks > math.MaxUint32 {
						return errors.New("Q8_0 mul_mat input block count exceeds uint32")
					}
					blockCount := uint32(blocks)
					if err := launchQ8InputQuantization(
						state, functions.quantizeQ8Input, right, q8Input.staging, blockCount,
					); err != nil {
						return err
					}
					q8Input.stagedNode = rightNode
				}
				function = functions.mulMatQ8Input
				quantizedRight = q8Input.staging
			}
			launchCount, err := quantMulMatLaunchCount(leftNode.Type, leftRows, rightRows)
			if err != nil {
				return err
			}
			if leftNode.Type == dtype.Q8_0 && rightRows == 1 {
				launchCount, err = q8InputMulMatLaunchCount(leftRows, rightRows)
				if err != nil {
					return err
				}
			}
			return launch1DABI(
				state, function, launchCount,
				&left, &quantizedRight, &output, &inner, &leftRows, &rightRows,
			)
		}
		if blas == nil {
			return errors.New("cuBLAS is unavailable for F32 mul_mat")
		}
		err = blas.library.SGEMM(
			blas.handle,
			cublas.OperationTranspose,
			cublas.OperationNone,
			int32(leftRows),
			int32(rightRows),
			int32(inner),
			1,
			left,
			int32(inner),
			right,
			int32(inner),
			0,
			output,
			int32(leftRows),
		)
		runtime.KeepAlive(left)
		runtime.KeepAlive(right)
		runtime.KeepAlive(output)
		return err
	case tensor.OpGroupedMulMat:
		leftNode := node.Inputs[0]
		rightNode := node.Inputs[1]
		inner, err := uint32Checked(leftNode.Shape.Dims[0], "grouped_mul_mat inner dimension")
		if err != nil {
			return err
		}
		leftRows, err := uint32Checked(leftNode.Shape.Dims[1], "grouped_mul_mat left rows")
		if err != nil {
			return err
		}
		groups, err := uint32Checked(leftNode.Shape.Dims[2], "grouped_mul_mat groups")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(rightNode.Shape.Dims[2], "grouped_mul_mat tokens")
		if err != nil {
			return err
		}
		traits, ok := leftNode.Type.Traits()
		if !ok || uint64(inner)%traits.BlockSize != 0 {
			return fmt.Errorf("%s grouped_mul_mat inner dimension is not block aligned", leftNode.Type)
		}
		leftMatrixBytes := uint64(inner) * uint64(leftRows) / traits.BlockSize * traits.TypeSize
		rightVectorBytes := uint64(inner) * 4
		outputVectorBytes := uint64(leftRows) * 4
		leftBase := pointers[leftNode]
		rightBase := pointers[rightNode]
		for token := uint32(0); token < tokens; token++ {
			for group := uint32(0); group < groups; group++ {
				left := leftBase + driver.DevicePtr(uint64(group)*leftMatrixBytes)
				right := rightBase + driver.DevicePtr((uint64(token)*uint64(groups)+uint64(group))*rightVectorBytes)
				groupOutput := output + driver.DevicePtr((uint64(token)*uint64(groups)+uint64(group))*outputVectorBytes)
				if leftNode.Type == dtype.F32 {
					if blas == nil {
						return errors.New("cuBLAS is unavailable for F32 grouped_mul_mat")
					}
					if err = blas.library.SGEMM(
						blas.handle, cublas.OperationTranspose, cublas.OperationNone,
						int32(leftRows), 1, int32(inner), 1, left, int32(inner),
						right, int32(inner), 0, groupOutput, int32(leftRows),
					); err != nil {
						return err
					}
					continue
				}
				if !nativeQuantizedType(leftNode.Type) || rightNode.Type != dtype.F32 {
					return fmt.Errorf("%s grouped_mul_mat inputs are unsupported", leftNode.Type)
				}
				function := quantKernels[leftNode.Type].mulMat(functions)
				rightRows := uint32(1)
				launchCount, launchErr := quantMulMatLaunchCount(leftNode.Type, leftRows, rightRows)
				if launchErr != nil {
					return launchErr
				}
				if err = launch1DABI(
					state, function, launchCount,
					&left, &right, &groupOutput, &inner, &leftRows, &rightRows,
				); err != nil {
					return err
				}
			}
		}
		runtime.KeepAlive(leftBase)
		runtime.KeepAlive(rightBase)
		runtime.KeepAlive(output)
		return nil
	case tensor.OpGetRows:
		attributes, ok := node.Attrs.(tensor.GetRowsAttributes)
		if !ok {
			return errors.New("invalid get_rows attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "get_rows width")
		if err != nil {
			return err
		}
		table := pointers[node.Inputs[0]]
		rows, ok := attributePointers[node]
		if !ok {
			return errors.New("get_rows row storage is unavailable")
		}
		if len(attributes.Rows) == 0 {
			return errors.New("get_rows row list is empty")
		}
		function := functions.getRows
		if descriptor, ok := quantKernels[node.Inputs[0].Type]; ok {
			traits, _ := node.Inputs[0].Type.Traits()
			if uint64(width)%traits.BlockSize != 0 {
				return fmt.Errorf("%s get_rows width is not block aligned", descriptor.label)
			}
			function = descriptor.getRows(functions)
		}
		return launch1DABI(state, function, count, &table, &rows, &output, &width, &count)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}

func q8InputMulMatLaunchCount(leftRows, rightRows uint32) (uint32, error) {
	const warpThreads = uint64(q8InputBlockWidth)
	warps := uint64(leftRows) * uint64(rightRows)
	if warps > math.MaxUint32/warpThreads {
		return 0, errors.New("Q8_0 input mul_mat launch size exceeds uint32")
	}
	return uint32(warps * warpThreads), nil
}

func launchQ8InputQuantization(
	state *device.State,
	function driver.Function,
	input, output driver.DevicePtr,
	blocks uint32,
) error {
	const threads = uint32(q8InputBlockWidth)
	return launchGridABI(
		state, function,
		driver.Dim3{X: blocks, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		&input, &output, &blocks,
	)
}

func launchWeightedRMSNorm(
	state *device.State,
	functions functionSet,
	q8Input *q8InputState,
	output *tensor.Tensor,
	fusion weightedRMSFusion,
	emitQ8 bool,
	pointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	attributes, ok := fusion.normalization.Attrs.(tensor.RMSNormAttributes)
	if !ok {
		return errors.New("invalid fused RMSNorm attributes")
	}
	width, rows, err := rowDimensions32(output.Shape)
	if err != nil {
		return err
	}
	input := pointers[fusion.normalization.Inputs[0]]
	weight := pointers[fusion.weight]
	result := pointers[output]
	epsilon := attributes.Epsilon
	if emitQ8 {
		if q8Input == nil || q8Input.staging == 0 {
			return errors.New("weighted RMS Q8 workspace is unavailable")
		}
		if err := launchNormalizationABI(
			state, functions.weightedRMSNormQ8, rows,
			&input, &weight, &result, &q8Input.staging, &width, &rows, &epsilon,
		); err != nil {
			return err
		}
		q8Input.stagedNode = output
		return nil
	}
	return launchNormalizationABI(
		state, functions.weightedRMSNorm, rows,
		&input, &weight, &result, &width, &rows, &epsilon,
	)
}

func launchActivatedGate(
	state *device.State,
	functions functionSet,
	q8Input *q8InputState,
	output *tensor.Tensor,
	fusion activatedGateFusion,
	emitQ8 bool,
	pointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	count, err := elementCount32(output.Shape)
	if err != nil {
		return err
	}
	gate, up, result := pointers[fusion.gate], pointers[fusion.up], pointers[output]
	kind := uint32(fusion.kind)
	if !emitQ8 {
		return launch1DABI(
			state, functions.activatedGate, count,
			&gate, &up, &result, &kind, &count,
		)
	}
	if q8Input == nil || q8Input.staging == 0 || uint64(count)%q8InputBlockWidth != 0 {
		return errors.New("activated-gate Q8 workspace is unavailable")
	}
	blocks := count / uint32(q8InputBlockWidth)
	const threads = uint32(q8InputBlockWidth)
	if err := launchGridABI(
		state, functions.activatedGateQ8,
		driver.Dim3{X: blocks, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		&gate, &up, &result, &q8Input.staging, &kind, &count,
	); err != nil {
		return err
	}
	q8Input.stagedNode = output
	return nil
}

const (
	q8ArgmaxWarpsPerBlock = uint64(8)
	q8ArgmaxPartialValues = uint64(2)
)

func q8ArgmaxPartialCount(rows uint64) (uint32, bool) {
	count := (rows + q8ArgmaxWarpsPerBlock - 1) / q8ArgmaxWarpsPerBlock
	return uint32(count), count <= math.MaxUint32
}

func launchQ8ArgmaxPartials(
	state *device.State,
	functions functionSet,
	q8Input *q8InputState,
	projection *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	leftNode, rightNode := projection.Inputs[0], projection.Inputs[1]
	inner, err := uint32Checked(leftNode.Shape.Dims[0], "Q8 argmax inner dimension")
	if err != nil {
		return err
	}
	rows, err := uint32Checked(leftNode.Shape.Dims[1], "Q8 argmax row count")
	if err != nil {
		return err
	}
	if q8Input == nil || q8Input.staging == 0 {
		return errors.New("Q8 argmax input workspace is unavailable")
	}
	right := pointers[rightNode]
	if q8Input.stagedNode != rightNode {
		blocks := uint64(inner) / q8InputBlockWidth
		if blocks > math.MaxUint32 {
			return errors.New("Q8 argmax input block count exceeds uint32")
		}
		if err := launchQ8InputQuantization(
			state, functions.quantizeQ8Input, right, q8Input.staging, uint32(blocks),
		); err != nil {
			return err
		}
		q8Input.stagedNode = rightNode
	}
	left, partials := pointers[leftNode], pointers[projection]
	partialCount, ok := q8ArgmaxPartialCount(uint64(rows))
	if !ok {
		return errors.New("Q8 argmax partial count exceeds uint32")
	}
	const threads = uint32(q8ArgmaxWarpsPerBlock * q8InputBlockWidth)
	return launchGridABI(
		state, functions.mulMatQ8Argmax,
		driver.Dim3{X: partialCount, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		&left, &q8Input.staging, &partials, &inner, &rows,
	)
}

func launchQ8ArgmaxReduction(
	state *device.State,
	functions functionSet,
	projection, selection *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	rows, err := uint32Checked(projection.Shape.Dims[0], "Q8 argmax row count")
	if err != nil {
		return err
	}
	partialCount, ok := q8ArgmaxPartialCount(uint64(rows))
	if !ok {
		return errors.New("Q8 argmax partial count exceeds uint32")
	}
	partials, output := pointers[projection], pointers[selection]
	return launchGridABI(
		state, functions.q8ArgmaxReduction,
		driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: 256, Y: 1, Z: 1},
		&partials, &output, &partialCount,
	)
}

func quantMulMatLaunchCount(storage dtype.Type, leftRows, rightRows uint32) (uint32, error) {
	const (
		q8DotProductThreads = uint32(32)
		q8VectorsPerWarp    = uint32(4)
	)
	warps := uint64(leftRows) * uint64(rightRows)
	if storage == dtype.Q8_0 {
		rightTiles := (rightRows-1)/q8VectorsPerWarp + 1
		warps = uint64(leftRows) * uint64(rightTiles)
	}
	if warps > uint64(math.MaxUint32/q8DotProductThreads) && storage == dtype.Q8_0 {
		return 0, errors.New("Q8_0 mul_mat launch size exceeds uint32")
	}
	if warps > math.MaxUint32 {
		return 0, errors.New("quantized mul_mat launch size exceeds uint32")
	}
	launches := uint32(warps)
	if storage == dtype.Q8_0 {
		launches *= q8DotProductThreads
	}
	return launches, nil
}

func launchNormalizationABI(
	state *device.State,
	function driver.Function,
	rows uint32,
	arguments ...any,
) error {
	if rows == 0 {
		return nil
	}
	const threads = uint32(256)
	return launchGridABI(
		state, function,
		driver.Dim3{X: rows, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		arguments...,
	)
}
