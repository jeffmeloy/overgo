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
		launchCount, err := normalizationLaunchCount(rows)
		if err != nil {
			return err
		}
		return launch1DABI(state, functions.rmsNorm, launchCount, &input, &output, &width, &rows, &epsilon)
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
		launchCount, err := normalizationLaunchCount(rows)
		if err != nil {
			return err
		}
		return launch1DABI(state, functions.layerNorm, launchCount, &input, &output, &width, &rows, &epsilon)
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
			count := leftRows * rightRows
			function := quantKernels[leftNode.Type].mulMat(functions)
			launchCount, err := quantMulMatLaunchCount(leftNode.Type, count)
			if err != nil {
				return err
			}
			return launch1DABI(
				state, function, launchCount, &left, &right, &output, &inner, &leftRows, &rightRows,
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
				launchCount, launchErr := quantMulMatLaunchCount(leftNode.Type, leftRows)
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

func quantMulMatLaunchCount(storage dtype.Type, outputs uint32) (uint32, error) {
	if storage != dtype.Q8_0 {
		return outputs, nil
	}
	const q8DotProductThreads = uint32(32)
	if outputs > math.MaxUint32/q8DotProductThreads {
		return 0, errors.New("Q8_0 mul_mat launch size exceeds uint32")
	}
	return outputs * q8DotProductThreads, nil
}

func normalizationLaunchCount(rows uint32) (uint32, error) {
	const normalizationThreadsPerRow = uint32(32)
	if rows > math.MaxUint32/normalizationThreadsPerRow {
		return 0, errors.New("normalization launch size exceeds uint32")
	}
	return rows * normalizationThreadsPerRow, nil
}
