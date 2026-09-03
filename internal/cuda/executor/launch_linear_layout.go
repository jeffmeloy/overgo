package executor

import (
	"errors"
	"fmt"
	"math"
	"runtime"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

const fp8KernelLaneWidth = uint32(4)

func launchLinearLayout(
	state *device.State,
	functions functionSet,
	blas *blasState,
	q8Input *q8InputState,
	node *tensor.Tensor,
	runtimeAttributes tensor.Attributes,
	pointers launchPointerFrame,
) error {
	output := pointers.output()
	switch node.Op {
	case tensor.OpRepeatHeads:
		attributes, ok := runtimeAttributes.(tensor.RepeatHeadsAttributes)
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
		input := pointers.input(0)
		heads := attributes.Heads
		return launch1DABI(
			state, functions[kernelRepeatHeadsF32], count,
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
		input := pointers.input(0)
		return launch1DABI(state, functions[kernelTranspose2dF32], count, &input, &output, &width, &rows, &count)
	case tensor.OpGroupSlice:
		attributes, ok := runtimeAttributes.(tensor.GroupSliceAttributes)
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
		input := pointers.input(0)
		return launch1DABI(
			state, functions[kernelGroupSliceF32], count,
			&input, &output, &inputWidth, &offset, &width, &groups, &stride, &count,
		)
	case tensor.OpFlatSlice:
		attributes, ok := runtimeAttributes.(tensor.FlatSliceAttributes)
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
		input := pointers.input(0)
		return launch1DABI(state, functions[kernelFlatSliceF32], count, &input, &output, &offset, &count)
	case tensor.OpRMSNorm:
		attributes, ok := runtimeAttributes.(tensor.RMSNormAttributes)
		if !ok {
			return errors.New("invalid RMSNorm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		epsilon := attributes.Epsilon
		return launchNormalizationABI(state, functions[kernelRmsNormF32], rows, &input, &output, &width, &rows, &epsilon)
	case tensor.OpMADNorm:
		attributes, ok := runtimeAttributes.(tensor.MADNormAttributes)
		if !ok {
			return errors.New("invalid MADNorm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		epsilon := attributes.Epsilon
		return launch1DABI(state, functions[kernelMadNormF32], rows, &input, &output, &width, &rows, &epsilon)
	case tensor.OpLayerNorm:
		attributes, ok := runtimeAttributes.(tensor.LayerNormAttributes)
		if !ok {
			return errors.New("invalid LayerNorm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		epsilon := attributes.Epsilon
		return launchNormalizationABI(state, functions[kernelLayerNormF32], rows, &input, &output, &width, &rows, &epsilon)
	case tensor.OpSoftmax:
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		return launch1DABI(state, functions[kernelSoftmaxF32], rows, &input, &output, &width, &rows)
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
		left := pointers.input(0)
		right := pointers.input(1)
		if leftNode.Type == dtype.F16 || leftNode.Type == dtype.BF16 {
			if tensorCoreMulMatAttributes(leftNode.Type, runtimeAttributes) && rightRows > 1 {
				// Tensor-core prefill: the resident half-precision weight is
				// multiplied as stored; the F32 activation is rounded to the
				// same dtype in the staging workspace (one pack per node,
				// shared by every weight it feeds) and cuBLAS accumulates
				// the products in F32. The weight is read once per forward.
				if rightNode.Type != dtype.F32 {
					return fmt.Errorf("%s tensor-core mul_mat has incompatible inputs", leftNode.Type)
				}
				if blas == nil || blas.staging == 0 {
					return fmt.Errorf("%s tensor-core mul_mat workspace is unavailable", leftNode.Type)
				}
				elements := uint64(inner) * uint64(rightRows)
				if elements > math.MaxUint32 || elements*bf16ScalarBytes > blas.stagingBytes {
					return fmt.Errorf("%s tensor-core mul_mat input exceeds workspace", leftNode.Type)
				}
				packKernel, operandType := kernelF32ToBf16, cublas.DataBF16
				if leftNode.Type == dtype.F16 {
					packKernel, operandType = kernelF32ToF16, cublas.DataF16
				}
				if blas.stagedNode != rightNode || blas.stagedType != leftNode.Type {
					count := uint32(elements)
					if err := launch1DABI(
						state, functions[packKernel], count,
						&right, &blas.staging, &count,
					); err != nil {
						return err
					}
					blas.stagedNode, blas.stagedType = rightNode, leftNode.Type
				}
				if traceExternalCall(
					state, traceTagGEMMEx,
					uint64(left), uint64(blas.staging), uint64(output),
					uint64(leftRows), uint64(rightRows), uint64(inner),
				) {
					return nil
				}
				return blas.library.GEMMEx(
					blas.handle, cublas.OperationTranspose, cublas.OperationNone,
					int32(leftRows), int32(rightRows), int32(inner), gemmProductScale,
					left, operandType, int32(inner),
					blas.staging, operandType, int32(inner), gemmAccumulatorScale,
					output, cublas.DataF32, int32(leftRows), cublas.ComputeF32, cublas.GemmDefault,
				)
			}
			// native half-precision residency, one mechanism parameterized by the
			// weight dtype: single-token decode reads the 2-byte weight directly
			// (lossless per-element upconvert in-kernel, F32 accumulate);
			// multi-token prefill upconverts the weight to F32 in a reused scratch
			// and runs the identical SGEMM as the F32 fallback. The upconvert
			// (F16 __half2float / BF16 bits<<16) reproduces the resident F32-copy
			// dequant bit-for-bit, so the SGEMM operands and result are bit-exact.
			decodeKernel := kernelMulMatF16F32
			upconvertKernel := kernelF16ToF32
			weightScalarBytes, _ := leftNode.Type.ScalarBytes()
			if leftNode.Type == dtype.BF16 {
				decodeKernel = kernelMulMatBf16F32
				upconvertKernel = kernelBf16ToF32
			}
			if nativeDecodeSpanFits(weightScalarBytes, rightRows) && inner%2 == 0 && rightNode.Type == dtype.F32 {
				launchCount, err := bf16MulMatLaunchCount(leftRows, rightRows)
				if err != nil {
					return err
				}
				return launch1DABI(
					state, functions[decodeKernel], launchCount,
					&left, &right, &output, &inner, &leftRows, &rightRows,
				)
			}
			if rightNode.Type != dtype.F32 {
				return fmt.Errorf("%s mul_mat right input has type %s", leftNode.Type, rightNode.Type)
			}
			if blas == nil || blas.staging == 0 {
				return fmt.Errorf("%s mul_mat weight workspace is unavailable", leftNode.Type)
			}
			blas.stagedNode = nil
			return launchStagedNativeMatMul(
				state, blas, inner, leftRows, rightRows, right, output,
				func(start, rows, count uint32) error {
					elementOffset := uint64(start) * uint64(inner)
					source := left + driver.DevicePtr(elementOffset*weightScalarBytes)
					return launch1DABI(
						state, functions[upconvertKernel], count,
						&source, &blas.staging, &count,
					)
				},
			)
		}
		if leftNode.Type == dtype.F8E4M3 {
			// native fp8 residency, same native-dtype contract as F16/BF16 but the
			// weight is packed e4m3 (1 byte/element) followed IN THE SAME resident
			// buffer by a per-output-row F32 scale [rows*inner e4m3 | rows*4 scale].
			// Decode (single token) reads e4m3 quads directly and folds scale[row]
			// once; prefill upconverts (e4m3 -> F32, scale folded per element) into
			// the reused weight staging and runs the identical SGEMM.
			if rightNode.Type != dtype.F32 {
				return fmt.Errorf("fp8 mul_mat right input has type %s", rightNode.Type)
			}
			if inner%fp8KernelLaneWidth != 0 {
				return fmt.Errorf("fp8 mul_mat inner dimension is not a multiple of %d", fp8KernelLaneWidth)
			}
			traits, _ := leftNode.Type.Traits()
			scaleOffset := uint64(inner) * uint64(leftRows) * traits.TypeSize
			scale := left + driver.DevicePtr(scaleOffset)
			if nativeDecodeSpanFits(traits.TypeSize, rightRows) {
				launchCount, err := bf16MulMatLaunchCount(leftRows, rightRows)
				if err != nil {
					return err
				}
				return launch1DABI(
					state, functions[kernelMulMatFp8F32], launchCount,
					&left, &scale, &right, &output, &inner, &leftRows, &rightRows,
				)
			}
			if blas == nil || blas.staging == 0 {
				return errors.New("fp8 mul_mat weight workspace is unavailable")
			}
			blas.stagedNode = nil
			return launchStagedNativeMatMul(
				state, blas, inner, leftRows, rightRows, right, output,
				func(start, rows, _ uint32) error {
					elementOffset := uint64(start) * uint64(inner)
					source := left + driver.DevicePtr(elementOffset)
					sourceScale := scale + driver.DevicePtr(uint64(start)*leftNode.Type.AuxiliaryRowBytes())
					return launchGridABI(
						state, functions[kernelFp8ToF32],
						kernel.Grid1D(int(rows)), kernel.DefaultBlock1D(),
						&source, &sourceScale, &blas.staging, &inner, &rows,
					)
				},
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
			if inputKernel, fast := q8InputMulMatKernels[leftNode.Type]; fast {
				if q8Input == nil || q8Input.staging == 0 {
					return fmt.Errorf("%s mul_mat input workspace is unavailable", leftNode.Type)
				}
				if q8Input.stagedNode != rightNode {
					blocks := uint64(inner) * uint64(rightRows) / q8InputTraits.BlockSize
					if blocks > math.MaxUint32 {
						return fmt.Errorf("%s mul_mat input block count exceeds uint32", leftNode.Type)
					}
					blockCount := uint32(blocks)
					if err := launchQ8InputQuantization(
						state, functions[kernelQuantizeQ80InputF32], right, q8Input.staging, blockCount,
					); err != nil {
						return err
					}
					q8Input.stagedNode = rightNode
				}
				return launchQ8InputSpans(
					state, functions[inputKernel], left, q8Input.staging, output, inner, leftRows, rightRows,
				)
			}
			function := functions[quantKernels[leftNode.Type].mulMat]
			launchCount, err := quantMulMatLaunchCount(leftNode.Type, leftRows, rightRows)
			if err != nil {
				return err
			}
			return launch1DABI(
				state, function, launchCount,
				&left, &right, &output, &inner, &leftRows, &rightRows,
			)
		}
		if blas == nil {
			return errors.New("cuBLAS is unavailable for F32 mul_mat")
		}
		if traceExternalCall(
			state, traceTagSGEMM,
			uint64(left), uint64(right), uint64(output),
			uint64(leftRows), uint64(rightRows), uint64(inner),
		) {
			return nil
		}
		err = blas.library.SGEMM(
			blas.handle,
			cublas.OperationTranspose,
			cublas.OperationNone,
			int32(leftRows),
			int32(rightRows),
			int32(inner),
			gemmProductScale,
			left,
			int32(inner),
			right,
			int32(inner),
			gemmAccumulatorScale,
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
		leftMatrixBytes, err := leftNode.Type.StorageBytes(uint64(inner)*uint64(leftRows), uint64(inner))
		if err != nil {
			return fmt.Errorf("%s grouped_mul_mat storage: %w", leftNode.Type, err)
		}
		rightVectorBytes := uint64(inner) * f32ScalarBytes
		outputVectorBytes := uint64(leftRows) * f32ScalarBytes
		leftBase := pointers.input(0)
		rightBase := pointers.input(1)
		for token := uint32(0); token < tokens; token++ {
			for group := uint32(0); group < groups; group++ {
				left := leftBase + driver.DevicePtr(uint64(group)*leftMatrixBytes)
				right := rightBase + driver.DevicePtr((uint64(token)*uint64(groups)+uint64(group))*rightVectorBytes)
				groupOutput := output + driver.DevicePtr((uint64(token)*uint64(groups)+uint64(group))*outputVectorBytes)
				if leftNode.Type == dtype.F32 {
					if blas == nil {
						return errors.New("cuBLAS is unavailable for F32 grouped_mul_mat")
					}
					if traceExternalCall(
						state, traceTagSGEMM,
						uint64(left), uint64(right), uint64(groupOutput),
						uint64(leftRows), 1, uint64(inner),
					) {
						continue
					}
					if err = blas.library.SGEMM(
						blas.handle, cublas.OperationTranspose, cublas.OperationNone,
						int32(leftRows), 1, int32(inner), gemmProductScale, left, int32(inner),
						right, int32(inner), gemmAccumulatorScale, groupOutput, int32(leftRows),
					); err != nil {
						return err
					}
					continue
				}
				if !nativeQuantizedType(leftNode.Type) || rightNode.Type != dtype.F32 {
					return fmt.Errorf("%s grouped_mul_mat inputs are unsupported", leftNode.Type)
				}
				function := functions[quantKernels[leftNode.Type].mulMat]
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
		attributes, ok := runtimeAttributes.(tensor.GetRowsAttributes)
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
		table := pointers.input(0)
		rows := pointers.attribute
		if rows == 0 {
			return errors.New("get_rows row storage is unavailable")
		}
		if len(attributes.Rows) == 0 {
			return errors.New("get_rows row list is empty")
		}
		function := functions[kernelGetRowsF32]
		if node.Inputs[0].Type == dtype.BF16 {
			function = functions[kernelGetRowsBf16F32]
		} else if descriptor, ok := quantKernel(node.Inputs[0].Type); ok {
			traits, _ := node.Inputs[0].Type.Traits()
			if _, aligned := traits.BlockCount(uint64(width)); !aligned {
				return fmt.Errorf("%s get_rows width is not block aligned", descriptor.label)
			}
			function = functions[descriptor.getRows]
		}
		return launch1DABI(state, function, count, &table, &rows, &output, &width, &count)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}

func launchStagedNativeMatMul(
	state *device.State,
	blas *blasState,
	inner, leftRows, rightRows uint32,
	right, output driver.DevicePtr,
	stage func(start, rows, count uint32) error,
) error {
	if blas == nil || blas.staging == 0 || stage == nil {
		return errors.New("native mul_mat staging is unavailable")
	}
	if inner > math.MaxInt32 || leftRows > math.MaxInt32 || rightRows > math.MaxInt32 {
		return errors.New("native mul_mat geometry exceeds cuBLAS")
	}
	rowBytes := uint64(inner) * f32ScalarBytes
	capacity := blas.stagingBytes / rowBytes
	if capacity == 0 {
		return errors.New("native mul_mat staging is smaller than one row")
	}
	for start := uint32(0); start < leftRows; {
		rows := min(leftRows-start, uint32(min(capacity, uint64(math.MaxUint32))))
		count64 := uint64(rows) * uint64(inner)
		if count64 > math.MaxUint32 {
			return errors.New("native mul_mat staging launch overflows")
		}
		if err := stage(start, rows, uint32(count64)); err != nil {
			return err
		}
		chunkOutput := output + driver.DevicePtr(uint64(start)*f32ScalarBytes)
		if !traceExternalCall(
			state, traceTagSGEMM,
			uint64(blas.staging), uint64(right), uint64(chunkOutput),
			uint64(rows), uint64(rightRows), uint64(inner),
		) {
			if err := blas.library.SGEMM(
				blas.handle, cublas.OperationTranspose, cublas.OperationNone,
				int32(rows), int32(rightRows), int32(inner), gemmProductScale,
				blas.staging, int32(inner), right, int32(inner), gemmAccumulatorScale,
				chunkOutput, int32(leftRows),
			); err != nil {
				return err
			}
		}
		start += rows
	}
	return nil
}

// tensorCoreMulMat reports whether a compiled mul_mat node multiplies on
// tensor cores with its activation staged in the weight's dtype: the
// explicit BF16 policy, or the native policy over an F16 or BF16 weight.
func tensorCoreMulMat(node *tensor.Tensor) bool {
	if node == nil || len(node.Inputs) == 0 {
		return false
	}
	return tensorCoreMulMatAttributes(node.Inputs[0].Type, node.Attrs)
}

func tensorCoreMulMatAttributes(weightType dtype.Type, attributes tensor.Attributes) bool {
	mulMat, ok := attributes.(tensor.MulMatAttributes)
	if !ok {
		return false
	}
	switch mulMat.Compute {
	case tensor.MulMatComputeBF16TensorCore:
		return weightType == dtype.BF16
	case tensor.MulMatComputeNativeTensorCore:
		return weightType == dtype.F16 || weightType == dtype.BF16
	}
	return false
}

// nativeDecodeSpanFits decides whether a native-dtype (F16, BF16, fp8)
// mul_mat with rightRows input columns runs through the decode kernel,
// which reads the resident weight once per column, or through the staged
// path, which reads the weight once, writes its F32 copy, and reads that
// copy back for SGEMM. The decode kernel wins while the bytes it moves
// (columns x weight bytes) stay below the staged path's traffic (weight
// bytes plus two F32 passes): four columns for 2-byte weights, eight for
// fp8. A grouped-choice continuation of a few tokens therefore never
// upconverts the whole model.
func nativeDecodeSpanFits(weightScalarBytes uint64, rightRows uint32) bool {
	if weightScalarBytes == 0 {
		return false
	}
	stagedTraffic := weightScalarBytes + 2*f32ScalarBytes
	return uint64(rightRows)*weightScalarBytes < stagedTraffic
}

func bf16MulMatLaunchCount(leftRows, rightRows uint32) (uint32, error) {
	warpThreads := uint64(kernel.WarpThreads())
	warps := uint64(leftRows) * uint64(rightRows)
	if warps > math.MaxUint32/warpThreads {
		return 0, errors.New("BF16 mul_mat launch size exceeds uint32")
	}
	return uint32(warps * warpThreads), nil
}

func q8InputMulMatLaunchCount(leftRows, rightRows uint32) (uint32, error) {
	warpThreads := q8InputTraits.BlockSize
	warps := uint64(leftRows) * uint64(rightRows)
	if warps > math.MaxUint32/warpThreads {
		return 0, errors.New("Q8_0 input mul_mat launch size exceeds uint32")
	}
	return uint32(warps * warpThreads), nil
}

// launchQ8InputSpans runs a quantized-weight mul_mat over the staged
// q8 right operand in spans of q8InputSpanColumns: one warp per weight
// row serves a whole span from a single weight read, so a prompt of N
// columns reads the weights ceil(N/span) times instead of N times --
// the difference between prefill at decode rate and prefill at the
// kernel's span throughput (the 27B BBH pass measured 33 ms per prompt
// token on the per-column kernel, one full weight stream per token).
// The staged input is row-major q8 blocks; the output is column-major
// [rightRows][leftRows] f32, so both advance by whole spans.
func launchQ8InputSpans(
	state *device.State,
	function boundKernel,
	left, staged, output driver.DevicePtr,
	inner, leftRows, rightRows uint32,
) error {
	launchCount, err := q8InputMulMatLaunchCount(leftRows, 1)
	if err != nil {
		return err
	}
	inputStride := uint64(inner) / q8InputTraits.BlockSize * q8InputTraits.TypeSize
	outputStride := uint64(leftRows) * f32ScalarBytes
	for start := uint32(0); start < rightRows; start += q8InputSpanColumns {
		columns := min(q8InputSpanColumns, rightRows-start)
		spanInput := staged + driver.DevicePtr(uint64(start)*inputStride)
		spanOutput := output + driver.DevicePtr(uint64(start)*outputStride)
		if err := launch1DABI(
			state, function, launchCount,
			&left, &spanInput, &spanOutput, &inner, &leftRows, &columns,
		); err != nil {
			return err
		}
	}
	return nil
}

func launchQ8InputQuantization(
	state *device.State,
	function boundKernel,
	input, output driver.DevicePtr,
	blocks uint32,
) error {
	threads := uint32(q8InputTraits.BlockSize)
	return launchGridABI(
		state, function,
		kernel.Grid1D(int(blocks)), kernel.Grid1D(int(threads)),
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
	pointers launchPointerFrame,
) error {
	attributes, ok := fusion.normalization.Attrs.(tensor.RMSNormAttributes)
	if !ok {
		return errors.New("invalid fused RMSNorm attributes")
	}
	width, rows, err := rowDimensions32(output.Shape)
	if err != nil {
		return err
	}
	input := pointers.input(0)
	weight := pointers.input(1)
	result := pointers.output()
	epsilon := attributes.Epsilon
	if fusion.addLeft != nil {
		left, right := pointers.input(2), pointers.input(3)
		if emitQ8 {
			if q8Input == nil || q8Input.staging == 0 {
				return errors.New("weighted RMS add Q8 workspace is unavailable")
			}
			if err := launchNormalizationABI(
				state, functions[kernelWeightedRmsNormAddQ80F32], rows,
				&left, &right, &weight, &result, &q8Input.staging, &width, &rows, &epsilon,
			); err != nil {
				return err
			}
			q8Input.stagedNode = output
			return nil
		}
		return launchNormalizationABI(
			state, functions[kernelWeightedRmsNormAddF32], rows,
			&left, &right, &weight, &result, &width, &rows, &epsilon,
		)
	}
	if emitQ8 {
		if q8Input == nil || q8Input.staging == 0 {
			return errors.New("weighted RMS Q8 workspace is unavailable")
		}
		if err := launchNormalizationABI(
			state, functions[kernelWeightedRmsNormQ80F32], rows,
			&input, &weight, &result, &q8Input.staging, &width, &rows, &epsilon,
		); err != nil {
			return err
		}
		q8Input.stagedNode = output
		return nil
	}
	return launchNormalizationABI(
		state, functions[kernelWeightedRmsNormF32], rows,
		&input, &weight, &result, &width, &rows, &epsilon,
	)
}

// launchGELUTanh: one fused pass over the exact tanh-GELU chain; a non-nil
// bias folds the preceding rank-1 broadcast bias add.
func launchGELUTanh(
	state *device.State,
	functions functionSet,
	output *tensor.Tensor,
	fusion geluTanhFusion,
	pointers launchPointerFrame,
) error {
	count, err := elementCount32(output.Shape)
	if err != nil {
		return err
	}
	input := pointers.input(0)
	result := pointers.output()
	var bias driver.DevicePtr
	biasWidth := uint32(0)
	if fusion.bias != nil {
		bias = pointers.input(1)
		width, widthErr := uint32Checked(fusion.bias.Shape.Dims[0], "gelu_tanh bias width")
		if widthErr != nil {
			return widthErr
		}
		biasWidth = width
	}
	cubic, inner, half := fusion.cubicCoefficient, fusion.innerScale, fusion.halfScale
	return launch1DABI(
		state, functions[kernelGeluTanhExactF32], count,
		&input, &bias, &result, &cubic, &inner, &half, &biasWidth, &count,
	)
}

// launchLayerNormModulate: layer norm with the modulation epilogue fused
// into its write-back pass.
func launchLayerNormModulate(
	state *device.State,
	functions functionSet,
	output *tensor.Tensor,
	fusion layerNormModulateFusion,
	pointers launchPointerFrame,
) error {
	attributes, ok := fusion.normalization.Attrs.(tensor.LayerNormAttributes)
	if !ok {
		return errors.New("invalid fused LayerNorm attributes")
	}
	width, rows, err := rowDimensions32(output.Shape)
	if err != nil {
		return err
	}
	input := pointers.input(0)
	scale := pointers.input(1)
	shift := pointers.input(2)
	result := pointers.output()
	epsilon := attributes.Epsilon
	adaptive := uint32(0)
	if fusion.adaptive {
		adaptive = 1
	}
	return launchNormalizationABI(
		state, functions[kernelLayerNormModulateF32], rows,
		&input, &scale, &shift, &result, &width, &rows, &epsilon, &adaptive,
	)
}

// launchBroadcastGateAdd: residual + value*gate joined in one pass.
func launchBroadcastGateAdd(
	state *device.State,
	functions functionSet,
	output *tensor.Tensor,
	fusion broadcastGateAddFusion,
	pointers launchPointerFrame,
) error {
	count, err := elementCount32(output.Shape)
	if err != nil {
		return err
	}
	width, err := uint32Checked(fusion.gate.Shape.Dims[0], "broadcast gate width")
	if err != nil {
		return err
	}
	value := pointers.input(0)
	gate := pointers.input(1)
	residual := pointers.input(2)
	result := pointers.output()
	return launch1DABI(
		state, functions[kernelBroadcastGateAddF32], count,
		&value, &gate, &residual, &result, &width, &count,
	)
}

func launchActivatedGate(
	state *device.State,
	functions functionSet,
	q8Input *q8InputState,
	output *tensor.Tensor,
	fusion activatedGateFusion,
	emitQ8 bool,
	pointers launchPointerFrame,
) error {
	count, err := elementCount32(output.Shape)
	if err != nil {
		return err
	}
	gate, up, result := pointers.input(0), pointers.input(1), pointers.output()
	kind := uint32(fusion.kind)
	if !emitQ8 {
		return launch1DABI(
			state, functions[kernelActivatedGateF32], count,
			&gate, &up, &result, &kind, &count,
		)
	}
	if q8Input == nil || q8Input.staging == 0 || uint64(count)%q8InputTraits.BlockSize != 0 {
		return errors.New("activated-gate Q8 workspace is unavailable")
	}
	blocks := count / uint32(q8InputTraits.BlockSize)
	threads := uint32(q8InputTraits.BlockSize)
	if err := launchGridABI(
		state, functions[kernelActivatedGateQ80F32],
		kernel.Grid1D(int(blocks)), kernel.Grid1D(int(threads)),
		&gate, &up, &result, &q8Input.staging, &kind, &count,
	); err != nil {
		return err
	}
	q8Input.stagedNode = output
	return nil
}

func launchWeightedRMSGate(
	state *device.State,
	functions functionSet,
	q8Input *q8InputState,
	output *tensor.Tensor,
	fusion weightedRMSGateFusion,
	emitQ8 bool,
	pointers launchPointerFrame,
) error {
	attributes, ok := fusion.normalization.Attrs.(tensor.RMSNormAttributes)
	if !ok {
		return errors.New("invalid fused gated RMSNorm attributes")
	}
	width, rows, err := rowDimensions32(output.Shape)
	if err != nil {
		return err
	}
	left := pointers.input(0)
	right := left
	useAdd := uint32(0)
	if fusion.addLeft != nil {
		left, right = pointers.input(3), pointers.input(4)
		useAdd = 1
	}
	gate := pointers.input(1)
	weight := pointers.input(2)
	result := pointers.output()
	kind := uint32(fusion.kind)
	epsilon := attributes.Epsilon
	if !emitQ8 {
		return launchNormalizationABI(
			state, functions[kernelWeightedRmsGateF32], rows,
			&left, &right, &gate, &weight, &result,
			&kind, &useAdd, &width, &rows, &epsilon,
		)
	}
	if q8Input == nil || q8Input.staging == 0 || width%uint32(q8InputTraits.BlockSize) != 0 {
		return errors.New("weighted RMS gate Q8 workspace is unavailable")
	}
	if err := launchNormalizationABI(
		state, functions[kernelWeightedRmsGateQ80F32], rows,
		&left, &right, &gate, &weight, &result, &q8Input.staging,
		&kind, &useAdd, &width, &rows, &epsilon,
	); err != nil {
		return err
	}
	q8Input.stagedNode = output
	return nil
}

func launchBF16ProjAdd(
	state *device.State,
	functions functionSet,
	output *tensor.Tensor,
	fusion bf16ProjAddFusion,
	pointers launchPointerFrame,
) error {
	leftNode := fusion.projection.Inputs[0]
	inner, err := uint32Checked(leftNode.Shape.Dims[0], "BF16 projection inner dimension")
	if err != nil {
		return err
	}
	rows, err := uint32Checked(leftNode.Shape.Dims[1], "BF16 projection row count")
	if err != nil {
		return err
	}
	launchCount, err := bf16MulMatLaunchCount(rows, 1)
	if err != nil {
		return err
	}
	left := pointers.input(0)
	right := pointers.input(1)
	addend := pointers.input(2)
	result := pointers.output()
	return launch1DABI(
		state, functions[kernelMulMatBf16AddF32], launchCount,
		&left, &right, &addend, &result, &inner, &rows,
	)
}

func launchBF16Gate(
	state *device.State,
	functions functionSet,
	output *tensor.Tensor,
	fusion bf16GateFusion,
	pointers launchPointerFrame,
) error {
	gateNode := fusion.gate.Inputs[0]
	inner, err := uint32Checked(gateNode.Shape.Dims[0], "BF16 gate inner dimension")
	if err != nil {
		return err
	}
	rows, err := uint32Checked(gateNode.Shape.Dims[1], "BF16 gate row count")
	if err != nil {
		return err
	}
	launchCount, err := bf16MulMatLaunchCount(rows, 1)
	if err != nil {
		return err
	}
	gate := pointers.input(0)
	up := pointers.input(1)
	right := pointers.input(2)
	result := pointers.output()
	kind := uint32(fusion.kind)
	return launch1DABI(
		state, functions[kernelMulMatBf16GateF32], launchCount,
		&gate, &up, &right, &result, &inner, &rows, &kind,
	)
}

func launchBF16Append(
	state *device.State,
	functions functionSet,
	node *tensor.Tensor,
	fusion bf16AppendFusion,
	pointers launchPointerFrame,
	attributePointer driver.DevicePtr,
) error {
	attributes, ok := node.Attrs.(tensor.CacheAppendAttributes)
	if !ok || attributes.Axis+1 != uint32(node.Shape.Rank) {
		return errors.New("invalid fused cache append attributes")
	}
	output := pointers.output()
	left := pointers.input(0)
	if output != left {
		leftCount, err := elementCount32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		if err := launch1DABI(
			state, functions[kernelCopyF32], leftCount, &left, &output, &leftCount,
		); err != nil {
			return err
		}
	}
	leftNode := fusion.projection.Inputs[0]
	inner, err := uint32Checked(leftNode.Shape.Dims[0], "BF16 append inner dimension")
	if err != nil {
		return err
	}
	rows, err := uint32Checked(leftNode.Shape.Dims[1], "BF16 append row count")
	if err != nil {
		return err
	}
	innerElements := uint64(1)
	for dimension := uint32(0); dimension < attributes.Axis; dimension++ {
		if innerElements > math.MaxUint64/node.Shape.Dims[dimension] {
			return errors.New("fused cache append offset overflows")
		}
		innerElements *= node.Shape.Dims[dimension]
	}
	appendInner, err := uint32Checked(innerElements, "fused cache append inner size")
	if err != nil {
		return err
	}
	offsetPointer := attributePointer
	if offsetPointer == 0 {
		return errors.New("fused cache append offset storage is unavailable")
	}
	launchCount, err := bf16MulMatLaunchCount(rows, 1)
	if err != nil {
		return err
	}
	weight := pointers.input(1)
	right := pointers.input(2)
	return launch1DABI(
		state, functions[kernelMulMatBf16AppendF32], launchCount,
		&weight, &right, &output, &offsetPointer, &inner, &rows, &appendInner,
	)
}

func launchBF16ArgmaxPartials(
	state *device.State,
	functions functionSet,
	projection *tensor.Tensor,
	pointers launchPointerFrame,
) error {
	leftNode := projection.Inputs[0]
	inner, err := uint32Checked(leftNode.Shape.Dims[0], "BF16 argmax inner dimension")
	if err != nil {
		return err
	}
	rows, err := uint32Checked(leftNode.Shape.Dims[1], "BF16 argmax row count")
	if err != nil {
		return err
	}
	partialCount, ok := q8ArgmaxPartialCount(uint64(rows))
	if !ok {
		return errors.New("BF16 argmax partial count exceeds uint32")
	}
	left := pointers.input(0)
	right := pointers.input(1)
	partials := pointers.output()
	const threads = uint32(q8ArgmaxWarpsPerBlock * 32)
	return launchGridABI(
		state, functions[kernelMulMatBf16ArgmaxPartialsF32],
		kernel.Grid1D(int(partialCount)), kernel.Grid1D(int(threads)),
		&left, &right, &partials, &inner, &rows,
	)
}

func launchRopeAppend(
	state *device.State,
	functions functionSet,
	node *tensor.Tensor,
	fusion ropeAppendFusion,
	pointers launchPointerFrame,
	positions driver.DevicePtr,
) error {
	appendAttributes, ok := node.Attrs.(tensor.CacheAppendAttributes)
	if !ok || appendAttributes.Axis+1 != uint32(node.Shape.Rank) {
		return errors.New("invalid fused cache append attributes")
	}
	ropeAttributes, ok := fusion.rope.Attrs.(tensor.RoPEAttributes)
	if !ok {
		return errors.New("invalid fused RoPE attributes")
	}
	output := pointers.output()
	left := pointers.input(0)
	if output != left {
		leftCount, err := elementCount32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		if err := launch1DABI(
			state, functions[kernelCopyF32], leftCount, &left, &output, &leftCount,
		); err != nil {
			return err
		}
	}
	count, err := elementCount32(fusion.rope.Shape)
	if err != nil {
		return err
	}
	width, err := uint32Checked(fusion.rope.Shape.Dims[0], "fused RoPE width")
	if err != nil {
		return err
	}
	heads, err := uint32Checked(fusion.rope.Shape.Dims[1], "fused RoPE heads")
	if err != nil {
		return err
	}
	tokens, err := uint32Checked(fusion.rope.Shape.Dims[2], "fused RoPE tokens")
	if err != nil {
		return err
	}
	innerElements := uint64(1)
	for dimension := uint32(0); dimension < appendAttributes.Axis; dimension++ {
		if innerElements > math.MaxUint64/node.Shape.Dims[dimension] {
			return errors.New("fused cache append offset overflows")
		}
		innerElements *= node.Shape.Dims[dimension]
	}
	inner, err := uint32Checked(innerElements, "fused cache append inner size")
	if err != nil {
		return err
	}
	input := pointers.input(1)
	var frequencyFactors driver.DevicePtr
	if len(fusion.rope.Inputs) == 2 {
		frequencyFactors = pointers.input(2)
	}
	if positions == 0 {
		return errors.New("fused RoPE position storage is unavailable")
	}
	if pointers.attribute == 0 {
		return errors.New("fused cache append offset storage is unavailable")
	}
	rotary := ropeAttributes.RotaryDimensions
	frequencyBase := ropeAttributes.FrequencyBase
	frequencyScale := ropeAttributes.FrequencyScale
	originalContext := ropeAttributes.OriginalContext
	extFactor := ropeAttributes.ExtFactor
	attentionFactor := ropeAttributes.AttentionFactor
	betaFast := ropeAttributes.BetaFast
	betaSlow := ropeAttributes.BetaSlow
	function := functions[kernelRopeAppendNormalF32]
	if fusion.rope.Op == tensor.OpRoPENeoX {
		function = functions[kernelRopeAppendNeoxF32]
	}
	return launch1DABI(
		state, function, count,
		&input, &positions, &frequencyFactors, &output, &pointers.attribute, &inner,
		&width, &heads, &tokens, &rotary,
		&frequencyBase, &frequencyScale, &originalContext, &extFactor, &attentionFactor,
		&betaFast, &betaSlow, &count,
	)
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
	pointers launchPointerFrame,
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
	right := pointers.input(1)
	if q8Input.stagedNode != rightNode {
		blocks := uint64(inner) / q8InputTraits.BlockSize
		if blocks > math.MaxUint32 {
			return errors.New("Q8 argmax input block count exceeds uint32")
		}
		if err := launchQ8InputQuantization(
			state, functions[kernelQuantizeQ80InputF32], right, q8Input.staging, uint32(blocks),
		); err != nil {
			return err
		}
		q8Input.stagedNode = rightNode
	}
	left, partials := pointers.input(0), pointers.output()
	partialCount, ok := q8ArgmaxPartialCount(uint64(rows))
	if !ok {
		return errors.New("Q8 argmax partial count exceeds uint32")
	}
	threads := uint32(q8ArgmaxWarpsPerBlock * q8InputTraits.BlockSize)
	return launchGridABI(
		state, functions[kernelMulMatQ80InputArgmaxPartialsF32],
		kernel.Grid1D(int(partialCount)), kernel.Grid1D(int(threads)),
		&left, &q8Input.staging, &partials, &inner, &rows,
	)
}

func launchQ8ArgmaxReduction(
	state *device.State,
	functions functionSet,
	projection, selection *tensor.Tensor,
	pointers launchPointerFrame,
) error {
	rows, err := uint32Checked(projection.Shape.Dims[0], "Q8 argmax row count")
	if err != nil {
		return err
	}
	partialCount, ok := q8ArgmaxPartialCount(uint64(rows))
	if !ok {
		return errors.New("Q8 argmax partial count exceeds uint32")
	}
	partials, output := pointers.input(0), pointers.output()
	return launchGridABI(
		state, functions[kernelArgmaxQ80InputPartialsF32],
		kernel.Grid1D(1), kernel.DefaultBlock1D(),
		&partials, &output, &partialCount,
	)
}

// q8InputSpanColumns: input columns one warp serves per weight read in
// the q8-input mul_mat kernels — mirrors the kernels' Q8_INPUT_SPAN_MAX
// and bounds the speculation verify batch; wider prefills chunk into
// spans of this width (launchQ8InputSpans).
const q8InputSpanColumns = 8

// q8InputMulMatKernels names the q8-input path per storage type: the
// right operand quantizes once to q8 blocks and the weight dot products
// run integer dp4a, decode and prefill alike. Types without an entry
// keep the float path.
var q8InputMulMatKernels = map[dtype.Type]kernelFunctionID{
	dtype.Q8_0: kernelMulMatQ80InputF32,
	dtype.Q4K:  kernelMulMatQ4KInputF32,
	dtype.Q5K:  kernelMulMatQ5KInputF32,
	dtype.Q6K:  kernelMulMatQ6KInputF32,
}

// q8InputFastPathType reports storage types whose single-vector decode
// routes through the q8-input integer kernels.
func q8InputFastPathType(storage dtype.Type) (ok bool) {
	_, ok = q8InputMulMatKernels[storage]
	return ok
}

// quantMulMatLaunchCount sizes the thread grid for the warp-cooperative
// quantized mul_mat kernels: one warp per output element, with Q8_0's
// kernel additionally tiling four input vectors per warp.
func quantMulMatLaunchCount(storage dtype.Type, leftRows, rightRows uint32) (uint32, error) {
	const (
		dotProductThreads = uint32(32)
		q8VectorsPerWarp  = uint32(4)
	)
	warps := uint64(leftRows) * uint64(rightRows)
	if storage == dtype.Q8_0 {
		rightTiles := (rightRows-1)/q8VectorsPerWarp + 1
		warps = uint64(leftRows) * uint64(rightTiles)
	}
	if warps > uint64(math.MaxUint32/dotProductThreads) {
		return 0, errors.New("quantized mul_mat launch size exceeds uint32")
	}
	return uint32(warps) * dotProductThreads, nil
}

func launchNormalizationABI(
	state *device.State,
	function boundKernel,
	rows uint32,
	arguments ...any,
) error {
	if rows == 0 {
		return nil
	}
	const threads = uint32(256)
	return launchGridABI(
		state, function,
		kernel.Grid1D(int(rows)), kernel.Grid1D(int(threads)),
		arguments...,
	)
}
