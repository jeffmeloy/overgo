//overgo:runtime-inputs caller

// Package cudnn binds the part of NVIDIA's cuDNN library a convolution over a
// five-dimensional tensor needs: a context on a stream, the descriptors of an
// input, a filter and an output, and the forward convolution. The library is
// loaded as cuBLAS is, from the search path or CUDA_PATH
// (internal/cuda/cublas), and a machine without it loads nothing: Open says
// so, and its caller keeps the path that needs no library.
package cudnn

import (
	"errors"
	"fmt"
)

// ErrAbsent is what Open returns on a machine that has no cuDNN to load.
var ErrAbsent = errors.New("cudnn: no cuDNN library is on the search path, under CUDA_PATH or beside overgo's data")

// Files are the files of cuDNN a convolution loads: the one a process asks
// for, and the parts of it that one loads when it first needs them.
var Files = []string{
	"cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
	"cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
}

// DataType is the element type of a tensor or a filter, or the type a
// convolution's sums are kept in, by cuDNN's own values of cudnnDataType_t.
type DataType int32

const (
	// DataF32 is single precision, CUDNN_DATA_FLOAT.
	DataF32 DataType = 0
	// DataF16 is half precision, CUDNN_DATA_HALF.
	DataF16 DataType = 2
)

const (
	// channelsFirst is CUDNN_TENSOR_NCHW: a tensor's or a filter's channels
	// are stored before its extents.
	channelsFirst = 0
	// implicitPrecomputedGEMM is
	// CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, the algorithm the
	// library runs on the device's tensor units for a convolution over three
	// extents.
	implicitPrecomputedGEMM = 1
	// crossCorrelation is CUDNN_CROSS_CORRELATION: the filter is applied as
	// stored, which is what a convolution layer computes.
	crossCorrelation = 1
	// tensorMathAllowConversion is CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION: the
	// library may compute on the device's tensor units, converting single
	// precision operands to half for them.
	tensorMathAllowConversion = 2
	// extents is how many extents a convolution here runs over, and dimensions
	// the dimensions of its tensors: a batch, channels and the extents.
	extents    = 3
	dimensions = extents + 2
)

// Shape is a convolution over frames, rows and columns: one clip of In
// channels and the extents Frames, Height and Width, convolved by Out filters
// of the extents Kernel at the strides Stride, with Pad zeros before and
// after each extent.
type Shape struct {
	In, Out               int
	Frames, Height, Width int
	Kernel, Stride, Pad   [extents]int
}

// The extents of a Shape's Kernel, Stride and Pad, in their order.
const (
	overFrames = iota
	overRows
	overColumns
)

// output is the extents the convolution writes.
func (shape Shape) output() (frames, height, width int) {
	extent := func(in, at int) int { return (in+shape.Pad[at]+shape.Pad[at]-shape.Kernel[at])/shape.Stride[at] + 1 }
	return extent(shape.Frames, overFrames), extent(shape.Height, overRows), extent(shape.Width, overColumns)
}

func (shape Shape) valid() bool {
	positive := shape.In > 0 && shape.Out > 0
	for at := range extents {
		positive = positive && shape.Kernel[at] > 0 && shape.Stride[at] > 0 && shape.Pad[at] >= 0
	}
	if !positive {
		return false
	}
	frames, height, width := shape.output()
	return frames > 0 && height > 0 && width > 0
}

// result is the failure of a cuDNN call that did not succeed.
func result(operation string, status uintptr) error {
	if int32(status) == 0 {
		return nil
	}
	return fmt.Errorf("%s: cuDNN status %d", operation, int32(status))
}
