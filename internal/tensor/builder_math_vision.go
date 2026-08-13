package tensor

import (
	"errors"
	"math"

	"overgo/internal/tensor/dtype"
)

func (b *Builder) Add(left, right *Tensor) *Tensor {
	return b.binary(OpAdd, left, right)
}

func (b *Builder) Multiply(left, right *Tensor) *Tensor {
	return b.binary(OpMultiply, left, right)
}

func (b *Builder) Divide(left, right *Tensor) *Tensor {
	return b.binary(OpDivide, left, right)
}

func (b *Builder) Scale(input *Tensor, value float32) *Tensor {
	return b.unary(OpScale, input, ScaleAttributes{Value: value})
}

func (b *Builder) Clamp(input *Tensor, minimum, maximum float32) *Tensor {
	if math.IsNaN(float64(minimum)) || math.IsNaN(float64(maximum)) ||
		math.IsInf(float64(minimum), 0) || math.IsInf(float64(maximum), 0) || minimum > maximum {
		b.setError(errors.New("clamp bounds are invalid"))
		return nil
	}
	return b.unary(OpClamp, input, ClampAttributes{Minimum: minimum, Maximum: maximum})
}

func (b *Builder) BF16Round(input *Tensor) *Tensor {
	return b.unary(OpBF16Round, input, nil)
}

func (b *Builder) RMSNorm(input *Tensor, epsilon float32) *Tensor {
	if epsilon <= 0 {
		b.setError(errors.New("RMSNorm epsilon must be positive"))
		return nil
	}
	return b.unary(OpRMSNorm, input, RMSNormAttributes{Epsilon: epsilon})
}

// WeightedRMSNorm: applies RMSNorm and learned per-channel weight
func (b *Builder) WeightedRMSNorm(input, weight *Tensor, epsilon float32) *Tensor {
	return b.Multiply(b.RMSNorm(input, epsilon), weight)
}

func (b *Builder) LayerNorm(input *Tensor, epsilon float32) *Tensor {
	if epsilon <= 0 {
		b.setError(errors.New("LayerNorm epsilon must be positive"))
		return nil
	}
	return b.unary(OpLayerNorm, input, LayerNormAttributes{Epsilon: epsilon})
}

// AffineLayerNorm: applies LayerNorm and learned per-channel weight and bias
func (b *Builder) AffineLayerNorm(input, weight, bias *Tensor, epsilon float32) *Tensor {
	return b.Add(b.Multiply(b.LayerNorm(input, epsilon), weight), bias)
}

func (b *Builder) Softmax(input *Tensor) *Tensor {
	return b.unary(OpSoftmax, input, nil)
}

func (b *Builder) SiLU(input *Tensor) *Tensor {
	return b.unary(OpSiLU, input, nil)
}

func (b *Builder) GELU(input *Tensor) *Tensor {
	return b.unary(OpGELU, input, nil)
}

func (b *Builder) GELUErf(input *Tensor) *Tensor {
	return b.unary(OpGELUErf, input, nil)
}

func (b *Builder) ReLU(input *Tensor) *Tensor {
	return b.unary(OpReLU, input, nil)
}

// Conv1DSame: odd-kernel stride-one convolution.
func (b *Builder) Conv1DSame(input, weight, bias *Tensor, depthwise bool) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || weight == nil || bias == nil {
		b.setError(errors.New("same Conv1D input is nil"))
		return nil
	}
	if input.Type != weight.Type || input.Type != bias.Type ||
		input.Shape.Rank != 2 || weight.Shape.Rank != 3 ||
		weight.Shape.Dims[0] == 0 || weight.Shape.Dims[0]%2 == 0 {
		b.setError(errors.New("same Conv1D tensor shape is invalid"))
		return nil
	}
	inputChannels := input.Shape.Dims[0]
	outputChannels := weight.Shape.Dims[2]
	if (depthwise && (weight.Shape.Dims[1] != 1 || outputChannels != inputChannels)) ||
		(!depthwise && weight.Shape.Dims[1] != inputChannels) {
		b.setError(errors.New("same Conv1D channel shape is incompatible"))
		return nil
	}
	biasOK := bias.Shape.Rank == 1 && bias.Shape.Dims[0] == outputChannels
	biasOK = biasOK || bias.Shape.Rank == 2 && bias.Shape.Dims[0] == 1 && bias.Shape.Dims[1] == outputChannels
	if !biasOK {
		b.setError(errors.New("same Conv1D bias shape is incompatible"))
		return nil
	}
	shape, err := NewShape(outputChannels, input.Shape.Dims[1])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", input.Type, shape, OpConv1DSame, []*Tensor{input, weight, bias}, Conv1DAttributes{Depthwise: depthwise})
}

// Conv2D: channel-first spatial convolution.
func (b *Builder) Conv2D(
	input, weight, bias *Tensor,
	strideX, strideY, padLeft, padRight, padTop, padBottom uint32,
	depthwise bool,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || weight == nil || strideX == 0 || strideY == 0 ||
		input.Shape.Rank != 3 || weight.Shape.Rank != 4 || input.Type != weight.Type {
		b.setError(errors.New("Conv2D tensor shape is invalid"))
		return nil
	}
	channelsIn, width, height := input.Shape.Dims[0], input.Shape.Dims[1], input.Shape.Dims[2]
	kernelW, kernelH := weight.Shape.Dims[0], weight.Shape.Dims[1]
	weightChannels, channelsOut := weight.Shape.Dims[2], weight.Shape.Dims[3]
	if kernelW == 0 || kernelH == 0 ||
		(depthwise && (weightChannels != 1 || channelsOut != channelsIn)) ||
		(!depthwise && weightChannels != channelsIn) {
		b.setError(errors.New("Conv2D channel shape is incompatible"))
		return nil
	}
	paddedW, paddedH := width+uint64(padLeft)+uint64(padRight), height+uint64(padTop)+uint64(padBottom)
	if paddedW < kernelW || paddedH < kernelH {
		b.setError(errors.New("Conv2D kernel exceeds padded input"))
		return nil
	}
	outputW := (paddedW-kernelW)/uint64(strideX) + 1
	outputH := (paddedH-kernelH)/uint64(strideY) + 1
	inputs := []*Tensor{input, weight}
	hasBias := bias != nil
	if hasBias {
		trailingSingleton := true
		for dimension := 1; dimension < int(bias.Shape.Rank); dimension++ {
			trailingSingleton = trailingSingleton && bias.Shape.Dims[dimension] == 1
		}
		if bias.Type != input.Type || bias.Shape.Dims[0] != channelsOut || !trailingSingleton ||
			(bias.Shape.Rank != 1 && bias.Shape.Rank != 2 && bias.Shape.Rank != 3) {
			b.setError(errors.New("Conv2D bias shape is incompatible"))
			return nil
		}
		inputs = append(inputs, bias)
	}
	shape, err := NewShape(channelsOut, outputW, outputH)
	if err != nil {
		b.setError(err)
		return nil
	}
	attributes := Conv2DAttributes{
		StrideX: strideX, StrideY: strideY, PadLeft: padLeft, PadRight: padRight,
		PadTop: padTop, PadBottom: padBottom, Depthwise: depthwise, HasBias: hasBias,
	}
	return b.add("", input.Type, shape, OpConv2D, inputs, attributes)
}

// WindowPartition2D: padded spatial windows.
func (b *Builder) WindowPartition2D(input *Tensor, window uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Shape.Rank != 3 || window == 0 ||
		input.Shape.Dims[1] > math.MaxUint32 || input.Shape.Dims[2] > math.MaxUint32 {
		b.setError(errors.New("WindowPartition2D input is invalid"))
		return nil
	}
	width, height := uint32(input.Shape.Dims[1]), uint32(input.Shape.Dims[2])
	windowsX, windowsY := (width+window-1)/window, (height+window-1)/window
	shape, err := NewShape(input.Shape.Dims[0], uint64(window)*uint64(window), uint64(windowsX)*uint64(windowsY))
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", input.Type, shape, OpWindowPartition2D, []*Tensor{input}, Window2DAttributes{
		Width: width, Height: height, Window: window,
	})
}

// WindowUnpartition2D: cropped spatial reconstruction.
func (b *Builder) WindowUnpartition2D(input *Tensor, width, height uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Shape.Rank != 3 || width == 0 || height == 0 {
		b.setError(errors.New("WindowUnpartition2D input is invalid"))
		return nil
	}
	window := uint32(math.Sqrt(float64(input.Shape.Dims[1])))
	windowsX, windowsY := (width+window-1)/window, (height+window-1)/window
	if window == 0 || uint64(window)*uint64(window) != input.Shape.Dims[1] ||
		uint64(windowsX)*uint64(windowsY) != input.Shape.Dims[2] {
		b.setError(errors.New("WindowUnpartition2D shape is incompatible"))
		return nil
	}
	shape, err := NewShape(input.Shape.Dims[0], uint64(width), uint64(height))
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", input.Type, shape, OpWindowUnpartition2D, []*Tensor{input}, Window2DAttributes{
		Width: width, Height: height, Window: window,
	})
}

// SAMAttention: batched decomposed-relative 2D attention.
func (b *Builder) SAMAttention(query, key, value, relativeW, relativeH *Tensor, scale, relativeScale float32, spatialSize uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if query == nil || key == nil || value == nil || relativeW == nil || relativeH == nil ||
		query.Shape.Rank != 4 || key.Shape.Rank != 4 || value.Shape.Rank != 4 ||
		relativeW.Shape.Rank != 2 || relativeH.Shape.Rank != 2 || spatialSize == 0 ||
		query.Type != key.Type || query.Type != value.Type || query.Type != relativeW.Type || query.Type != relativeH.Type ||
		query.Shape.Dims[0] != key.Shape.Dims[0] || key.Shape.Dims[1] != value.Shape.Dims[1] ||
		query.Shape.Dims[2] != key.Shape.Dims[2] || key.Shape.Dims[2] != value.Shape.Dims[2] ||
		query.Shape.Dims[3] != key.Shape.Dims[3] || key.Shape.Dims[3] != value.Shape.Dims[3] ||
		query.Shape.Dims[1]%key.Shape.Dims[1] != 0 ||
		query.Shape.Dims[2] != uint64(spatialSize)*uint64(spatialSize) ||
		relativeW.Shape.Dims[0] != query.Shape.Dims[0] || relativeH.Shape.Dims[0] != query.Shape.Dims[0] ||
		relativeW.Shape.Dims[1] == 0 || relativeH.Shape.Dims[1] == 0 ||
		scale <= 0 || relativeScale <= 0 || math.IsNaN(float64(scale)) || math.IsNaN(float64(relativeScale)) {
		b.setError(errors.New("SAMAttention shape or attributes are invalid"))
		return nil
	}
	shape, err := NewShape(value.Shape.Dims[0], query.Shape.Dims[1], query.Shape.Dims[2], query.Shape.Dims[3])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", query.Type, shape, OpSAMAttention, []*Tensor{query, key, value, relativeW, relativeH}, SAMAttentionAttributes{
		Scale: scale, RelativeScale: relativeScale, SpatialSize: spatialSize,
	})
}

// GroupNorm: channel groups across one sequence.
func (b *Builder) GroupNorm(input, weight, bias *Tensor, groups uint32, epsilon float32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || weight == nil || bias == nil {
		b.setError(errors.New("group norm input is nil"))
		return nil
	}
	if epsilon <= 0 || groups == 0 || input.Shape.Rank != 2 ||
		input.Shape.Dims[0]%uint64(groups) != 0 ||
		input.Type != weight.Type || input.Type != bias.Type {
		b.setError(errors.New("group norm configuration is invalid"))
		return nil
	}
	channels := input.Shape.Dims[0]
	weightOK := weight.Shape.Rank == 1 && weight.Shape.Dims[0] == channels
	weightOK = weightOK || weight.Shape.Rank == 2 && weight.Shape.Dims[0] == 1 && weight.Shape.Dims[1] == channels
	biasOK := bias.Shape.Rank == 1 && bias.Shape.Dims[0] == channels
	biasOK = biasOK || bias.Shape.Rank == 2 && bias.Shape.Dims[0] == 1 && bias.Shape.Dims[1] == channels
	if !weightOK || !biasOK {
		b.setError(errors.New("group norm affine shape is incompatible"))
		return nil
	}
	return b.add("", input.Type, input.Shape, OpGroupNorm, []*Tensor{input, weight, bias}, GroupNormAttributes{Groups: groups, Epsilon: epsilon})
}

// HyperConnectionInit: replicate embedding across hyper streams.
func (b *Builder) HyperConnectionInit(input *Tensor, hyperConnections uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank != 2 || hyperConnections == 0 {
		b.setError(errors.New("hyper-connection init input is invalid"))
		return nil
	}
	shape, err := NewShape(input.Shape.Dims[0], uint64(hyperConnections), input.Shape.Dims[1])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpHyperConnectionInit, []*Tensor{input}, HyperConnectionAttributes{HyperConnections: hyperConnections})
}

// HyperConnectionPre: hyper-stream branch input.
func (b *Builder) HyperConnectionPre(
	input, fn, scale, base *Tensor,
	hyperConnections, sinkhornIterations uint32,
	normEpsilon, epsilon float32,
) *Tensor {
	if !b.validateHyperConnection(input, fn, scale, base, hyperConnections, sinkhornIterations, normEpsilon, epsilon, false) {
		return nil
	}
	shape, _ := NewShape(input.Shape.Dims[0], input.Shape.Dims[2])
	return b.add("", dtype.F32, shape, OpHyperConnectionPre, []*Tensor{input, fn, scale, base}, HyperConnectionAttributes{
		HyperConnections: hyperConnections, SinkhornIterations: sinkhornIterations, NormEpsilon: normEpsilon, Epsilon: epsilon,
	})
}

// HyperConnectionPost: branch merge into hyper streams.
func (b *Builder) HyperConnectionPost(
	branch, residual, fn, scale, base *Tensor,
	hyperConnections, sinkhornIterations uint32,
	normEpsilon, epsilon float32,
) *Tensor {
	if !b.validateHyperConnection(residual, fn, scale, base, hyperConnections, sinkhornIterations, normEpsilon, epsilon, false) {
		return nil
	}
	if branch == nil || branch.Type != dtype.F32 || branch.Shape.Rank != 2 ||
		branch.Shape.Dims[0] != residual.Shape.Dims[0] || branch.Shape.Dims[1] != residual.Shape.Dims[2] {
		b.setError(errors.New("hyper-connection post branch shape is invalid"))
		return nil
	}
	return b.add("", dtype.F32, residual.Shape, OpHyperConnectionPost, []*Tensor{branch, residual, fn, scale, base}, HyperConnectionAttributes{
		HyperConnections: hyperConnections, SinkhornIterations: sinkhornIterations, NormEpsilon: normEpsilon, Epsilon: epsilon,
	})
}

// HyperConnectionHead: collapse final hyper streams.
func (b *Builder) HyperConnectionHead(input, fn, scale, base *Tensor, hyperConnections uint32, normEpsilon, epsilon float32) *Tensor {
	if !b.validateHyperConnection(input, fn, scale, base, hyperConnections, 1, normEpsilon, epsilon, true) {
		return nil
	}
	shape, _ := NewShape(input.Shape.Dims[0], input.Shape.Dims[2])
	return b.add("", dtype.F32, shape, OpHyperConnectionHead, []*Tensor{input, fn, scale, base}, HyperConnectionAttributes{
		HyperConnections: hyperConnections, SinkhornIterations: 1, NormEpsilon: normEpsilon, Epsilon: epsilon,
	})
}

func (b *Builder) validateHyperConnection(
	input, fn, scale, base *Tensor,
	hyperConnections, sinkhornIterations uint32,
	normEpsilon, epsilon float32,
	head bool,
) bool {
	if b.err != nil {
		return false
	}
	if input == nil || fn == nil || scale == nil || base == nil ||
		input.Type != dtype.F32 || fn.Type != dtype.F32 || scale.Type != dtype.F32 || base.Type != dtype.F32 ||
		input.Shape.Rank != 3 || fn.Shape.Rank != 2 || scale.Shape.Rank != 1 || base.Shape.Rank != 1 ||
		hyperConnections == 0 || sinkhornIterations == 0 || normEpsilon <= 0 || epsilon <= 0 ||
		math.IsNaN(float64(normEpsilon)) || math.IsInf(float64(normEpsilon), 0) ||
		math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) ||
		input.Shape.Dims[1] != uint64(hyperConnections) || fn.Shape.Dims[0] != input.Shape.Dims[0]*uint64(hyperConnections) {
		b.setError(errors.New("hyper-connection metadata or input shape is invalid"))
		return false
	}
	mix := uint64(hyperConnections)
	scaleWidth := uint64(1)
	if !head {
		mix *= uint64(2 + hyperConnections)
		scaleWidth = 3
	}
	if fn.Shape.Dims[1] != mix || scale.Shape.Dims[0] != scaleWidth || base.Shape.Dims[0] != mix {
		b.setError(errors.New("hyper-connection weight shape is invalid"))
		return false
	}
	return true
}

func (b *Builder) XIELU(input *Tensor, alphaN, alphaP, beta, epsilon float32) *Tensor {
	parameters := []float32{alphaN, alphaP, beta, epsilon}
	for _, parameter := range parameters {
		if math.IsNaN(float64(parameter)) || math.IsInf(float64(parameter), 0) {
			b.setError(errors.New("xIELU parameters must be finite"))
			return nil
		}
	}
	return b.unary(OpXIELU, input, XIELUAttributes{
		AlphaN: alphaN, AlphaP: alphaP, Beta: beta, Epsilon: epsilon,
	})
}

func (b *Builder) ReLUSquared(input *Tensor) *Tensor {
	return b.unary(OpReLUSquared, input, nil)
}

func (b *Builder) Sigmoid(input *Tensor) *Tensor {
	return b.unary(OpSigmoid, input, nil)
}

func (b *Builder) Softplus(input *Tensor) *Tensor {
	return b.unary(OpSoftplus, input, nil)
}

func (b *Builder) Tanh(input *Tensor) *Tensor {
	return b.unary(OpTanh, input, nil)
}

func (b *Builder) Exp(input *Tensor) *Tensor {
	return b.unary(OpExp, input, nil)
}

func (b *Builder) L2Norm(input *Tensor, epsilon float32) *Tensor {
	if epsilon < 0 || math.IsNaN(float64(epsilon)) {
		b.setError(errors.New("L2Norm epsilon must be non-negative"))
		return nil
	}
	return b.unary(OpL2Norm, input, L2NormAttributes{Epsilon: epsilon})
}
