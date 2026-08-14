package reference

import (
	"errors"
	"math"
	"slices"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

func loraMerge(shape tensor.Shape, inputs []Value, attributes tensor.LoRAMergeAttributes) (Value, error) {
	if len(inputs) != 3 || math.IsNaN(float64(attributes.Scale)) || math.IsInf(float64(attributes.Scale), 0) {
		return Value{}, errors.New("LoRA merge input count is invalid")
	}
	base := inputs[0]
	output := slices.Clone(base.Data)
	k, m, groups := int(shape.Dims[0]), int(shape.Dims[1]), 1
	if shape.Rank == 3 {
		groups = int(shape.Dims[2])
	}
	a, b := inputs[1], inputs[2]
	rank := int(a.Shape.Dims[1])
	if k <= 0 || m <= 0 || rank <= 0 || len(a.Data) != k*rank*groups || len(b.Data) != rank*m*groups {
		return Value{}, errors.New("LoRA merge pair shape is invalid")
	}
	for group := 0; group < groups; group++ {
		for row := 0; row < m; row++ {
			for column := 0; column < k; column++ {
				var delta float64
				for inner := 0; inner < rank; inner++ {
					aIndex := (group*rank+inner)*k + column
					bIndex := (group*m+row)*rank + inner
					delta += float64(a.Data[aIndex]) * float64(b.Data[bIndex])
				}
				output[(group*m+row)*k+column] += attributes.Scale * float32(delta)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func conv1DSame(shape tensor.Shape, input, weight, bias Value, depthwise bool) (Value, error) {
	channelsIn := int(input.Shape.Dims[0])
	tokens := int(input.Shape.Dims[1])
	kernel := int(weight.Shape.Dims[0])
	channelsOut := int(weight.Shape.Dims[2])
	output := make([]float32, channelsOut*tokens)
	padding := kernel / 2
	for token := 0; token < tokens; token++ {
		for channelOut := 0; channelOut < channelsOut; channelOut++ {
			sum := float64(bias.Data[channelOut])
			for tap := 0; tap < kernel; tap++ {
				sourceToken := token + tap - padding
				if sourceToken < 0 || sourceToken >= tokens {
					continue
				}
				if depthwise {
					sum += float64(input.Data[sourceToken*channelsIn+channelOut]) *
						float64(weight.Data[channelOut*kernel+tap])
					continue
				}
				for channelIn := 0; channelIn < channelsIn; channelIn++ {
					weightOffset := (channelOut*channelsIn+channelIn)*kernel + tap
					sum += float64(input.Data[sourceToken*channelsIn+channelIn]) * float64(weight.Data[weightOffset])
				}
			}
			output[token*channelsOut+channelOut] = float32(sum)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func conv2D(shape tensor.Shape, input, weight, bias Value, attributes tensor.Conv2DAttributes) (Value, error) {
	channelsIn, inputW, inputH := int(input.Shape.Dims[0]), int(input.Shape.Dims[1]), int(input.Shape.Dims[2])
	kernelW, kernelH := int(weight.Shape.Dims[0]), int(weight.Shape.Dims[1])
	weightChannels, channelsOut := int(weight.Shape.Dims[2]), int(weight.Shape.Dims[3])
	outputW, outputH := int(shape.Dims[1]), int(shape.Dims[2])
	output := make([]float32, channelsOut*outputW*outputH)
	for y := 0; y < outputH; y++ {
		for x := 0; x < outputW; x++ {
			for channelOut := 0; channelOut < channelsOut; channelOut++ {
				var sum float64
				if attributes.HasBias {
					sum = float64(bias.Data[channelOut])
				}
				for ky := 0; ky < kernelH; ky++ {
					sourceY := y*int(attributes.StrideY) + ky - int(attributes.PadTop)
					if sourceY < 0 || sourceY >= inputH {
						continue
					}
					for kx := 0; kx < kernelW; kx++ {
						sourceX := x*int(attributes.StrideX) + kx - int(attributes.PadLeft)
						if sourceX < 0 || sourceX >= inputW {
							continue
						}
						if attributes.Depthwise {
							inputOffset := channelOut + channelsIn*(sourceX+inputW*sourceY)
							weightOffset := kx + kernelW*(ky+kernelH*channelOut)
							sum += float64(input.Data[inputOffset]) * float64(weight.Data[weightOffset])
							continue
						}
						for channelIn := 0; channelIn < channelsIn; channelIn++ {
							inputOffset := channelIn + channelsIn*(sourceX+inputW*sourceY)
							weightOffset := kx + kernelW*(ky+kernelH*(channelIn+weightChannels*channelOut))
							sum += float64(input.Data[inputOffset]) * float64(weight.Data[weightOffset])
						}
					}
				}
				output[channelOut+channelsOut*(x+outputW*y)] = float32(sum)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func windowPartition2D(shape tensor.Shape, input Value, attributes tensor.Window2DAttributes, reverse bool) (Value, error) {
	channels := int(shape.Dims[0])
	width, height, window := int(attributes.Width), int(attributes.Height), int(attributes.Window)
	if channels <= 0 || width <= 0 || height <= 0 || window <= 0 {
		return Value{}, errors.New("invalid window partition dimensions")
	}
	windowsX := (width + window - 1) / window
	output := make([]float32, mustElements(shape))
	if !reverse {
		for windowY := 0; windowY < (height+window-1)/window; windowY++ {
			for windowX := 0; windowX < windowsX; windowX++ {
				batch := windowX + windowsX*windowY
				for localY := 0; localY < window; localY++ {
					for localX := 0; localX < window; localX++ {
						x, y := windowX*window+localX, windowY*window+localY
						if x >= width || y >= height {
							continue
						}
						for channel := 0; channel < channels; channel++ {
							output[channel+channels*(localX+window*(localY+window*batch))] =
								input.Data[channel+channels*(x+width*y)]
						}
					}
				}
			}
		}
		return Value{Shape: shape, Data: output}, nil
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			windowX, windowY := x/window, y/window
			localX, localY := x%window, y%window
			batch := windowX + windowsX*windowY
			for channel := 0; channel < channels; channel++ {
				output[channel+channels*(x+width*y)] =
					input.Data[channel+channels*(localX+window*(localY+window*batch))]
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func mustElements(shape tensor.Shape) int {
	elements, err := shape.Elements()
	if err != nil || elements > uint64(^uint(0)>>1) {
		panic("invalid reference tensor shape")
	}
	return int(elements)
}

func samRelativeValue(table Value, channel, targetIndex, targetLength int) float64 {
	sourceLength := int(table.Shape.Dims[1])
	if sourceLength == targetLength {
		return float64(table.Data[channel+int(table.Shape.Dims[0])*targetIndex])
	}
	coordinate := (float64(targetIndex)+0.5)*float64(sourceLength)/float64(targetLength) - 0.5
	low := max(0, min(sourceLength-1, int(math.Floor(coordinate))))
	high := max(0, min(sourceLength-1, low+1))
	weight := max(0.0, min(1.0, coordinate-float64(low)))
	width := int(table.Shape.Dims[0])
	return float64(table.Data[channel+width*low])*(1-weight) + float64(table.Data[channel+width*high])*weight
}

func samAttention(shape tensor.Shape, inputs []Value, attributes tensor.SAMAttentionAttributes) (Value, error) {
	if len(inputs) != 5 {
		return Value{}, errors.New("SAMAttention requires five inputs")
	}
	query, key, value, relativeW, relativeH := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	keyWidth, valueWidth := int(query.Shape.Dims[0]), int(value.Shape.Dims[0])
	queryHeads, keyHeads := int(query.Shape.Dims[1]), int(key.Shape.Dims[1])
	tokens, batches, spatial := int(query.Shape.Dims[2]), int(query.Shape.Dims[3]), int(attributes.SpatialSize)
	if keyWidth <= 0 || valueWidth <= 0 || queryHeads <= 0 || keyHeads <= 0 || tokens != spatial*spatial || batches <= 0 {
		return Value{}, errors.New("invalid SAMAttention dimensions")
	}
	group := queryHeads / keyHeads
	output := make([]float32, mustElements(shape))
	scores := make([]float64, tokens)
	for batch := 0; batch < batches; batch++ {
		for queryToken := 0; queryToken < tokens; queryToken++ {
			queryX, queryY := queryToken%spatial, queryToken/spatial
			for head := 0; head < queryHeads; head++ {
				keyHead := head / group
				queryBase := keyWidth * (head + queryHeads*(queryToken+tokens*batch))
				maximum := math.Inf(-1)
				for keyToken := 0; keyToken < tokens; keyToken++ {
					keyX, keyY := keyToken%spatial, keyToken/spatial
					keyBase := keyWidth * (keyHead + keyHeads*(keyToken+tokens*batch))
					var score float64
					for channel := 0; channel < keyWidth; channel++ {
						q := float64(query.Data[queryBase+channel])
						score += q*float64(key.Data[keyBase+channel])*float64(attributes.Scale) +
							q*samRelativeValue(relativeW, channel, queryX-keyX+spatial-1, 2*spatial-1)*float64(attributes.RelativeScale) +
							q*samRelativeValue(relativeH, channel, queryY-keyY+spatial-1, 2*spatial-1)*float64(attributes.RelativeScale)
					}
					scores[keyToken] = score
					maximum = math.Max(maximum, score)
				}
				var sum float64
				for keyToken := range tokens {
					scores[keyToken] = math.Exp(scores[keyToken] - maximum)
					sum += scores[keyToken]
				}
				for channel := 0; channel < valueWidth; channel++ {
					var result float64
					for keyToken := range tokens {
						valueBase := valueWidth * (keyHead + keyHeads*(keyToken+tokens*batch))
						result += scores[keyToken] * float64(value.Data[valueBase+channel])
					}
					output[channel+valueWidth*(head+queryHeads*(queryToken+tokens*batch))] = float32(result / sum)
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func groupNorm(shape tensor.Shape, input, weight, bias Value, groups uint32, epsilon float32) (Value, error) {
	channels := int(input.Shape.Dims[0])
	tokens := int(input.Shape.Dims[1])
	channelsPerGroup := channels / int(groups)
	valuesPerGroup := channelsPerGroup * tokens
	output := make([]float32, len(input.Data))
	for group := 0; group < int(groups); group++ {
		firstChannel := group * channelsPerGroup
		var mean float64
		for token := 0; token < tokens; token++ {
			for channel := firstChannel; channel < firstChannel+channelsPerGroup; channel++ {
				mean += float64(input.Data[token*channels+channel])
			}
		}
		mean /= float64(valuesPerGroup)
		var variance float64
		for token := 0; token < tokens; token++ {
			for channel := firstChannel; channel < firstChannel+channelsPerGroup; channel++ {
				delta := float64(input.Data[token*channels+channel]) - mean
				variance += delta * delta
			}
		}
		inverse := 1 / math.Sqrt(variance/float64(valuesPerGroup)+float64(epsilon))
		for token := 0; token < tokens; token++ {
			for channel := firstChannel; channel < firstChannel+channelsPerGroup; channel++ {
				offset := token*channels + channel
				output[offset] = float32((float64(input.Data[offset])-mean)*inverse)*weight.Data[channel] + bias.Data[channel]
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func float16Round(value float32) float32 {
	bits := math.Float32bits(value)
	sign := uint16(bits >> 16 & 0x8000)
	exponent := int32(bits>>23&0xff) - 127 + 15
	mantissa := bits & 0x7fffff
	var half uint16
	switch {
	case exponent <= 0:
		if exponent < -10 {
			half = sign
			break
		}
		mantissa |= 0x800000
		shift := uint32(14 - exponent)
		rounded := mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)
		half = sign | uint16(rounded>>shift)
	case exponent >= 31:
		if mantissa == 0 {
			half = sign | 0x7c00
		} else {
			half = sign | 0x7e00
		}
	default:
		rounded := mantissa + 0xfff + ((mantissa >> 13) & 1)
		if rounded&0x800000 != 0 {
			rounded = 0
			exponent++
		}
		if exponent >= 31 {
			half = sign | 0x7c00
		} else {
			half = sign | uint16(exponent<<10) | uint16(rounded>>13)
		}
	}
	sign32 := uint32(half&0x8000) << 16
	exp16 := uint32(half>>10) & 0x1f
	frac16 := uint32(half & 0x03ff)
	if exp16 == 0 {
		if frac16 == 0 {
			return math.Float32frombits(sign32)
		}
		exp := int32(-14)
		for frac16&0x0400 == 0 {
			frac16 <<= 1
			exp--
		}
		frac16 &= 0x03ff
		return math.Float32frombits(sign32 | uint32(exp+127)<<23 | frac16<<13)
	}
	if exp16 == 31 {
		return math.Float32frombits(sign32 | 0x7f800000 | frac16<<13)
	}
	return math.Float32frombits(sign32 | (exp16-15+127)<<23 | frac16<<13)
}

func elementwiseBroadcast(
	shape tensor.Shape,
	left, right Value,
	operation func(float32, float32) float32,
) (Value, error) {
	elements, err := shape.Elements()
	if err != nil {
		return Value{}, err
	}
	if elements > uint64(math.MaxInt) {
		return Value{}, errors.New("elementwise output is too large")
	}
	output := make([]float32, int(elements))
	for i := range output {
		leftIndex := broadcastIndex(uint64(i), shape, left.Shape)
		rightIndex := broadcastIndex(uint64(i), shape, right.Shape)
		if leftIndex >= uint64(len(left.Data)) || rightIndex >= uint64(len(right.Data)) {
			return Value{}, errors.New("elementwise broadcast index is out of range")
		}
		output[i] = operation(left.Data[leftIndex], right.Data[rightIndex])
	}
	return Value{Shape: shape, Data: output}, nil
}

func broadcastIndex(index uint64, outputShape, inputShape tensor.Shape) uint64 {
	var result uint64
	var stride uint64 = 1
	for axis := range tensor.MaxDimensions {
		coordinate := index % outputShape.Dims[axis]
		index /= outputShape.Dims[axis]
		if inputShape.Dims[axis] != 1 {
			result += coordinate * stride
		}
		stride *= inputShape.Dims[axis]
	}
	return result
}

func rmsNorm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid RMSNorm row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		var sumSquares float64
		for _, value := range input.Data[row : row+width] {
			sumSquares += float64(value) * float64(value)
		}
		inverse := float32(1 / math.Sqrt(sumSquares/float64(width)+float64(epsilon)))
		for column, value := range input.Data[row : row+width] {
			output[row+column] = value * inverse
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func madNorm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid MADNorm row width")
	}
	output, err := hostmath.MADNorm(input.Data, len(input.Data)/width, width, float64(epsilon))
	return Value{Shape: shape, Data: output}, err
}

func layerNorm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid LayerNorm row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		var sum float64
		for _, value := range input.Data[row : row+width] {
			sum += float64(value)
		}
		mean := sum / float64(width)
		var sumSquares float64
		for _, value := range input.Data[row : row+width] {
			centered := float64(value) - mean
			sumSquares += centered * centered
		}
		inverse := float32(1 / math.Sqrt(sumSquares/float64(width)+float64(epsilon)))
		for column, value := range input.Data[row : row+width] {
			output[row+column] = (value - float32(mean)) * inverse
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func l2Norm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid L2Norm row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		var sumSquares float64
		for _, value := range input.Data[row : row+width] {
			sumSquares += float64(value) * float64(value)
		}
		denominator := math.Max(math.Sqrt(sumSquares), float64(epsilon))
		inverse := float32(1 / denominator)
		for column, value := range input.Data[row : row+width] {
			output[row+column] = value * inverse
		}
	}
	return Value{Shape: shape, Data: output}, nil
}
