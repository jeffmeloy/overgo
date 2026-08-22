package hostmath

import (
	"fmt"
	"math"
)

const channelRMSNormZeroGuard = 1e-12

func ChannelRMSNormZeroGuard() float64 { return channelRMSNormZeroGuard }

// ChannelRMSNormF64Into: channel norm; fixed per-position accumulation order.
func ChannelRMSNormF64Into(out, input, scale []float32, channels, positions int) error {
	if channels <= 0 || positions <= 0 ||
		len(input) != channels*positions || len(out) != len(input) || len(scale) != channels {
		return fmt.Errorf("channel RMS norm: invalid shape")
	}
	normalizer := math.Sqrt(float64(channels))
	for position := range positions {
		var sumSquares float64
		for channel := range channels {
			value := float64(input[channel*positions+position])
			sumSquares += value * value
		}
		norm := max(math.Sqrt(sumSquares), channelRMSNormZeroGuard)
		for channel := range channels {
			index := channel*positions + position
			out[index] = float32(float64(input[index]) / norm * normalizer * float64(scale[channel]))
		}
	}
	return nil
}

// AddResidualF64Into adds an identity residual or a learned channel projection
// to output in place.
func AddResidualF64Into(output, input, projection, bias []float32, inputChannels, outputChannels, positions int) error {
	if inputChannels == outputChannels {
		if len(output) != len(input) {
			return fmt.Errorf("residual: identity storage differs")
		}
		for index := range output {
			output[index] += input[index]
		}
		return nil
	}
	projected := make([]float32, len(output))
	if err := ChannelMixF64Into(projected, input, projection, bias, inputChannels, outputChannels, positions); err != nil {
		return err
	}
	for index := range output {
		output[index] += projected[index]
	}
	return nil
}

// ZeroCenteredRMSNormF64InPlace applies x/sqrt(mean(x²)+eps)*(1+weight)
// independently to row-major rows.
func ZeroCenteredRMSNormF64InPlace(values []float64, weight []float32, rows, width int, epsilon float64) {
	for row := range rows {
		current := values[row*width : (row+1)*width]
		var sumSquares float64
		for _, value := range current {
			sumSquares += value * value
		}
		inverse := 1 / math.Sqrt(sumSquares/float64(width)+epsilon)
		for index := range current {
			current[index] *= inverse * (1 + float64(weight[index]))
		}
	}
}

// StandardRMSNormF64 applies x/sqrt(mean(x²)+eps)*weight independently to
// row-major rows and returns new storage.
func StandardRMSNormF64(values []float64, weight []float32, rows, width int, epsilon float64) []float64 {
	output := make([]float64, len(values))
	for row := range rows {
		source := values[row*width : (row+1)*width]
		destination := output[row*width : (row+1)*width]
		var sumSquares float64
		for _, value := range source {
			sumSquares += value * value
		}
		inverse := 1 / math.Sqrt(sumSquares/float64(width)+epsilon)
		for index := range source {
			destination[index] = source[index] * inverse * float64(weight[index])
		}
	}
	return output
}

// LayerNormF64 applies non-parametric layer normalization independently to
// row-major rows and returns new storage.
func LayerNormF64(values []float64, rows, width int, epsilon float64) []float64 {
	output := make([]float64, len(values))
	for row := range rows {
		source := values[row*width : (row+1)*width]
		destination := output[row*width : (row+1)*width]
		var mean float64
		for _, value := range source {
			mean += value
		}
		mean /= float64(width)
		var variance float64
		for _, value := range source {
			delta := value - mean
			variance += delta * delta
		}
		inverse := 1 / math.Sqrt(variance/float64(width)+epsilon)
		for index, value := range source {
			destination[index] = (value - mean) * inverse
		}
	}
	return output
}

// AdaptiveAffineF64 applies per-feature scale and shift formed by adding a
// conditioning vector to the corresponding halves of a learned table.
func AdaptiveAffineF64(values, conditioning []float64, table []float32, rows, width int) []float64 {
	output := make([]float64, len(values))
	for row := range rows {
		for feature := range width {
			scale := conditioning[feature] + float64(table[feature])
			shift := conditioning[feature] + float64(table[width+feature])
			index := row*width + feature
			output[index] = (1+scale)*values[index] + shift
		}
	}
	return output
}

// ApplyRotaryInterleavedF64 rotates adjacent channel pairs using interleaved
// cosine and sine tables.
func ApplyRotaryInterleavedF64(vector, cosine, sine []float64) {
	for pair := range len(vector) / 2 {
		first, second := 2*pair, 2*pair+1
		x0, x1 := vector[first], vector[second]
		vector[first] = x0*cosine[first] - x1*sine[first]
		vector[second] = x1*cosine[second] + x0*sine[second]
	}
}

// NormalizeHeadsRotaryHalfF64 applies standard per-head RMS normalization and
// half-split rotary embedding to row-major projected heads.
func NormalizeHeadsRotaryHalfF64(values []float64, weight []float32, rows, heads, headWidth int, epsilon float64, cosine, sine []float64) {
	half := headWidth / 2
	for row := range rows {
		cosineRow := cosine[row*headWidth : (row+1)*headWidth]
		sineRow := sine[row*headWidth : (row+1)*headWidth]
		for head := range heads {
			vector := values[(row*heads+head)*headWidth : (row*heads+head+1)*headWidth]
			var sumSquares float64
			for _, value := range vector {
				sumSquares += value * value
			}
			inverse := 1 / math.Sqrt(sumSquares/float64(headWidth)+epsilon)
			for index := range vector {
				vector[index] *= inverse * float64(weight[index])
			}
			for index := range half {
				first, second := vector[index], vector[half+index]
				vector[index] = first*cosineRow[index] - second*sineRow[index]
				vector[half+index] = second*cosineRow[half+index] + first*sineRow[half+index]
			}
		}
	}
}

// LinearFloat64 computes a row-major affine projection with float64 inputs and
// outputs and float32 model weights.
func LinearFloat64(input []float64, weight, bias []float32, rows, inputWidth, outputWidth int) []float64 {
	output := make([]float64, rows*outputWidth)
	ParallelRangeF64(rows, inputWidth*outputWidth, func(startRow, endRow int) {
		for row := startRow; row < endRow; row++ {
			source := input[row*inputWidth : (row+1)*inputWidth]
			destination := output[row*outputWidth : (row+1)*outputWidth]
			for column := range outputWidth {
				var value float64
				if bias != nil {
					value = float64(bias[column])
				}
				weights := weight[column*inputWidth : (column+1)*inputWidth]
				for index := range inputWidth {
					value += source[index] * float64(weights[index])
				}
				destination[column] = value
			}
		}
	})
	return output
}

// LinearFloat64Backward accumulates input, weight, and bias gradients for
// LinearFloat64.
func LinearFloat64Backward(dInput []float64, dWeight, dBias []float32, input []float64, weight []float32, dOutput []float64, rows, inputWidth, outputWidth int) {
	for row := range rows {
		for output := range outputWidth {
			gradient := dOutput[row*outputWidth+output]
			dBias[output] += float32(gradient)
			weightRow := weight[output*inputWidth : (output+1)*inputWidth]
			dWeightRow := dWeight[output*inputWidth : (output+1)*inputWidth]
			for inputIndex := range inputWidth {
				dWeightRow[inputIndex] += float32(gradient * input[row*inputWidth+inputIndex])
				dInput[row*inputWidth+inputIndex] += gradient * float64(weightRow[inputIndex])
			}
		}
	}
}

// AdaptiveAffineBackwardF64 accumulates the table gradient and returns the
// gradient with respect to the normalized input.
func AdaptiveAffineBackwardF64(dInput []float64, dTable []float32, normalized, conditioning []float64, table []float32, dOutput []float64, rows, width int) {
	for row := range rows {
		for feature := range width {
			index := row*width + feature
			gradient := dOutput[index]
			scale := conditioning[feature] + float64(table[feature])
			dInput[index] = gradient * (1 + scale)
			dTable[feature] += float32(gradient * normalized[index])
			dTable[width+feature] += float32(gradient)
		}
	}
}

// ZeroCenteredRMSNormWeightGradientF64 accumulates the learned weight gradient
// for ZeroCenteredRMSNormF64InPlace.
func ZeroCenteredRMSNormWeightGradientF64(dWeight []float32, input, dOutput []float64, rows, width int, epsilon float64) {
	for row := range rows {
		source := input[row*width : (row+1)*width]
		var sumSquares float64
		for _, value := range source {
			sumSquares += value * value
		}
		inverse := 1 / math.Sqrt(sumSquares/float64(width)+epsilon)
		for feature := range width {
			dWeight[feature] += float32(dOutput[row*width+feature] * source[feature] * inverse)
		}
	}
}

// SpatialAttentionF64Into: independent frame-local attention rows.
func SpatialAttentionF64Into(out, qkv []float32, channels, frames, positions int) error {
	volume := channels * frames * positions
	if channels <= 0 || frames <= 0 || positions <= 0 || len(qkv) != 3*volume || len(out) != volume {
		return fmt.Errorf("spatial attention: invalid shape")
	}
	scale := 1 / math.Sqrt(float64(channels))
	keyOffset, valueOffset := volume, 2*volume
	rows := frames * positions
	ParallelRangeF64(rows, positions*channels*2, func(start, end int) {
		scores := make([]float32, positions)
		for row := start; row < end; row++ {
			frame, query := row/positions, row%positions
			for key := range positions {
				var dot float64
				for channel := range channels {
					queryIndex := (channel*frames+frame)*positions + query
					keyIndex := keyOffset + (channel*frames+frame)*positions + key
					dot += float64(qkv[queryIndex]) * float64(qkv[keyIndex])
				}
				scores[key] = float32(dot * scale)
			}
			SoftmaxInPlace(scores)
			for channel := range channels {
				var sum float64
				for key, probability := range scores {
					valueIndex := valueOffset + (channel*frames+frame)*positions + key
					sum += float64(probability) * float64(qkv[valueIndex])
				}
				out[(channel*frames+frame)*positions+query] = float32(sum)
			}
		}
	})
	return nil
}
