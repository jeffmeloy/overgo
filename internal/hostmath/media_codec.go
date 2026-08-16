package hostmath

import (
	"fmt"
	"math"
)

// ChannelRMSNormF64Into: channel norm; fixed per-position accumulation order.
func ChannelRMSNormF64Into(out, input, scale []float32, channels, positions int, zeroGuard float64) error {
	if channels <= 0 || positions <= 0 || zeroGuard <= 0 ||
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
		norm := max(math.Sqrt(sumSquares), zeroGuard)
		for channel := range channels {
			index := channel*positions + position
			out[index] = float32(float64(input[index]) / norm * normalizer * float64(scale[channel]))
		}
	}
	return nil
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
