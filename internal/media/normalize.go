package media

import (
	"math"

	"overgo/internal/tensor"
)

func SignedUnitBounds32() (float32, float32) {
	upper := float32(tensor.SingletonExtent)
	return -upper, upper
}

// ClampNormalizedF32InPlace bounds normalized samples to their signed unit
// interval without assigning an image-, video-, or audio-family policy.
func ClampNormalizedF32InPlace(values []float32) {
	lower, upper := SignedUnitBounds32()
	for index, value := range values {
		values[index] = max(lower, min(value, upper))
	}
}

// NormalizedF32ToU8 maps signed-unit samples to the full unsigned byte range.
func NormalizedF32ToU8(values []float32) []uint8 {
	output := make([]uint8, len(values))
	for index, value := range values {
		output[index] = SignedUnitF32ToU8(value)
	}
	return output
}

func SignedUnitF32ToU8(value float32) uint8 {
	lower := float64(tensor.FirstOffset)
	upper := float64(tensor.SingletonExtent)
	centerScale := upper / float64(tensor.PairedExtent)
	normalized := (float64(value) + upper) * centerScale
	return unitFloat64ToU8(max(lower, min(normalized, upper)))
}

func UnitF32ToU8(value float32) uint8 {
	lower := float64(tensor.FirstOffset)
	upper := float64(tensor.SingletonExtent)
	return unitFloat64ToU8(max(lower, min(float64(value), upper)))
}

func unitFloat64ToU8(value float64) uint8 {
	byteMaximum := float64(^uint8(tensor.FirstOffset))
	return uint8(math.Round(value * byteMaximum))
}
