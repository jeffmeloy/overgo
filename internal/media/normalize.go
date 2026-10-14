package media

import (
	"math"

	"overgo/internal/tensor"
)

// SignedUnitBounds32 returns the inclusive normalized sample interval.
func SignedUnitBounds32() (float32, float32) {
	upper := float32(tensor.SingletonExtent)
	return -upper, upper
}

// SignedUnitF32ToU8 maps one signed-unit sample to an unsigned byte.
func SignedUnitF32ToU8(value float32) uint8 {
	lower := float64(tensor.FirstOffset)
	upper := float64(tensor.SingletonExtent)
	centerScale := upper / float64(tensor.PairedExtent)
	normalized := (float64(value) + upper) * centerScale
	return unitFloat64ToU8(max(lower, min(normalized, upper)))
}

// U8ToSignedUnitF32 maps one unsigned byte back to a signed-unit sample,
// the inverse of SignedUnitF32ToU8 up to the byte quantization.
func U8ToSignedUnitF32(value uint8) float32 {
	byteMaximum := float64(^uint8(tensor.FirstOffset))
	upper := float64(tensor.SingletonExtent)
	centerScale := upper / float64(tensor.PairedExtent)
	return float32(float64(value)/byteMaximum/centerScale - upper)
}

// U8ToUnitF32 maps one unsigned byte back to a unit-interval sample, the
// inverse of UnitF32ToU8 up to the byte quantization.
func U8ToUnitF32(value uint8) float32 {
	return float32(float64(value) / float64(^uint8(tensor.FirstOffset)))
}

// UnitF32ToU8 maps one unit-interval sample to an unsigned byte.
func UnitF32ToU8(value float32) uint8 {
	lower := float64(tensor.FirstOffset)
	upper := float64(tensor.SingletonExtent)
	return unitFloat64ToU8(max(lower, min(float64(value), upper)))
}

func unitFloat64ToU8(value float64) uint8 {
	byteMaximum := float64(^uint8(tensor.FirstOffset))
	return uint8(math.Round(value * byteMaximum))
}
