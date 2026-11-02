package dtype

import "math"

const (
	float32Magnitude = 0x7fffffff // every bit of an F32 but its sign
	float32Infinity  = 0x7f800000 // its magnitude bits: above them lies a NaN
	bf16NaNBit       = 0x0040     // the top mantissa bit of a BF16
	bf16Shift        = 16         // the F32 bits below a BF16's
)

// Float32ToBF16: round-to-nearest-even BF16 storage, as the device's bf16_pack does. A NaN stays one: its own mantissa may
// lie below BF16's, and would truncate to an infinity.
func Float32ToBF16(value float32) uint16 {
	bits := math.Float32bits(value)
	if bits&float32Magnitude > float32Infinity {
		return uint16(bits>>bf16Shift) | bf16NaNBit
	}
	bits += 0x7fff + ((bits >> bf16Shift) & 1)
	return uint16(bits >> bf16Shift)
}

// BF16ToFloat32: exact BF16 expansion.
func BF16ToFloat32(value uint16) float32 {
	return math.Float32frombits(uint32(value) << 16)
}

// RoundBF16: round F32 through BF16 storage.
func RoundBF16(value float32) float32 {
	return BF16ToFloat32(Float32ToBF16(value))
}

// RoundBF16Slice rounds values in place.
func RoundBF16Slice(values []float32) {
	for index, value := range values {
		values[index] = RoundBF16(value)
	}
}
