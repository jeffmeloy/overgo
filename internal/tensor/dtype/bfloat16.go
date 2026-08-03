package dtype

import "math"

// Float32ToBF16: round-to-nearest-even BF16 storage.
func Float32ToBF16(value float32) uint16 {
	bits := math.Float32bits(value)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + ((bits >> 16) & 1)
	}
	return uint16(bits >> 16)
}

// BF16ToFloat32: exact BF16 expansion.
func BF16ToFloat32(value uint16) float32 {
	return math.Float32frombits(uint32(value) << 16)
}

// RoundBF16: round F32 through BF16 storage.
func RoundBF16(value float32) float32 {
	return BF16ToFloat32(Float32ToBF16(value))
}
