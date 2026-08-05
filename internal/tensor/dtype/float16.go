package dtype

import "math"

// Float16ToFloat32: exact IEEE 754 binary16 expansion.
func Float16ToFloat32(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := uint32(value>>10) & 0x1f
	mantissa := uint32(value & 0x03ff)

	var bits uint32
	switch exponent {
	case 0:
		if mantissa == 0 {
			bits = sign
			break
		}
		exponent = 113
		for mantissa&0x0400 == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= 0x03ff
		bits = sign | exponent<<23 | mantissa<<13
	case 0x1f:
		bits = sign | 0x7f800000 | mantissa<<13
	default:
		bits = sign | (exponent+112)<<23 | mantissa<<13
	}
	return math.Float32frombits(bits)
}
