package dtype

import "math"

// F8E4M3ToFloat32: OCP F8_E4M3FN decode -- 1 sign, 4 exponent (bias 7), 3
// mantissa; NO infinities; the sole NaN pattern is S.1111.111. Exact by
// construction: every e4m3 value is representable in f32, so this is pure
// integer bit assembly with no float rounding. Bit-identical to the CUDA
// fp8_e4m3_decode device function (shared native-dtype decode fact).
func F8E4M3ToFloat32(b byte) float32 {
	s := (uint32(b) & 0x80) << 24 // e4m3 sign bit7 -> f32 bit31
	e := (uint32(b) >> 3) & 0x0f
	m := uint32(b) & 0x07
	if e == 0 {
		// subnormal: m * 2^-9 (m in 0..7, exact in f32); sign OR'd back in
		v := float32(m) * (1.0 / 512.0)
		return math.Float32frombits(s | math.Float32bits(v))
	}
	if e == 0x0f && m == 0x07 {
		return math.Float32frombits(s | 0x7fc00000) // E4M3FN NaN, sign preserved
	}
	// normal: (1 + m/8) * 2^(e-7); exponent field e-7+127 = e+120, mantissa m<<20
	return math.Float32frombits(s | ((e + 120) << 23) | (m << 20))
}
