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

// Float32ToF8E4M3: OCP F8_E4M3FN encode with round-to-nearest-even, the
// inverse of F8E4M3ToFloat32 on every representable value. Magnitudes above
// the format maximum (448) saturate to it — E4M3FN has no infinity and the
// sole NaN pattern is reserved, so saturation is the encode contract. NaN
// inputs encode the NaN pattern with the input's sign.
func Float32ToF8E4M3(f float32) byte {
	bits := math.Float32bits(f)
	sign := byte(bits >> 31 << 7)
	if math.IsNaN(float64(f)) {
		return sign | 0x7f
	}
	a := math.Abs(float64(f))
	if a == 0 {
		return sign
	}
	if a >= 448 {
		return sign | 0x7e // 1111.110 = 448
	}
	exponent := int(math.Floor(math.Log2(a)))
	if exponent < -6 {
		// subnormal: units of 2^-9, round half to even
		units := a * 512
		m := int(math.RoundToEven(units))
		if m == 0 {
			return sign
		}
		if m <= 7 {
			return sign | byte(m)
		}
		exponent = -6 // rounded up into the first normal binade
		a = math.Ldexp(1, -6)
		_ = a
		return sign | (1 << 3)
	}
	mantissa := int(math.RoundToEven((a/math.Ldexp(1, exponent) - 1) * 8))
	if mantissa == 8 {
		exponent++
		mantissa = 0
	}
	if exponent > 8 || (exponent == 8 && mantissa > 6) {
		return sign | 0x7e
	}
	return sign | byte(exponent+7)<<3 | byte(mantissa)
}
