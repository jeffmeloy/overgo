package dtype

import (
	"math"
	"testing"
)

// referenceE4M3 independently decodes an OCP F8_E4M3FN byte via floating-point
// arithmetic (distinct from the bit-assembly under test). Every e4m3 value is
// exactly representable in f32, so the two forms must agree bit-for-bit.
func referenceE4M3(b byte) (value float32, isNaN bool) {
	sign := 1.0
	if b&0x80 != 0 {
		sign = -1.0
	}
	exponent := int(b>>3) & 0x0f
	mantissa := int(b & 0x07)
	if exponent == 0x0f && mantissa == 0x07 {
		return 0, true
	}
	if exponent == 0 {
		return float32(sign * float64(mantissa) / 512.0), false // subnormal: m * 2^-9
	}
	return float32(sign * (1 + float64(mantissa)/8) * math.Pow(2, float64(exponent-7))), false
}

func TestF8E4M3ToFloat32ExactForAllCodes(t *testing.T) {
	for code := range 256 {
		got := F8E4M3ToFloat32(byte(code))
		want, isNaN := referenceE4M3(byte(code))
		if isNaN {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("code 0x%02x: got %v, want NaN", code, got)
			}
			// NaN sign must be preserved.
			if math.Signbit(float64(got)) != (byte(code)&0x80 != 0) {
				t.Fatalf("code 0x%02x: NaN sign mismatch", code)
			}
			continue
		}
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("code 0x%02x: got %v (0x%08x), want %v (0x%08x)",
				code, got, math.Float32bits(got), want, math.Float32bits(want))
		}
	}
}
