package dtype

import (
	"math"
	"testing"
)

func TestBFloat16Conversions(t *testing.T) {
	for _, value := range []float32{0, 1, -2.5, 1.00390625, float32(math.Inf(1)), float32(math.NaN())} {
		encoded := Float32ToBF16(value)
		got := BF16ToFloat32(encoded)
		if math.IsNaN(float64(value)) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("NaN round trip = %v", got)
			}
			continue
		}
		if got != RoundBF16(value) {
			t.Fatalf("round trip %v = %v, round = %v", value, got, RoundBF16(value))
		}
	}
}

// TestFloat32ToBF16KeepsEveryNaN: a NaN stays one through the pack, a signalling one whose mantissa lies below BF16's among
// them, and an infinity stays an infinity.
func TestFloat32ToBF16KeepsEveryNaN(t *testing.T) {
	for _, bits := range []uint32{0x7fffffff, 0xffffffff, 0x7fc00000, 0x7f800001, 0x7f80ffff, 0xff800001} {
		if got := BF16ToFloat32(Float32ToBF16(math.Float32frombits(bits))); !math.IsNaN(float64(got)) {
			t.Errorf("%#08x packs to %v", bits, got)
		}
	}
	for _, sign := range []int{1, -1} {
		if got := BF16ToFloat32(Float32ToBF16(float32(math.Inf(sign)))); !math.IsInf(float64(got), sign) {
			t.Errorf("an infinity packs to %v", got)
		}
	}
}
