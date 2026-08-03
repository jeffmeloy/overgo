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
