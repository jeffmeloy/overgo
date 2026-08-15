package dtype

import (
	"math"
	"testing"
)

func TestFloat64SliceToFloat32OwnsNarrowedStorage(t *testing.T) {
	values := []float64{-math.MaxFloat64, -1, 0, 1, math.MaxFloat64}
	converted := Float64SliceToFloat32(values)
	for index, value := range values {
		if converted[index] != float32(value) {
			t.Fatalf("converted[%d] = %g, want %g", index, converted[index], float32(value))
		}
	}
	values[0] = 0
	if converted[0] == 0 {
		t.Fatal("conversion retained source storage")
	}
}
