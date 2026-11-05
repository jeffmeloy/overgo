package hostmath

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestDownsample2DChannelsF64Into(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 0, 1, 1, 0, 0, 0, 0}
	out := make([]float32, 1)
	testcheck.NoError(t, Downsample2DChannelsF64Into(out, input, weight, []float32{1}, 1, 1, 2, 2))
	if out[0] != 11 {
		t.Fatalf("output = %v", out)
	}
}
