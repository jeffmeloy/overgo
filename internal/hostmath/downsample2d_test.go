package hostmath

import "testing"

func TestDownsample2DChannelsF64Into(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 0, 1, 1, 0, 0, 0, 0}
	out := make([]float32, 1)
	if err := Downsample2DChannelsF64Into(out, input, weight, []float32{1}, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if out[0] != 11 {
		t.Fatalf("output = %v", out)
	}
}
