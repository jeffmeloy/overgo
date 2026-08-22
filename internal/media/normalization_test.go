package media

import "testing"

func TestValidateChannelMoments(t *testing.T) {
	if err := ValidateChannelMoments([]float64{0, 1}, []float64{1, 2}, 2); err != nil {
		t.Fatal(err)
	}
	if err := ValidateChannelMoments([]float32{0}, []float32{0}, 1); err == nil {
		t.Fatal("accepted zero standard deviation")
	}
	if err := ValidateChannelMoments([]float32{0}, []float32{1}, 2); err == nil {
		t.Fatal("accepted mismatched channels")
	}
}

func TestNormalizePlanarChannelsInto(t *testing.T) {
	output := make([]float32, 4)
	err := NormalizePlanarChannelsInto(output, []float32{1, 3, 6, 10}, []float32{1, 2}, []float32{2, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 0 || output[1] != 1 || output[2] != 1 || output[3] != 2 {
		t.Fatalf("normalized=%v", output)
	}
}
