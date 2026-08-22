package trainingprogram

import "testing"

func TestMeanSquaredErrorF32(t *testing.T) {
	loss, gradient, err := MeanSquaredErrorF32([]float32{1, 3}, []float32{0, 1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if loss != 2.5 || len(gradient) != 2 || gradient[0] != 1 || gradient[1] != 2 {
		t.Fatalf("loss=%g gradient=%v", loss, gradient)
	}
}
