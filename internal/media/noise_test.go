package media

import "testing"

func TestFreshNoiseMix(t *testing.T) {
	mix := FreshNoiseMix()
	values := []float32{7}
	MixNoiseBF16Into(values, []float32{2}, mix)
	if values[0] != 2 {
		t.Fatalf("mixed value = %v", values)
	}
}
