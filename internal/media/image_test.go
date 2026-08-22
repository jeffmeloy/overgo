package media

import "testing"

func TestEncodePlanarRGB8Into(t *testing.T) {
	destination := make([]byte, RGBChannels)
	err := EncodePlanarRGB8Into(destination, []float32{1, 2, 3}, 1, 1, func(value float32) uint8 { return uint8(value) })
	if err != nil {
		t.Fatal(err)
	}
	if destination[0] != 1 || destination[1] != 2 || destination[2] != 3 {
		t.Fatalf("RGB=%v", destination)
	}
}

func TestAffineRGB(t *testing.T) {
	input := []float32{0.5, 0.25, 0.75, 1, 0, 0.5}
	scale := [RGBChannels]float32{2, 4, 0.5}
	bias := [RGBChannels]float32{-1, -1, 0.25}
	want := []float32{0, 0, 0.625, 1, -1, 0.5}
	got := AffineRGB(input, scale, bias)
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("affine[%d]=%g want=%g", index, got[index], want[index])
		}
	}
}
