package projector

import "testing"

func TestGemma4TowerVisionInputAffineUsesProfile(t *testing.T) {
	pixels := []float32{0.5, 0.25, 0.75, 1, 0, 0.5}
	got := gemma4TowerInputAffine(
		pixels,
		[3]float32{2, 4, 0.5},
		[3]float32{-1, -1, 0.25},
	)
	want := []float32{0, 0, 0.625, 1, -1, 0.5}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("normalized[%d]=%g want=%g", index, got[index], want[index])
		}
	}
}
