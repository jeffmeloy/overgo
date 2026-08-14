package projector

import "testing"

func TestGemma4TowerVisionNormalizationUsesProfile(t *testing.T) {
	pixels := []float32{0.5, 0.25, 0.75, 1, 0, 0.5}
	got := gemma4TowerNormalizePixels(
		pixels,
		[3]float32{0.25, 0.5, 0.75},
		[3]float32{0.25, 0.25, 0.5},
	)
	want := []float32{1, -1, 0, 3, -2, -0.5}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("normalized[%d]=%g want=%g", index, got[index], want[index])
		}
	}
}
