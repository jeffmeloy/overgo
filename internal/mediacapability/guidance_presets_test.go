package mediacapability

import (
	"slices"
	"testing"

	"overgo/internal/latentvideo"
	"overgo/internal/testcheck"
)

// TestWansPageOffersItsGuidancePresets: Wan's page offers CFG-Zero* and the guidance interval beside the model card's
// preset, as the choices its preset control takes from the profile (latentvideo.PresetNames, which offerPresets
// reads), each a preset the flow guider runs step by step (guidance-ops/guider).
func TestWansPageOffersItsGuidancePresets(t *testing.T) {
	t.Parallel()
	presets := testcheck.Must(latentvideo.PresetNames(t.TempDir()))(t)
	for _, preset := range []string{"cfg-zero-star", "guidance-interval", "model-card"} {
		if !slices.Contains(presets, preset) {
			t.Errorf("Wan's preset control offers %v, want %s among them", presets, preset)
		}
	}
}
