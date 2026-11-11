package mediacapability

import (
	"slices"
	"testing"

	"overgo/internal/latentvideo"
	"overgo/internal/testcheck"
)

// TestWanDeclaresItsGuidancePresets: Wan declares CFG-Zero* and the guidance interval beside the model card's
// preset in its profile (latentvideo.PresetNames), which its page offers once their readings admit them (offerPresets),
// each a preset the flow guider runs step by step (guidance-ops/guider).
func TestWanDeclaresItsGuidancePresets(t *testing.T) {
	t.Parallel()
	presets := testcheck.Must(latentvideo.PresetNames(t.TempDir()))(t)
	for _, preset := range []string{"cfg-zero-star", "guidance-interval", "model-card"} {
		if !slices.Contains(presets, preset) {
			t.Errorf("Wan's preset control offers %v, want %s among them", presets, preset)
		}
	}
}
