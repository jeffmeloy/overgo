//go:build windows

package mediacapability

import (
	"slices"
	"testing"

	"overgo/internal/modelrecipe"
)

// TestSenseNovasPageOffersItsGuidancePresets: SenseNova's page offers APG, CFG-Zero* and the guidance interval as the
// choices of its preset control, a tuning control beside the authors' recipe, as Wan's and MOSS's pages offer theirs
// (guidance-ops/sensenova-presets).
func TestSenseNovasPageOffersItsGuidancePresets(t *testing.T) {
	t.Parallel()
	controls, refusal := Controls(modelrecipe.ModuleRoutedImagePrepare, t.TempDir())
	if refusal != "" {
		t.Fatal(refusal)
	}
	held := slices.IndexFunc(controls, func(c Control) bool { return c.Name == presetControl })
	if held < 0 || !controls[held].Advanced || !slices.Equal(controls[held].Choices, []string{"apg", "cfg-zero-star", "guidance-interval"}) {
		t.Fatalf("SenseNova's page offers %+v, want its three guidance presets on a tuning control", controls)
	}
}
