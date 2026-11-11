//go:build windows

package mediacapability

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/sensenovarecipe"
	"overgo/internal/testcheck"
)

// TestAPresetUnderReadingIsOfferedOnlyOnceAdmitted holds a model's page to the readings of its presets
// (guidance-ops/preset-gate), read through SenseNova's page, whose presets its recipe declares with no model files: a
// preset the readings name is offered only once admitted, pending or refused it is not, and the page drops its preset
// control when it offers none. The readings, read from the file the page's code embeds, hold only the three statuses,
// name for Wan and SenseNova only presets their declarations hold, and name every guidance preset the campaign reads.
func TestAPresetUnderReadingIsOfferedOnlyOnceAdmitted(t *testing.T) {
	t.Parallel()
	var readings map[string]map[string]string
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile("preset_readings.json"))(t), &readings))
	declared := map[string][]string{
		"latentvideo": testcheck.Must(latentvideo.PresetNames(t.TempDir()))(t),
		"sensenova":   testcheck.Must(sensenovarecipe.PresetNames())(t),
	}
	read := map[string][]string{
		"latentimage": {"apg", "cfg-zero-star", "guidance-interval"},
		"latentvideo": {"cfg-zero-star", "guidance-interval", "model-card"},
		"sensenova":   {"apg", "cfg-zero-star", "guidance-interval"},
	}
	for family, presets := range read {
		for _, preset := range presets {
			status, named := readings[family][preset]
			if !named || !slices.Contains([]string{"pending", "admitted", "refused"}, status) {
				t.Errorf("the readings hold %s's preset %s as %q (named: %t); want it named pending, admitted or refused", family, preset, status, named)
			}
		}
		for preset := range readings[family] {
			if declarations, known := declared[family]; known && !slices.Contains(declarations, preset) {
				t.Errorf("the readings name %s's preset %s, which it does not declare (%v)", family, preset, declarations)
			}
		}
	}

	controls, refusal := Controls(modelrecipe.ModuleRoutedImagePrepare, t.TempDir())
	if refusal != "" {
		t.Fatal(refusal)
	}
	var offered []string
	if held := slices.IndexFunc(controls, func(c Control) bool { return c.Name == presetControl }); held >= 0 {
		offered = controls[held].Choices
	}
	for preset, status := range readings["sensenova"] {
		if slices.Contains(offered, preset) != (status == "admitted") {
			t.Errorf("SenseNova's page offers %v; its preset %s, read as %s, is offered only once admitted", offered, preset, status)
		}
	}
}
