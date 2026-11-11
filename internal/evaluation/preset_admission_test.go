package evaluation

import (
	"fmt"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAPresetIsAdmittedOnlyAtAPersonsSize holds a preset's admission to a suite drawn at the size a person asks for
// (judge-calibration/user-size): a reading whose either arm was drawn smaller than the person's request (the Wan student
// was first read at 17 frames and 416 x 240 against a person's 81 at 832 x 480) or in another format is refused as its
// evidence; at the person's size a preset that loses no case beyond chance is admitted, and one that loses many is not.
func TestAPresetIsAdmittedOnlyAtAPersonsSize(t *testing.T) {
	t.Parallel()
	person := map[string]float64{"frames": 81, "width": 832, "height": 480, "steps": 50}
	smaller := map[string]float64{"frames": 17, "width": 416, "height": 240, "steps": 20}
	reference := PresetArm{Settings: person, Format: "mp4"}
	candidate := PresetArm{Preset: "cfg-zero-star", Settings: person, Format: "mp4"}
	outcomes := func(right int) map[string]bool {
		cases := map[string]bool{}
		for at := range 40 {
			cases[fmt.Sprintf("case-%02d", at)] = at < right
		}
		return cases
	}
	held := testcheck.Must(AdmitPreset(person, reference, candidate, outcomes(30), outcomes(30)))(t)
	if !held.Admitted {
		t.Errorf("a preset that loses no case was refused: %+v", held)
	}
	worse := testcheck.Must(AdmitPreset(person, reference, candidate, outcomes(30), outcomes(10)))(t)
	if worse.Admitted || worse.Paired.Losses != 20 {
		t.Errorf("a preset that loses 20 cases of 40 was admitted, or its losses miscounted: %+v", worse)
	}
	for name, arms := range map[string][2]PresetArm{
		"a reference drawn smaller": {{Settings: smaller, Format: "mp4"}, candidate},
		"a preset drawn smaller":    {reference, {Preset: "cfg-zero-star", Settings: smaller, Format: "mp4"}},
		"arms in two formats":       {reference, {Preset: "cfg-zero-star", Settings: person, Format: "gif"}},
		"a reference with a preset": {{Preset: "guidance-interval", Settings: person, Format: "mp4"}, candidate},
	} {
		if _, err := AdmitPreset(person, arms[0], arms[1], outcomes(30), outcomes(30)); err == nil {
			t.Errorf("%s was taken as a preset's evidence", name)
		}
	}
	if _, err := AdmitPreset(person, reference, candidate, outcomes(30), map[string]bool{"case-00": true}); err == nil || !strings.Contains(err.Error(), "case") {
		t.Errorf("arms of different cases were paired: %v", err)
	}
}
