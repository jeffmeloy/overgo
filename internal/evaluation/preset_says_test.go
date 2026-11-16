package evaluation

import (
	"fmt"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAPresetVerdictSaysItScreensHarm holds a preset's verdict to saying what its admission measured: cfg-zero-star's
// reading, 5 wins and 6 losses among ties, is admitted on no loss beyond chance and says it shows no gain; a preset that
// wins many cases more than it loses is admitted with a gain beyond chance; and one that loses many is refused for its
// loss, each with its wins, losses and ties (admission-words/do).
func TestAPresetVerdictSaysItScreensHarm(t *testing.T) {
	t.Parallel()
	person := map[string]float64{"width": 1024, "height": 1024, "steps": 30}
	reference := PresetArm{Settings: person, Format: "png"}
	candidate := PresetArm{Preset: "cfg-zero-star", Settings: person, Format: "png"}
	// arms are 400 cases, the reference right on the first half, the preset winning the cases from 200 on up to wins
	// and losing the first losses.
	arms := func(wins, losses int) (map[string]bool, map[string]bool) {
		referenceRight, presetRight := map[string]bool{}, map[string]bool{}
		for at := range 400 {
			name := fmt.Sprintf("case-%03d", at)
			referenceRight[name] = at < 200
			presetRight[name] = at < 200 && at >= losses || at >= 200 && at < 200+wins
		}
		return referenceRight, presetRight
	}
	for _, reading := range []struct {
		name          string
		wins, losses  int
		admitted      bool
		says, counted string
	}{
		{"cfg-zero-star's", 5, 6, true, "admitted on no loss beyond chance, which shows no gain", "5 wins, 6 losses and 389 ties over 400 media"},
		{"a gain's", 40, 5, true, "admitted: a gain beyond chance", "40 wins, 5 losses and 355 ties over 400 media"},
		{"a loss's", 5, 40, false, "refused: a loss beyond chance", "5 wins, 40 losses and 355 ties over 400 media"},
	} {
		referenceRight, presetRight := arms(reading.wins, reading.losses)
		verdict := testcheck.Must(AdmitPreset(person, reference, candidate, referenceRight, presetRight, alone(referenceRight)))(t)
		if verdict.Admitted != reading.admitted || !strings.HasPrefix(verdict.Says, reading.says) || !strings.HasSuffix(verdict.Says, reading.counted) {
			t.Errorf("%s reading: admitted %t, says %q; want admitted %t, saying %q with %q", reading.name, verdict.Admitted, verdict.Says, reading.admitted, reading.says, reading.counted)
		}
	}
}
