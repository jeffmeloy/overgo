package mediacapability

import (
	"slices"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/testcheck"
)

// TestAGuidancePresetIsOfferedOnlyOnAGain holds a page's read presets to a measured gain (common-controls/gain-only,
// the owner's direction of 2026-11-16: a canned choice is a speed tier, and a guidance method reaches a page only on a
// gain). A preset admitted on no loss beyond chance and showing no gain is not offered, though its reading admits it;
// one whose question or pairwise verdict shows a gain is, where neither shows a loss; one with a gain on one
// instrument and a loss on the other is not; a pending one is not; and a preset the record does not read, a step count
// or a scale from its authors, is offered as declared. On the embedded record every read preset a page offers shows a
// gain.
func TestAGuidancePresetIsOfferedOnlyOnAGain(t *testing.T) {
	t.Parallel()
	tie := evaluation.PresetVerdict{Wins: 12, Losses: 9, Ties: 59, Media: 80, Tail: 0.33, Admitted: true}
	gain := evaluation.PresetVerdict{Wins: 40, Losses: 5, Ties: 35, Media: 80, Tail: 0.0001, Admitted: true}
	loss := evaluation.PresetVerdict{Wins: 3, Losses: 55, Ties: 22, Media: 80, Tail: 1, Admitted: false}
	record := presetRecord{Readings: map[string]map[string]presetReading{"family": {
		"tied":            {Verdict: &tie, Pairwise: &pairwiseReading{Verdict: tie}},
		"gaining":         {Verdict: &tie, Pairwise: &pairwiseReading{Verdict: gain}},
		"asked":           {Verdict: &gain},
		"split":           {Verdict: &gain, Pairwise: &pairwiseReading{Verdict: loss}},
		"tied unpairwise": {Verdict: &tie},
		"pending":         {},
	}}}
	declared := []string{"asked", "draft", "gaining", "pending", "split", "tied", "tied unpairwise"}
	if offered := offeredPresets(record, "family", declared); !slices.Equal(offered, []string{"asked", "draft", "gaining"}) {
		t.Errorf("a page offers %v; want the presets that gained on a reading with no loss, and the one not read", offered)
	}

	embedded := testcheck.Must(presetReadings())(t)
	for family, presets := range embedded.Readings {
		for preset, reading := range presets {
			names := []string{preset}
			if offered := offeredPresets(embedded, family, names); len(offered) == 1 &&
				!(reading.Verdict.Gained() || reading.Pairwise != nil && reading.Pairwise.Verdict.Gained()) {
				t.Errorf("%s's preset %s is offered with no gain on any reading", family, preset)
			}
		}
	}
}
