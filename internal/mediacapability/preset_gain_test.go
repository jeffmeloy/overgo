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
// instrument and a loss on the other is not; a pending one is not; the speed preset, a speed tier, is offered on no
// loss beyond chance, gain or none, and not on a loss; and a preset the record does not read, a step count or a scale
// from its authors, is offered as declared, as is each preset of a model the record reads nothing of. On the embedded
// record every read preset a page offers but the speed preset shows a gain.
func TestAGuidancePresetIsOfferedOnlyOnAGain(t *testing.T) {
	t.Parallel()
	tie := evaluation.PresetVerdict{Wins: 12, Losses: 9, Ties: 59, Media: 80, Tail: 0.33, Admitted: true}
	gain := evaluation.PresetVerdict{Wins: 40, Losses: 5, Ties: 35, Media: 80, Tail: 0.0001, Admitted: true}
	loss := evaluation.PresetVerdict{Wins: 3, Losses: 55, Ties: 22, Media: 80, Tail: 1, Admitted: false}
	record := presetRecord{Readings: map[string]map[string]presetReading{"Model": {
		"tied":            {Verdict: &tie, Pairwise: &pairwiseReading{Verdict: tie}},
		"gaining":         {Verdict: &tie, Pairwise: &pairwiseReading{Verdict: gain}},
		"asked":           {Verdict: &gain},
		"split":           {Verdict: &gain, Pairwise: &pairwiseReading{Verdict: loss}},
		"tied unpairwise": {Verdict: &tie},
		"pending":         {},
		speedPreset:       {Verdict: &tie, Pairwise: &pairwiseReading{Verdict: tie}},
	}, "Losing": {
		speedPreset: {Verdict: &tie, Pairwise: &pairwiseReading{Verdict: loss}},
	}}}
	declared := []string{"asked", "draft", speedPreset, "gaining", "pending", "split", "tied", "tied unpairwise"}
	if offered := offeredPresets(record, "Model", declared); !slices.Equal(offered, []string{"asked", "draft", speedPreset, "gaining"}) {
		t.Errorf("a page offers %v; want the presets that gained on a reading with no loss, the speed preset with no loss, and the one not read", offered)
	}
	if offered := offeredPresets(record, "Losing", []string{speedPreset}); len(offered) != 0 {
		t.Errorf("a page offers %v, a speed preset whose pairwise reading shows a loss", offered)
	}
	// Another model's page reads none of these readings, so it offers each preset as declared.
	if offered := offeredPresets(record, "Other", declared); !slices.Equal(offered, declared) {
		t.Errorf("another model's page offers %v; want every preset it declares", offered)
	}

	embedded := testcheck.Must(presetReadings())(t)
	for model, presets := range embedded.Readings {
		for preset, reading := range presets {
			names := []string{preset}
			if offered := offeredPresets(embedded, model, names); len(offered) == 1 && preset != speedPreset &&
				!(reading.Verdict.Gained() || reading.Pairwise != nil && reading.Pairwise.Verdict.Gained()) {
				t.Errorf("%s's preset %s is offered with no gain on any reading", model, preset)
			}
		}
	}
}
