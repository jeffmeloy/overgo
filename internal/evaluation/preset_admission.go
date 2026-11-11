package evaluation

import (
	"errors"
	"fmt"
	"maps"
	"slices"

	"overgo/internal/tensorstats"
)

// PresetArm is one arm of a preset's reading on a suite: the preset its renders were drawn with, none for the model's
// reference (the authors' recipe), the settings they were drawn at (frames, width, height, steps) and their format.
type PresetArm struct {
	Preset   string
	Settings map[string]float64
	Format   string
}

// PresetVerdict is a preset's reading against its model's reference on one suite: the paired cases (a win the preset
// right where the reference was wrong) and whether it is admitted.
type PresetVerdict struct {
	Preset   string
	Paired   tensorstats.PairedBinary
	Admitted bool
}

// AdmitPreset reads a preset against its model's reference on one suite (judge-calibration/user-size). Its evidence
// is a suite drawn at the size a person asks for, person giving the settings PersonRequest asks with: an arm drawn at
// any other setting of person's, in a format the other arm was not drawn in, or a reference drawn with a preset is
// refused, as the Wan student was first read at 17 frames and 416 x 240 against a person's 81 at 832 x 480. The cases
// pair by name, both arms answering the same ones. The preset is admitted when the exact paired test finds no loss
// beyond chance (falseGain): a preset that does not lower quality is offered, one that does is not.
func AdmitPreset(person map[string]float64, reference, candidate PresetArm, referenceRight, candidateRight map[string]bool) (PresetVerdict, error) {
	if reference.Preset != "" || candidate.Preset == "" {
		return PresetVerdict{}, fmt.Errorf("evaluation: a preset is read against its model's reference, not %q against %q", candidate.Preset, reference.Preset)
	}
	if reference.Format != candidate.Format {
		return PresetVerdict{}, fmt.Errorf("evaluation: %s was drawn as %q and its reference as %q; a reading compares one format", candidate.Preset, candidate.Format, reference.Format)
	}
	for _, arm := range []PresetArm{reference, candidate} {
		for _, setting := range slices.Sorted(maps.Keys(person)) {
			if drawn, stated := arm.Settings[setting]; !stated || drawn != person[setting] {
				return PresetVerdict{}, fmt.Errorf("evaluation: the arm of preset %q was drawn at %s %v, and a person asks for %v; a suite drawn at another size is no preset's evidence",
					arm.Preset, setting, drawn, person[setting])
			}
		}
	}
	if !slices.Equal(slices.Sorted(maps.Keys(referenceRight)), slices.Sorted(maps.Keys(candidateRight))) {
		return PresetVerdict{}, errors.New("evaluation: the preset and its reference answered different cases, which do not pair")
	}
	var wins, losses, ties uint64
	for name, right := range referenceRight {
		switch {
		case candidateRight[name] == right:
			ties++
		case candidateRight[name]:
			wins++
		default:
			losses++
		}
	}
	// The tail is the chance of at least these losses among the discordant cases were the preset no worse.
	lost := tensorstats.ExactPairedBinary(losses, wins, ties)
	verdict := PresetVerdict{Preset: candidate.Preset, Paired: tensorstats.ExactPairedBinary(wins, losses, ties)}
	verdict.Admitted = lost.Tail > falseGain
	return verdict, nil
}
