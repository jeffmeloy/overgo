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

// PresetVerdict is a preset's reading against its model's reference on one suite: its cases' wins (the preset right
// where the reference was wrong), losses and ties, the media they ask about, the grouped tail of a gain (Tail: the
// chance, each medium's cases flipped together, of a net at least this one were the arms alike), whether it is
// admitted, and what the reading says in words (Says). Admission screens harm: a preset that loses no medium's worth
// beyond chance is admitted, which shows no gain, as cfg-zero-star's 5 wins and 6 losses and SenseNova's APG's 13 and
// 15, among about 97% ties, were admitted (admission-words/do).
type PresetVerdict struct {
	Preset   string
	Wins     uint64
	Losses   uint64
	Ties     uint64
	Media    int
	Tail     float64
	Admitted bool
	Says     string
}

// AdmitPreset reads a preset against its model's reference on one suite (judge-calibration/user-size). Its evidence
// is a suite drawn at the size a person asks for, person giving the settings PersonRequest asks with: an arm drawn at
// any other setting of person's, in a format the other arm was not drawn in, or a reference drawn with a preset is
// refused, as the Wan student was first read at 17 frames and 416 x 240 against a person's 81 at 832 x 480. The cases
// pair by name, both arms answering the same ones, and media names what each case asks about (CaseMedia): the several
// questions a suite asks of one clip are one unit of evidence, so the null flips each medium's net at once
// (tensorstats.GroupedSignTail), where 524 questions of 80 clips were read as 524 independent pairs (SQA-342). The
// preset is admitted when that test finds no loss beyond chance (falseGain): a preset that does not lower quality is
// offered, one that does is not.
func AdmitPreset(person map[string]float64, reference, candidate PresetArm, referenceRight, candidateRight map[string]bool, media map[string]string) (PresetVerdict, error) {
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
	nets := map[string]int64{}
	for name, right := range referenceRight {
		medium, named := media[name]
		if !named {
			return PresetVerdict{}, fmt.Errorf("evaluation: case %s asks about no medium the suite names, so its evidence has no unit", name)
		}
		net := nets[medium]
		switch {
		case candidateRight[name] == right:
			ties++
		case candidateRight[name]:
			wins++
			net++
		default:
			losses++
			net--
		}
		nets[medium] = net
	}
	return presetVerdict(candidate.Preset, wins, losses, ties, nets), nil
}

// Gained says the verdict shows a gain beyond chance: were the preset no better, a net gain at least its own over its
// media would come by chance at most falseGain of the time.
func (verdict PresetVerdict) Gained() bool { return verdict.Tail <= falseGain }

// presetVerdict is the verdict on a preset whose evidence is its wins, losses and ties and each medium's net: the
// grouped tail of a gain, admission where no loss is beyond chance (falseGain), and both in words.
func presetVerdict(preset string, wins, losses, ties uint64, nets map[string]int64) PresetVerdict {
	verdict := PresetVerdict{Preset: preset, Wins: wins, Losses: losses, Ties: ties}
	gained := slices.Collect(maps.Values(nets))
	lost := make([]int64, len(gained))
	for at, net := range gained {
		lost[at] = -net
	}
	verdict.Media, verdict.Tail = len(nets), tensorstats.GroupedSignTail(gained)
	// The loss's tail is the chance of a net loss at least this one were the preset no worse.
	verdict.Admitted = tensorstats.GroupedSignTail(lost) > falseGain
	counts := fmt.Sprintf("%d wins, %d losses and %d ties over %d media", verdict.Wins, verdict.Losses, verdict.Ties, verdict.Media)
	switch {
	case !verdict.Admitted:
		verdict.Says = "refused: a loss beyond chance, " + counts
	case verdict.Gained():
		verdict.Says = "admitted: a gain beyond chance, " + counts
	default:
		verdict.Says = "admitted on no loss beyond chance, which shows no gain: " + counts
	}
	return verdict
}
