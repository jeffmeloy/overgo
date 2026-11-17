package evaluation

import (
	"errors"
	"fmt"
	"image"
	"image/draw"
	"maps"
	"slices"
	"strings"
)

// The pairwise reading's suite: its source and schema, as a suite of TIFA questions names its own.
const (
	pairwiseSource = "pairwise-preset"
	pairwiseSchema = "pairwise-preset/v1"
)

// pairwiseSides are the sides of a pairwise composite, by the letter the judge chooses: the left picture is A, the
// right B. Each prompt is asked once with the preset on each side (BuildPairwise), so a judge's leaning to a side
// counts for neither arm.
var pairwiseSides = []string{"A", "B"}

// PairwiseMedium is the name of the composite that sets a prompt's medium, medium, beside its reference's with the
// preset on side, a letter of pairwiseSides: the file the composer writes and the suite's case asks about.
func PairwiseMedium(medium, side string) string {
	return strings.NewReplacer("/", "_", "\\", "_", " ", "_").Replace(medium) + ".preset-" + side + ".png"
}

// ComposePair sets two pictures of one size side by side, left then right, into one picture the judge reads whole.
// Pictures of different sizes are refused: arms drawn at one size a person asks for are what a reading compares
// (judge-calibration/user-size), and a smaller picture beside a larger one would be read for its size.
func ComposePair(left, right image.Image) (*image.RGBA, error) {
	size := left.Bounds().Size()
	if right.Bounds().Size() != size || size.X == 0 || size.Y == 0 {
		return nil, fmt.Errorf("evaluation: a pair is set side by side at one size, not %v beside %v", size, right.Bounds().Size())
	}
	composite := image.NewRGBA(image.Rect(0, 0, 2*size.X, size.Y))
	draw.Draw(composite, image.Rect(0, 0, size.X, size.Y), left, left.Bounds().Min, draw.Src)
	draw.Draw(composite, image.Rect(size.X, 0, 2*size.X, size.Y), right, right.Bounds().Min, draw.Src)
	return composite, nil
}

// BuildPairwise is the suite of a pairwise reading of a preset against its model's reference (pairwise-reading/do):
// for each medium both arms drew, by the prompt it was drawn from, one case for each side the preset is set on, asking
// which of the two pictures better shows the prompt, its answer the preset's letter, its group the medium. Arms drawn
// at other settings or in other formats, or a reference drawn with a preset, are refused, as AdmitPreset refuses them.
// The suite names no judge: the judge answers as itself about composites no model drew, as it answers its calibration
// suites, where a judged suite's pictures are each the evaluated recipe's own drawing (guidance-ops/pairwise-judge).
func BuildPairwise(reference, candidate PresetArm, prompts map[string]string) (GeneratedAnswerSuite, error) {
	switch {
	case reference.Preset != "" || candidate.Preset == "":
		return GeneratedAnswerSuite{}, fmt.Errorf("evaluation: a preset is read against its model's reference, not %q against %q", candidate.Preset, reference.Preset)
	case reference.Format != candidate.Format || !maps.Equal(reference.Settings, candidate.Settings):
		return GeneratedAnswerSuite{}, fmt.Errorf("evaluation: %s was drawn at %v as %q and its reference at %v as %q; a pair is drawn alike but the preset",
			candidate.Preset, candidate.Settings, candidate.Format, reference.Settings, reference.Format)
	case len(prompts) == 0:
		return GeneratedAnswerSuite{}, errors.New("evaluation: a pairwise reading asks about at least one prompt")
	}
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: pairwiseSchema, Source: pairwiseSource,
		Transforms: []string{TransformTrimSpace, transformLowercase},
	}
	for _, medium := range slices.Sorted(maps.Keys(prompts)) {
		question := fmt.Sprintf("The picture holds two pictures side by side, A on the left and B on the right. Which of them better shows this prompt: %q?\nA. the left picture\nB. the right picture\nAnswer with the letter only.", prompts[medium])
		for _, side := range pairwiseSides {
			suite.Cases = append(suite.Cases, GeneratedAnswerCase{
				Name:      pairwiseSource + "/" + medium + "/preset-" + side,
				Prompt:    mediaPrompt(MediaImage, PairwiseMedium(medium, side), question),
				MaxTokens: tifaAnswerTokens, Answers: []string{side}, Choices: pairwiseSides, Group: medium,
			})
		}
	}
	return suite, nil
}

// ReadPairwise is a preset's pairwise verdict from its judged suite (BuildPairwise): chose says, by case, that the
// judge chose the preset's picture, and media names each case's medium (CaseMedia). A medium counts for the preset
// where both its cases chose the preset, against it where both chose the reference, and ties where they split, so a
// judge that answers by side ties every medium; the media's nets are read as AdmitPreset reads them (presetVerdict).
func ReadPairwise(preset string, chose map[string]bool, media map[string]string) (PresetVerdict, error) {
	asked := map[string][]bool{}
	for name, chosen := range chose {
		medium, named := media[name]
		if !named {
			return PresetVerdict{}, fmt.Errorf("evaluation: case %s asks about no medium the suite names, so its evidence has no unit", name)
		}
		asked[medium] = append(asked[medium], chosen)
	}
	var wins, losses, ties uint64
	nets := map[string]int64{}
	for medium, chosen := range asked {
		if len(chosen) != len(pairwiseSides) {
			return PresetVerdict{}, fmt.Errorf("evaluation: medium %s was asked %d times; a pairwise reading asks each once with the preset on each side", medium, len(chosen))
		}
		switch {
		case chosen[0] && chosen[1]:
			wins, nets[medium] = wins+1, 1
		case !chosen[0] && !chosen[1]:
			losses, nets[medium] = losses+1, -1
		default:
			ties, nets[medium] = ties+1, 0
		}
	}
	return presetVerdict(preset, wins, losses, ties, nets), nil
}
