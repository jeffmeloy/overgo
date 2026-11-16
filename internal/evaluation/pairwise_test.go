package evaluation

import (
	"image"
	"image/color"
	"testing"

	"overgo/internal/testcheck"
)

// TestAPairwiseReadingCountsAPromptByBothOrders holds the pairwise reading of a preset against its model's reference
// (pairwise-reading/do): the composite holds each arm's picture on its own side, whole, and pictures of two sizes are
// refused; the suite asks each prompt once with the preset on each side, its answer the preset's letter; and a judge
// scripted to choose the preset's picture of one prompt in both orders, the reference's of another, and the left
// picture of a third reads one win, one loss and one tie, a judge that answers by side alone tying every prompt. Arms
// drawn at other settings, a reference drawn with a preset, and a prompt asked in one order alone are refused.
func TestAPairwiseReadingCountsAPromptByBothOrders(t *testing.T) {
	t.Parallel()
	solid := func(width, height int, colour color.RGBA) image.Image {
		picture := image.NewRGBA(image.Rect(0, 0, width, height))
		for y := range height {
			for x := range width {
				picture.SetRGBA(x, y, colour)
			}
		}
		return picture
	}
	red, blue := color.RGBA{R: 255, A: 255}, color.RGBA{B: 255, A: 255}
	composite := testcheck.Must(ComposePair(solid(3, 2, red), solid(3, 2, blue)))(t)
	if composite.Bounds().Size() != image.Pt(6, 2) || composite.RGBAAt(2, 1) != red || composite.RGBAAt(3, 0) != blue {
		t.Fatalf("a red and a blue 3 x 2 picture composed as %v with %v and %v at the seam; want 6 x 2, red left, blue right",
			composite.Bounds().Size(), composite.RGBAAt(2, 1), composite.RGBAAt(3, 0))
	}
	if _, err := ComposePair(solid(3, 2, red), solid(2, 2, blue)); err == nil {
		t.Error("pictures of two sizes were set side by side")
	}

	settings := map[string]float64{"width": 512, "height": 512, "steps": 8}
	reference, preset := PresetArm{Settings: settings}, PresetArm{Preset: "apg", Settings: settings}
	prompts := map[string]string{"fox": "a red fox", "hen": "a white hen", "owl": "an owl at night"}
	suite := testcheck.Must(BuildPairwise(reference, preset, prompts, "judge"))(t)
	testcheck.Must(CompileGeneratedAnswer(suite))(t)
	if len(suite.Cases) != 2*len(prompts) {
		t.Fatalf("the suite asks %d cases of %d prompts; want each prompt in both orders", len(suite.Cases), len(prompts))
	}
	media := map[string]string{}
	for _, testCase := range suite.Cases {
		media[testCase.Name] = testCase.Group
		_, named, _, _ := ParseMediaPrompt(testCase.Prompt)
		if named != PairwiseMedium(testCase.Group, testCase.Answers[0]) {
			t.Errorf("case %s asks about %s; want the composite with the preset on its answer's side", testCase.Name, named)
		}
	}
	// read is the judged suite of a judge answering each case's letter as answer says.
	read := func(answer func(GeneratedAnswerCase) string) PresetVerdict {
		chose := map[string]bool{}
		for _, testCase := range suite.Cases {
			chose[testCase.Name] = answer(testCase) == testCase.Answers[0]
		}
		return testcheck.Must(ReadPairwise(preset.Preset, chose, media))(t)
	}
	other := map[string]string{"A": "B", "B": "A"}
	scripted := read(func(testCase GeneratedAnswerCase) string {
		switch testCase.Group {
		case "fox":
			return testCase.Answers[0]
		case "hen":
			return other[testCase.Answers[0]]
		}
		return "A"
	})
	if scripted.Wins != 1 || scripted.Losses != 1 || scripted.Ties != 1 || scripted.Media != 3 {
		t.Errorf("a judge choosing the preset's fox, the reference's hen and the left owl read %+v; want one win, one loss and one tie over three media", scripted)
	}
	if bySide := read(func(GeneratedAnswerCase) string { return "A" }); bySide.Wins != 0 || bySide.Losses != 0 || bySide.Ties != 3 {
		t.Errorf("a judge answering by side read %+v; want every prompt a tie", bySide)
	}

	if _, err := BuildPairwise(reference, PresetArm{Preset: "apg", Settings: map[string]float64{"width": 256, "height": 256, "steps": 8}}, prompts, "judge"); err == nil {
		t.Error("arms drawn at other sizes were paired")
	}
	if _, err := BuildPairwise(preset, preset, prompts, "judge"); err == nil {
		t.Error("a reference drawn with a preset was paired")
	}
	delete(media, suite.Cases[0].Name)
	once := map[string]bool{}
	for name := range media {
		once[name] = true
	}
	if _, err := ReadPairwise(preset.Preset, once, media); err == nil {
		t.Error("a prompt asked in one order alone was read")
	}
}
