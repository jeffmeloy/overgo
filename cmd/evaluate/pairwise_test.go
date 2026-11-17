package main

import (
	"encoding/json"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAPairwiseSuiteSetsEachRequestsPicturesSideBySide holds -pairwise-preset's composition (pairwise-reading/do): two
// renders of one request but the preset, each drawn under its own key, pair by the request; each pair is written in
// both orders, the preset's picture on side A's left and side B's right, and the suite asks the judge about each with
// the request's prompt. A render of another seed is no pair.
func TestAPairwiseSuiteSetsEachRequestsPicturesSideBySide(t *testing.T) {
	t.Parallel()
	painter, painting := testutil.ArtifactID(t, artifact.KindModel, "painter"), testutil.ArtifactID(t, artifact.KindRecipe, "painting")
	red, blue := color.RGBA{R: 255, A: 255}, color.RGBA{B: 255, A: 255}
	render := func(preset, key string, seed int, colour color.RGBA) string {
		t.Helper()
		request := map[string]any{"prompt": "a river at dawn", "seed": seed, "width": 4, "height": 2}
		if preset != "" {
			request["preset"] = preset
		}
		directory := t.TempDir()
		picture := image.NewRGBA(image.Rect(0, 0, 4, 2))
		for y := range 2 {
			for x := range 4 {
				picture.SetRGBA(x, y, colour)
			}
		}
		testcheck.NoError(t, writePNG(filepath.Join(directory, key+".png"), picture))
		manifest := renderManifest{Model: painter, Recipe: painting, Task: recipe.TaskImageGen, Preset: preset, Draws: map[string]renderedMedia{
			key: {File: key + ".png", Request: testcheck.Must(json.Marshal(request))(t),
				Input: testutil.ArtifactID(t, artifact.KindFile, "asked"), Output: testutil.ArtifactID(t, artifact.KindOutput, "drawn")},
		}}
		testcheck.NoError(t, jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, renderFileMode))
		return directory
	}
	reference, preset := render("", "river", 7, red), render("apg", "river-apg", 7, blue)
	media, suitePath := t.TempDir(), filepath.Join(t.TempDir(), "pairwise.json")
	testcheck.NoError(t, writePairwise(reference, preset, media, suitePath))
	var suite evaluation.GeneratedAnswerSuite
	testcheck.NoError(t, jsonfile.DecodeStrict(suitePath, &suite))
	if len(suite.Cases) != 2 {
		t.Fatalf("one request paired into %d cases; want it asked in both orders", len(suite.Cases))
	}
	for _, testCase := range suite.Cases {
		_, named, question, _ := evaluation.ParseMediaPrompt(testCase.Prompt)
		composite := testcheck.Must(readPicture(media, named))(t)
		left, right := color.RGBAModel.Convert(composite.At(0, 0)), color.RGBAModel.Convert(composite.At(7, 1))
		presetLeft := testCase.Answers[0] == "A"
		if (left == blue) != presetLeft || (right == blue) == presetLeft || testCase.Group != "river" {
			t.Errorf("case %s, the preset's answer %s, holds %v left and %v right under group %s; want the preset's blue on its side, the reference's red on the other, grouped by the reference's key",
				testCase.Name, testCase.Answers[0], left, right, testCase.Group)
		}
		if !strings.Contains(question, "a river at dawn") {
			t.Errorf("case %s asks %q; want the request's prompt", testCase.Name, question)
		}
	}
	if err := writePairwise(reference, render("apg", "river-apg", 8, blue), t.TempDir(), filepath.Join(t.TempDir(), "other.json")); err == nil {
		t.Error("renders of two seeds were paired")
	}
	if _, err := os.Stat(filepath.Join(media, evaluation.PairwiseMedium("river", "A"))); err != nil {
		t.Errorf("the composite with the preset on the left is not where the suite names it: %v", err)
	}
}
