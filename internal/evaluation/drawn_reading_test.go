package evaluation

import (
	"math"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestDrawnTextScoresTheCharactersReadBack holds the character score to what it counts. Each case of a drawn-text suite
// names the text it asks about by its answer's letter, and that text is the brief's own. A picture whose reading holds
// every asked text, however its rows break a line, scores 1; one whose reading has one character wrong scores below it,
// by that one character of all those asked; a judged reading of the suite records their mean. Readings that miss a
// picture the suite asks about, or name no reader, are refused.
func TestDrawnTextScoresTheCharactersReadBack(t *testing.T) {
	t.Parallel()
	items := testcheck.Must(PickDrawnText(2, 7))(t)
	images := map[string]string{}
	for _, item := range items {
		images[item.ID] = item.ID + ".png"
	}
	suite := testcheck.Must(BuildDrawnText(items, images, "judge"))(t)
	// Readings are kept by picture, since a suite built before cases had groups asks about many pictures under none.
	for index := range suite.Cases {
		suite.Cases[index].Group = ""
	}
	asked := testcheck.Must(askedTexts(suite.Cases))(t)
	for _, item := range items {
		for index, question := range item.Questions {
			if asked[images[item.ID]][index] != question.Answer {
				t.Fatalf("case %d of %s asks about %q, want the brief's %q", index, item.ID, asked[images[item.ID]][index], question.Answer)
			}
		}
	}
	clean, flawed := images[items[0].ID], images[items[1].ID]
	// A reader breaks a picture's text into rows where the picture breaks it: the clean reading puts each word of the
	// title on a row of its own.
	cleanRead := strings.Join(append(strings.Fields(asked[clean][0]), asked[clean][1:]...), "\n")
	title := []rune(asked[flawed][0])
	// A sign no title begins with, so folding to lower case leaves it wrong.
	title[0] = '#'
	flawedRead := strings.Join(append([]string{string(title)}, asked[flawed][1:]...), "\n")
	var characters int
	for _, text := range asked[flawed] {
		characters += len([]rune(strings.Join(strings.Fields(text), " ")))
	}
	whole, short := characterShare(asked[clean], cleanRead), characterShare(asked[flawed], flawedRead)
	t.Logf("a reading of every asked text scores %.4f; one with a character wrong of %d scores %.4f", whole, characters, short)
	if whole != 1 || math.Abs(short-float64(characters-1)/float64(characters)) > 1e-12 {
		t.Fatalf("the clean reading scores %v and the one with a character wrong %v, want 1 and %d of %d", whole, short, characters-1, characters)
	}

	reader := MediaRuntime{
		Model: testutil.ArtifactID(t, artifact.KindModel, "reader"), Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "inference"),
		Projection: testutil.ArtifactID(t, artifact.KindRecipe, "projection"),
	}
	suite.Readings = &PictureReadings{Reader: reader, Read: map[string]string{clean: cleanRead, flawed: flawedRead}}
	testcheck.Must(CompileGeneratedAnswer(suite))(t)
	recorded := map[string]float64{}
	for _, metric := range generatedAnswerMetrics(suite, 1, 1) {
		recorded[metric.Name] = metric.Value
	}
	if math.Abs(recorded[CharacterMetric]-(whole+short)/2) > 1e-12 {
		t.Fatalf("a judged reading records %v, want the pictures' mean share %v", recorded, (whole+short)/2)
	}
	for name, readings := range map[string]PictureReadings{
		"a picture unread": {Reader: reader, Read: map[string]string{clean: cleanRead}},
		"no reader":        {Read: map[string]string{clean: cleanRead, flawed: flawedRead}},
	} {
		suite.Readings = &readings
		if _, err := CompileGeneratedAnswer(suite); err == nil {
			t.Errorf("readings with %s were admitted", name)
		}
	}
}
