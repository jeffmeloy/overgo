package evaluation

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestLocalEditSuiteChecksTheChangeAndTheRestApart holds the measure of what
// an editor that preserves what it is not told to change is for. A clip is
// picked only when its caption names one worn thing by one colour and does not
// already hold the colour it would be changed to; the instruction names the
// thing by its colour and the colour it is to have. The one question offers
// four captions that tell the change and the rest apart: the clip's own with
// the new colour in the old one's place, which alone is the right answer; its
// own as it is, which is true of a clip given back unchanged; and two other
// videos' captions as distractors, never another scene of the clip's own source video.
// Caption matching does not measure preservation of unmentioned scene details.
// The same seed picks the same clips with the same questions.
//
// A caption that names no worn thing by a colour, that names two, that names
// one by two colours or a shade, that already holds the new colour, or whose
// colour takes "an" is passed over.
func TestLocalEditSuiteChecksTheChangeAndTheRestApart(t *testing.T) {
	t.Parallel()
	for caption, want := range map[string][]string{
		"A man wearing a blue shirt repairs a car door.":                         {"blue", "shirt", "A man wearing a yellow shirt repairs a car door."},
		"A woman in a Black Jacket walks. The black jacket is zipped.":           {"black", "jacket", "A woman in a yellow Jacket walks. The yellow jacket is zipped."},
		"A boy in a gray t-shirt kicks a ball on green grass.":                   {"gray", "t-shirt", "A boy in a red t-shirt kicks a ball on green grass."},
		"A man in a gray shirt talks to a man in a blue shirt.":                  nil,
		"A girl in a blue dress holds a yellow balloon.":                         nil,
		"A woman wearing a pink tank top walks through a parking lot.":           nil,
		"A man in an orange jacket climbs a ladder.":                             nil,
		"A man in a black and white jacket drives a car.":                        nil,
		"A woman in a light blue dress sits by a window.":                        nil,
		"Two cats sleep on a sofa.":                                              nil,
		"A woman paints a wall blue while wearing a hat.":                        nil,
		"A cook in a white coat stirs a pot beside a red kettle and a white cup": nil,
	} {
		colour, thing, found := wornColour(caption)
		if found != (want != nil) {
			t.Errorf("%q names a worn thing by a colour: %t, want %t", caption, found, want != nil)
			continue
		}
		if found && !slices.Equal([]string{colour, thing, recoloured(caption, colour, thing, recolours[colour])}, want) {
			t.Errorf("%q names %q %q and is recoloured %q, want %q", caption, colour, thing, recoloured(caption, colour, thing, recolours[colour]), want)
		}
	}

	directory := t.TempDir()
	var rows []captionRow
	for index, caption := range []string{
		"A man wearing a blue shirt repairs a car door in a garage.", "A woman in a black jacket walks a dog along a beach.",
		"A boy in a green cap rides a bicycle down a hill.", "A chef in a white coat slices onions in a kitchen.",
		"A girl in a pink dress waters flowers in a garden.", "A man in a brown hat plays a guitar on a porch.",
		"Children play football in a park.", "Two cats sleep on a sofa.", "A train crosses a bridge over a river.",
		"A man in a gray shirt talks to a man in a blue shirt.", "A woman in an orange jacket climbs a ladder.", "A girl in a blue dress holds a yellow balloon.",
	} {
		vid := fmt.Sprintf("clip%02d%s0001", index, sceneMarker)
		rows = append(rows, captionRow{Vid: vid, Caption: caption})
		if err := os.WriteFile(filepath.Join(directory, vid+clipExtension), []byte("clip"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// Another scene of the first clip's source video, which shows the same man
	// in the same garage: no question may offer it.
	sibling := "The same man in the garage closes the car door and wipes his hands."
	rows = append(rows, captionRow{Vid: "clip00" + sceneMarker + "0002", Caption: sibling})
	data, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	captions := filepath.Join(t.TempDir(), "captions.json")
	if err := os.WriteFile(captions, data, 0o600); err != nil {
		t.Fatal(err)
	}
	// Six of the twelve clips' captions name one worn thing by a colour an edit can change.
	const editable = 6
	items, err := PickLocalVideoEdits(directory, captions, editable, 17)
	if err != nil {
		t.Fatal(err)
	}
	again, err := PickLocalVideoEdits(directory, captions, editable, 17)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != editable {
		t.Fatalf("%d clips were picked, want %d", len(items), editable)
	}
	for index, item := range items {
		if item.ID != again[index].ID || item.Instruction != again[index].Instruction || !slices.Equal(item.Question.Options, again[index].Question.Options) {
			t.Fatalf("the same seed picked %+v and then %+v", item, again[index])
		}
		colour, thing, found := wornColour(item.Caption)
		if !found {
			t.Fatalf("picked %q, which names no one worn thing by a colour", item.Caption)
		}
		changed := recolours[colour]
		if want := "Change the " + colour + " " + thing + " to a " + changed + " " + thing + "."; item.Instruction != want {
			t.Fatalf("%q is told %q, want %q", item.Caption, item.Instruction, want)
		}
		options := slices.Clone(item.Question.Options)
		right := recoloured(item.Caption, colour, thing, changed)
		if len(options) != editChoices || options[item.Question.Answer] != right || right == item.Caption || !slices.Contains(options, item.Caption) || slices.Contains(options, sibling) {
			t.Fatalf("%q is asked %q with answer %d, want its caption with the new colour right, its caption as it is, and two other videos'", item.Caption, options, item.Question.Answer)
		}
		slices.Sort(options)
		if len(slices.Compact(options)) != editChoices {
			t.Fatalf("%q is offered a caption twice: %q", item.Caption, item.Question.Options)
		}
	}
	if _, err := PickLocalVideoEdits(directory, captions, editable+1, 17); err == nil {
		t.Fatal("a clip whose caption names no one worn thing by a colour was counted among those to edit")
	}

	videos := map[string]string{}
	for _, item := range items {
		videos[item.ID] = item.ID + ".gif"
	}
	suite, err := BuildGeneratedVideoLocalEdit(items, videos, "a judge")
	if err != nil {
		t.Fatal(err)
	}
	first := suite.Cases[0]
	if len(suite.Cases) != editable || suite.Source != localEditSource || first.Name != localEditSource+"/"+items[0].ID ||
		!slices.Equal(first.Answers, []string{string(captionLetters[items[0].Question.Answer])}) || !strings.Contains(first.Prompt, items[0].ID+".gif") {
		t.Fatalf("the suite holds %d cases of %s, the first %+v", len(suite.Cases), suite.Source, first)
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil {
		t.Fatalf("the suite does not compile: %v", err)
	}
	delete(videos, items[0].ID)
	if _, err := BuildGeneratedVideoLocalEdit(items, videos, "a judge"); err == nil {
		t.Fatal("a suite was built with a clip that was not edited")
	}
}
