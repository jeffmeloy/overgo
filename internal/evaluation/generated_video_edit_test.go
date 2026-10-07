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

// TestGeneratedVideoEditAsksWhatChangedAndWhatStayed holds the suite to its
// construction: each picked clip is one the directory holds, with one instruction
// from the list; its one question offers the clip's own caption and one other, each
// with that change made and without it, and the answer is the clip's own with the
// change made, so a clip given back as it was and a changed clip of another scene
// are both wrong; a clip whose caption already shows a change is never picked; the
// same seed picks the same items; and each clip yields one case that names its
// edited clip and is answered by a letter.
func TestGeneratedVideoEditAsksWhatChangedAndWhatStayed(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 6, 6)
	items, err := PickVideoEdits(spec.Clips, spec.Captions, 5, 17)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := PickVideoEdits(spec.Clips, spec.Captions, 5, 17)
	if len(items) != 5 || fmt.Sprint(items) != fmt.Sprint(again) {
		t.Fatalf("the same seed picked %v and then %v", items, again)
	}
	videos := map[string]string{}
	for _, item := range items {
		videos[item.ID] = item.ID + ".gif"
		// The fixture writes a clip file for the first half of its sources only.
		if !strings.HasPrefix(item.ID, "src00") && !strings.HasPrefix(item.ID, "src01") && !strings.HasPrefix(item.ID, "src02") {
			t.Fatalf("item %s is not a clip the directory holds", item.ID)
		}
		made := slices.IndexFunc(videoChanges, func(change videoChange) bool { return change.instruction == item.Instruction })
		if made < 0 {
			t.Fatalf("item %s is given %q, not an instruction of the list", item.ID, item.Instruction)
		}
		change := videoChanges[made]
		options := item.Question.Options
		if len(options) != editChoices || options[item.Question.Answer] != item.Caption+" "+change.statement ||
			!slices.Contains(options, item.Caption+" "+change.absence) || len(slices.Compact(slices.Sorted(slices.Values(options)))) != editChoices {
			t.Fatalf("item %s question = %+v, want its caption with %q made as the answer and without it as another, four statements in all", item.ID, item.Question, item.Instruction)
		}
		made, own := 0, 0
		for _, option := range options {
			if strings.HasSuffix(option, " "+change.statement) {
				made++
			}
			if strings.HasPrefix(option, item.Caption+" ") {
				own++
			}
		}
		if made != 2 || own != 2 {
			t.Fatalf("item %s offers the change made in %d statements and its own caption in %d, want two of each", item.ID, made, own)
		}
	}
	suite, err := BuildGeneratedVideoEdit(items, videos, "judge-a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil || suite.Judge != "judge-a" || len(suite.Cases) != len(items) {
		t.Fatalf("suite = %d cases judged by %q (%v), want one a clip that compile", len(suite.Cases), suite.Judge, err)
	}
	kind, name, question, ok := ParseMediaPrompt(suite.Cases[0].Prompt)
	if !ok || kind != MediaVideo || name != items[0].ID+".gif" || !strings.Contains(question, items[0].Caption) ||
		suite.Cases[0].Answers[0] != string(captionLetters[items[0].Question.Answer]) || !strings.HasSuffix(suite.Cases[0].Name, "/"+items[0].ID) {
		t.Fatalf("first case = %s answering %v", suite.Cases[0].Name, suite.Cases[0].Answers)
	}
	if _, err := BuildGeneratedVideoEdit(items, map[string]string{}, "judge-a"); err == nil {
		t.Error("a suite was built for clips with no edited clip")
	}
	if _, err := BuildGeneratedVideoEdit(items, videos, ""); err == nil {
		t.Error("a suite was built with no judge named")
	}
	if _, err := PickVideoEdits(spec.Clips, spec.Captions, 500, 17); err == nil {
		t.Error("more clips than the directory holds were picked")
	}
}

// TestVideoEditsPassOverClipsThatShowAChangeAlready holds the pick to clips an
// instruction can change: a caption that already says snow, night or a cartoon
// leaves the judge nothing to tell an edit by.
func TestVideoEditsPassOverClipsThatShowAChangeAlready(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	var rows []captionRow
	for index, caption := range []string{
		"A man walks a dog along a beach.", "Children play football in a park.", "A chef slices onions in a kitchen.",
		"Cars drive through heavy snow.", "A street at night with neon signs.", "A cartoon rabbit eats a carrot.",
		"A woman paints a wall blue.", "Two cats sleep on a sofa.", "A train crosses a bridge in thick fog.",
		"A boy rides a bicycle down a hill.", "A pencil sketch of a horse comes to life.", "Rain falls on a tin roof.",
	} {
		vid := fmt.Sprintf("clip%02d%s0001", index, sceneMarker)
		rows = append(rows, captionRow{Vid: vid, Caption: caption})
		if err := os.WriteFile(filepath.Join(directory, vid+clipExtension), []byte("clip"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	data, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	captions := filepath.Join(t.TempDir(), "captions.json")
	if err := os.WriteFile(captions, data, 0o600); err != nil {
		t.Fatal(err)
	}
	// Six of the twelve captions say snow, night, a cartoon, fog, a sketch or rain.
	const unchanged = 6
	items, err := PickVideoEdits(directory, captions, unchanged, 17)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range items {
		for _, word := range []string{"snow", "night", "cartoon", "fog", "sketch", "Rain"} {
			if strings.Contains(item.Caption, word) {
				t.Fatalf("picked %q, which shows a change already", item.Caption)
			}
		}
	}
	if _, err := PickVideoEdits(directory, captions, unchanged+1, 17); err == nil {
		t.Fatal("a clip that shows a change already was counted among those to edit")
	}
	for _, change := range videoChanges {
		if change.instruction == "" || change.statement == "" || change.absence == "" || change.statement == change.absence || len(change.words) == 0 {
			t.Errorf("change %+v is incomplete", change)
		}
	}
}
