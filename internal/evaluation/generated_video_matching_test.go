package evaluation

import (
	"fmt"
	"strings"
	"testing"
)

// TestGeneratedVideoMatchingAsksWhichCaptionAClipWasDrawnFrom holds the suite to
// its construction: each picked caption has four distinct options of which only one
// is its own, from other source videos, the clip is named by its file, and the
// answer is the letter of the caption the clip was drawn from.
func TestGeneratedVideoMatchingAsksWhichCaptionAClipWasDrawnFrom(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 6, 6)
	items, err := PickVideoCaptions(spec.Captions, 5, 17)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := PickVideoCaptions(spec.Captions, 5, 17)
	if len(items) != 5 || fmt.Sprint(items) != fmt.Sprint(again) {
		t.Fatalf("the same seed picked %v and then %v", items, again)
	}
	videos := map[string]string{}
	for _, item := range items {
		videos[item.ID] = item.ID + ".gif"
		if len(item.Options) != captionChoices || item.Options[item.Answer] != item.Caption {
			t.Fatalf("item %+v does not offer its own caption among %d", item, captionChoices)
		}
		seen := map[string]bool{}
		for _, option := range item.Options {
			if seen[option] {
				t.Fatalf("item %s offers %q twice", item.ID, option)
			}
			seen[option] = true
		}
		source, _, _ := strings.Cut(item.ID, sceneMarker)
		for _, option := range item.Options {
			if option != item.Caption && strings.HasPrefix(option, "caption of "+source+sceneMarker) {
				t.Fatalf("item %s offers %q, a caption of its own source video", item.ID, option)
			}
		}
	}
	suite, err := BuildGeneratedVideoMatching(items, videos, "judge-a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil || suite.Judge != "judge-a" || len(suite.Cases) != 5 {
		t.Fatalf("suite = %d cases judged by %q (%v), want 5 judged by judge-a that compile", len(suite.Cases), suite.Judge, err)
	}
	kind, name, _, ok := ParseMediaPrompt(suite.Cases[0].Prompt)
	if !ok || kind != MediaVideo || name != items[0].ID+".gif" || suite.Cases[0].Answers[0] != string(captionLetters[items[0].Answer]) {
		t.Fatalf("first case = %q answering %v", suite.Cases[0].Prompt, suite.Cases[0].Answers)
	}
	if _, err := BuildGeneratedVideoMatching(items, map[string]string{}, "judge-a"); err == nil {
		t.Error("a suite was built for captions with no clip")
	}
	if _, err := BuildGeneratedVideoMatching(items, videos, ""); err == nil {
		t.Error("a suite was built with no judge named")
	}
	if _, err := PickVideoCaptions(spec.Captions, 500, 17); err == nil {
		t.Error("more captions than the file holds were picked")
	}
}
