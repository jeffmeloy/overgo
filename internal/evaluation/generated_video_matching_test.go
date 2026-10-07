package evaluation

import (
	"fmt"
	"strings"
	"testing"
)

// TestGeneratedVideoMatchingAsksWhichCaptionAClipWasDrawnFrom holds the suite to
// its construction: each picked caption has two questions of four distinct options of
// which only one is its own; the distractors are captions of other scenes of the same
// source video while it has scenes left, a round never repeating the round before's, and
// captions of other videos after; the clip is named by its file; and the answer is the
// letter of the caption the clip was drawn from.
func TestGeneratedVideoMatchingAsksWhichCaptionAClipWasDrawnFrom(t *testing.T) {
	t.Parallel()
	// Six scenes a source video: five siblings, so the first question takes three and the
	// second the other two and one caption of another video.
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
		source, _, _ := strings.Cut(item.ID, sceneMarker)
		sibling := func(option string) bool { return strings.HasPrefix(option, "caption of "+source+sceneMarker) }
		if len(item.Questions) != generatedVideoRounds {
			t.Fatalf("item %s has %d questions, want %d", item.ID, len(item.Questions), generatedVideoRounds)
		}
		offered := map[string]int{}
		for round, question := range item.Questions {
			if len(question.Options) != captionChoices || question.Options[question.Answer] != item.Caption {
				t.Fatalf("item %s question %d does not offer its own caption among %d: %+v", item.ID, round, captionChoices, question)
			}
			siblings := 0
			for _, option := range question.Options {
				offered[option]++
				if option != item.Caption && sibling(option) {
					siblings++
				}
			}
			if want := []int{3, 2}[round]; siblings != want {
				t.Fatalf("item %s question %d offers %d sibling scenes, want %d", item.ID, round, siblings, want)
			}
		}
		for option, times := range offered {
			if option != item.Caption && times != 1 {
				t.Fatalf("item %s offers the distractor %q %d times across its questions", item.ID, option, times)
			}
		}
	}
	suite, err := BuildGeneratedVideoMatching(items, videos, "judge-a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil || suite.Judge != "judge-a" || len(suite.Cases) != 5*generatedVideoRounds {
		t.Fatalf("suite = %d cases judged by %q (%v), want %d judged by judge-a that compile", len(suite.Cases), suite.Judge, err, 5*generatedVideoRounds)
	}
	kind, name, _, ok := ParseMediaPrompt(suite.Cases[1].Prompt)
	if !ok || kind != MediaVideo || name != items[0].ID+".gif" || suite.Cases[1].Answers[0] != string(captionLetters[items[0].Questions[1].Answer]) ||
		!strings.HasSuffix(suite.Cases[1].Name, items[0].ID+"/1") {
		t.Fatalf("second case = %s %q answering %v", suite.Cases[1].Name, suite.Cases[1].Prompt, suite.Cases[1].Answers)
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
