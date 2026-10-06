package evaluation

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// captionFixture writes clips and a caption file: sources of scenes each, the
// first source's scenes the only ones with clips on disk.
func captionFixture(t *testing.T, sources, scenes int) CaptionMatchingSpec {
	t.Helper()
	directory := t.TempDir()
	var rows []captionRow
	for source := range sources {
		for scene := range scenes {
			vid := fmt.Sprintf("src%02d%s%04d", source, sceneMarker, scene)
			rows = append(rows, captionRow{Vid: vid, Caption: "caption of " + vid})
			if source < sources/2 {
				if err := os.WriteFile(filepath.Join(directory, vid+clipExtension), []byte("clip"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
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
	return CaptionMatchingSpec{Clips: directory, Captions: captions, Cases: 5, Seed: 17}
}

// TestCaptionMatchingAnswersAreTheTrueCaptionsLetter holds each case to its
// construction: four distinct lettered captions, the answer the letter of the
// caption of the clip the prompt names, and the distractors from the same
// source video when it has scenes enough.
func TestCaptionMatchingAnswersAreTheTrueCaptionsLetter(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 4, 6)
	suite, err := BuildCaptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	if suite.Kind != GeneratedAnswerKind || len(suite.Cases) != spec.Cases {
		t.Fatalf("suite = %s with %d cases, want %s with %d", suite.Kind, len(suite.Cases), GeneratedAnswerKind, spec.Cases)
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil {
		t.Fatalf("the suite does not compile: %v", err)
	}
	for _, testCase := range suite.Cases {
		kind, name, question, ok := ParseMediaPrompt(testCase.Prompt)
		if !ok || kind != MediaVideo || !strings.HasSuffix(name, clipExtension) {
			t.Fatalf("case %s names no clip: %q", testCase.Name, testCase.Prompt)
		}
		vid := strings.TrimSuffix(name, clipExtension)
		source, _, _ := strings.Cut(vid, sceneMarker)
		letters := map[string]string{}
		for line := range strings.SplitSeq(question, "\n") {
			if letter, caption, listed := strings.Cut(line, ". "); listed && len(letter) == 1 {
				letters[letter] = caption
			}
		}
		if len(letters) != captionChoices || letters[testCase.Answers[0]] != "caption of "+vid {
			t.Fatalf("case %s: answer %v does not letter the caption of %s among %v", testCase.Name, testCase.Answers, vid, letters)
		}
		for _, caption := range letters {
			if !strings.HasPrefix(caption, "caption of "+source) {
				t.Errorf("case %s offers %q, a caption of another source video, though %s has scenes enough", testCase.Name, caption, source)
			}
		}
	}
}

// TestCaptionMatchingIsReproducible holds the suite to its spec: the same spec
// builds the same cases, and another seed picks others.
func TestCaptionMatchingIsReproducible(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 6, 6)
	first, err := BuildCaptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	again, err := BuildCaptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	spec.Seed++
	other, err := BuildCaptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, _ := json.Marshal(first)
	againJSON, _ := json.Marshal(again)
	otherJSON, _ := json.Marshal(other)
	if string(firstJSON) != string(againJSON) || string(firstJSON) == string(otherJSON) {
		t.Fatal("the same spec must build the same suite, and another seed another")
	}
}

// TestCaptionMatchingRefusesWhatItCannotBuild holds the builder to what the
// directory and the caption file hold.
func TestCaptionMatchingRefusesWhatItCannotBuild(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 4, 6)
	for name, change := range map[string]func(*CaptionMatchingSpec){
		"no cases":              func(spec *CaptionMatchingSpec) { spec.Cases = 0 },
		"more cases than clips": func(spec *CaptionMatchingSpec) { spec.Cases = 1000 },
		"no clip directory":     func(spec *CaptionMatchingSpec) { spec.Clips = filepath.Join(spec.Clips, "absent") },
		"no caption file":       func(spec *CaptionMatchingSpec) { spec.Captions = filepath.Join(spec.Clips, "absent.json") },
	} {
		broken := spec
		change(&broken)
		if _, err := BuildCaptionMatching(broken); err == nil {
			t.Errorf("a suite was built with %s", name)
		}
	}
}
