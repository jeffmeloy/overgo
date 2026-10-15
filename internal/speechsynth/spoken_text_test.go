package speechsynth

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/testutil"
)

// spokenTextScript writes the fixture this test reads, from the model's own
// code. It is kept where Python is kept, on the machine that ran it and not
// in git: the fixture is the data.
const spokenTextScript = "scripts/python_golden_references/pockettts_spoken_text.py"

// TestSpokenTextMatchesUpstream holds the first stretch of the path a speech
// request takes, from its text to the token ids the model reads and to how
// long it speaks on after the end, to the model's own code: its preparation of
// a text, its tokenizer, and its rule for the frames generated after the end
// of speech is first seen. The texts are every text of the round-trip
// evaluation, and texts that try the preparation: short ones, lower-case
// openings, endings with and without punctuation, line breaks and runs of
// spaces, digits, and letters that are not ASCII.
//
// A served request used to tokenize the text as given and stop at the frame
// the end was first seen in, where that code generates three frames more, and
// five after a text of four words or fewer.
func TestSpokenTextMatchesUpstream(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(testutil.FixturePath(t, "pockettts_spoken_text.json"))
	if err != nil {
		t.Fatalf("the upstream spoken texts are absent (%v); write them with %s", err, spokenTextScript)
	}
	var fixture struct {
		Cases []struct {
			Text           string `json:"text"`
			Spoken         string `json:"spoken"`
			IDs            []int  `json:"ids"`
			FramesAfterEnd int    `json:"frames_after_end"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("the fixture holds no text")
	}
	short, long := 0, 0
	for _, reference := range fixture.Cases {
		spoken, frames, err := spokenText(reference.Text)
		if err != nil {
			t.Fatal(err)
		}
		if spoken != reference.Spoken || frames != reference.FramesAfterEnd {
			t.Errorf("%q is prepared as %q with %d frames after the end, and the model's own code prepares %q with %d", reference.Text, spoken, frames, reference.Spoken, reference.FramesAfterEnd)
			continue
		}
		if frames == shortTextGuess+framesBeyondGuess {
			short++
		} else {
			long++
		}
	}
	if short == 0 || long == 0 {
		t.Fatalf("the fixture holds %d short texts and %d longer ones, and both rules need a text", short, long)
	}
	if _, _, err := spokenText(" \n "); err == nil {
		t.Error("a text of spaces was prepared to be spoken")
	}
	t.Logf("%d texts, %d of them short: preparation, post-EOS budget and blank refusal checked before tokenizer admission", len(fixture.Cases), short)
	t.Run("tokenizer", func(t *testing.T) {
		tokenizer, err := LoadTokenizer(filepath.Join(artifactDir(t), "tokenizer.model"))
		if err != nil {
			t.Skipf("UNAVAILABLE: pocket-tts tokenizer unavailable: %v; parity NOT verified", err)
		}
		for _, reference := range fixture.Cases {
			ids, err := tokenizer.Encode(reference.Spoken)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(ids, reference.IDs) {
				t.Errorf("%q is read as %v, and the model's own tokenizer reads %v", reference.Spoken, ids, reference.IDs)
			}
		}
	})

}
