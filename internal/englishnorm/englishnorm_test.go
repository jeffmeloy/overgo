package englishnorm

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// normalizerScript writes the fixture this test reads and the tables the
// package embeds, from the normalizer's own code. It is kept where Python is
// kept, on the machine that ran it and not in git: the fixture and the tables
// are the data.
const normalizerScript = "scripts/python_golden_references/whisper_english_normalizer.py"

// TestTextMatchesUpstream holds the port to the normalizer's own code, text
// for text and character for character: the texts of the speech round-trip
// evaluation, the references and the stored transcripts of the recognizers'
// held-out evaluation, a fifth of LibriSpeech test-clean's references, and
// texts written to try each rule of the number reader, the contractions and
// titles, and the removal of symbols and diacritics.
func TestTextMatchesUpstream(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(testutil.FixturePath(t, "whisper_english_normalizer.json"))
	if err != nil {
		t.Fatalf("the normalizer's own output is absent (%v); write it with %s", err, normalizerScript)
	}
	var fixture struct {
		Spellings string `json:"spellings"`
		Cases     []struct {
			Text    string `json:"text"`
			Written string `json:"written"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("the fixture holds no text")
	}
	if !strings.HasPrefix(fixture.Spellings, "none") {
		t.Fatalf("the fixture was written with a table of spellings (%s), and the port has none", fixture.Spellings)
	}
	changed, digits := 0, 0
	for _, reference := range fixture.Cases {
		written := Text(reference.Text)
		if written != reference.Written {
			t.Errorf("%q is written %q, and the normalizer's own code writes %q", reference.Text, written, reference.Written)
			continue
		}
		if strings.TrimSpace(written) != strings.ToLower(strings.TrimSpace(reference.Text)) {
			changed++
		}
		if strings.ContainsAny(written, "0123456789") && !strings.ContainsAny(reference.Text, "0123456789") {
			digits++
		}
	}
	if changed == 0 || digits == 0 {
		t.Fatalf("the normalizer changes %d of the fixture's texts beyond their case and writes digits into %d: the fixture tries too little", changed, digits)
	}
	t.Logf("%d texts are written as the normalizer's own code writes them; it changes %d beyond their case and writes digits into %d", len(fixture.Cases), changed, digits)
}

// TestTextWritesWhatWasSaidOneWay holds the reason the normalizer exists: a
// text and what a recognizer writes on hearing it read the same once both are
// normalized, where they do not read the same lowercased and stripped of
// punctuation.
func TestTextWritesWhatWasSaidOneWay(t *testing.T) {
	t.Parallel()
	for said, heard := range map[string]string{
		"It wasn't ten minutes.":                   "it was not 10 minutes",
		"Five hundred thousand throats.":           "500000 throats",
		"I called a third time, Mr. Conseil.":      "I called a 3rd time, Mister Conseil!",
		"The goat's spirit; let's go, we'll stay.": "the goat is spirit let us go we will stay",
		"Um, the café [laughter] is closed.":       "The cafe is closed",
		"Twenty dollars and fifty cents":           "$20.50",
		"nineteen eighty four":                     "1984",
	} {
		if written, counted := strings.Fields(Text(said)), strings.Fields(Text(heard)); strings.Join(written, " ") != strings.Join(counted, " ") {
			t.Errorf("%q is written %q and %q is written %q", said, written, heard, counted)
		}
	}
	if Text("") != "" {
		t.Errorf("an empty text is written %q", Text(""))
	}
}
