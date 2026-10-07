package evaluation

import (
	"fmt"
	"strings"
	"testing"
)

// TestWikiTextPassagesSpanTheSplitWithoutOverlap holds the suite to its
// construction: sixty-four passages, each starting on a line of the split and
// ending on a word, none overlapping the one before, counted in characters so
// that text outside ASCII neither splits a character nor shifts a passage;
// and a split too short for them is refused rather than cut into passages
// that share text.
func TestWikiTextPassagesSpanTheSplitWithoutOverlap(t *testing.T) {
	t.Parallel()
	var lines []string
	for line := range 4000 {
		lines = append(lines, fmt.Sprintf(" line %d holds a café, a naïve piñata and some filler words to read \n", line))
	}
	built, count, err := BuildWikiTextPassages(lines)
	if err != nil {
		t.Fatal(err)
	}
	suite := built.(sequenceScoringSuite)
	if count != wikitextPassages || len(suite.Cases) != wikitextPassages || suite.Kind != sequenceScoringKind {
		t.Fatalf("suite = %d cases of kind %q, want %d", len(suite.Cases), suite.Kind, wikitextPassages)
	}
	text := strings.Join(lines, "")
	last := 0
	for index, passage := range suite.Cases {
		at := strings.Index(text[last:], passage.Text)
		if at < 0 {
			t.Fatalf("passage %d is not text of the split after the passage before it", index)
		}
		start := last + at
		if start > 0 && text[start-1] != '\n' {
			t.Fatalf("passage %d starts mid-line", index)
		}
		if after := text[start+len(passage.Text):]; !strings.HasPrefix(after, " ") {
			t.Fatalf("passage %d does not end on a word", index)
		}
		if characters := len([]rune(passage.Text)); characters >= wikitextPassageCharacters || characters < wikitextPassageCharacters/2 {
			t.Fatalf("passage %d holds %d characters, want just under %d", index, characters, wikitextPassageCharacters)
		}
		last = start + len(passage.Text)
	}
	if again, _, _ := BuildWikiTextPassages(lines); fmt.Sprint(again) != fmt.Sprint(built) {
		t.Fatal("the same split built two different suites")
	}
	if _, _, err := BuildWikiTextPassages(lines[:40]); err == nil {
		t.Fatal("a split too short for the passages built a suite")
	}
}
