package tokenizer

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"unicode/utf8"
)

// The class intervals come from the pinned Hugging Face regex engine, queried
// over every valid Unicode scalar before the native scanner was implemented.
func TestRxBrainUnicodeClassesMatchReference(t *testing.T) {
	const fixtureSHA = "7db1e9a46d0478d3ae4c3d77276cec9c636a8fdb041ff30d2aa6d701d00c0e66"
	const sourceSHA = "ae5ca95eeb8e9a8774513a996e4e820a05c27db32966aa8e071c10828402cb73"
	raw, err := os.ReadFile(filepath.Join("testdata", "rxbrain-hf-unicode-classes.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != fixtureSHA {
		t.Fatalf("Unicode class fixture drifted: %s", got)
	}
	var fixture struct {
		SourceSHA string               `json:"source_sha256"`
		Version   string               `json:"reference_version"`
		Classes   map[string][][2]rune `json:"classes"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.SourceSHA != sourceSHA || fixture.Version != "0.22.2" || len(fixture.Classes) != 6 {
		t.Fatal("Unicode class provenance or cardinality changed")
	}
	for _, check := range []struct {
		name     string
		contains func(rune) bool
	}{
		{"L", rxLetter}, {"M", rxMark}, {"N", rxNumber},
		{"P", rxPunct}, {"S", rxSymbol}, {"space", rxWhite},
	} {
		t.Run(check.name, func(t *testing.T) {
			spans := fixture.Classes[check.name]
			if len(spans) == 0 {
				t.Fatal("empty class")
			}
			index := 0
			for r := rune(0); r <= utf8.MaxRune; r++ {
				if r >= 0xD800 && r <= 0xDFFF {
					continue
				}
				for index < len(spans) && spans[index][1] < r {
					index++
				}
				want := index < len(spans) && spans[index][0] <= r
				if got := check.contains(r); got != want {
					t.Fatalf("U+%04X in %s: got=%t want=%t", r, check.name, got, want)
				}
			}
		})
	}
}
