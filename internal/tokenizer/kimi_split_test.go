package tokenizer

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestKimiDeclaredSplit(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "kimi-declared-split.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Source          string    `json:"source"`
		SourceSHA256    string    `json:"source_sha256"`
		PatternSHA256   string    `json:"pattern_sha256"`
		HanSourceSHA256 string    `json:"han_source_sha256"`
		HanRanges       [][2]rune `json:"han_ranges"`
		Pattern         string    `json:"pattern"`
		Cases           []struct {
			Input  string   `json:"input"`
			Pieces []string `json:"pieces"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Source != "colibri/c/tests/tok_kimi_tiny.json" ||
		fixture.SourceSHA256 != "99b5ff4e6cc9f14bcdc9c4e1d854c35d4afc087af8e825f3264b1ebbc10132eb" ||
		fixture.PatternSHA256 != "de5781783b193d5ccf5b1b28edfa70fa816ce78d54603fdc422cfd8d4ea4411f" ||
		fmt.Sprintf("%x", sha256.Sum256([]byte(fixture.Pattern))) != fixture.PatternSHA256 || len(fixture.Cases) != 13 || fixture.HanSourceSHA256 != "15acd2c92e4aec4db416bc19e195dd9c77ffae5feb0d104afd0a5dd637a0611f" || !slices.Equal(fixture.HanRanges, kimiHanRanges[:]) {
		t.Fatal("Kimi declaration provenance or test cardinality changed")
	}
	for _, span := range fixture.HanRanges {
		if !kimiHan(span[0]) || !kimiHan(span[1]) || kimiHan(span[0]-1) || kimiHan(span[1]+1) {
			t.Fatalf("Han range boundary changed: %#v", span)
		}
	}
	split, complete, err := CompileBPESplit(fixture.Pattern)
	if err != nil || complete {
		t.Fatalf("Kimi declaration or coverage claim wrong: complete=%t error=%v", complete, err)
	}
	for _, sample := range fixture.Cases {
		got := split(sample.Input)
		if !slices.Equal(got, sample.Pieces) {
			t.Errorf("split %q = %q, want %q", sample.Input, got, sample.Pieces)
		}
		if strings.Join(got, "") != sample.Input {
			t.Errorf("split %q lost or added bytes", sample.Input)
		}
	}
	changed := strings.Replace(fixture.Pattern, `\p{Han}`, `\p{L}`, 1)
	if _, _, err := CompileBPESplit(changed); err == nil {
		t.Fatal("altered Kimi regex acquired the exact scanner")
	}
}
