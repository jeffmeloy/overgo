package tokenizer

import (
	"strings"
	"testing"
	"unicode"
)

// stripMarks mirrors preprocessWPM's category-M deletion (wpm.go).
func stripMarks(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.Is(unicode.M, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// TestNFDParityStripAccents proves the stdlib nfdString, followed by the same
// mark-stripping preprocessWPM performs, is byte-identical to
// golang.org/x/text's norm.NFD followed by mark-stripping, over a broad Unicode
// corpus. The x/text reference is snapshotted in nfd_parity_fixtures_test.go, so
// this committed test needs no x/text at runtime.
func TestNFDParityStripAccents(t *testing.T) {
	for _, c := range nfdParityCases {
		got := stripMarks(nfdString(c.in))
		if got != c.want {
			t.Errorf("strip(nfdString(%q)) = %q, x/text reference = %q", c.in, got, c.want)
		}
	}
	if len(nfdParityCases) < 500 {
		t.Fatalf("parity corpus too small: %d", len(nfdParityCases))
	}
}

// TestNFDTableFullyExpanded verifies table entries are terminal: no value rune
// is itself a table key (recursive decomposition already applied).
func TestNFDTableFullyExpanded(t *testing.T) {
	for r, d := range nfdTable {
		for _, c := range d {
			if _, ok := nfdTable[c]; ok {
				t.Errorf("rune 0x%04X decomposes to 0x%04X which is itself decomposable", r, c)
			}
		}
	}
}
