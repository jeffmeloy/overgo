package tokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
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
	for r, d := range nfdTable() {
		for _, c := range d {
			if _, ok := nfdTable()[c]; ok {
				t.Errorf("rune 0x%04X decomposes to 0x%04X which is itself decomposable", r, c)
			}
		}
	}
}

// TestNFDTableIsPacked proves the packed table and the Hangul arithmetic
// decompose every scalar value exactly as the generated map literal they
// replaced: the digest covers each decomposable rune and its decomposition in
// rune order, and the map literal's 13,233 entries, 11,172 of them Hangul
// syllables, hashed to mapLiteralDigest. The table records no Hangul syllable.
func TestNFDTableIsPacked(t *testing.T) {
	const mapLiteralDigest = "a6af2e6c9f96801811848bdf2147ce9ced3570c8a859e101606dfd763bfaec6f"
	digest := sha256.New()
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if values := appendNFD(nil, r); len(values) != 1 || values[0] != r {
			fmt.Fprintf(digest, "%X %X\n", r, values)
		}
	}
	if got := hex.EncodeToString(digest.Sum(nil)); got != mapLiteralDigest {
		t.Errorf("decompositions hash to %s, the replaced map literal's to %s", got, mapLiteralDigest)
	}
	for r := range nfdTable() {
		if r >= hangulSBase && r < hangulSBase+hangulSCount {
			t.Errorf("the packed table records Hangul syllable %U, which decomposes by arithmetic", r)
		}
	}
}
