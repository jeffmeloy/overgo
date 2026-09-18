package artifact

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

// TestValidHexDigest proves the shared digest predicate: a full-length hex
// sha256 in either case is valid, and anything shorter, longer, non-hex, or
// empty is not. Manifests, records and closure evidence share this instead of
// each re-deriving the decode-and-length test.
func TestValidHexDigest(t *testing.T) {
	sum := sha256.Sum256([]byte("overgo"))
	lower := hex.EncodeToString(sum[:])
	valid := []string{lower, strings.ToUpper(lower)}
	for _, value := range valid {
		if !ValidHexDigest(value) {
			t.Fatalf("valid digest rejected: %q", value)
		}
	}
	invalid := []string{
		"",
		lower[:len(lower)-2],         // one byte short
		lower + "ab",                 // one byte long
		lower[:len(lower)-1] + "g",   // non-hex rune
		hex.EncodeToString(sum[:16]), // half length
		"sha256:" + lower,            // carries a kind prefix
	}
	for _, value := range invalid {
		if ValidHexDigest(value) {
			t.Fatalf("invalid digest accepted: %q", value)
		}
	}
}
