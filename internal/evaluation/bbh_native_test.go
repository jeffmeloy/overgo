package evaluation

import (
	"slices"
	"testing"
)

// TestBBHNativeProfileRefusesChange holds the native BBH profile to its
// committed bytes: a changed profile is refused before any case is built. It
// reads no retained evidence, so the landing runs it; the protocol against
// the native oracle is the cycle's.
func TestBBHNativeProfileRefusesChange(t *testing.T) {
	t.Parallel()
	if _, err := decodeNativeBBH(bbhNativeProfileData); err != nil {
		t.Fatal(err)
	}
	changed := slices.Clone(bbhNativeProfileData)
	changed[len(changed)-1] = ' '
	if _, err := decodeNativeBBH(changed); err == nil {
		t.Fatal("changed native profile admitted")
	}
}
