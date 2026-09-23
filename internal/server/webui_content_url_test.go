package server

import (
	"strings"
	"testing"
)

// TestArtifactContentURLHasOneOwner keeps the artifact content route in
// boot.js's contentURL: eight modules once built it by hand, and a gallery
// link that bypassed the shared client opened the bearer route unauthenticated.
func TestArtifactContentURLHasOneOwner(t *testing.T) {
	t.Parallel()
	for name, source := range webuiJavaScript(t) {
		if name != "boot.js" && strings.Contains(source, "/artifacts/content?") {
			t.Errorf("%s builds an artifact content URL; call overgo.contentURL", name)
		}
	}
}
