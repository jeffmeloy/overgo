package storepolicy

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestIdentitiesReadEmbeddedIDs holds the identity scan to the syntax
// artifact.ParseID owns: an identity inside prose, JSON or after a leading
// digit is found once, and a truncated or unknown-kind one is not.
func TestIdentitiesReadEmbeddedIDs(t *testing.T) {
	t.Parallel()
	digest := strings.Repeat("ab", 32)
	evidence := testcheck.Must(artifact.ParseID("evidence:sha256:" + digest))(t)
	recipe := testcheck.Must(artifact.ParseID("recipe:sha256:" + strings.Repeat("cd", 32)))(t)
	text := `{"result": "evidence:sha256:` + digest + `"} cites 2recipe:sha256:` + strings.Repeat("cd", 32) +
		" and evidence:sha256:" + digest + " again; nothing:sha256:" + digest + " evidence:sha256:abc"
	want := []artifact.ID{evidence, recipe}
	slices.SortFunc(want, artifact.CompareID)
	if got := Identities([]byte(text)); !slices.Equal(got, want) {
		t.Fatalf("Identities = %v, want %v", got, want)
	}
}
