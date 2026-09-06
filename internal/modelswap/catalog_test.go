package modelswap

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/testutil"
)

// TestMatchServableRoutesRemoteLocations: proxy remote route. Remote entry
// matches by the location's last segment (case-insensitive) or the model
// id; served reference = the remote location (server resolves via relay).
func TestMatchServableRoutesRemoteLocations(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "remote-swap")
	entries := []discovery.CatalogEntry{
		{Model: modelID, Location: "remote://fake/vendor/relay-model", Present: true},
		{Model: testutil.ArtifactID(t, artifact.KindModel, "absent"), Location: "remote://fake/vendor/absent", Present: false},
	}
	for _, name := range []string{"relay-model", "RELAY-MODEL", modelID.String()} {
		servable, found := matchServable(entries, name)
		if !found || servable.Location != "remote://fake/vendor/relay-model" || servable.Name != "relay-model" || servable.Model != modelID.String() {
			t.Fatalf("match %q = %+v found=%v", name, servable, found)
		}
	}
	if _, found := matchServable(entries, "absent"); found {
		t.Fatal("an absent remote entry matched")
	}
}
