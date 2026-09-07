package modelswap

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestCatalogRefreshPreservesMemo(t *testing.T) {
	root := t.TempDir()
	writer, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	resolver := CatalogResolver{Store: root}
	reader, err := resolver.open(t.Context(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	memo := resolver.memo
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(path, []byte("initial weights"), 0600); err != nil {
		t.Fatal(err)
	}
	before, err := discovery.WeightsIdentity(path, memo)
	if err != nil {
		t.Fatal(err)
	}
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/catalog-refresh-test/v1"), struct{ Value string }{"new declaration"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Commit(t.Context(), artifact.Batch{Key: "catalog/refresh", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	updated, err := resolver.open(t.Context(), true)
	if err != nil || updated != reader || resolver.memo != memo {
		t.Fatalf("refresh replaced resident state: reader=%t memo=%t error=%v", updated == reader, resolver.memo == memo, err)
	}
	if _, found, err := updated.Artifact(t.Context(), content.Descriptor.ID); err != nil || !found {
		t.Fatalf("new committed declaration missing: found=%t error=%v", found, err)
	}
	if same, err := discovery.WeightsIdentity(path, resolver.memo); err != nil || same != before {
		t.Fatalf("unchanged identity=%s want=%s error=%v", same, before, err)
	}
	if err := os.WriteFile(path, []byte("changed and longer weights"), 0600); err != nil {
		t.Fatal(err)
	}
	if after, err := discovery.WeightsIdentity(path, resolver.memo); err != nil || after == before {
		t.Fatalf("changed bytes reused stale identity: %s error=%v", after, err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := resolver.open(ctx, true); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled refresh: %v", err)
	}
}

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
