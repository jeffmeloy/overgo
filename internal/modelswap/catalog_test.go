package modelswap

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

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
	reader, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	resolver := CatalogResolver{Store: reader, Limit: 100}
	if _, _, err := resolver.Catalog(t.Context()); err != nil {
		t.Fatal(err)
	}
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
	_, _, err = resolver.Catalog(t.Context())
	if err != nil || resolver.Store != reader || resolver.memo != memo {
		t.Fatalf("refresh replaced resident state: reader=%t memo=%t error=%v", resolver.Store == reader, resolver.memo == memo, err)
	}
	if _, found, err := reader.Artifact(t.Context(), content.Descriptor.ID); err != nil || !found {
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
	if _, _, err := resolver.Catalog(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled refresh: %v", err)
	}
	var unbound CatalogResolver
	if err := unbound.refresh(t.Context()); err == nil {
		t.Fatal("missing borrowed store accepted")
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
		servable, found, err := matchServable(entries, false, name)
		if err != nil || !found || servable.Location != "remote://fake/vendor/relay-model" || servable.Name != "relay-model" || servable.Model != modelID.String() {
			t.Fatalf("match %q = %+v found=%v", name, servable, found)
		}
	}
	if _, found, err := matchServable(entries, false, "absent"); err != nil || found {
		t.Fatal("an absent remote entry matched")
	}
}

func TestCatalogProgressNamesStagesAndKeepsSlowest(t *testing.T) {
	root := t.TempDir()
	writer, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	reader, err := overgodb.OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	resolver := CatalogResolver{Store: reader, Limit: 100}
	if progress := resolver.Progress(); progress.Stage != "" || progress.Waiting != 0 || len(progress.Slowest) != 0 {
		t.Fatalf("idle resolver reports %+v", progress)
	}
	if _, _, err := resolver.Catalog(t.Context()); err != nil {
		t.Fatal(err)
	}
	progress := resolver.Progress()
	if progress.Stage != "" || progress.Waiting != 0 {
		t.Fatalf("resolver still inside a stage after the catalog: %+v", progress)
	}
	for _, stage := range []string{"refresh", "memo", "catalog"} {
		if _, ran := progress.Slowest[stage]; !ran {
			t.Fatalf("stage %s not timed: %+v", stage, progress)
		}
	}
	// A caller inside a stage is reported with its wait, so a stalled
	// picker names the stage it waits behind.
	holding := make(chan struct{})
	release := make(chan struct{})
	go func() {
		defer resolver.acquire()()
		defer resolver.begin("catalog")()
		close(holding)
		<-release
	}()
	<-holding
	waiting := make(chan struct{})
	go func() {
		defer resolver.acquire()()
		close(waiting)
	}()
	for resolver.Progress().Waiting == 0 {
		time.Sleep(time.Millisecond)
	}
	if progress := resolver.Progress(); progress.Stage != "catalog" || progress.Waiting != 1 || progress.Elapsed <= 0 {
		t.Fatalf("held resolver reports %+v", progress)
	}
	close(release)
	<-waiting
}
