package dataset

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
)

func registerFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "clip-a.mp4"), make([]byte, 4096), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "clip-b.mp4"), make([]byte, 4096), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "captions.json"), []byte(`[{"vid":"clip-a","caption":"a"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	return root
}

// TestRegisterDirectoryDataset pins the new-dataset path: a directory
// registers with a byte-dominant modality, the active catalog carries
// the entry beside existing ones, re-registration is idempotent, and
// coverage reports the dataset available on disk.
func TestRegisterDirectoryDataset(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	legacyRoot, legacyContent := legacyCatalogFixture(t)
	if _, err := PublishLegacyCatalog(ctx, store, legacyRoot, legacyContent); err != nil {
		t.Fatal(err)
	}
	before, err := InspectCatalog(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	root := registerFixture(t)
	registered, err := RegisterDirectoryDataset(ctx, store, "clip-corpus", root)
	if err != nil {
		t.Fatal(err)
	}
	if !registered.Changed || registered.Files != 3 {
		t.Fatalf("registration = %+v", registered)
	}
	coverage, err := InspectCatalog(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if coverage.Registered != before.Registered+1 || !coverage.Complete {
		t.Fatalf("coverage = registered %d complete %t, want the legacy entries plus the new one",
			coverage.Registered, coverage.Complete)
	}
	var entry *CatalogCoverageEntry
	for index := range coverage.Entries {
		if coverage.Entries[index].Entry.Name == "clip-corpus" {
			entry = &coverage.Entries[index]
		}
	}
	if entry == nil || entry.Status != CatalogEntryPublished || !entry.Available {
		t.Fatalf("entry = %+v, want the corpus published and available", entry)
	}
	if entry.Entry.Modality != "video" {
		t.Fatalf("modality = %q, want video to outweigh the caption sidecar", entry.Entry.Modality)
	}
	repeat, err := RegisterDirectoryDataset(ctx, store, "clip-corpus", root)
	if err != nil {
		t.Fatal(err)
	}
	if repeat.Changed {
		t.Fatalf("idempotent re-registration changed the store: %+v", repeat)
	}
}
