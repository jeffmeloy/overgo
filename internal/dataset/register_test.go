package dataset

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
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
// registers with its exact modality set (mixed here), the active
// catalog carries the entry beside existing ones, re-registration is
// idempotent, and coverage reports the dataset available on disk.
func TestRegisterDirectoryDataset(t *testing.T) {
	ctx := t.Context()
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
	if entry.Entry.Modality != MixedModality || !slices.Equal(entry.Entry.Modalities, []string{"structured", "video"}) {
		t.Fatalf("modality = %q set = %v, want a mixed corpus with the exact set, not a byte-volume winner",
			entry.Entry.Modality, entry.Entry.Modalities)
	}
	repeat, err := RegisterDirectoryDataset(ctx, store, "clip-corpus", root)
	if err != nil {
		t.Fatal(err)
	}
	if repeat.Changed {
		t.Fatalf("idempotent re-registration changed the store: %+v", repeat)
	}
}

// TestRegisterContentIdentity pins the reopened finding: two files
// with identical size and modification time but different bytes must
// produce different dataset identities, because identity follows a
// content digest, not path plus metadata.
func TestRegisterContentIdentity(t *testing.T) {
	ctx := t.Context()
	build := func(payload []byte) artifact.ID {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		root := t.TempDir()
		path := filepath.Join(root, "clip.mp4")
		if err := os.WriteFile(path, payload, 0o600); err != nil {
			t.Fatal(err)
		}
		fixed := time.Unix(1700000000, 0)
		if err := os.Chtimes(path, fixed, fixed); err != nil {
			t.Fatal(err)
		}
		registered, err := RegisterDirectoryDataset(ctx, store, "clip-corpus", root)
		if err != nil {
			t.Fatal(err)
		}
		return registered.Dataset
	}
	first := build(bytes.Repeat([]byte{0xAA}, 4096))
	second := build(bytes.Repeat([]byte{0xBB}, 4096))
	if first == second {
		t.Fatal("changed content with identical metadata kept the same dataset identity")
	}
}

// TestRegisterTouchDoesNotReidentify pins the other half of content
// identity: bumping an unchanged file's modification time must leave
// the dataset identity untouched -- re-registration after a touch is
// an idempotent no-op, because identity follows bytes alone.
func TestRegisterTouchDoesNotReidentify(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	root := t.TempDir()
	path := filepath.Join(root, "clip.mp4")
	if err := os.WriteFile(path, bytes.Repeat([]byte{0xCC}, 4096), 0o600); err != nil {
		t.Fatal(err)
	}
	first, err := RegisterDirectoryDataset(ctx, store, "clip-corpus", root)
	if err != nil {
		t.Fatal(err)
	}
	touched := time.Unix(1800000000, 0)
	if err := os.Chtimes(path, touched, touched); err != nil {
		t.Fatal(err)
	}
	second, err := RegisterDirectoryDataset(ctx, store, "clip-corpus", root)
	if err != nil {
		t.Fatal(err)
	}
	if second.Changed || second.Dataset != first.Dataset {
		t.Fatalf("touch re-identified: first=%s second=%s changed=%t", first.Dataset, second.Dataset, second.Changed)
	}
}

// TestRegisterModalitySetAndDeclaration pins the modality-set contract:
// a single-modality corpus reports that modality, a declaration selects
// the primary when present and is refused when absent, and the primary
// of an undeclared mixed corpus is "mixed", never a byte-volume winner.
func TestRegisterModalitySetAndDeclaration(t *testing.T) {
	ctx := t.Context()
	mixedCorpus := func() string {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "a.mp4"), make([]byte, 2048), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "b.json"), []byte("[]"), 0o600); err != nil {
			t.Fatal(err)
		}
		return root
	}
	primaryOf := func(store *overgodb.Store, name string) CatalogEntry {
		catalog, found, err := ResolveCatalog(ctx, store)
		if err != nil || !found {
			t.Fatalf("catalog = (%t, %v)", found, err)
		}
		for _, entry := range catalog.Catalog {
			if entry.Name == name {
				return entry
			}
		}
		t.Fatalf("entry %q absent", name)
		return CatalogEntry{}
	}

	declaredStore, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer declaredStore.Close()
	if _, err := RegisterDirectoryDatasetAs(ctx, declaredStore, "corpus", mixedCorpus(), "video"); err != nil {
		t.Fatal(err)
	}
	if entry := primaryOf(declaredStore, "corpus"); entry.Modality != "video" ||
		!slices.Equal(entry.Modalities, []string{"structured", "video"}) {
		t.Fatalf("declared entry = %+v, want video primary over the full set", entry)
	}
	if _, err := RegisterDirectoryDatasetAs(ctx, declaredStore, "corpus", mixedCorpus(), "audio"); err == nil {
		t.Fatal("declared modality absent from the corpus was accepted")
	}

	undeclaredStore, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer undeclaredStore.Close()
	if _, err := RegisterDirectoryDataset(ctx, undeclaredStore, "corpus", mixedCorpus()); err != nil {
		t.Fatal(err)
	}
	if entry := primaryOf(undeclaredStore, "corpus"); entry.Modality != MixedModality {
		t.Fatalf("undeclared mixed primary = %q, want %q", entry.Modality, MixedModality)
	}
	single := t.TempDir()
	if err := os.WriteFile(filepath.Join(single, "only.mp4"), make([]byte, 1024), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := RegisterDirectoryDataset(ctx, undeclaredStore, "single", single); err != nil {
		t.Fatal(err)
	}
	if entry := primaryOf(undeclaredStore, "single"); entry.Modality != "video" ||
		!slices.Equal(entry.Modalities, []string{"video"}) {
		t.Fatalf("single-modality entry = %+v, want video primary and set", entry)
	}
}
