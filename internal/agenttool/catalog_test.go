package agenttool

import (
	"testing"

	"overgo/internal/overgodb"
)

func testManuals(t *testing.T) []Manual {
	t.Helper()
	search, err := NewManual(validManual())
	if err != nil {
		t.Fatal(err)
	}
	write := Manual{
		Name:        "repo.write",
		Description: "Write one file under the workspace root.",
		Effect:      EffectMutation,
		Arguments: []Field{
			{Name: "path", Kind: FieldString, Required: true},
			{Name: "content", Kind: FieldString, Required: true},
		},
		Transport: Transport{Kind: TransportBuiltin},
	}
	mutation, err := NewManual(write)
	if err != nil {
		t.Fatal(err)
	}
	return []Manual{search, mutation}
}

func TestPublishManualCatalogIsAtomicAndIdempotent(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manuals := testManuals(t)
	empty, err := InspectManualCatalog(ctx, store, manuals)
	if err != nil {
		t.Fatal(err)
	}
	if empty.Complete || empty.Published != 0 || empty.Registered != len(manuals) {
		t.Fatalf("empty coverage = %+v", empty)
	}
	published, err := PublishManualCatalog(ctx, store, manuals)
	if err != nil {
		t.Fatal(err)
	}
	if !published.Changed || !published.Coverage.Complete || published.Coverage.Published != len(manuals) {
		t.Fatalf("publication = %+v", published)
	}
	repeated, err := PublishManualCatalog(ctx, store, manuals)
	if err != nil {
		t.Fatal(err)
	}
	if repeated.Changed {
		t.Fatalf("repeat publication changed the store: %+v", repeated)
	}
	resolved, err := ResolveRegisteredManual(ctx, store, "repo.write")
	if err != nil {
		t.Fatal(err)
	}
	if resolved.ID != manuals[1].ID || resolved.Effect != EffectMutation {
		t.Fatalf("resolved = %+v, want the published mutation manual", resolved)
	}
	if _, err := ResolveRegisteredManual(ctx, store, "repo.delete"); err == nil {
		t.Fatal("unregistered tool resolved")
	}
}

func TestPublishManualCatalogSupersedesStaleBindings(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manuals := testManuals(t)
	if _, err := PublishManualCatalog(ctx, store, manuals); err != nil {
		t.Fatal(err)
	}
	revised := validManual()
	revised.Description = "Search the repository for a pattern, bounded."
	updated, err := NewManual(revised)
	if err != nil {
		t.Fatal(err)
	}
	manuals[0] = updated
	stale, err := InspectManualCatalog(ctx, store, manuals)
	if err != nil {
		t.Fatal(err)
	}
	if stale.Complete || stale.Entries[0].Status != CatalogEntryMismatched {
		t.Fatalf("stale coverage = %+v, want the revised manual mismatched", stale)
	}
	published, err := PublishManualCatalog(ctx, store, manuals)
	if err != nil {
		t.Fatal(err)
	}
	if !published.Changed || !published.Coverage.Complete {
		t.Fatalf("supersede publication = %+v", published)
	}
	resolved, err := ResolveRegisteredManual(ctx, store, "repo.search")
	if err != nil {
		t.Fatal(err)
	}
	if resolved.ID != updated.ID {
		t.Fatalf("resolved %s, want the superseding manual %s", resolved.ID, updated.ID)
	}
}

func TestPublishManualCatalogRefusesDuplicatesAndEmpty(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := PublishManualCatalog(ctx, store, nil); err == nil {
		t.Fatal("empty catalog published")
	}
	manuals := testManuals(t)
	if _, err := PublishManualCatalog(ctx, store, append(manuals, manuals[0])); err == nil {
		t.Fatal("duplicate manual published")
	}
}
