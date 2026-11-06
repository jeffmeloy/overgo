package agenttool

import (
	"testing"

	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func testManuals(t *testing.T) []Manual {
	t.Helper()
	search := testcheck.Must(NewManual(validManual()))(t)
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
	mutation := testcheck.Must(NewManual(write))(t)
	return []Manual{search, mutation}
}

func TestPublishManualCatalogIsAtomicAndIdempotent(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	manuals := testManuals(t)
	empty := testcheck.Must(InspectManualCatalog(ctx, store, manuals))(t)
	if empty.Complete || empty.Published != 0 || empty.Registered != len(manuals) {
		t.Fatalf("empty coverage = %+v", empty)
	}
	published := testcheck.Must(PublishManualCatalog(ctx, store, manuals))(t)
	if !published.Changed || !published.Coverage.Complete || published.Coverage.Published != len(manuals) {
		t.Fatalf("publication = %+v", published)
	}
	repeated := testcheck.Must(PublishManualCatalog(ctx, store, manuals))(t)
	if repeated.Changed {
		t.Fatalf("repeat publication changed the store: %+v", repeated)
	}
	resolved := testcheck.Must(ResolveRegisteredManual(ctx, store, "repo.write"))(t)
	if resolved.ID != manuals[1].ID || resolved.Effect != EffectMutation {
		t.Fatalf("resolved = %+v, want the published mutation manual", resolved)
	}
	if _, err := ResolveRegisteredManual(ctx, store, "repo.delete"); err == nil {
		t.Fatal("unregistered tool resolved")
	}
}

func TestPublishManualCatalogSupersedesStaleBindings(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	manuals := testManuals(t)
	if _, err := PublishManualCatalog(ctx, store, manuals); err != nil {
		t.Fatal(err)
	}
	revised := validManual()
	revised.Description = "Search the repository for a pattern, bounded."
	updated := testcheck.Must(NewManual(revised))(t)
	manuals[0] = updated
	stale := testcheck.Must(InspectManualCatalog(ctx, store, manuals))(t)
	if stale.Complete || stale.Entries[0].Status != CatalogEntryMismatched {
		t.Fatalf("stale coverage = %+v, want the revised manual mismatched", stale)
	}
	published := testcheck.Must(PublishManualCatalog(ctx, store, manuals))(t)
	if !published.Changed || !published.Coverage.Complete {
		t.Fatalf("supersede publication = %+v", published)
	}
	resolved := testcheck.Must(ResolveRegisteredManual(ctx, store, "repo.search"))(t)
	if resolved.ID != updated.ID {
		t.Fatalf("resolved %s, want the superseding manual %s", resolved.ID, updated.ID)
	}
}

func TestPublishManualCatalogRefusesDuplicatesAndEmpty(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	if _, err := PublishManualCatalog(ctx, store, nil); err == nil {
		t.Fatal("empty catalog published")
	}
	manuals := testManuals(t)
	if _, err := PublishManualCatalog(ctx, store, append(manuals, manuals[0])); err == nil {
		t.Fatal("duplicate manual published")
	}
}
