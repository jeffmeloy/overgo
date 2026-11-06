package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestArchitectureProfileCatalogRegistryParity(t *testing.T) {
	documents, err := compileArchitectureProfileCatalog()
	testcheck.NoError(t, err)
	names := model.SupportedArchitectures()
	if len(documents) != len(names) {
		t.Fatalf("profiles = %d, registered = %d", len(documents), len(names))
	}
	for index, document := range documents {
		if document.Architecture != names[index] || document.Policy.Name != names[index] {
			t.Fatalf("profile[%d] = %q, registered = %q", index, document.Architecture, names[index])
		}
	}
}

func TestPublishArchitectureProfileCatalogIsIdempotent(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)

	empty, err := InspectArchitectureProfileCatalog(ctx, store)
	testcheck.NoError(t, err)
	if empty.Registered == 0 || empty.Published != 0 || empty.Complete {
		t.Fatalf("empty coverage = %+v", empty)
	}
	if _, err := ResolveRegisteredArchitectureProfile(ctx, store, empty.Entries[0].Architecture); err == nil {
		t.Fatal("missing registered profile resolved")
	}
	first, err := PublishArchitectureProfileCatalog(ctx, store)
	testcheck.NoError(t, err)
	if !first.Changed || !first.Coverage.Complete || first.Coverage.Published != first.Coverage.Registered {
		t.Fatalf("first publication = %+v", first)
	}
	resolved, err := ResolveRegisteredArchitectureProfile(ctx, store, first.Coverage.Entries[0].Architecture)
	testcheck.NoError(t, err)
	if resolved.ID != first.Coverage.Entries[0].Expected {
		t.Fatalf("resolved profile = %s, expected %s", resolved.ID, first.Coverage.Entries[0].Expected)
	}
	_, sequence := store.Head()
	testcheck.NoError(t, store.Close())
	store = overgodbtest.Create(t, root)
	second, err := PublishArchitectureProfileCatalog(ctx, store)
	testcheck.NoError(t, err)
	_, repeatedSequence := store.Head()
	if second.Changed || second.Commit != first.Commit || repeatedSequence != sequence {
		t.Fatalf("repeated publication = %+v, sequence %d -> %d", second, sequence, repeatedSequence)
	}
}

func TestPublishArchitectureProfileCatalogSupersedesStaleAlias(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	documents, err := compileArchitectureProfileCatalog()
	testcheck.NoError(t, err)
	stale, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindProfile, "test/stale-profile/v1"),
		map[string]string{"state": "stale"},
	)
	testcheck.NoError(t, err)
	alias := registeredProfileAlias(documents[0].Architecture)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/profile-catalog/stale", Contents: []artifact.Content{stale},
		Aliases: []artifact.AliasBinding{{Name: alias, Target: stale.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	publication, err := PublishArchitectureProfileCatalog(ctx, store)
	testcheck.NoError(t, err)
	if !publication.Changed || !publication.Coverage.Complete {
		t.Fatalf("publication = %+v", publication)
	}
	target, found, err := store.ResolveAlias(ctx, alias)
	if err != nil || !found || target != documents[0].ID {
		t.Fatalf("alias target = %s, found=%t, err=%v", target, found, err)
	}
}
