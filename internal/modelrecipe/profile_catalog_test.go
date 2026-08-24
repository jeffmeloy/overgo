package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/overgodb"
)

func TestArchitectureProfileCatalogRegistryParity(t *testing.T) {
	documents, err := CompileArchitectureProfileCatalog()
	if err != nil {
		t.Fatal(err)
	}
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
	ctx := context.Background()
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}

	empty, err := InspectArchitectureProfileCatalog(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if empty.Registered == 0 || empty.Published != 0 || empty.Complete {
		t.Fatalf("empty coverage = %+v", empty)
	}
	if _, err := ResolveRegisteredArchitectureProfile(ctx, store, empty.Entries[0].Architecture); err == nil {
		t.Fatal("missing registered profile resolved")
	}
	first, err := PublishArchitectureProfileCatalog(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if !first.Changed || !first.Coverage.Complete || first.Coverage.Published != first.Coverage.Registered {
		t.Fatalf("first publication = %+v", first)
	}
	resolved, err := ResolveRegisteredArchitectureProfile(ctx, store, first.Coverage.Entries[0].Architecture)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.ID != first.Coverage.Entries[0].Expected {
		t.Fatalf("resolved profile = %s, expected %s", resolved.ID, first.Coverage.Entries[0].Expected)
	}
	_, sequence := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	second, err := PublishArchitectureProfileCatalog(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	_, repeatedSequence := store.Head()
	if second.Changed || second.Commit != first.Commit || repeatedSequence != sequence {
		t.Fatalf("repeated publication = %+v, sequence %d -> %d", second, sequence, repeatedSequence)
	}
}

func TestPublishArchitectureProfileCatalogSupersedesStaleAlias(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	documents, err := CompileArchitectureProfileCatalog()
	if err != nil {
		t.Fatal(err)
	}
	stale, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindProfile, "test/stale-profile/v1"),
		map[string]string{"state": "stale"},
	)
	if err != nil {
		t.Fatal(err)
	}
	alias := registeredProfileAlias(documents[0].Architecture)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/profile-catalog/stale", Contents: []artifact.Content{stale},
		Aliases: []artifact.AliasBinding{{Name: alias, Target: stale.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	publication, err := PublishArchitectureProfileCatalog(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if !publication.Changed || !publication.Coverage.Complete {
		t.Fatalf("publication = %+v", publication)
	}
	target, found, err := store.ResolveAlias(ctx, alias)
	if err != nil || !found || target != documents[0].ID {
		t.Fatalf("alias target = %s, found=%t, err=%v", target, found, err)
	}
}
