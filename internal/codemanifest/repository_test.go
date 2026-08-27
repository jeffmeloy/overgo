package codemanifest

import (
	"context"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

func TestPublishAndLoad(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/example.go"
	writeGeneratorFixture(t, root, name, "package example\nfunc Value() int { return 1 }\n")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := Generate(snapshot, []repoanalysis.BuildSelection{{
		Context: "linux/amd64", Root: root, Files: map[string]bool{name: true}, Packages: map[string]string{name: "overgo/internal/example"},
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := Publish(context.Background(), store, manifest); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(context.Background(), store, manifest.ID)
	if err != nil || loaded.ID != manifest.ID || loaded.SourceIdentity != snapshot.Identity() {
		t.Fatalf("loaded = %+v, %v", loaded, err)
	}
}

// TestDigestPublicationKeepsClaimNotCache pins the store diet: a gate
// run records the manifest's digest, not its derived content, and the
// digest round-trips by manifest identity.
func TestDigestPublicationKeepsClaimNotCache(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/example.go"
	writeGeneratorFixture(t, root, name, "package example\nfunc Value() int { return 2 }\n")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := Generate(snapshot, []repoanalysis.BuildSelection{{
		Context: "linux/amd64", Root: root, Files: map[string]bool{name: true}, Packages: map[string]string{name: "overgo/internal/example"},
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := PublishDigest(context.Background(), store, manifest); err != nil {
		t.Fatal(err)
	}
	// Idempotent republication must not error.
	if _, err := PublishDigest(context.Background(), store, manifest); err != nil {
		t.Fatal(err)
	}
	digest, err := LoadDigest(context.Background(), store, manifest.ID)
	if err != nil || digest.ID != manifest.ID || digest.SourceIdentity != snapshot.Identity() || digest.Files == 0 {
		t.Fatalf("digest = %+v, %v", digest, err)
	}
	// The cache itself must be absent: only the claim is durable.
	if _, err := Load(context.Background(), store, manifest.ID); err == nil {
		t.Fatal("digest publication stored the full manifest content")
	}
}
