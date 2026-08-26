package discovery

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestIdentityMemoPersistence pins the persisted digest memo: identities
// hashed once publish to the store, a fresh memo loaded from the store
// answers from stat checks without re-hashing, a changed file misses so
// the claim stays exact, and an unchanged memo does not republish.
func TestIdentityMemoPersistence(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	path := filepath.Join(t.TempDir(), "model.bin")
	payload := []byte("model bytes that would normally cost a full hash pass")
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	digest, size, err := artifact.Identify(artifact.KindModel, file)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}

	hashed := NewMemo()
	key := path + "\x00" + artifact.KindModel.String()
	hashed.record(key, info, fileIdentity{id: digest, size: size, present: true})
	if err := PublishMemo(ctx, store, hashed); err != nil {
		t.Fatal(err)
	}

	// A fresh memo loaded from the store serves the identity from the
	// stat match alone -- the re-hash never happens.
	loaded := LoadMemo(ctx, store)
	identity, hit := loaded.lookup(key, info)
	if !hit || identity.id != digest || identity.size != size || !identity.present {
		t.Fatalf("loaded lookup = (%+v, %t), want the persisted identity", identity, hit)
	}

	// Loading alone leaves nothing to publish; the alias must not move.
	before, bound, err := artifact.ResolveAlias(ctx, store, IdentityEvidenceAlias)
	if err != nil || !bound {
		t.Fatalf("alias after publish = (%v, %t, %v)", before, bound, err)
	}
	if err := PublishMemo(ctx, store, loaded); err != nil {
		t.Fatal(err)
	}
	after, _, err := artifact.ResolveAlias(ctx, store, IdentityEvidenceAlias)
	if err != nil || after != before {
		t.Fatalf("clean memo republished: %v -> %v (%v)", before, after, err)
	}

	// A file whose bytes changed misses: the persisted claim is bound to
	// the stat under which it was hashed.
	if err := os.WriteFile(path, append(payload, " grown"...), 0o644); err != nil {
		t.Fatal(err)
	}
	changed, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, hit := loaded.lookup(key, changed); hit {
		t.Fatal("changed file served a stale identity")
	}

	// New identities recorded on a loaded memo publish under a moved alias.
	if err := os.Chtimes(path, time.Now(), time.Now()); err != nil {
		t.Fatal(err)
	}
	file, err = os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	regrown, regrownSize, err := artifact.Identify(artifact.KindModel, file)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	refreshed, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	loaded.record(key, refreshed, fileIdentity{id: regrown, size: regrownSize, present: true})
	if err := PublishMemo(ctx, store, loaded); err != nil {
		t.Fatal(err)
	}
	final, _, err := artifact.ResolveAlias(ctx, store, IdentityEvidenceAlias)
	if err != nil || final == before {
		t.Fatalf("dirty memo did not republish: %v (%v)", final, err)
	}
	reloaded := LoadMemo(ctx, store)
	if identity, hit := reloaded.lookup(key, refreshed); !hit || identity.id != regrown {
		t.Fatalf("reloaded lookup = (%+v, %t), want the refreshed identity", identity, hit)
	}
}
