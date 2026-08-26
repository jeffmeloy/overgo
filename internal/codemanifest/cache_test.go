package codemanifest

import (
	"slices"
	"testing"
)

func TestCacheIdentity(t *testing.T) {
	manifest := cacheManifestFixture(t, "package example\nfunc Value() int { return 1 }\n")
	key := CacheKey{manifest.SourceIdentity, manifest.Analyzer, Schema, manifest.BuildContexts, manifest.ExternalInputs}
	cache, _ := NewCache(2)
	if err := cache.put(key, manifest); err != nil {
		t.Fatal(err)
	}
	loaded, found := cache.get(key)
	if !found || loaded.ID != manifest.ID {
		t.Fatalf("cache load = %s, %t", loaded.ID, found)
	}
}

func TestCacheCanonicalizesAuthorityBeforeComparison(t *testing.T) {
	value := fixtureManifest()
	value.ExternalInputs = append(value.ExternalInputs, ExternalInput{
		Path: "architecture_profiles.json", ContentID: fixtureDigest,
		Kind: "architecture-profiles", Owner: "internal/modelrecipe",
	})
	manifest, err := codec.New(value)
	if err != nil {
		t.Fatal(err)
	}
	key := CacheKey{manifest.SourceIdentity, manifest.Analyzer, Schema,
		slices.Clone(manifest.BuildContexts), slices.Clone(manifest.ExternalInputs)}
	slices.Reverse(key.BuildContexts)
	slices.Reverse(key.ExternalInputs)
	cache, _ := NewCache(2)
	if err := cache.put(key, manifest); err != nil {
		t.Fatalf("canonical authority rejected: %v", err)
	}
	if loaded, found := cache.get(key); !found || loaded.ID != manifest.ID {
		t.Fatalf("canonical cache load = %s, %t", loaded.ID, found)
	}
}

func TestCacheInvalidation(t *testing.T) {
	manifest := cacheManifestFixture(t, "package example\nfunc Value() int { return 1 }\n")
	key := CacheKey{manifest.SourceIdentity, manifest.Analyzer, Schema, manifest.BuildContexts, manifest.ExternalInputs}
	cache, _ := NewCache(2)
	if err := cache.put(key, manifest); err != nil {
		t.Fatal(err)
	}
	key.Analyzer.Version += "-changed"
	if _, found := cache.get(key); found {
		t.Fatal("changed analyzer reused a manifest")
	}
}

func TestSchemaChangeInvalidatesCache(t *testing.T) {
	manifest := cacheManifestFixture(t, "package example\nfunc Value() int { return 1 }\n")
	key := CacheKey{manifest.SourceIdentity, manifest.Analyzer, Schema, manifest.BuildContexts, manifest.ExternalInputs}
	cache, _ := NewCache(2)
	if err := cache.put(key, manifest); err != nil {
		t.Fatal(err)
	}
	key.Schema += "-next"
	if _, found := cache.get(key); found {
		t.Fatal("changed schema reused a manifest")
	}
}

func TestBoundedEviction(t *testing.T) {
	cache, _ := NewCache(1)
	first := cacheManifestFixture(t, "package example\nfunc Value() int { return 1 }\n")
	second := cacheManifestFixture(t, "package example\nfunc Value() int { return 2 }\n")
	firstKey := CacheKey{first.SourceIdentity, first.Analyzer, Schema, first.BuildContexts, first.ExternalInputs}
	secondKey := CacheKey{second.SourceIdentity, second.Analyzer, Schema, second.BuildContexts, second.ExternalInputs}
	if err := cache.put(firstKey, first); err != nil {
		t.Fatal(err)
	}
	if err := cache.put(secondKey, second); err != nil {
		t.Fatal(err)
	}
	if len(cache.entries) != 1 {
		t.Fatalf("cache length = %d", len(cache.entries))
	}
	if _, found := cache.get(firstKey); found {
		t.Fatal("least-recently used entry survived bounded eviction")
	}
}

func cacheManifestFixture(t *testing.T, source string) Manifest {
	t.Helper()
	root := t.TempDir()
	name := "internal/example/example.go"
	writeGeneratorFixture(t, root, name, source)
	return generateSingleFile(t, root, name)
}
